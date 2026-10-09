# PR #27997 A/B harness (Volta, D=256 prefill config)

Upstream: https://github.com/ggml-org/llama.cpp/pull/27997 (draft, by mistrjirka). It adds a
Volta-only row for `DKQ=DV=256, ncols=64` instead of falling back to the Ampere row.

## Review of the change (code reading, 2026-10-09, llama.cpp 79e2e74)

| field | Ampere fallback | PR |
| --- | --- | --- |
| nbatch_V2 | 128 | 64 |
| Q_in_reg | true | false |
| everything else | 128 threads, occupancy 2, nbatch_fa 32, nbatch_K2 128, combine 128, nstages 2 | same |

- **Valid.** Passes the macro static asserts and the kernel's loop constraints
  (`DV % (2*nbatch_V2) == 0`, `% 32` checks). Host and device pick the same Volta config
  (`fattn-mma-f16.cuh:234-266`). `nstages_target=2` has no effect on Volta: no `cp.async`, so
  `ggml_cuda_fattn_mma_get_nstages` returns 0 and the single-stage KV path is used.
- **The gain is not occupancy.** Shared memory per block (`fattn-mma-f16.cuh:2033-2043`,
  Volta: `cols_per_warp=32`, no swizzle, single stage):

  | variant | Q | KV | combine | total | blocks/SM (96 KB) |
  | --- | --- | --- | --- | --- | --- |
  | upstream (Q in regs) | 33.0 KB | 16.5 KB | 66.0 KB | **66.0 KB** | 1 |
  | `pr` (Q in smem) | 33.0 KB | 16.5 KB | 66.0 KB | **66.0 KB** | 1 |
  | `pr_comb64` | 33.0 KB | 16.5 KB | 34.0 KB | **50.1–54.5 KB** | 1 |
  | `comb64` (Q in regs) | 33.0 KB | 16.5 KB | 34.0 KB | **34.0 KB** | **2** |

  The 66 KB `nbatch_combine` buffer pins every variant with `combine=128` to 1 block/SM, so the
  row's `occupancy=2` is never reached. The PR's speedup must come from the register side: the
  PR reports stack 552 → 56 B at 255 registers, i.e. spills removed by not keeping Q in registers.
- **Untested alternatives:** `comb64` is the upstream row with only `nbatch_combine` halved: it fits
  2 blocks/SM but keeps Q in registers (the spilling layout). `pr_k64_c64` is the PR row with
  `nbatch_K2` and `nbatch_combine` also halved: the K tile shrinks enough that Q-in-smem + K/V + mask
  is 42.1–46.5 KB, so it is the only variant with **both** low spills and 2 blocks/SM, at the cost of
  more K-load iterations per tile. Which wins is a hardware question; the harness times all of them.

## Static results (real compile, no GPU)

nvcc 12.8.93 (conda-forge, `WITH_NVCC=1 bin/setup-dev.sh`), `sm_70`, llama.cpp 79e2e74, upstream's
own flags (`-O3 -use_fast_math -extended-lambda`), the four `ncols=64` instance files
(`fattn-mma-f16-instance-ncols1_{64,32,16,8}-ncols2_{1,2,4,8}.cu`); `cuobjdump -res-usage` on the nine
D=256 kernels they contain. Shared memory is the launch-time formula (`fattn-mma-f16.cuh:2033-2043`),
range over ncols1.

| variant | registers | stack (spill) | smem / block | blocks/SM (smem, regs) |
| --- | --- | --- | --- | --- |
| upstream | 255 | 464–640 B | 66.0 KB | 1 |
| `pr` (#27997) | 253–255 | 48–112 B | 66.0 KB | 1 |
| `comb64` | 255 | 464–640 B | 34.0 KB | 2 |
| `pr_comb64` | 254–255 | 48–112 B | 50.1–54.5 KB | 1 |
| `pr_k64_c64` | 252–255 | 48–160 B | 42.1–46.5 KB | 2 |

Artifact: `bench/results/pr27997-static-20261009-182401/` (summary, per-variant `cuobjdump` output, and a head-dim survey of
upstream's other ncols=64 kernels: D=512 also carries 544–568 B of stack on Volta).

This confirms the PR's register claim on every D=256 ncols=64 instance (it reported one kernel:
552 → 56 B). Two blocks/SM at 255 registers × 128 threads uses exactly the 64K-register file.

- **Nits:** the PR's row is not column-aligned like its neighbours (CONTRIBUTING asks for vertical
  alignment); only `ncols=64` gets a Volta row, so `ncols=8/16/32` still use Ampere rows.

## Running (on the V100, CUDA 12.8)

```bash
sudo apt-get install -y ccache    # llama.cpp picks it up; variants 2-4 then rebuild only the FA files
bench/pr27997/run.sh                                         # kernel-level A/B: 5 builds, 3 interleaved timing rounds
MODEL=/models/<d256-model>.gguf bench/pr27997/run.sh         # + llama-bench pp1024 at depth 4k/16k/64k
```

Per variant it records: the exact patch, `cuobjdump -res-usage` for the D=256 kernels (registers,
stack), `test-backend-ops test` for all `hsk=256` cases (correctness), and `test-backend-ops perf`
for local-only D=256 prefill cases (nb=512, kv 4k/16k/64k, GQA 2 and 4). The extra perf cases are
added to the variant checkouts only; nothing here is for upstream.

## Posting results

llama.cpp requires disclosure of AI use and bans AI-written PR text, comments and replies
(CONTRIBUTING.md "AI Usage Policy", AGENTS.md). Any comment on #27997 must be written by the
owner in their own words, with AI use disclosed as their template asks. This harness only
produces the numbers.
