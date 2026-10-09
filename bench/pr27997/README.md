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
  | `pr_comb64` | 33.0 KB | 16.5 KB | 34.0 KB | **50.8–54.5 KB** | 1 |
  | `comb64` (Q in regs) | 33.0 KB | 16.5 KB | 34.0 KB | **34.0 KB** | **2** |

  The 66 KB `nbatch_combine` buffer pins every variant with `combine=128` to 1 block/SM, so the
  row's `occupancy=2` is never reached. The PR's speedup must come from the register side: the
  PR reports stack 552 → 56 B at 255 registers, i.e. spills removed by not keeping Q in registers.
- **Untested alternative:** `comb64` is the upstream row with only `nbatch_combine` halved. It is
  the only variant that fits 2 blocks/SM, but it keeps Q in registers (the spilling layout). Which
  of "fewer spills" (`pr`) and "2× occupancy" (`comb64`) wins is a hardware question.
- **Nits:** the PR's row is not column-aligned like its neighbours (CONTRIBUTING asks for vertical
  alignment); only `ncols=64` gets a Volta row, so `ncols=8/16/32` still use Ampere rows.

## Running (on the V100, CUDA 12.8)

```bash
sudo apt-get install -y ccache    # llama.cpp picks it up; variants 2-4 then rebuild only the FA files
bench/pr27997/run.sh                                         # kernel-level A/B, ~4 builds
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
