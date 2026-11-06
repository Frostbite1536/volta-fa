# Hyperkernel: design ideas that transfer to Volta (desk research, 2026-11-06)

Desk research only: nothing here was run on a V100 or any other GPU. Every speed effect below is "not run".

Source: `Frostbite1536/Hyperkernel` at `14d1e14` (2026-11-05, MIT). Shallow clone, read only. The kernel
metadata quoted here comes from decompressing the `kernels/b300/c1/` bundle (`reference.json`, `rank0.json`,
`entry.sass`). The llama.cpp claim was checked against upstream `master` (`e14a779`, 2026-11-06),
not our pinned commit.

## What the repository is

"Hyperserve" is a Rust host runtime (~9.7K lines) for **closed, precompiled megakernels for NVIDIA B300**
(Blackwell). It serves DeepSeek-V4.1-Flash on 4 GPUs, plus openpilot, Evo 2 and Matrix-Game 3.0. There is **no
kernel source**: `kernels/b300/*` holds zstd-compressed cubins, their SASS listings and load-time data. There is
nothing to port, and the hard rules would block most of it anyway (see the last section). The value is in the
design, which the metadata and SASS show:

| feature | evidence (`c1` preset) |
| --- | --- |
| one persistent kernel for the whole model | `kernel.entries = [mega_main]`, `cooperative: true`, 384 threads, 232,448 B dynamic smem, `cluster: 16`, 7,200 B stack |
| stages synchronised by memory counters, not kernel boundaries | `spine_wait` called from 551 sites, `spine_land` from 601, `spine_done` from 140 in `entry.sass` |
| warp-specialised dense matmul | `DnCommon<…>::role_copy`, `role_mma`, `role_init`, `role_fini` |
| weights rewritten into tile layout once, at load | 9 loader kernels: `hd_pretile`, `pt_codes`, `pt_codes4`, `pt_scales`, `pt_mx_codes`, `pt_mx_scales`, `pt_wide`, `sp_copy`, `tma_map_at`. 1,017 of 6,934 memory homes are `staged` |
| GPUs write straight into each other's memory | `src/gpu/ranks.rs`: each rank's arena is exported and mapped into one symmetric VA range (`base + rank × stride`); a `fx_peers` table (64 B per peer); NVLink multicast object |
| host ↔ running kernel through polled rings | `src/gpu/ring.rs`: request/reply rings in mapped pinned host memory, head/tail words at offsets 0/64; the host grants work through a control word (`feed.rs`, `ctl.word(8)`) |
| speculative decoding and sampling on the GPU | `model.block = 6`, `drafting` request message, `spine::topk_run`, 33 logprob slots per token |
| paged KV with a kept sliding window | `pool.page_rows = 256`, `window_keep = 2048` |

## Ideas that transfer, ranked

### 1. Repack quantised weights once at load (P-9 option (b))

Hyperkernel is a second independent example, after the llama-superfast-v100 fork's QPN matmul, of the same
choice: a one-off load-time pass rewrites quantised weights (codes and scales separately) into the order the matmul
consumes. Our own measurements already point the same way (`docs/design/P9-small-batch-qmatmul.md`,
"Phase 2, third part"):
- loads: 801 GB/s tile-contiguous vs 685 GB/s for the native staged layout;
- the remaining gap: 46.2 ms vs the fork's 32.8 ms per 16-token Qwen3.8-27B evaluation.

This is the largest measured gap still open. It stays an **owner decision**: Hyperkernel shows the technique is
the common answer, but it says nothing about whether llama.cpp would accept a load-time repack in the CUDA backend.

### 2. One persistent kernel with counters instead of kernel boundaries

- **What Volta has:** cooperative launch (sm_60+) and flag counters using sm_70's scoped acquire/release memory
  operations (`ld.acquire.gpu`, `red.release.gpu`).
- **What Volta lacks:** clusters, TMA and mbarrier transaction counts. Hyperkernel's in-block pipelines depend on
  these.

**Correction to an assumption:** upstream llama.cpp already runs CUDA graphs on Volta. The architecture gate in
`ggml_cuda_graph_set_enabled` (`ggml-cuda.cu:5118` at `e14a779`) only disables graphs for `cc < GGML_CUDA_CC_VOLTA`.
So host launch overhead is mostly gone already. What a persistent kernel could still remove is the idle time
between dependent kernels while one drains and the next fills.

**Next step (cheap, needs a V100):** measure that idle time per decode token from an nsys timeline. Sum of
(next kernel start − previous kernel end) per token, at 1 and 8–16 tokens, f16 and q8_0 KV. Graphs on, which is
the default. If it is a small share of the step time, stop here.

**Scope warning:** a full megakernel is a model-specific rewrite, which is Hyperkernel's approach and the opposite
of our upstream-patch strategy. The realistic form is local fusion of adjacent kernels. "Pairing matmuls that share
an input" is already on the ROADMAP Next list.

### 3. Direct GPU-to-GPU exchange inside a kernel (Phase 3)

Hyperkernel maps every rank's memory into one address range, and kernels write partial results directly into peer
memory, then signal with flags. The dual-V100 PCIe equivalent is:
- `cudaDeviceEnablePeerAccess`;
- the producing kernel stores its partial sums to the peer's buffer through the peer pointer;
- it then publishes a release flag that the consumer polls with acquire loads.

That is a one-shot all-reduce that can overlap with the matmul epilogue. It targets exactly the gap in
`llama-cpp-volta-baseline.md`: upstream's internal all-reduce (`GGML_CUDA_ALLREDUCE=internal`) is staged through
pinned host memory and never uses P2P.

- **Precondition:** the two home-box V100s must support P2P with each other (same PCIe switch or root complex). The
  P-7 topology probe (`include/volta_fa/p2p.h`, `bench/p2p_bandwidth`) answers this on day one.
- Not run.
- Phase 3 already excludes NVLink and more than 2 GPUs. Multicast has no PCIe equivalent.

### Lower priority

- **Host ↔ kernel rings.** These only pay off together with idea 2, and on a PCIe V100 every poll of mapped pinned
  memory crosses the bus.
- **Speculative decoding on the GPU.** P-6 already covers attention for verifying draft tokens. One gap: a draft of
  k tokens gives (k+1)-token batches. Patch 0008 routes Q4_K from 5 tokens, Q5_K from 6, IQ4_XS from 8 and Q8_0 from
  9, so short drafts on most types stay on the current path. That falls under the 2–4-token item already on the
  ROADMAP Next list.

## Not applicable to sm_70

- **Forbidden by the hard rules (CLAUDE.md):** thread-block clusters, TMA (`tma_map_at`), and the Blackwell async
  barrier ops (`SYNCS`, 2,541 in `entry.sass`). If someone thinks we need one of these, they stop and raise it.
- **Not available on V100 PCIe:** NVLink multicast.
- **No Volta hardware support:** FP4/MX scale formats (`pt_mx_codes`, `pt_mx_scales`).
- **Irrelevant on a single-die GPU:** per-die SM-ID remapping (`MEGA_SMID_PERM4`).

## Recommendation

1. Owner decision on P-9 option (b) (idea 1): it has the most measured evidence behind it.
2. On the home box, measure the inter-kernel idle time before any design work on idea 2.
3. Fold idea 3 into the Phase 3 design once the P2P probe has run on the dual-V100 machine.
