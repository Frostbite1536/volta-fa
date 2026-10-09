# llama.cpp on Volta (sm_70): FA and multi-GPU baseline

Source: `third_party/llama.cpp` @ 79e2e74 (2026-10-09), read only. Paths are relative to `ggml/src/ggml-cuda/` unless stated. Nothing here was run on a V100. "(inferred)" marks conclusions drawn from the code rather than stated in it.

## 1. Kernel selection on cc 7.0

`ggml_cuda_get_best_fattn_kernel` (fattn.cu:541). On Volta the Turing branch (fattn.cu:638) is skipped, and the Volta branch decides (fattn.cu:667-681):

- `gqa_ratio_eff` = largest power of two dividing the GQA ratio, capped at 8 (16 for 576/192, 32 for 320) and forced to 1 if ALiBi is on (fattn.cu:667-671, since #29803).
- **VEC** if the head dim is 64/128/256 (and KV is padded to 256) and `Q->ne[1]*gqa_ratio_eff <= 2` (fattn.cu:635, 674).
- **TILE** if `Q->ne[1]*gqa_ratio_eff <= 16`. The comment says "On Volta tensor cores are only faster for sufficiently large matrices" (fattn.cu:677-678).
- **MMA** otherwise. Head dims 40 and 72 always use TILE.

| Case | Volta kernel |
|---|---|
| Decode, MHA or GQA 2, D 64/128/256 | VEC |
| Decode, GQA ≥4 (Llama-3, Qwen) | TILE: FP16 FMA, no tensor cores |
| Decode, D 80/96/112/576 | TILE (VEC has instances only for 64/128/256, fattn.cu:428-431) |
| Prefill (Q×eff > 16) | MMA |
| ncols2 on MMA | GQA divisible by 8 → 8, by 4 → 4, by 2 (D ≤ 256) → 2, else 1 (fattn.cu:223-246) |

**Quantised KV.** TILE and MMA always dequantise the **whole** K/V to f16 before the kernel runs (fattn.cu:737-742; fattn-common.cuh:1031-1085). Turing+ sends batch-1 quantised decode to VEC no matter the GQA ratio (fattn.cu:649-658). Volta has no quantised special case, so **q8_0/q4_0 decode with GQA ≥4 goes to TILE and pays for a full-cache dequant every token** (inferred cost, but the path is clear). Only q4_0-q4_0, q8_0-q8_0, f16-f16 and bf16-bf16 VEC instances are compiled by default (ggml/CMakeLists.txt:208). Mixed K/V types fall back to f16 with a "(slow)" warning (fattn.cu:502-510).

**No FA at all (returns NONE):**
- 192/128 unless GQA % 8 == 0 with the GQA optimisation on (fattn.cu:589-595; open PR #26404).
- 512/576 without a mask or with an unpadded KV.
- `mask->ne[2] != 1`.
- K/V types other than f32/f16/bf16/q4_0/q4_1/q5_0/q5_1/q8_0.

**Possible hard failures on Volta (inferred):**
- **512/512 with GQA 2:** MMA is chosen when Q > 8, but the Volta ncols2 switch reaches `GGML_ABORT` (fattn.cu:243-244).
- **No mask (or KV not padded to 256) with GQA ≥ 2 and 16/eff < Q ≤ 16:** `gqa_ratio_eff` ignores the mask, but `use_gqa_opt` does not (fattn.cu:207). The launch is then ncols2=1, ncols1=16, which trips the Volta `NO_DEVICE_CODE` (`__trap`) for fewer than 32 columns (fattn-mma-f16.cuh:1847-1852).
- Neither shape is in `test-backend-ops` (nb ∈ {1,3,32,75}, tests/test-backend-ops.cpp:11593).

## 2. The Volta MMA path

- **Instruction:** PTX `mma.sync.aligned.m8n8k4`. KQ uses `.f32.f16.f16.f32` and VKQ uses `.f16.f16.f16.f16`, issued as two instructions per k=8 (mma.cuh:1504-1542). So VKQ accumulates in FP16 (Turing does too).
- **Layout:** "each warp is doing 4 8x8 mma operations in parallel… stack 4 input tiles in I direction and mirror the B tile" (mma.cuh:21-24). FA tiles are A_KQ = `tile<8,4,half2,I_MAJOR_MIRRORED>` and B_KQ = `tile<32,4>`, giving C = 32×8 (fattn-mma-f16.cuh:1178-1186).
  - **A warp always covers 32 Q columns** (`get_cols_per_warp` = 32, fattn-mma-f16.cuh:340-347), so the minimum tile is 32 columns.
  - Converting FP32 to FP16 tiles needs a `__shfl_xor` because the layouts differ (mma.cuh:768-771).
- **No ldmatrix:** fragments come from plain 16-byte shared loads (mma.cuh:861-863). The column-major V fragment uses **8-byte** loads (mma.cuh:891-895, 998-1003).
- **Tile configs:** `ggml_cuda_fattn_mma_get_config_volta` has its own rows only for 320/512/576. Everything else is `// TODO tune specifically for Volta` → Ampere table (fattn-mma-f16.cuh:112-127). Kernel is `__launch_bounds__(nthreads, occupancy)` from that table (fattn-mma-f16.cuh:1797).
  - Ampere rows for D ≤ 128: 128 threads, occupancy 2, nbatch_fa 64 for ncols ≥ 16 (fattn-mma-f16.cuh:39-62).
- **Disabled on Volta:**
  - cp.async multi-stage pipelining: nstages=0, synchronous loads (fattn-mma-f16.cuh:364-381).
  - Swizzle: padded strides (+4 half2) instead (fattn-mma-f16.cuh:350-360).
  - Thin 8-column tiles (fattn.cu:171).
  - Sparse FA, Turing+ only (fattn.cu:147; PR #28887 to enable it was closed under the AI policy).
  - Stream-K heuristics are Ada-specific (fattn-common.cuh ~1150).
- **Shared memory (computed from fattn-mma-f16.cuh:2033-2043):** the combine buffer scales with `cols_per_warp`, so it is **2× larger on Volta than on Ampere**.
  - D=128, ncols=64: 34.8 KB → 2 blocks (8 warps) per SM.
  - D=256, ncols=64: 67.6 KB → **1 block of 4 warps per SM**, even though the launch-bounds hint asks for 2. Q_in_reg=true also spills: open PR #27997 reports 255 regs + 552 B stack, cut to 56 B by a Volta row, with **+20-28 % prefill at 65-100K depth** on Qwen 27B (V100).
- **TILE kernel:** uses the P100-tuned FP16 config (fattn-tile.cuh:7, 21, 322-323). A comment there still assumes TILE "is only used for GPUs that can't use the mma kernel (Pascal and older)" (fattn-tile.cuh:1257-1259), which is stale for Volta decode.

## 3. History and open work

- **#17505** (JohannesGaessler, merged 2025-12-03): "generalized (mma) FA, add Volta support". It replaced the WMMA kernel on Volta.
  - V100-PCIE-32GB, pp512@d32768 against the old WMMA kernel: llama 8B Q4_0 **1.33× at ub=1, 1.84× at ub=4, 2.97× at ub=512** (932 t/s); gemma 2B 1.4-2.0×; deepseek2 16B up to 1.8×.
  - The author calls the mma.cuh interface "suboptimal".
- **#29803** (2026-10-01): fixed 2 broken Volta FA cases (320/256 smem overflow → issue #28661 crash; ALiBi routing).
- **#26289** (merged 2026-09-27): tile FP16 configs for D 40-112, V100 kernel gains +9-24 % on some rows.
- **Open issue #28037**: Volta FA analysis on Xavier. It says config tuning gave nothing for decode and calls the m8n8k4/no-ldmatrix gap "fundamental". It proposes a split-D WMMA kernel after 1Cat-vLLM's FLASH_ATTN_V100 (claimed ~35 %). A comment points at the fishlikeX/sm70-attn port: "no gain in decode… up to 39.9 % prefill".
- **Multi-GPU/toolchain issues:**
  - #27366: `-sm tensor` hang and VRAM leak on V100 (closed as stale).
  - #28416 (open): **CUDA 12.9 sm_70 builds fail weight allocation; 12.8 works.** This matters for TOOLCHAIN.md.

## 4. Multi-GPU

- **Split modes:** `none`, `layer` (default, pipeline-parallel, `GGML_SCHED_MAX_COPIES=4`) and `tensor` (EXPERIMENTAL, "meta" backend, #19378, 2026-04-09; ggml-backend-meta.cpp).
  - **`row` was removed from CUDA** in #24216 (2026-07-06). llama-model.cpp:1166-1185 now throws "does not support split buffers".
- **Peer access:** P2P is opt-in through `GGML_CUDA_P2P`, which calls `cudaDeviceEnablePeerAccess` for every pair at init (ggml-cuda.cu:386-399, plus VMM access at :600-637). Docs warn about IOMMU (docs/multi-gpu.md:104-112).
- **Cross-device copies** (layer boundaries, meta-backend moves) are `cudaMemcpyPeerAsync` on the source stream plus an event wait (ggml-cuda.cu:2596-2619). Without peer access the driver stages them through the host (inferred).
- **Tensor-mode AllReduce** is chosen in ggml-cuda.cu:1209-1245, chain NCCL → internal → butterfly:
  - **NCCL** (Linux default): FP32 for small tensors, **BF16 for large** ones (ggml-cuda.cu:996-1058).
  - **Internal** (`GGML_CUDA_ALLREDUCE=internal`, #22299): two GPUs only, **staged through pinned host memory and never P2P**, "targets setups without NVLink" (allreduce.cu:13-37). Small tensors take an in-kernel chunked path with a host-flag spin; tensors over 1 MB take a copy-engine D2H+H2D path. BF16 on the wire by default (allreduce.cu:435).
  - **Butterfly:** generic `cpy_tensor_async` + ADD (ggml-backend-meta.cpp:2395-2440).
- **Room for a P2P project:** upstream has no direct device-to-device AllReduce. Every reduction is a sync point between graph segments, and the internal path makes two PCIe crossings through host RAM (inferred).
  - The superfast fork's `allreduce-p2p.cu` (below) shows the idea pays off on 2× V100 PCIe.

## 5. External repos

- **mylordmonkeyman/llama-superfast-v100**: created 2026-10-09, one squashed import based on upstream `6ba30d0`, mirroring codeberg justanagent/llama-superfast-v100. It targets Qwen3.8-27B (D=256, GQA 6) on V100 32 GB PCIe, CUDA 12.8. New CUDA files include:
  - `fattn-gqa6.cu`: prefill attention ported from 1Cat-vLLM's CUTLASS sm70 kernels. The 6 heads are packed as GEMM rows; QK GEMM with a softmax epilogue writing **FP16 probability tiles to a workspace**, then a PV GEMM. That is not a fused kernel. It claims to be "faster at every bench cell: 256 to 2,048 rows, 4,096 to 245,760 keys" but gives no number.
  - `fattn-verify-stream.cu`: a 1-16-token streaming kernel.
  - A route that sends **Q ≤ 8, GQA 3-8, KV ≥ 4096 to MMA ncols2=8 instead of upstream's TILE**, with "each KV head streamed once".
  - `allreduce-p2p.cu`: a Lamport-style push into peer memory, with copy-engine fallback for prefill; needs `iommu=pt`.
  - Claims: 935 t/s cold 16K prefill and 742 t/s at 64K on one V100; ~99-116 t/s decode with MTP; "the attention kernels' fast paths read only f16 and bf16 caches".
- **thc1006/qwen3.6-speculative-decoding-v100**: DFlash/MTP decode sweep on 2× V100, llama.cpp b9860, `-fa on -c 4096`. Baseline decode is 37.6 t/s (27B IQ4_XS) and 96.6 t/s (35B-A3B). **There are no FA or prefill measurements.**

## Candidate gaps to measure in Phase 0 (ranked)

1. **Decode/verify at long context, GQA ≥ 4 (TILE path).** Upstream uses FP16-FMA TILE; superfast moved this to MMA for KV ≥ 4096.
   - `llama-bench -m <llama8B|qwen27B> -fa 1 -p 0 -n 128 -d 0,8192,32768,65536 -ub 1,2,4,8`
   - `test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=256.*nr23=\[12,1\]'`
2. **Quantised KV decode with GQA ≥ 4** (TILE plus full dequant per token).
   - Compare `llama-bench ... -ctk q8_0 -ctv q8_0` with f16 at `-d 32768 -n 128`, and check nsys for `dequantize`/`convert` kernels.
   - `test-backend-ops perf -o FLASH_ATTN_EXT -p 'type_K=q8_0'`
3. **D=256 prefill occupancy/spill** (Ampere config, 1 block/SM, stack spill).
   - `llama-bench -m <qwen27B> -fa 1 -p 512 -d 4096,32768,65536 -ub 512` before and after the #27997 row.
   - Check `--ptxas-options=-v` on the `<256,256,16,4>` / `<…,8,8>` instances.
4. **D=128 prefill against peak.** Compare MMA throughput with V100's ~112 TFLOPS FP16 roofline.
   - `llama-bench -p 512 -d 0,16384,32768 -ub 512`
   - ncu on the `flash_attn_ext_f16` kernels: tensor-pipe utilisation, shared bank conflicts from the 8-byte V loads.
5. **Tensor-split AllReduce.** Run `-sm layer` against `-sm tensor` with `GGML_CUDA_ALLREDUCE=nccl|internal` and `GGML_CUDA_P2P` on/off; use nsys to measure AllReduce share. Correctness check: `test-backend-ops -o FLASH_ATTN_EXT` across all cases on Volta.

## Open questions

- Is TILE actually slower than MMA for Volta decode at 8-64K? The upstream routing comment says no; the superfast fork says yes. Only a measurement settles it.
- Do the two inferred crash shapes (512/512 GQA 2; maskless GQA small batch) reproduce? They need a local test case.
- Do the 1Cat/fishlikeX split-D prefill claims (35-40 %) hold against upstream after #27997?
- NCCL on our PCIe topology: P2P or SHM? Is its BF16 path emulated on sm_70?
- CUDA 12.9 allocation failure (#28416) against our TOOLCHAIN pin.
