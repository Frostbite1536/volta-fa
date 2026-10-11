# Roadmap

Phases have explicit exclusions. Exit criteria for Phases 0–3 are measured on a real V100.

**Starting premise (corrected 2026-10-09):** llama.cpp already has a Volta-specific FlashAttention path. `fattn-mma-f16.cuh` runs on sm_70 through PTX `mma.m8n8k4` (`mma.cuh`, `VOLTA_MMA_AVAILABLE`), with Volta-specific dispatch in `fattn.cu`. The old WMMA kernel has been removed. "No FlashAttention on Volta" is therefore false. Any custom kernel has to beat a maintained, Volta-aware kernel on specific shapes. See `docs/research/llama-cpp-volta-baseline.md`.

## Pre-hardware track (now → V100 box arrives, ~2026-11)

**Goal**: Make day one on the hardware about measuring, not setup.
**Includes**:
- P-1 Toolchain without a GPU: NVRTC + ptxas static reports for sm_70 (`bin/sm70-report.py`). ✅
- P-2 Reference + tests: CPU attention reference with the full shape/edge-case matrix, and a CPU *tiled online-softmax model* that copies the kernel's numerics (FP16 P, FP32 m/l, tile order). It gives an empirical error budget before any GPU code exists, so tolerances come from data rather than guesses. ✅ Tolerances set in `include/volta_fa/tolerance.h`; FP16 QK accumulate ruled out (INV-NUM-005).
- P-3 Desk research: how llama.cpp picks the FA kernel on cc 7.0, its tile configs, history, multi-GPU modes; a ranked list of gaps to measure. ✅ `docs/research/llama-cpp-volta-baseline.md`. Headline candidates (unmeasured):
  1. Small-batch attention at long context with GQA ≥ 4 (decode with GQA, speculative verify) routes to the non-tensor-core TILE kernel on Volta (`fattn.cu:673-681`). The llama-superfast-v100 fork moves KV ≥ 4096 to MMA.
  2. Quantised-KV decode with GQA ≥ 4 converts the whole K/V cache to F16 on every call on the TILE path.
  3. Volta reuses Ampere tile configs ("TODO tune specifically for Volta", `fattn-mma-f16.cuh:126`); D=256 prefill looks smem/spill-limited. Open PR #27997 claims +20–28% at 65–100K depth with one Volta config row.
  4. Multi-GPU: `-sm tensor` all-reduce is NCCL or a host-staged fallback; upstream has no direct P2P all-reduce.
  Prior art: PR #17505 (Volta MMA FA, 1.3–3.0× over the old WMMA kernel); issue #28037 (config tuning didn't help decode).
  Static compile (nvcc 12.8, no GPU, `bench/pr27997/static.sh`): upstream's D=256 ncols=64 kernels spill 464–640 B at 255 registers; #27997 cuts that to 48–112 B; a further variant (`pr_k64_c64`) keeps spills at 48–160 B **and** fits 2 blocks/SM. Timing pending (cloud V100). The same survey shows upstream's D=512 ncols=64 kernels at 544–568 B of stack despite their Volta rows: a fifth candidate (`bench/results/pr27997-static-20261009-182401/headdim-survey.txt`).
- P-4 Planning calculator: smem bytes, register estimate and theoretical occupancy for V100 for a candidate tile config (`bin/plan-tiles.py`). ✅
- P-5 Phase 0 measurement kit: `bench/phase0.sh` runs FA support/correctness/perf via test-backend-ops plus the llama-bench grid (depth, KV quant, ubatch, dual-GPU split modes) into `bench/results/`. Drafted, not yet run (needs the box).
- P-6 Small-batch GQA attention on Volta: **designed and implemented** (`docs/design/P6-small-batch-gqa.md`, `third_party/patches/0001..0003`), statically verified for sm_70. Not a new kernel: upstream's MMA kernel already masks padded heads; Volta's dispatch just never padded.
- P-7 P2P code that needs no hardware: topology probe, capability check with staged fallback, bandwidth test driver. ✅ `include/volta_fa/p2p.h`, `bench/p2p_bandwidth`; staged path verified on a real (Maxwell) GPU, peer path not run until 2 GPUs. Known limit: large staged copies block the host on staging-buffer reuse.
- P-8 Quantised-KV decode on Volta: **designed and implemented** (`docs/design/P8-quantised-kv-decode.md`, `third_party/patches/0004..0006`). Cause, from the measured rows (9 of 9 within ±4%) and end to end (+14.0 ms/token modelled vs +12.8 ms measured): every decode token converts the whole K/V view to F16 with a launch of one 256-thread block per (token, head) row, which is bound by the block count. Fix: single-token decode uses VEC on these types (0004, Volta-only); faster conversions for everything that still converts (0005 geometry, 0006 vectorised q4_0/q8_0, both architecture-independent). Conversion kernels bit-exact and 2.6–5.2× faster on a Maxwell GPU (not a V100). Change 4 (GQA-packed VEC) is gated on the V100 numbers.
- **V100 session 2 (2026-10-10): done.** `docs/research/patches-cloud-2026-10-10.md`. All 6 patches: 0 new failures (FA, MUL_MAT); crash fix confirmed. P-6: D=256 1.0–3.9× kernel, Qwen3.8-27B +7.6% / +19.9% pp8 / pp16 @32K (as predicted: FA is ~16 of its 65 layers); **D=128 GQA 3/6 at 8–16 tokens regresses (0.69–0.91×), so 0002 needs a D=128 rule before upstreaming.** P-8: qkv 2.1–3.3× at 1 token; conv8 2× at D=128, 1.3–1.4× at D=256; Llama-3.1-8B tg64 @16K +16.7% (q8_0, below the +25% target) / +36.2% (q4_0), no f16 change. Change 4 gate triggered.
- **Session 2 extension (2026-10-10/11, same instance): done.** `docs/research/pad-rule-vecpack-v100-2026-10-10.md`. (a) 0002 rule: the D=128 regressions are a 4-K/V-head parallelism effect (8 K/V heads: padding wins in every cell, 1.04–3.17×); 0002 now keeps exact `<16,2>` for GQA 6 above 8 tokens at D ≤ 128 with < 8 K/V heads (1.00× there, was 0.69/0.86×); two 4K cells remain (0.83×, 0.91×). (b) Change 4 = patch 0007 (packed VEC, 2/4 heads per block, key loops unrolled 4×; the first unrolled-1 prototype lost on the V100): Llama-3.1-8B tg64 @16K vs upstream q8_0 **+26.8%** (P-8 target met), q4_0 +56.2%, f16 0.0%; Qwen3.8-27B q8_0 +5.8%. Full series 0001–0007: FA 4334/4334 on sm_70. New small cost: D=256 f16 single-token VEC at GQA 3/5/6/7 0.96–0.98× (not attributed). Lesson: never build on the host while timing (doubles short-kernel times).
- **(c)+(e) on the same instance (2026-10-11): done.** `docs/research/qwen-smallbatch-and-vecpack-heads-v100-2026-10-11.md`. (c) The fork's Qwen3.8-27B small-batch gain (+85–89% at 8–16 tokens) is its QPN matmul (`mmvq-qpn.cu`: Volta m8n8k4 tensor-core skinny GEMM on K-quant weights repacked once into fragment order): 68.5 → 32.8 ms of a 78.7 ms upstream step (~95% of the gap); with `LLAMA_MMVQ_QPN=0` the fork is −2% / −35% vs upstream at 8 / 16 tokens. (e) The D=256 f16 single-token 0.95–0.98× is 0007's (conv8 = upstream); scheduling, ~0.1–0.7% of Qwen f16 decode, not fixed. 0007's rule now depends on K/V heads (≥ 8: GQA 2 packed, GQA 3 four per block): 1.02–2.53× with 8 K/V heads, no loss; FA 4358/4358 on sm_70.
- P-9 Small-batch quantised matmul on Volta tensor cores: **designed, Phase 0 done on a V100** (`docs/design/P9-small-batch-qmatmul.md`). Upstream MUL_MAT streams weights at 570–775 GB/s at 1 token but 205–365 GB/s at 9–16 (dp4a MMQ): Qwen3.8-27B ≈ 58.7 ms and Llama-3.1-8B ≈ 21.7 ms of matmul per 16-token evaluation against floors of 20.6 / 9.9 ms. Layout microbenchmark (`bench/micro/qload.cu`): the native GGUF layout staged through shared memory with 128-byte row reads reaches 0.76–0.90× of a repacked fragment-order layout with the decode + m8n8k4 work included (naive native: 0.29–0.64×), so Phase 1 needs no weight repacking and no ggml/llama changes. Tools: `bin/qmatmul-table.py` (shapes, perf → GB/s, per-evaluation estimate).
- **Fixed (P-7 test, 2026-10-11):** `p2p_buffer_reuse` failed intermittently: `tests/test_p2p.cpp` filled device buffers with a pageable `cudaMemcpy`, which can return before the DMA completes, then read them on non-blocking streams. A `cudaDeviceSynchronize()` after each of the three fills fixes it (test-only; the library is unaffected). Laptop GPU under load (the layout microbenchmark running): old binary 75/150 failures, fixed 0/150, interleaved. Found in the P100 repo (`~/Github/P100` `d8b7351`).
- **Open (from the P100 repo, `~/Github/P100/docs/research/pre-hardware-findings-2-2026-10-10.md` §2, §5):** (1) llama.cpp's FA TILE kernel accumulates VKQ in half2 when `FAST_FP16_AVAILABLE`, which includes sm_70 for TILE-routed shapes; a CPU model of the key split with V biased +2 on 8 channels at 32K keys gives up to 33% max relative error (58% unsplit). P100's patch flushes to FP32 per KV tile but adds 4–32 B of spills in sm_70 D=256 configs (4x8, 8x4, 32x1), so it needs work before it passes the spill rule here. Not run on a GPU with FP16 FA. (2) test-backend-ops' CPU reference for FLASH_ATTN_EXT also accumulates F16 V in FP16 (`ops.cpp` VKQ16), so long-context biased-V cases fail against the reference even when the GPU is right; P100 patch 0003 accumulates in FP32 only in reference mode. Needed before volta-fa adds long-context FA tests.
- **P-9 Phase 1 (2026-10-11): done for Q4_K/Q5_K/IQ4_XS** (patch 0008, `docs/design/P9-small-batch-qmatmul.md` "Phase 1 result"). Qwen3.8-27B +21.6% / +21.8% at 8 / 16 tokens end to end, unchanged at 1–4; per matmul 1.12–1.72x at 8 tokens, 1.48–1.67x at 12–16; MUL_MAT 2549/2549 on sm_70.
- **Next:** P-9 Phase 2 (Q8_0 for Llama, Q6_K, perplexity); (d) upstream contributions for the owner to write: 0001 crash fix, 0005/0006 conversions, 0003 configs (with #27997); 0007 needs measurements on other architectures (or a Volta-only switch) first; (e) open: the D=256 f16 single-token 0.95–0.98× from 0007 (needs counters or a code-layout experiment); a GQA 2 model end to end.
**Excludes**: any performance claim; tuning by guesswork.
**Cloud V100 session (2026-10-10, V100-PCIE-32GB): done.** Results in `docs/research/phase0-cloud-2026-10-10.md`. Measured gaps that now drive P-6, ranked:
  1. **Small-batch / speculative-verify attention at depth.** Upstream uses the non-tensor-core TILE path; the fork's Qwen-specific kernel shows +80–112% for pp8/pp16 on D=256 GQA 6 but regresses D=128 GQA 4 by 23–27%. A general Volta tensor-core small-batch kernel is the first kernel target.
  2. **Quantised-KV decode.** Up to 7.2× slower than f16 at kernel level, −36 to −39% tg128 end to end on Llama-3.1-8B at 16K.
  3. **D=256 prefill.** The one-row `pr_k64_c64` config gives 2.3–2.65× kernel / +45% end to end at 64K; that is an upstream contribution (via PR #27997), not a custom kernel.
  4. **Volta small-batch crash** (16-column MMA instance without sm_70 code): one-line fix verified; an upstream bug report.
  Not measured: multi-GPU/P2P (needs the home box), ncu metrics (container), perplexity.

## Phase 0: Baseline and gap analysis (first week on hardware)

**Goal**: Find where the existing Volta path is actually slow.
**Includes**: run the P-5 kit: `llama-bench` over context depth (`-d 0,4096,16384,32768,65536`), prompt sizes, batch sizes, FA on/off, a few models (different head dims and GQA ratios), f16 and quantised KV; `test-backend-ops -o FLASH_ATTN_EXT -b CUDA0 perf` for kernel-level numbers; nsys timeline of a long-context prefill; ncu on the FA kernels; `nvidia-smi topo -m` + P2P bandwidth test.
**Excludes**: writing new kernels.
**Exit**: `bench/BASELINE.md` + `profiles/phase0-*.txt`, and a written go/no-go per shape naming the measured bottleneck. Also re-check the upstream state (#27997, superfast fork) so we don't rebuild what has landed. If no shape has a meaningful gap, redirect the project (for example to P2P/tensor-parallel, quantised-KV FA, or contributing Volta tuning upstream).

## Phase 1: Standalone fused attention kernel (only for "go" shapes)

**Goal**: A correct Volta kernel that beats the llama.cpp kernel on the Phase 0 gap shapes.
**Includes**: kernel; tile-size sweep; static report → ncu loop until there are 0 spills and bank conflicts are addressed; microbench vs. the llama.cpp kernel on identical inputs.
**Excludes**: llama.cpp integration, multi-GPU, backward pass.
**Exit**: parity tests pass; microbench shows a gain (or honestly fails to) with artifacts in `bench/results/`.

## Phase 2: llama.cpp integration

**Goal**: End-to-end prefill/decode gain on real GGUF models.
**Includes**: patch series on the pinned commit behind a flag; end-to-end `llama-bench` before/after; perplexity check vs. baseline.
**Excludes**: P2P work. An upstream PR is preferred if the gain is real; decide after results.
**Exit**: end-to-end numbers + perplexity delta published.

## Phase 3: P2P tensor-parallel communication

**Goal**: Reduce dual-V100 PCIe stalls.
**Includes**: start from what llama.cpp's split modes already do (P-3); peer-copy microbench; overlap of partial-result exchange with compute; staged fallback; nsys timelines showing overlap.
**Excludes**: NVLink paths; more than 2 GPUs.
**Exit**: measured tok/s change on a real dual-GPU run, or a documented negative result.

## Phase 4: Publish

README results table, reproduction script, write-up using only INV-PERF-001-compliant numbers.

## Resume bullet gating

Bullets are written last, from artifacts. Placeholder percentages ("by X%") and borrowed numbers (e.g. "150+ tok/s") do not ship.
