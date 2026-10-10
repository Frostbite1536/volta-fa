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
- P-6 Small-batch GQA attention on Volta: **designed and implemented** (`docs/design/P6-small-batch-gqa.md`, `third_party/patches/0001..0003`), statically verified for sm_70. Not a new kernel: upstream's MMA kernel already masks padded heads; Volta's dispatch just never padded. **Next: one V100 session with `bench/p6/run.sh`** (correctness + interleaved kernel A/B + Qwen/Llama small-batch llama-bench). Quantised-KV decode (gap 2) is the next design after that.
- P-7 P2P code that needs no hardware: topology probe, capability check with staged fallback, bandwidth test driver. ✅ `include/volta_fa/p2p.h`, `bench/p2p_bandwidth`; staged path verified on a real (Maxwell) GPU, peer path not run until 2 GPUs. Known limit: large staged copies block the host on staging-buffer reuse.
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
