# Roadmap

Phases have explicit exclusions. Exit criteria for Phases 0–3 are measured on a real V100.

**Starting premise (corrected 2026-10-09):** llama.cpp already has a Volta-specific FlashAttention path. `fattn-mma-f16.cuh` runs on sm_70 through PTX `mma.m8n8k4` (`mma.cuh`, `VOLTA_MMA_AVAILABLE`), with Volta-specific dispatch in `fattn.cu`. The old WMMA kernel has been removed. "No FlashAttention on Volta" is therefore false. Any custom kernel has to beat a maintained, Volta-aware kernel on specific shapes. See `docs/research/llama-cpp-volta-baseline.md`.

## Pre-hardware track (now → V100 box arrives, ~2026-11)

**Goal**: Make day one on the hardware about measuring, not setup.
**Includes**:
- P-1 Toolchain without a GPU: NVRTC + ptxas static reports for sm_70 (`bin/sm70-report.py`). ✅
- P-2 Reference + tests: CPU attention reference with the full shape/edge-case matrix, and a CPU *tiled online-softmax model* that copies the kernel's numerics (FP16 P, FP32 m/l, tile order). It gives an empirical error budget before any GPU code exists, so tolerances come from data rather than guesses.
- P-3 Desk research: how llama.cpp picks the FA kernel on cc 7.0, its tile configs, history, multi-GPU modes; a ranked list of gaps to measure.
- P-4 Planning calculator: smem bytes, register estimate and theoretical occupancy for V100 for a candidate tile config (`bin/plan-tiles.py`). ✅
- P-5 Phase 0 measurement kit: `bench/phase0.sh` runs FA support/correctness/perf via test-backend-ops plus the llama-bench grid (depth, KV quant, ubatch, dual-GPU split modes) into `bench/results/`. Drafted, not yet run (needs the box).
- P-6 Kernel design docs for the top gap(s) from P-3, with static sm70 reports for draft kernels (regs/spill only, no timing).
- P-7 P2P code that needs no hardware: topology probe, capability check with staged fallback, bandwidth test driver.
**Excludes**: any performance claim; tuning by guesswork.
**Optional**: rent a single cloud V100 for a few hours to run P-5 early. Cloud multi-V100 instances usually have NVLink, so they say nothing about PCIe P2P.

## Phase 0: Baseline and gap analysis (first week on hardware)

**Goal**: Find where the existing Volta path is actually slow.
**Includes**: run the P-5 kit: `llama-bench` over context depth (`-d 0,4096,16384,32768,65536`), prompt sizes, batch sizes, FA on/off, a few models (different head dims and GQA ratios), f16 and quantised KV; `test-backend-ops -o FLASH_ATTN_EXT -b CUDA0 perf` for kernel-level numbers; nsys timeline of a long-context prefill; ncu on the FA kernels; `nvidia-smi topo -m` + P2P bandwidth test.
**Excludes**: writing new kernels.
**Exit**: `bench/BASELINE.md` + `profiles/phase0-*.txt`, and a written go/no-go per shape naming the measured bottleneck. If no shape has a meaningful gap, redirect the project (for example to P2P/tensor-parallel, quantised-KV FA, or contributing Volta tuning upstream).

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
