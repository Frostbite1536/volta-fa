# System Invariants

Non-negotiable rules. A change that violates one is a bug or an explicit product decision, never an implementation detail. Changes to this file require a justification in the change log at the bottom.

## Architecture invariants

### INV-ARCH-001: sm_70 only
**Rule**: Device code uses only features available on compute capability 7.0. Tensor cores are reached through either `nvcuda::wmma` (16×16×16 half, FP16 or FP32 accumulate) or inline PTX `mma.sync.aligned.m8n8k4` (the Volta-native shape). Shared memory ≤ 96 KB per block (above 48 KB needs `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`).
**Rationale**: The project exists for V100. Turing/Ampere code either fails to compile for sm_70 or silently takes a slow emulated path.
**Valid**: WMMA half fragments; `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` (llama.cpp's Volta FA path uses this shape, see `mma.cuh`).
**Invalid**: BF16, FP8, `cp.async`, `ldmatrix`/`stmatrix` (sm_75+), `mma.sync` m16n8k8/m16n8k16 (sm_75/80+), thread-block clusters, TMA, `wgmma`.
**Note on WMMA vs m8n8k4**: WMMA fragment element-to-thread mapping is unspecified, so per-row softmax statistics need a round trip through shared memory. m8n8k4 has a documented register layout (PTX ISA, "Matrix fragments for mma.m8n8k4") that lets the row max/sum stay in registers. Which to use per kernel is a design decision recorded in ARCHITECTURE.md.
**Enforcement**: `bin/sm70-report.py` compiles at `compute_70` (rejects most sm_75+ intrinsics); `bin/check-forbidden.sh` greps inline PTX and arch-guarded code; CI runs both.

### INV-ARCH-003: CUDA 12.x toolchain
**Rule**: Build with CUDA 12.x (12.9 pinned) and an R580-or-older driver branch. Never CUDA 13+.
**Rationale**: CUDA 13.0 removed offline compilation support for sm_70 (and NVRTC 12.9 already warns it is deprecated); R580 is the last driver branch that supports Volta. See docs/TOOLCHAIN.md.
**Enforcement**: `bin/sm70-report.py` pins the NVRTC 12.9 wheel; the V100 box install doc pins the toolkit.

### INV-ARCH-002: Per-block shared memory budget
**Rule**: Static + dynamic shared memory per block ≤ 96 KB, and occupancy target per kernel is documented next to its launch config.
**Rationale**: V100 has 96 KB configurable shared memory per SM; exceeding the budget silently costs occupancy.
**Enforcement**: `--ptxas-options=-v` output is checked in the profiling loop.

## Correctness invariants

### INV-NUM-001: No materialised score matrix
**Rule**: The fused path never writes the N×N scores/probabilities to global memory. Only O (and the log-sum-exp statistic if needed for backward/split-KV) leave the SM.
**Enforcement**: Memory-traffic check in `bench/` (DRAM bytes ≈ Q+K+V+O); ncu `dram__bytes` captured in `profiles/`.

### INV-NUM-002: Online softmax is numerically safe
**Rule**: Running max and sum are kept in FP32; fully masked rows produce O = 0 and LSE = -inf, never NaN; no intermediate overflows for logits up to the FP16 range.
**Enforcement**: `tests/` includes fully-masked-row, very-large-logit and all-equal-logit cases.

### INV-NUM-004: Causal alignment is bottom-right
**Rule**: With `causal`, query row `i` (0-based) may attend key `j` iff `j <= i + (seq_kv - seq_q)`. This matches a KV cache where the new queries are the last `seq_q` positions.
**Rationale**: Top-left alignment is the classic silent bug when `seq_q < seq_kv` (decode, chunked prefill).
**Enforcement**: reference + tests with `seq_q < seq_kv`.

### INV-NUM-003: Reference parity
**Rule**: Output matches an FP32 reference within the documented tolerance for every supported shape (head dim, GQA ratio, causal/non-causal, seq lens including non-multiples of the tile).
**Enforcement**: `ctest`. Tolerances live in one header and change only with a manager decision.

## Performance honesty invariants

### INV-PERF-001: No claim without an artifact
**Rule**: Every speedup/percentage in README, resume bullets or PRs links to a committed file in `profiles/` or `bench/results/` with hardware, driver, CUDA version, commit, and baseline commit recorded.
**Enforcement**: Review checklist in REVIEW.md.

### INV-PERF-002: Compare against the real baseline
**Rule**: The baseline is the best existing Volta path (llama.cpp's current FlashAttention dispatch on cc 7.0 at a pinned commit, which uses the m8n8k4 mma kernel for most prefill shapes), not a naive implementation and not llama.cpp with FA off.
**Enforcement**: Baseline commit pinned in `bench/BASELINE.md`.

### INV-PERF-003: No spills by default
**Rule**: Shipped kernels have 0 bytes spill stores/loads unless a documented exception exists.
**Enforcement**: ptxas `-v` output reviewed on every kernel PR.

## P2P / multi-GPU invariants

### INV-P2P-001: Capability is checked, with a fallback
**Rule**: Use `cudaDeviceCanAccessPeer` before enabling P2P; fall back to staged host copies and log which path is active. Never assume the topology (PCIe switch vs. root complex changes everything).
**Enforcement**: `tests/` covers the fallback path via a forced-disable env var.

### INV-P2P-002: Streams and events, not device syncs
**Rule**: Communication overlaps compute through streams/events; no `cudaDeviceSynchronize()` in the hot path.
**Enforcement**: Review + nsys timeline in `profiles/`.

### INV-P2P-003: Buffer lifetime
**Rule**: IPC/peer buffers are freed only after all consuming streams have passed an event recorded after the last use.

## Agent-process invariants

### INV-PROC-001: Workers do not decide
**Rule**: Workers implement and report; they do not decide ship/no-ship or loosen invariants/tolerances. Stop conditions are honoured.

### INV-PROC-002: "Not run" is a valid result
**Rule**: Anything not executed on real V100 hardware is labelled "not run", never inferred.

## Change log

| Date | Invariant | Change | Justification |
| --- | --- | --- | --- |
| 2026-10-09 | all | Initial draft | Repo created |
| 2026-10-09 | ARCH-001, ARCH-003, NUM-002, NUM-004, PERF-002 | Allow PTX m8n8k4; forbid ldmatrix; pin CUDA 12.x; define masked-row output and causal alignment | Review: m8n8k4 is Volta-native and what llama.cpp uses; CUDA 13 dropped sm_70 |
