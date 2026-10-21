# volta-fa — making llama.cpp faster on the V100

*Share brief for V100 owner communities. Every number below was measured on real
hardware and links to committed raw data; nothing extrapolated.*

**Repo:** https://github.com/Frostbite1536/volta-fa (AGPL-3.0)

## What it is

A research project targeting one specific, aging GPU: the **NVIDIA Tesla V100 (Volta,
sm_70)**. Modern llama.cpp performance work targets Ampere and newer, but V100s —
especially 32 GB PCIe cards — are cheap, common, and still have tensor cores that are
barely used at small batch sizes. This repo measures where llama.cpp actually loses
time on sm_70 and fixes it as a series of **eight small patches against a pinned
llama.cpp commit** (`79e2e74`, 2026-10-09), plus the full benchmark harness, raw
result data, and design docs behind every change.

Discipline that may interest this group: every performance claim links to committed
raw data; A/B comparisons run interleaved in one session; correctness suites
(`test-backend-ops` FLASH_ATTN_EXT / MUL_MAT) must show zero new failures before
anything is timed; regressions are recorded, not hidden.

## The gap that motivated it

An initial code-reading pass ([research/llama-cpp-volta-baseline.md](research/llama-cpp-volta-baseline.md))
corrected the founding assumption — llama.cpp *does* have a Volta FlashAttention path
(PTX `mma.m8n8k4`, PR #17505). So Phase 0 measured instead of rewrote, and found four
real problems on a rented V100-PCIE-32GB:

1. **A crash:** upstream's FA test suite *aborts* on sm_70 — kernel selection and
   dispatch disagree about the GQA optimization, and a 16-column MMA instance gets
   launched that Volta compiles as `NO_DEVICE_CODE`.
2. **Small-batch GQA attention at depth:** for GQA 3/5/6/7 at 3–16 tokens, Volta
   streams K/V once per 2-head group — the same cache read 3–4×.
3. **Quantised-KV decode:** with GQA ≥ 4, the whole quantised K/V view is converted
   to F16 *on every token*; up to 7.2× slower than f16 at kernel level, −25/−39%
   end-to-end on Llama-3.1-8B at 16K.
4. **D=256 prefill:** Volta borrows Ampere tile configs; the kernels spill 464–640 B
   of stack at 255 registers and run at ~19% of the V100's FP16 peak (open PR #27997
   addresses part of this).

A later profiling session found a fifth, bigger one: at 5–16 tokens, quantised matmul
streams weights at only 205–365 GB/s (vs 570–775 at 1 token) because Volta's MMQ is
dp4a — no int8 tensor cores on sm_70.

## Benchmark results (why you'd try it)

All numbers: one Tesla V100-PCIE-32GB (cloud), CUDA 12.8, llama.cpp `79e2e74`,
patches 0001–0008 applied, interleaved A/B, medians. Full data in `bench/results/`,
summaries in [research/](research/) and [PATCH_DATASHEETS.md](PATCH_DATASHEETS.md).

| Workload | vs upstream |
| --- | --- |
| **Qwen3.8-27B small batches** (8 / 12 / 16 tokens) | **+34% / +39% / +38%** |
| **Llama-3.1-8B** (12 / 16 tokens) | **+28% / +27%** |
| **Long-document prefill**, Qwen 27B @64K context | **+45%** (kernel level 2.3–2.65×; #27997's row alone gives +22%) |
| **Quantised-KV decode**, Llama 8B tg64 @16K | **+27% (q8_0) / +56% (q4_0)** — quantised cache now 5% behind f16 instead of 25% |
| **GQA 3–7 small-batch attention kernel** (D=256, 32K) | 1.9–3.9× (e.g. GQA 6 @16 tokens: 1610 → 411 µs) |
| **FA crash** | upstream suite aborts after 3964 cases → **4358/4358 pass** |

The headline patch (0008) is a new Volta tensor-core matmul for
Q4_K/Q5_K/Q6_K/IQ4_XS/Q8_0 at 5–16 tokens: weights read once in native GGUF layout,
exact FP16 fragment decode, zero spills. Bonus: it's *more accurate* than the current
path — mean KL divergence vs an FP32 reference is 6–16× lower, because it doesn't
quantise activations to 8 bits.

**Honest footnotes:** three small regressions are recorded (D=128 GQA 3/6 at 4K,
0.83–0.91×, 5–11 µs/call; D=256 f16 single-token decode 0.96–0.98× from patch 0007,
~0.1–0.7% of a Qwen token). The `llama-superfast-v100` fork is still faster on Qwen
small-batches — its remaining edge comes from repacking weights at load time, which
upstream won't accept; this repo's analysis of *why* it's faster is in
[research/qwen-smallbatch-and-vecpack-heads-v100-2026-10-11.md](research/qwen-smallbatch-and-vecpack-heads-v100-2026-10-11.md).

## How to try it

```bash
git clone https://github.com/Frostbite1536/volta-fa
# llama.cpp at bench/LLAMA_COMMIT (79e2e74), then:
git am third_party/patches/00*.patch   # 0001–0008, in order
```

Build with **CUDA 12.8** and `-DGGML_CUDA_ARCHITECTURES=70` (12.9 has an open sm_70
weight-allocation bug, llama.cpp #28416; CUDA 13 dropped Volta). Patches are not
rebased past the pinned commit. The repo's own harness (`bench/phase0.sh`,
`bench/pr27997/run.sh`) reproduces every table above.

## Future work (recorded in [ROADMAP.md](ROADMAP.md))

- **Repacked-weight matmul** — the remaining gap to the superfast fork (46.2 vs
  32.8 ms matmul per 16-token Qwen step) needs load-time weight repacking; design
  option (b), not yet started.
- **Smaller matmul items:** 2–4 token batches, IQ4_NL / Q3_K support, pairing
  matmuls that share an input.
- **Upstreaming:** 0001 (crash fix) is the clearest candidate; 0005/0006 conversions
  are architecture-independent and need one non-Volta timing; 0003 extends PR #27997
  with measured data.
- **Dual-GPU PCIe P2P tensor-parallel** — the P2P code exists and is unit-tested,
  but needs two cards to measure (the target machine arrives ~Nov 2026). Upstream's
  tensor-split all-reduce has no direct device-to-device path.
- **Open questions:** the D=256 f16 single-token 0.96× scheduling artifact; a GQA 2
  model end-to-end (Gemma 2/3).

Sibling efforts apply the same method to the P100 and Intel Arc A770.
