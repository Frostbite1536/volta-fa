# Patch data sheets (0001–0008, plus local 0009)

Facts for the owner to decide what to upstream and to write any PR text themselves. llama.cpp requires AI-use
disclosure and does not accept AI-written PR text, comments or replies, so this file gives data only, not prose to
post.

**Common to all patches**
- Base: llama.cpp `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (`bench/LLAMA_COMMIT`, pinned 2026-10-09). The
  patches apply in order with `git apply` (or `git am`) on that commit. None of them has been rebased onto a newer
  master.
- Hardware measured: one Tesla V100-PCIE-32GB (Vast.ai, 1380 MHz, CUDA 12.8.93, driver 580.173.02). Conversion
  patches were also checked on a Quadro M2200 (Maxwell, sm_52). **Nothing was measured on Turing, Ampere, Ada,
  Hopper or Pascal.**
- Models: Qwen3.8-27B-UD-Q4_K_XL (hybrid: 16 of 65 layers use attention, D=256, 4 K/V heads, GQA 6) and
  Llama-3.1-8B-Instruct-Q8_0 (D=128, 8 K/V heads, GQA 4).
- Method: A/B comparisons interleaved within one session. Kernel times come from `test-backend-ops perf`. End to
  end is `llama-bench` with the first repetition dropped (it runs 25–35% slow on that host), reporting medians.

**Before posting any of these (checklist)**
1. Rebase onto current master and check whether an equivalent change has landed (especially #27997 for 0003).
2. Re-run `test-backend-ops -o FLASH_ATTN_EXT` and `-o MUL_MAT` on the rebased tree.
3. For patches that change code shared with other architectures (marked below), get numbers on at least one
   non-Volta GPU, or restrict the change to Volta.
4. Disclose AI use as llama.cpp's contribution rules require.

## Summary

| patch | what | scope | headline result (V100) | ready? |
| --- | --- | --- | --- | --- |
| 0001 | FA kernel-selection crash fix | Volta dispatch | upstream FA suite aborts on sm_70; fixed, 0 failures | **yes**, the clearest candidate |
| 0002 | pad non-power-of-two GQA for small batches | Volta dispatch + shared sink guard | Qwen +5.8 / +7.6 / +19.9% at pp4 / 8 / 16 @32K (with 0003) | with 0003 only; 2 small 4K regressions |
| 0003 | Volta FA configs for D=256, ncols 32/64 | Volta config rows | 2.26–2.65× kernel, +45% Qwen prefill @64K | overlaps #27997; check status first |
| 0004 | single-token quantised/BF16 decode uses VEC | Volta dispatch | 2.1–3.3× kernel; part of Llama +26.8% | yes (small, mirrors Turing) |
| 0005 | conversion launch geometry for short rows | **all NVIDIA archs** | 1.15–4.0× conversion, bit-identical | needs a non-Volta check |
| 0006 | vectorised q4_0/q8_0 dequant, non-contiguous | **all NVIDIA archs** | 1.67–6.92× conversion, bit-identical | needs a non-Volta check |
| 0007 | VEC FA packs Q heads per K/V head (quantised K, 1 token) | **all archs using VEC** | Llama q8_0 decode @16K +26.8% (series), q4_0 +56.2% | needs other-arch numbers or a Volta-only switch |
| 0008 | Volta tensor-core matmul for K-quants/IQ4_XS/Q8_0 at 5–16 tokens | Volta only (new file) | Qwen +33.8 / +38.8 / +37.9% at 8 / 12 / 16 tokens | technically ready; large (770-line file) |
| 0009 | activation-range MUL_MAT tests | local only | 35/35 with 0008; current path 18/35 | **never upstream** (fails on upstream) |

## 0001 — Volta FA: one heads-per-tile rule for kernel selection and dispatch

- **Problem:** kernel selection assumed the GQA optimisation whenever there is no ALiBi. The Volta dispatch also
  turns it off without a mask, for unpadded K/V, or for misaligned strides. Small batches then selected MMA but
  dispatched a 16-column instance, which sm_70 builds as `NO_DEVICE_CODE`, so the process aborts.
- **Change:** one helper used by both selection and dispatch, and never a column instance below 32 on Volta.
  `ggml/src/ggml-cuda/fattn.cu` (+47 lines changed), `tests/test-backend-ops.cpp` (+7).
- **Evidence:**
  - Upstream's full FLASH_ATTN_EXT suite aborts after 3964 OK: `fattn-mma-f16.cuh:1849 … no device code
    compatible with CUDA arch 700`. This was first seen in the 2026-10-10 Phase 0 session.
  - With 0001: 4004/4004, exit 0. This adds 9 crash-class tests.
- **Tests:** the 9 new cases abort on unmodified upstream by design (they are the regression test), so they must
  ship together with the fix.
- **Results:** `bench/results/patches-20261010-173522-*`. Details: `docs/research/patches-cloud-2026-10-10.md` §1,
  `docs/research/phase0-cloud-2026-10-10.md`.
- **Reproduce:** `test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT` on any sm_70 GPU, with and without the patch.
- **Note:** the selection helper is shared code. The behaviour change is meant to affect only Volta, but other
  architectures were not run.

## 0002 — Volta FA: pad non-power-of-two GQA ratios for small batches

- **Problem:** for GQA 3/5/6/7 at 3–16 query tokens, Volta streamed K/V once per 2-head group, i.e. G/2 times.
- **Change:** pack all Q heads of a K/V head into one MMA tile, padded to the next power of two with the padded heads
  masked. The sink read is guarded for masked heads. Exception: D ≤ 128 with fewer than 8 K/V heads × sequences keeps
  the exact 2-head tiles for GQA 6 above 8 tokens. `fattn.cu` (+20), `fattn-mma-f16.cuh` (+5, the sink guard),
  tests +16 lines (170 cases).
- **Kernel results** (f16 K/V, 4 K/V heads, with 0003):
  - D=256 GQA 6 at 32K: 2.48× at 8 tokens, 3.92× at 16.
  - D=128 GQA 5/7: 1.00–3.54×.
  - With 8 K/V heads, padding wins every cell, 1.04–3.17×.
- **Remaining regressions** (accepted, 4 K/V heads only, 5–11 µs per call):
  - D=128 GQA 3 at 16 tokens, 4K: 0.83×.
  - D=128 GQA 6 at 8 tokens, 4K: 0.91×.
  - The same cells gain 1.13–2.2× from 8K up.
- **End to end:**
  - Qwen3.8-27B f16 KV @32K: +5.8% / +7.6% / +19.9% at pp4 / pp8 / pp16, as predicted from the kernel times.
  - Llama-3.1-8B (GQA 4) unchanged (−1.5…+3.0%, noise).
- **Dependency:** ship only together with 0003. Without 0003 the padded D=256 instances spill, and 0002 alone is
  slower on several D=256 shapes (down to 0.83×).
- **Results:** `patches-20261010-*`, `pad-sweep-20261010-224625`, `pad-sweep-nh8-20261011-000501`,
  `final-v100-20261011-005730`. Docs: `docs/design/P6-small-batch-gqa.md`,
  `docs/research/pad-rule-vecpack-v100-2026-10-10.md` §1.
- **Not measured:** 2 or 16 K/V heads; several sequences; D = 80/96/112; end to end on a D ≤ 128 model with GQA 3 or 6.

## 0003 — Volta FA configs for D=256 with ncols 32 and 64

- **Change:** two config rows, `GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, {32, 64}, 128, 2, 32, 64, 64, 64, 2,
  false)`: Q in shared memory, nbatch_K2/V2/combine 64, 2 blocks per SM. `fattn-mma-f16.cuh` (+3).
- **Static (nvcc 12.8):**
  - Upstream's Ampere-fallback rows reach 255 registers with 464–640 B of stack at ncols=64.
  - With 0003: ncols=64 has 48–160 B of stack and 2 blocks per SM (the `pr_k64_c64` variant). ncols=32 (`<4,8>`)
    goes from 448 to 64 B of stack.
  - **Flag:** the commit message says "no register spills", but stack is not zero. Check with `-Xptxas -v` whether
    the remaining stack is spill or local arrays, and correct the message if needed.
- **Kernel results** (D=256 prefill, batch 512, GQA 2/4/6, KV 4K–128K):
  - #27997's row (V2 64, Q in smem): 1.41–1.45×.
  - 0003's ncols=64 row: **2.26–2.65×**, e.g. GQA 6 at 16K went from 11.62 ms to 4.40 ms.
  - Small batch: D=256 GQA 4/8 also gain 1.36–2.70× at 4–16 tokens (no spills, 2 blocks per SM).
- **End to end, Qwen3.8-27B prefill** (pp1024):

  | depth | change |
  | --- | --- |
  | 64K | **+45%** (#27997 alone +22%, which reproduces the PR author's +21.88%) |
  | 16K | +9% |
  | 4K | 0.91×, inconclusive (per-run stddev 4–9%, one cool-GPU outlier) |

- **Relationship to PR #27997:** the ncols=64 row extends that PR's change (K2 and combine 64 on top of V2 64 and Q
  in smem). The data could go to that PR as a measured suggestion rather than a separate PR. It was open as of
  2026-10-09; check its status now.
- **Results:** `pr27997-20261010-025229-*`, `pr27997-20261010-034851-*`, `pr27997-static-20261009-182401`. Tool:
  `bench/pr27997/` (`run.sh`, `static.sh`, README).
- **To do before posting:** more repetitions at 4K depth.

## 0004 — Volta FA: vector kernel for single-token decode with quantised or BF16 K/V

- **Problem:** the tile and MMA kernels convert the whole quantised/BF16 K/V view to F16 on every call. For one
  token, the conversion costs more than the attention. Turing already sends batch-1 quantised decode to VEC.
- **Change:** Volta does the same when the VEC instance for those K/V types is compiled. `fattn.cu` (+8), tests +16
  lines (D 128/256, q8_0/q4_0/bf16, GQA 4/6/8, 1–2 tokens, sinks, unpadded KV, permuted K/V, softcap).
- **Kernel results** (1 token, 32K, GQA 4):

  | D | q8_0 | q4_0 |
  | --- | --- | --- |
  | 128 | 3.31× | 2.86× |
  | 256 | 2.08× | 2.11× |

- **End to end** (0001–0004 vs upstream, Llama-3.1-8B tg64 @16K): q8_0 +16.7%, q4_0 +36.2%, f16 0.0%. Qwen
  (GQA 6) unchanged.
- **Results:** `patches-20261010-*`. Docs: `docs/design/P8-quantised-kv-decode.md`,
  `docs/research/patches-cloud-2026-10-10.md` §3.2, §4.2.

## 0005 — CUDA: fewer idle threads when converting short non-contiguous rows

- **Problem:** `dequantize_block_cuda` and `convert_unary_cuda` used 256 threads along a row and one block per row.
  A K/V cache view has one 64–256 value row per token and head, so most threads sat idle and the launch was bound
  by the block count.
- **Change:** only as many threads per row as needed, and several rows per block. Rows of ≥ 512 values (dequantize)
  or ≥ 256 (convert), and the contiguous paths, launch exactly as before. `convert.cu` (+25).
- **Scope:** **architecture-independent** (all NVIDIA GPUs).
- **Microbenchmark** (`bench/micro/dequant_nc.cu`, one K or V view per call), outputs bit-identical:
  - D=64: 3.79–4.04×.
  - Llama @16K D=128: 2.09–2.16×.
  - D=256: 1.14–1.16×.
  - Maxwell M2200 (0005 + 0006 together): 2.6–5.2×.
- **End to end:** contained in the 0004–0007 numbers. It speeds up the multi-token paths that still convert
  (tile/MMA with a quantised cache).
- **Results:** `patches-20261010-173522-*`. Docs: `docs/design/P8-quantised-kv-decode.md`.
- **To do before posting:** a timing on at least one non-Volta GPU (Ampere or newer). The sibling P100 and A770
  repos carry ports of 0005/0006 (`conv`/`conv8` variants), not yet run on their hardware either.

## 0006 — CUDA: vectorised dequantisation of non-contiguous q4_0/q8_0 rows

- **Change:** for the K/V views that FlashAttention converts to F16, convert 8 values per thread with 2-byte
  aligned loads and 8/16-byte stores. The arithmetic is the same as `dequantize_q4_0/q8_0`, so results are
  bit-identical. It falls back to the generic kernel if a row is not a whole number of blocks or the plane exceeds
  INT_MAX threads. `convert.cu` (+84).
- **Scope:** **architecture-independent**.
- **Microbenchmark** (vs upstream, q8_0 / q4_0), bit-identical:
  - D=64: **5.80× / 6.92×**.
  - Llama @16K: 3.08× / 3.51×.
  - Qwen @32K D=256: 1.74× / 1.93×.
  - Faster than an f16 device-to-device copy of the same output.
- **Kernel, 8 tokens @32K** (0001–0006 vs 0001–0004): D=128 1.95–2.21×, D=256 1.31–1.39×. Over all quantised
  cases, the series is min 1.00×, median 1.61×, max 3.37× vs upstream.
- **End to end:** Llama q8_0/q4_0 pp8 @16K did not follow the kernel gain (−1.7…+0.2%; not explained).
- **Results / to do:** as 0005.

## 0007 — CUDA: FA vector kernel packs the Q heads of a K/V head (single-token decode, quantised K)

- **Problem:** with GQA, VEC ran one block per Q head, so each K/V head was read and dequantised once per Q head.
- **Change:**
  - 2 or 4 Q heads per block, with padded heads (and their sinks) masked. Grid mapping as in the MMA kernel.
  - Rule: 4 heads for GQA ratios that are a multiple of 4, and for 7. GQA 3 gets 4 with ≥ 8 K/V heads × sequences,
    else 2. GQA 2 gets 2 with ≥ 8, else unpacked. Other ratios get 2; GQA 1 is unpacked.
  - Q is kept as q8_1, key loops are unrolled 4×, and sink columns are indexed at compile time.
  - `fattn-vec.cuh` (+120 lines changed), tests +19 lines.
- **Static (sm_70):** 76–163 registers, 0 spill, 0 local memory. It also removes most local memory from the
  existing 2-token kernels.
- **Kernel results** (vs unpacked):
  - 4 K/V heads: up to 2.45× (D=256 GQA 8 q8_0, 32K). Some 4K cells are below 1: GQA 3 down to 0.88×, D=64 GQA 5
    q8_0 0.95×.
  - 8 K/V heads: 1.02–2.53×, median 1.52×, no cell below 0.97×.
- **End to end** (0001–0007 vs upstream, tg64 @16K):

  | model, KV | change |
  | --- | --- |
  | Llama-3.1-8B q8_0 | **+26.8%** (packing alone +8.6%) |
  | Llama-3.1-8B q4_0 | **+56.2%** (packing alone +14.4%) |
  | Llama-3.1-8B f16 | 0.0% |
  | Qwen3.8-27B q8_0 | +5.8% |

  With the series, Llama with a quantised cache is 5% behind f16; upstream is 25% behind.
- **Known cost:** D=256 f16 single-token decode at GQA 3/5/6/7 runs at 0.95–0.98× (it shares code with the unpacked
  instances; not fixed, about 0.1–0.7% of Qwen f16 decode).
- **Correctness:** FLASH_ATTN_EXT 4358/4358 on sm_70. On Maxwell (sm_52), 307 quantised single-token cases pass and
  compute-sanitizer memcheck reports 0 errors.
- **Scope:** **VEC is used on all architectures.** The packing rule was tuned on a V100 only, so it needs
  measurements on other GPUs or a Volta-only switch before upstreaming.
- **Results:** `vecpack-u4-v100-20261011-002409`, `final-v100-20261011-005730`, `rev4-v100-20261011-032246`. Docs:
  `docs/design/P8-change4-packed-vec.md`, `docs/research/pad-rule-vecpack-v100-2026-10-10.md` §2,
  `docs/research/qwen-smallbatch-and-vecpack-heads-v100-2026-10-11.md`.

## 0008 — CUDA: Volta tensor-core matmul for Q4_K/Q5_K/Q6_K/IQ4_XS/Q8_0 weights at 5–16 tokens

- **Problem:** on Volta at 5–16 tokens, quantised matmuls use MMVQ (which redoes the dequantisation for each column)
  or MMQ, which is dp4a on Volta (no int8 tensor cores). Weights stream at 205–365 GB/s at 9–16 tokens, against
  570–775 GB/s at 1 token.
- **Change:** new file `mmq-volta.cu` (770 lines) and header (+11), a routing branch in `ggml-cuda.cu` (+5), and
  tests (+29).
  - Weights are read once in their GGUF layout, staged through shared memory, and decoded to FP16
    `mma.sync.m8n8k4` fragments. The decode tricks are exact: OR into 0x6400 plus one HFMA2; Q6_K with a single
    HFMA2.
  - Activations are converted to FP16 after a power-of-two scale per 256 values, which is exact and undone in FP32.
  - K is split across blocks according to the kernel's occupancy, with a deterministic last-block reduction.
- **Routing:**
  - Volta only, for K % 256 == 0 (Q6_K: K % 1024), contiguous 2-D weights and ≤ 16 tokens.
  - N ≥ 4096 from 5 (Q4_K), 6 (Q5_K), 7 (Q6_K), 8 (IQ4_XS) or 9 (Q8_0) tokens.
  - N ≤ 128 wherever MMQ would run.
  - Not used when the graph asks for F32/BF16 activations.
  - `GGML_CUDA_NO_MMQ_VOLTA=1` turns it off.
- **Per matmul** (vs the current path, minimum of 3 rounds):
  - Q4_K/Q5_K/IQ4_XS: 1.21–1.76× at 8 tokens, 1.32–1.84× at 16 (Qwen shapes).
  - Q6_K: 1.10–1.45×.
  - Q8_0: 1.33–1.62× at 16 tokens.
  - N = 48: 1.06–1.74×.
- **End to end:**

  | model | 1–4 tokens | 8 | 12 | 16 |
  | --- | --- | --- | --- | --- |
  | Qwen3.8-27B | unchanged | +33.8% | +38.8% | +37.9% |
  | Llama-3.1-8B | unchanged | unchanged | +27.7% | +27.0% |

- **Accuracy** (llama-perplexity KL divergence vs an FP32 reference, i.e. dequantise + SGEMM, 4096 tokens at
  batch 16):

  | model | mean KLD, 0008 | mean KLD, current path |
  | --- | --- | --- |
  | Qwen | 0.000104 | 0.0017 |
  | Llama | 0.000053 | 0.00030 |

  0008 is closer to exact than the current path because activations are not quantised to 8 bits.
- **Static:** 167–255 registers, 0 spill, 0 stack, 0 local memory in all 15 instances.
- **Correctness:**
  - Committed series on a clean tree: MUL_MAT 2628/2628 (plus 35 local range cases), MUL_MAT_ID 931/931.
  - The new upstream-list tests use normal-range inputs, which the existing paths handle. They were not re-run on
    unmodified upstream for the final list, so do that on the rebased tree.
- **Results:** `p9-final4-*`, `p9-scale2-20261011-195231`, `p9-verify-20261011-210439` and earlier `p9-*`. Doc:
  `docs/design/P9-small-batch-qmatmul.md`.
- **Considerations:**
  - It is a 770-line Volta-only kernel. Maintainers' appetite for that much Volta-specific code is unknown, although
    upstream did take a Volta MMA FA path (#17505).
  - The remaining gap to the llama-superfast-v100 fork (46.2 vs 32.8 ms of matmul per 16-token Qwen evaluation)
    needs repacked weights. That is not proposed.
- **Not measured:** other models or weight types (IQ4_NL, Q3_K are not routed); multi-GPU.

## 0009 (local only) — tests: MUL_MAT with activations beyond FP16's range

- `test_mul_mat_range`: 35 cases, with src1 scaled by 1e5 (above 65504) or 1e−5 (FP16 subnormal), in three
  magnitudes per token.
- Results:
  - With 0008: 35/35.
  - Current path: 18/35, because block_q8_1 stores its scale and sum in FP16.
  - CPU reference: fails Q8_0 at 1e−5, so those cases are skipped.
- **Do not upstream:** the cases fail on unmodified upstream. The q8_1 overflow is an upstream finding the owner may
  want to report separately, in their own words.
- File: `third_party/patches-local/0009-*.patch`.
