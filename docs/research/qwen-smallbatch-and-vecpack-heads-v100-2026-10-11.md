# Where the fork's Qwen small-batch gain comes from; packing vs K/V-head count (V100, 2026-10-11)

Same Vast.ai instance as `patches-cloud-2026-10-10.md` and `pad-rule-vecpack-v100-2026-10-10.md` (Tesla
V100-PCIE-32GB, 1380 MHz, CUDA 12.8.93, llama.cpp `79e2e74`, fork llama-superfast-v100 `70df871` built with
session 1's flags: gcc-14, `-DGGML_CUDA_CUB_3DOT2=ON`). All builds finished before any timing; clocks
1230–1380 MHz, 34–48 °C (`run.log`). Nsight Systems 2024.6.2 (`cuda-nsight-systems-12-8`) for kernel timelines;
Nsight Compute counters are blocked in this container (`ERR_NVGPUCTRPERM`). Raw `.nsys-rep` files are not committed;
their per-kernel sums are (`kern-*_cuda_gpu_kern_sum.csv`).

Raw data: `bench/results/ce-v100-20261011-024315/` (e1, e2, Qwen llama-bench, nsys, QPN on/off) and
`bench/results/rev4-v100-20261011-032246/` (final 0007: full FA suite, rule vs unpacked).

**llama-bench note:** in every process the first of the 3 repetitions is 25–35% slower than the other two (e.g.
Qwen pp16 upstream 148 / 206 / 206 t/s). Medians of the samples are used below; `bench/analyze.py lb` averages
`avg_ts`, which includes the slow repetition (it gives 187 t/s for the same runs). Comparisons hold either way.

## 1. (c) Qwen3.8-27B small batches: the gain is the fork's quantised matmul

llama-bench, `-p 1,4,8,16 -n 0 -d 0 -fa on`, 2 interleaved rounds × 3 reps, median of samples, t/s:

| | pp1 | pp4 | pp8 | pp16 |
| --- | --- | --- | --- | --- |
| upstream | 32.6 | 96.6 | 132.8 | 205.3 |
| fork | 30.5 (−6%) | 122.8 (+27%) | 245.7 (**+85%**) | 387.7 (**+89%**) |
| fork, `LLAMA_MMVQ_QPN=0` | 36.2 (+11%) | 111.8 (+16%) | 130.3 (−2%) | 133.7 (−35%) |

The gain is depth-independent (session 1: +81% at 4K, +112% at 32K for pp16; here +89% at depth 0), so it is
not attention. Per-kernel GPU time per 16-token evaluation (nsys, 5 runs + warmup = 6 evaluations; the fork's
one-time weight repack, 463 launches / 177 ms in both the pp1 and pp16 runs, is excluded):

| | upstream | fork | fork, QPN off |
| --- | --- | --- | --- |
| quantised matmul | **68.5 ms** (`mul_mat_q`, mmq_x 16) | **32.8 ms** (`qpn_mul*`) | 114.8 ms (`mul_mat_vec_q` + quantize) |
| everything else | 10.2 ms | 8.3 ms | 6.7 ms |
| total GPU | 78.7 ms | 41.1 ms | 121.4 ms |

So **~95% of the fork's small-batch gain (35.7 of 37.6 ms per step) is its matmul**, and with that path off the
fork is no faster than upstream at 8–16 tokens. Attention, copies, RoPE and the SSM (gated delta net) layers
differ by ~2 ms in total.

What the fork does (`ggml/src/ggml-cuda/mmvq-qpn.cu`, 3322 lines; header comment): Volta tensor-core products
(`mma.sync.m8n8k4`) for K-quants (Q4_K/Q5_K/Q6_K, also Q3_K, Q8_0, IQ4_XS/NL) at 1–8 tokens per pass, on weights
**repacked once at load, in place and at the same size, into mma fragment order** (one 16-byte chunk per lane, no
shared-memory staging), codes turned into fp16 with one HFMA2, activations in fp16 with a power-of-two range scale
per 256 columns (non-finite slices fall back to fp32), split-K across warps. It credits NInfer's QPN kernels
(Apache-2.0) and "v100-skinny". Its matmul time is nearly flat from 1 to 16 tokens (29.5 → 32.8 ms per evaluation:
weight-bandwidth bound), while upstream's `mul_mat_q` on Volta uses dp4a (no int8 tensor cores on sm_70) and is
2.7× slower at 16 tokens than its own 1-token `mul_mat_vec_q` (25.0 ms).

Implication: on this model the remaining small-batch gap is not an attention problem. A Volta skinny quantised
GEMM on tensor cores is in this repo's scope (sm_70, m8n8k4) and is the next design candidate (P-9), with the
fork's QPN as prior art. It needs a repacked weight layout, which upstream's CUDA backend has no mechanism for
today (the fork adds a tensor flag and aborts if any other op reads a repacked tensor), so the design question is
as much ggml plumbing as kernel.

## 2. (e1) The 0.96–0.98× on D=256 f16 single-token decode comes from 0007

f16, 1 token, 4 K/V heads, 3 interleaved rounds (per-cell range median 0.3–0.5%, max 2.6%), upstream / x:

- `conv8` (0001–0006) vs upstream: 0.995–1.006 in every cell.
- final (0001–0007): **D=256 GQA 3/5/6/7: 0.95–0.98** (GQA 7 32K 434.8 → 456.7 µs), **D=128 GQA 3/5/6/7: up to
  1.05 faster** (GQA 7 32K 246.9 → 235.6 µs); GQA 4/8 (TILE path) 1.00.

These are the unpacked VEC instances, whose source 0007 changes (indexing helpers, compile-time sink loop,
dst/meta indexing). Registers 200 → 202 (D=256), same shared memory; SASS of the D=256 f16 kernel has 2728 vs
2760 instructions with the same loads, shuffles and FMAs, so it is scheduling, not added work. For a single
column the old and new sink loops do the same work (and the perf cases have no sinks), so restoring the old loop
was not tried. Impact estimate for Qwen3.8-27B f16 decode (16 attention layers of this shape): +0.03 ms per token
at 4K, +0.23 ms at 32K, i.e. 0.1–0.7% of a ~33–35 ms token. Recorded, not fixed.

## 3. (e2) Packing vs number of K/V heads

With 8 K/V heads (D 128/256, q8_0/q4_0, GQA 2–8, KV 4K/32K) packing never lost: 1.00–2.53× under the previous
rule. Two choices differed from the 4-head optimum:
- **GQA 2 packed 2 per block: 1.10–1.75×** with 8 K/V heads; with 4 K/V heads 0.90–0.91× at D=64 q8_0 (8K, 16K)
  and 0.78× at D=256 q8_0 4K (earlier run), 1.00–1.47× otherwise.
- **GQA 3: 4 per block beats 2** with 8 K/V heads (D=128 q8_0: 1.21 vs 0.99× at 4K, 1.44 vs 1.25× at 32K).

Same pattern as 0002: packing divides the block count, which only hurts with few K/V heads. **0007's rule now:**
4 heads per block for multiples of 4 and 7; GQA 3 → 4 with ≥ 8 K/V heads × sequences, else 2; GQA 2 → 2 with
≥ 8, else unpacked; other ratios → 2; GQA 1 unpacked. Tests added: GQA 2/3 with 8 K/V heads × D × type × sinks (24).

Final 0007 vs unpacked (`rev4-v100`, 2 rounds, median spread 0.46%, p95 4.9%):
- 8 K/V heads: **1.02–2.53×, median 1.52×, no cell below 0.97** (GQA 2: 1.02–1.75; GQA 3: 1.07–1.87).
- 4 K/V heads: 0.85–2.44×, median 1.24×; below 0.97: GQA 3 at 4K (D=64 0.85/0.87, D=256 q8_0 0.93), D=64 GQA 5
  q8_0 4K 0.93, and two GQA 1–2 cells (0.93, 0.94) where both binaries run the same unpacked code (noise).

Correctness: full FLASH_ATTN_EXT **4358/4358** on sm_70 (4334 + 24); laptop sm_52: the 307 quantised single-token
cases pass, memcheck 0 errors on the 24 new ones. The installed `third_party/patches/0007-*.patch` is the file
tested here (sha256 `e8404e84f3c23bdb…`, `rev4-v100/patches.sha256`); 0001–0006 unchanged.

End to end is unchanged from `pad-rule-vecpack-v100-2026-10-10.md` §4 for the measured models (Llama-3.1-8B GQA 4
and Qwen3.8-27B GQA 6 route as before); no GQA 2/3 model with a quantised cache was measured.

## 4. Not measured

A GQA 2 model end to end (e.g. Gemma 2/3, D=256, 8–16 K/V heads), 16+ K/V heads, the fork's QPN accuracy
(perplexity) and its memory/latency cost of the load-time repack beyond the 177 ms of GPU time seen here.
