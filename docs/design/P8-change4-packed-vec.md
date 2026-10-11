# P-8 Change 4: GQA-packed VEC kernel for quantised-KV decode

Status: **patch `third_party/patches/0007-*.patch`, measured on a V100** (`docs/research/pad-rule-vecpack-v100-2026-10-10.md`).
Full FA suite 4334/4334 on sm_70 and on the laptop's sm_52 (memcheck clean). Llama-3.1-8B tg64 @16K vs upstream:
q8_0 **+26.8%**, q4_0 +56.2%, f16 0.0%; Qwen3.8-27B q8_0 +5.8%. Kernel: 1.08–2.45× at 32K for GQA 3–8 vs
unpacked. Known costs: GQA 3 at 4K (0.88–0.92×; D=64 GQA 5 q8_0 0.95×) and D=256 f16 single-token decode at
GQA 3/5/6/7 (0.96–0.98×, unpacked instances whose code 0007 touches; not attributed).
In the harness as variant `vecpack` (0001–0007).

## Why

With `0004`, single-token decode with a quantised cache uses VEC on Volta. VEC runs one block per Q head, so a
K/V head is read and dequantised once per Q head. On the V100 the unpacked time per Q head is about constant
(D=128 q8_0 32K: ~37 µs at GQA 4, 6 and 8), which tripped the Change 4 gate in `P8-quantised-kv-decode.md`.

## What the patch does (`fattn-vec.cuh`, plus tests)

- The kernel template gets `ncols2` (Q heads per block) next to the existing token count `ncols1`. Block
  column `jc` is token `jc / ncols2`, Q head `jc % ncols2`. The grid mapping is the MMA kernel's,
  `blockIdx.z = (sequence, K/V head, head group)`, with padded heads masked. For `ncols2 == 1` it reduces
  exactly to the old `(sequence, head)` indexing.
- Per column: the Q pointer, the validity check (token in the batch, head not padding), the mask row (shared
  by the heads of a token), the sink (guarded for padded heads), and the output and meta indices. ALiBi
  is excluded (one slope per block).
- **Rule** (quantised K, one token, no softcap, no ALiBi): GQA ≥ 3; **4 heads per block when the GQA ratio is a
  multiple of 4 or is 7** (7 padded to 8), **otherwise 2** (3 → 4, 5 → 6, 6 exact). GQA 1–2 unpacked. Chosen by
  scoring rules against 72 measured V100 cells (D 64/128/256 × q8_0/q4_0 × GQA 3–8 × KV 4K/32K): 8 heads per
  block was never faster than 4; GQA 6 is fastest as 3 blocks of 2; packing GQA 2 gains 1.15–1.45× at 32K but
  loses 0.78× at D=256 4K.
- Quantised K keeps Q as q8_1 (a few registers per column); F16/BF16 K keep Q as floats and are not packed.
- **The key loops of packed instances are unrolled 4×** (fully unrolled they spill; not unrolled they were
  1.19–1.59× slower on the V100, median 1.40×). Registers on sm_70 (nvcc 12.8), all with 0 spill and 0 stack:

  | D | 2 heads (q8_0 / q4_0) | 4 heads (q8_0 / q4_0) |
  | --- | --- | --- |
  | 64 | 76 / 78 | 118 / 125 |
  | 128 | 80 / 80 | 126 / 128 |
  | 256 | 120 / 120 | 161 / 163 |

- The sink loop indexes columns at compile time (indexing per-column register arrays with `threadIdx.y` put
  them in local memory). The same rewrite cuts upstream's 2-token instances from 48–80 B of stack to 0–16 B.
- The KQ shared buffer is sized for the softmax values (`ncols × nthreads`), not just `ncols × D`. Upstream's
  ≤ 2 columns always fit; 8 columns at D=64 did not (14 inf/NaN failures in the first build, which packed 8).
- Architecture scope: the packing is not Volta-specific. Any GPU whose selection picks VEC for quantised decode
  (Turing+, Pascal, Maxwell at ≤ 2 tokens) uses it. Measured only on the V100 and (correctness, relative timing
  of the first prototype) the laptop's Maxwell GPU; needs measurements elsewhere or a Volta-only switch before
  upstreaming.

## History (why the first prototype was wrong)

The first version packed 4 heads for GQA 3–4 and 8 for GQA ≥ 5, with the key loops not unrolled to save
registers. On Maxwell it looked good (D=256 2.8–4.3×). On the V100 it lost in 42 of 72 cells (to 0.46×) and
gave Llama q8_0 +0.6%: the per-block latency, not the K/V re-reads, dominated. Unrolling 4× fixed the latency,
and the V100 grid replaced the 4/8 rule. Bench-only switches used for this are in the results' `*.patch` files.

## Tests (60 new cases)

D 64/128/256 × q8_0/q4_0 × GQA 3/5/7/12 × sinks on/off, plus 2 sequences (GQA 6) and ALiBi (GQA 8, stays
unpacked). With the final rule these cover 2 heads padded (GQA 3, 5) and exact (GQA 6, 2 sequences), 4 heads
padded (GQA 7) and exact (GQA 12; GQA 4/8 come from `0004`'s cases).

## Not measured

8 K/V heads at kernel level (the Llama end-to-end result has 8), GQA > 8, several sequences, f16 Qwen3.8-27B
decode with 0007, perplexity, other architectures.
