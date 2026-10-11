# P-8 Change 4: GQA-packed VEC kernel for quantised-KV decode

Status: **patch `third_party/patches/0007-*.patch`, measured on a V100** (`docs/research/pad-rule-vecpack-v100-2026-10-10.md`).
Full FA suite 4358/4358 on sm_70 (laptop sm_52: quantised single-token cases pass, memcheck clean). Llama-3.1-8B tg64 @16K vs upstream:
q8_0 **+26.8%**, q4_0 +56.2%, f16 0.0%; Qwen3.8-27B q8_0 +5.8%. Kernel: 1.08–2.45× at 32K for GQA 3–8 vs
unpacked with 4 K/V heads, 1.02–2.53× (median 1.52×, no loss) with 8. Known costs: with 4 K/V heads, GQA 3 at 4K
(0.85–0.93×) and D=64 GQA 5 q8_0 4K (0.93×); D=256 f16 single-token decode at GQA 3/5/6/7 0.95–0.98× (shown to come
from 0007's changes to the unpacked instances; scheduling, not added instructions; ~0.1–0.7% of Qwen3.8-27B f16
decode). D=128 at the same shapes is up to 1.05× faster.
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
- **Rule** (quantised K, one token, no softcap, no ALiBi): **4 heads per block when the GQA ratio is a multiple
  of 4 or is 7** (7 padded to 8), **otherwise 2** (5 → 6, 6 exact); with **≥ 8 K/V heads × sequences, GQA 3 uses 4
  and GQA 2 uses 2**, with fewer GQA 3 uses 2 (padded to 4) and GQA 2 stays unpacked. GQA 1 unpacked. Chosen by
  scoring rules against 72 measured V100 cells with 4 K/V heads (8 heads per block never faster than 4; GQA 6
  fastest as 3 blocks of 2), then refined with 8 K/V heads, where packing never lost and GQA 2 (1.10–1.75×) and
  GQA 3 at 4 per block won (`docs/research/qwen-smallbatch-and-vecpack-heads-v100-2026-10-11.md`). Packing divides
  the block count, which only hurts with few K/V heads and short K/V.
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

## Tests (84 new cases)

D 64/128/256 × q8_0/q4_0 × GQA 3/5/7/12 × sinks on/off, plus 2 sequences (GQA 6) and ALiBi (GQA 8, stays
unpacked), plus GQA 2/3 with 8 K/V heads × sinks (24 more, 84 in total). With the final rule these cover 2 heads padded (GQA 3, 5) and exact (GQA 6, 2 sequences), 4 heads
padded (GQA 7) and exact (GQA 12; GQA 4/8 come from `0004`'s cases).

## Not measured

16+ K/V heads, GQA > 8, several sequences, a GQA 2 model end to end, f16 Qwen3.8-27B decode with 0007,
perplexity, other architectures.
