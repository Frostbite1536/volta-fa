# Padding rule for 0002 and packed VEC (0007) on a V100 (2026-10-10/11, session 2 extension)

**Later (2026-10-11):** 0007's rule was refined with 8-K/V-head data (GQA 2 and 3 pack with ≥ 8 K/V heads), and the
D=256 f16 cost in §4 was attributed to 0007: `qwen-smallbatch-and-vecpack-heads-v100-2026-10-11.md`.

Same Vast.ai instance as `patches-cloud-2026-10-10.md` (Tesla V100-PCIE-32GB, 1380 MHz, CUDA 12.8.93, llama.cpp
`79e2e74`), used after that write-up while it was still rented. Clocks 1230–1380 MHz and 32–58 °C in every
timed pass (`run.log` in each results directory).

The two design questions use one binary per question with a **bench-only environment switch** (in each results
directory's `*.patch`, not in the patch series), so the compared modes differ only in the routing decision:
- `VFA_NOPAD=1` turns off 0002's padding (exact divisors, as upstream) in a 0001–0006 build.
- `VFA_VEC_NCOLS2=1|2|4|8` forces the Q heads per VEC block in a 0001–0006 + 0007 build (1 = unpacked = `conv8`).

The final series was then measured without switches against upstream (§4).

Raw data (`bench/results/`):

| directory | what | rounds |
| --- | --- | --- |
| `pad-sweep-20261010-224625` | padded vs exact, f16, 4 K/V heads, D 64/128, GQA 3/5/6/7, KV 4–32K, 3–16 tokens | 3 (r1 contaminated, see below) |
| `pad-sweep-nh8-20261011-000501` | same, 8 K/V heads, KV 2–8K, 6–16 tokens | 3 |
| `vecpack-v100-20261010-231820` | heads per VEC block 1/2/4/8, first prototype (key loops not unrolled); Llama-3.1-8B tg64 | 3 (r1 contaminated) + 2×3 reps |
| `vecpack-u4-v100-20261011-002409` | same with key loops unrolled 4× | 2 + 2×3 reps |
| `rev-correctness-20261011-001850` | full FA + MUL_MAT, intermediate rule | – |
| `rev2-correctness-20261011-005141` | full FA + MUL_MAT, final 0001–0006 (+ first 0007) | – |
| `final-v100-20261011-005730` | final 0001–0007: full FA, kernel A/B vs upstream and vs unpacked, llama-bench vs upstream | 2 (+ 2×3 reps) |

**Measurement caveat found here:** a CPU-heavy build running on the host (even at `nice 19`) roughly doubles
these short kernels' times (round 1 of the 4-head sweep: 1.5–2.1× slower in *both* modes; round 1 of the first
vecpack run: 1.4–1.5× in the packed modes). Rounds without a concurrent build agree to a median 0.2–0.7%.
The tables below use only clean rounds (4-head sweep: medians of 3, where the contaminated round is always the
slowest and so never the median; first vecpack run: rounds 2–3 only). Never build while timing.

## 1. Padding (0002) vs exact divisors, f16 K/V

Ratio = exact time / padded time (> 1: padding faster).

**4 K/V heads** (as in the session 2 kernel A/B): padding wins from 8K up everywhere except one block of cells.
- **GQA 6 above 8 tokens (10/12/16), D=64 and D=128: padding loses at every KV length**, 0.70–0.87×
  (D=128: 4K 0.70, 8K 0.76–0.78, 16K 0.84–0.86, 32K 0.85–0.87). These are the shapes where the exact divisor
  already runs on tensor cores (`<16,2>`, n_q × 2 > 16), and padding cuts the number of tiles by 3×.
- At 4K only: D=128 GQA 3 at 10–16 tokens 0.85–0.88×, D=128 GQA 6 at 6–8 tokens 0.91–0.94×, D=64 GQA 6 at
  8 tokens 0.99×. From 8K up these cells are 1.13–2.2×.
- GQA 5 and 7 (the ratios of real D=128 models, e.g. Qwen2.5/3) win in every cell: 1.00–3.54×.

**8 K/V heads** (KV 2K/4K/8K, 6–16 tokens): **padding wins in every cell, 1.04–3.17×**, including GQA 6 above 8
tokens (D=128: 1.11–1.15× at 2K, 1.28–1.30× at 4K, 1.37–1.38× at 8K) and the cells that lost at 4K with 4 heads
(GQA 3 at 10–16 tokens 1.26–1.30×; GQA 6 at 6–8 tokens 1.38–1.42×).

The 4-head losses are therefore a parallelism effect (too few blocks for 80 SMs with 4 K/V heads), not a cost of
the padded columns. A first rule, "never pad where the exact divisor reaches MMA, for D ≤ 128", was written from
the 4-head data alone; the 8-head sweep showed it would give up 1.04–1.38× on 8-head models, so it was replaced.

**Rule now in 0002:** pad as before, except for D ≤ 128 with fewer than 8 K/V heads × sequences, where GQA 6 above
8 tokens keeps the exact 2-head MMA tiles. Confirmed in §4 (GQA 6, 16 tokens: 1.00× vs upstream at 4K and 32K,
was 0.69× / 0.86×). Remaining and accepted: D=128 GQA 3 at 16 tokens, 4K (0.83×) and GQA 6 at 8 tokens, 4K
(0.91×), 4 K/V heads only, 5–11 µs per call; the same cells gain 1.13–2.2× from 8K up and with 8 K/V heads.
Removing them would need a KV-length or head-count threshold set from data at 2–8 K/V heads × 4–8K; not done.

Not measured: 2 or 16 K/V heads, several sequences, D=80/96/112 (the rule treats them like 64/128), end to end on
a D ≤ 128 model with GQA 3 or 6 (none in the model set).

## 2. Packed VEC (0007), quantised K/V, 1 token, 4 K/V heads

**The first prototype lost.** Without unrolled key loops (chosen for occupancy) the 8-head instance took ~250 µs
at D=128 32K at every GQA, and its rule (4 heads for GQA 3–4, 8 for GQA ≥ 5) was slower than unpacked in 42 of
72 packed cells (to 0.46×, mostly at 4K and D=256); Llama-3.1-8B tg64 @16K: q8_0 +0.6% (within range), q4_0 +6.0%. The per-head cost
was not K/V re-reads as the Change 4 gate assumed: unpacked time per Q head is ~37 µs at D=128 32K whether GQA
is 4, 6 or 8, and packing alone did not remove it.

**Unrolled 4×** (sm_70: 0 spill, 0 stack, 118–163 registers at 4 heads, 76–120 at 2), the packed instances are
1.19–1.59× faster than the first prototype in the same cells (10th–90th percentile, median 1.40×). 8 heads per block was never faster than 4. Rules were scored
against all 72 cells (`vecpack-u4`): the chosen one — **GQA ≥ 3; 4 heads per block for a multiple of 4 and for 7,
otherwise 2; GQA 1–2 unpacked** — has 4 cells below 0.97×, all at 4K (GQA 3: 0.86–0.94×; D=64 GQA 5 q8_0 0.93×).
GQA 2 packed would gain 1.15–1.45× at 32K but loses 0.78× at D=256 4K (the Gemma shape), so it stays unpacked.

Final patch vs unpacked (`final-v100`, same session, 2 rounds), unpacked/final:

| D | type | GQA 3 4K / 32K | GQA 4 | GQA 5 | GQA 6 | GQA 7 | GQA 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | q8_0 | 0.88 / 1.14 | 1.02 / 1.49 | 0.95 / 1.28 | 1.13 / 1.34 | 1.15 / 1.44 | 1.31 / 1.59 |
| 128 | q8_0 | 1.04 / 1.17 | 1.31 / 1.51 | 1.09 / 1.31 | 1.49 / 1.59 | 1.29 / 1.78 | 1.38 / 1.86 |
| 256 | q8_0 | 0.92 / 1.41 | 1.01 / 2.03 | 1.10 / 1.63 | 1.42 / 1.90 | 1.35 / 2.20 | 1.46 / 2.45 |
| 64 | q4_0 | 0.90 / 1.08 | 1.10 / 1.38 | 1.03 / 1.22 | 1.21 / 1.41 | 1.26 / 1.55 | 1.49 / 1.74 |
| 128 | q4_0 | 1.31 / 1.39 | 1.51 / 1.73 | 1.23 / 1.50 | 1.73 / 1.85 | 1.48 / 1.96 | 1.62 / 2.19 |
| 256 | q4_0 | 1.14 / 1.38 | 1.31 / 2.05 | 1.15 / 1.52 | 1.59 / 2.04 | 1.59 / 2.31 | 1.76 / 2.32 |

GQA 1–2 (unpacked path, code shared with the packed kernel): 0.98–1.07× except q8_0 D=128 GQA 1 at 32K, 0.94×.

## 3. Correctness on sm_70

| build | FLASH_ATTN_EXT | MUL_MAT |
| --- | --- | --- |
| intermediate rule + first 0007 (`rev-correctness`) | 4332/4332, rc 0 | 2444/2444, rc 0 |
| final 0001–0006 + first 0007 (`rev2-correctness`) | 4334/4334, rc 0 | 2444/2444, rc 0 |
| **final 0001–0007** (`final-v100`) | **4334/4334, rc 0** | (0001–0006 identical to rev2) |

4334 = session 2's 4200 + 74 from the extended 0002 tests (D=64 added, 12 tokens, two 8-head cases) + 60 from
0007. (Session 2's write-up counted "4202 / 2446 OK": those include two "OK" summary lines; the test counts were
4200 / 2444.) Laptop (Quadro M2200, sm_52) for the final 0007: 4334/4334, compute-sanitizer memcheck 0 errors on
42 packed cases.

## 4. Final series vs upstream (same session, interleaved)

Kernel, f16 small batches (`final-v100`, 2 rounds, median spread 0.24%): as in session 2 except that **D=128
GQA 6 at 16 tokens is now 1.00× (4K and 32K)**. Remaining below 0.98×: D=128 GQA 3 16 tokens 4K 0.83×,
D=128 GQA 6 8 tokens 4K 0.91× (§1), and **D=256 f16 single-token decode at GQA 3/5/6/7: 0.96–0.98×** (e.g.
GQA 6 32K 382 → 395 µs). The last is new with 0007, which shares code with these unpacked VEC instances: same
registers (200 → 202) and shared memory, and the SASS has slightly *fewer* instructions (2728 vs 2760, same
loads/FMAs), so it is not added work; not attributed further (no profiler in the container). D=128 at the same
shapes is 1.03–1.05× faster.

End to end, llama-bench `-fa on -p 0 -n 64 -d 16384`, 2 rounds × 3 reps, median [min–max]:

| model | KV | upstream t/s | final t/s | change |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B | q8_0 | 49.27 [48.80–49.34] | 62.48 [61.91–62.63] | **+26.8%** |
| Llama-3.1-8B | q4_0 | 40.16 [39.79–40.18] | 62.72 [61.93–62.74] | **+56.2%** |
| Llama-3.1-8B | f16 | 65.79 [65.08–65.85] | 65.81 [65.05–65.85] | 0.0% |
| Qwen3.8-27B | q8_0 | 29.42 [28.94–29.44] | 31.13 [30.41–31.17] | +5.8% |

Llama-3.1-8B with a q8_0 cache is now 5% behind f16 (upstream: 25%), with q4_0 also 5%. Packing alone (same
binary, `vecpack-u4`): q8_0 57.62 → 62.59 (+8.6%), q4_0 54.74 → 62.64 (+14.4%). Qwen3.8-27B's quantised decode
had not moved with 0001–0006 (§4.2 of the session 2 write-up); its GQA 6 attention now packs 2 heads per block.

## 5. Against the acceptance criteria (changes from session 2)

- P-6 "no f16 shape slower": the GQA 6 / 16-token regressions are fixed. Two 4K cells at 4 K/V heads remain
  (0.83×, 0.91×), plus the 0.96–0.98× D=256 single-token f16 cells from 0007.
- P-8 "Llama-3.1-8B q8_0 tg64 @16K ≥ +25%": **now met (+26.8%)** with 0007.
- Change 4: designed, built and measured; the gate's premise (re-reads dominate) was only partly right — packing
  needed the unrolled loops to pay.

## 6. Not measured

Packing with 8 K/V heads at kernel level (Llama end to end has 8), GQA > 8 (rule gives 4 heads for 12/16, 2 for
odd ratios), more than one sequence, f16 Qwen3.8-27B decode with 0007 (the 0.96–0.98× kernel cells), perplexity.
