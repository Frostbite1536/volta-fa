# Patch series 0001–0006 on a cloud V100 (2026-10-10, session 2)

Hardware: Vast.ai, **Tesla V100-PCIE-32GB** (same part as session 1 and the home box), 1380 MHz max SM clock,
250 W, driver 580.173.02, CUDA 12.8.93, Ubuntu 24.04 container, 24 vCPUs. llama.cpp `79e2e74`
(`bench/LLAMA_COMMIT`), patches as committed in `8533d81`. Single GPU. Clocks held at 1380 MHz, 44–55 °C,
no throttle reasons logged in any round (`run.log`).

Raw data:
- Run 1 (correctness, microbench, kernel A/B, 3 interleaved rounds): `bench/results/patches-20261010-173522-9180cf2b1457/`
- Run 2 (llama-bench, 2 interleaved rounds × 3 reps): `bench/results/patches-20261010-205002-9180cf2b1457/`
- f16-KV decode baseline added after run 2 (same builds, same session): `bench/results/patches-f16base-20261010-212029/`
  (also `gguf-meta.txt`: layer layout of both models)

Variants are cumulative: `fix` = 0001, `p6` = 0001–0002, `p6cfg` = 0001–0003, `qkv` = 0001–0004,
`conv` = 0001–0005, `conv8` = 0001–0006.

## 1. Correctness

| variant | FLASH_ATTN_EXT (full suite) | MUL_MAT |
| --- | --- | --- |
| upstream | **abort** after 3964 OK (`fattn-mma-f16.cuh:1849 … no device code compatible with CUDA arch 700`, the session 1 §4 crash) | 2446 OK, 0 FAIL |
| fix | 4004 OK, 0 FAIL, exit 0 (+9 crash-class tests) | – |
| p6, p6cfg | 4100 OK, 0 FAIL, exit 0 (+96 padded-GQA tests) | – |
| qkv, conv, conv8 | 4202 OK, 0 FAIL, exit 0 (+102 quantised-decode tests) | conv8: 2446 OK, 0 FAIL |

0 new failures for every patched variant. The crash fix (0001) is confirmed on sm_70 hardware.

## 2. Conversion microbenchmark (`bench/micro/dequant_nc.cu`, one K or V view per call)

| shape | upstream | 0005 rows/block | 0006 vec8 | f16 D2D copy |
| --- | --- | --- | --- | --- |
| D=64, KV 7680, 8 heads (q8_0 / q4_0) | 110 µs | 3.79× / 4.04× | **5.80× / 6.92×** | 22 µs |
| Llama-3.1-8B @16K, D=128 (q8_0 / q4_0) | 232 µs | 2.09× / 2.16× | **3.08× / 3.51×** | 85 µs |
| D=256, KV 16K, 2 heads (q8_0 / q4_0) | 63 µs | 1.15× / 1.16× | **1.67× / 1.88×** | 44 µs |
| Qwen3.8-27B @32K, D=256 (q8_0 / q4_0) | 240 µs | 1.14× / 1.15× | **1.74× / 1.93×** | 167 µs |

All outputs bit-identical to upstream. With 0006 the conversion is faster than an f16 device-to-device copy
of the same output (it reads fewer bytes).

## 3. Kernel timing (test-backend-ops perf, nh=4 KV heads, median of 3 interleaved rounds)

Per-case spread (max−min)/median is < 1% for almost every case; the largest is 17% on one upstream q4_0
2-token case. A 2% threshold is used below for "within noise".

### 3.1 P-6: small batches, f16 K/V

GQA 6 at 32K (the acceptance shape; Qwen3.8-27B's attention is D=256, 4 KV heads × GQA 6):

| D | tokens | upstream | p6 | p6cfg | p6cfg vs upstream |
| --- | --- | --- | --- | --- | --- |
| 256 | 8 | 646.6 µs | 682.8 µs | 260.7 µs | **2.48×** |
| 256 | 16 | 1609.5 µs | 1104.3 µs | 410.6 µs | **3.92×** |
| 128 | 8 | 334.1 µs | 150.8 µs | 150.8 µs | **2.22×** |
| 128 | 16 | 205.9 µs | 240.2 µs | 240.0 µs | **0.86×** |

Across all f16 cases (D 128/256 × GQA 3–8 × KV 4K/32K × 1–16 tokens), `p6cfg` vs upstream:
- D=256: 1.00× to 3.92×, never slower. GQA 3/5/6/7 at 4–16 tokens: 1.9–3.9× at 32K. GQA 4 and 8 also gain
  1.36–2.70× at 4–16 tokens: that is 0003's configs (no spills, 2 blocks/SM), not padding.
- D=128 GQA 3/5/7 at 4–16 tokens: 1.8–3.5× at 32K. GQA 4/8: all within 2% (worst 0.98×, GQA 4, 8 tokens, 32K).
- **Regressions, all at D=128 (GQA 3 and 6), largest at 4K:**

| case | upstream | p6cfg | ratio |
| --- | --- | --- | --- |
| D=128 GQA 6, 16 tokens, 4K | 44.5 µs | 64.0 µs | **0.69×** |
| D=128 GQA 3, 16 tokens, 4K | 54.5 µs | 65.8 µs | **0.83×** |
| D=128 GQA 6, 16 tokens, 32K | 205.9 µs | 240.0 µs | **0.86×** |
| D=128 GQA 6, 8 tokens, 4K | 51.3 µs | 56.5 µs | **0.91×** |
| D=128 GQA 6, 4 tokens, 4K | 37.5 µs | 38.4 µs | 0.98× |

- `p6` without 0003 is slower than upstream on several D=256 shapes (down to 0.83×: GQA 3, 8 tokens, 32K):
  the padded instances at D=256 need 0003's configs. Ship 0002 only together with 0003.

### 3.2 P-8: quantised K/V

| criterion shape (32K) | p6cfg | qkv | ratio |
| --- | --- | --- | --- |
| D=128 GQA 4, 1 token, q8_0 | 522.1 µs | 157.5 µs | **3.31×** |
| D=128 GQA 4, 1 token, q4_0 | 517.5 µs | 180.7 µs | **2.86×** |
| D=256 GQA 4, 1 token, q8_0 | 616.9 µs | 296.6 µs | **2.08×** |
| D=256 GQA 4, 1 token, q4_0 | 611.6 µs | 289.3 µs | **2.11×** |

`conv8` vs `qkv` at 8 tokens, 32K (TILE + conversion path):

| D | GQA 4 q8_0 / q4_0 | GQA 6 q8_0 / q4_0 | GQA 8 q8_0 / q4_0 |
| --- | --- | --- | --- |
| 128 | **2.06× / 2.21×** | 1.96× / 2.08× | 1.95× / 2.07× |
| 256 | **1.32× / 1.39×** | 1.31× / 1.38× | 1.31× / 1.37× |

At 2 tokens the same pattern holds (D=128 1.99–2.39×, D=256 1.30–1.44×). Over all quantised cases,
`conv8` vs upstream is min 1.00×, median 1.61×, max 3.37×. No f16 case moves by more than 2% between
`p6cfg` and `qkv`/`conv`/`conv8`.

GQA 6 at 1 token is unchanged by 0004 (D=128 q8_0: 231 µs upstream, 236 µs qkv); upstream is already ~2× faster
there than at GQA 4 (520 µs). Which kernel upstream picks for it was not traced in this session.

**Change 4 gate** (qkv, 1 token, 32K, time at GQA 8 ÷ time at GQA 4, equal KV bytes):
D=128 q8_0 1.89, q4_0 1.94; D=256 q8_0 1.88, q4_0 1.97. All ≥ 1.6: the VEC kernel re-reads K/V per Q head.

## 4. End to end (llama-bench `-fa on`, median of 2 interleaved rounds × 3 reps)

Run-to-run ranges on this host are small (mostly < 1%; decode ranges < 0.1 t/s). Session 1's decode
spread (±5–20%) did not reproduce here; see §6.

### 4.1 Small batches at depth (f16 K/V)

| model | test | upstream | p6 | p6cfg | conv8 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | pp4 @32K | 74.43 | −1.9% | **+5.8%** | +5.9% |
| Qwen3.8-27B | pp8 @32K | 104.06 | −1.0% | **+7.6%** | +8.6% |
| Qwen3.8-27B | pp16 @32K | 144.81 | +8.1% | **+19.9%** | +21.0% |
| Qwen3.8-27B | pp8 / pp16 @4K | 115.59 / 179.76 | 0.0% / +1.2% | +0.5% / +2.6% | +0.7% / +2.7% |
| Llama-3.1-8B | pp1–pp16 @4K and @32K | – | −1.5…+2.6% | −1.5…+3.0% | −2.2…+2.6% |

Qwen3.8-27B is a hybrid model: `qwen35.block_count = 65`, `full_attention_interval = 4`, SSM layers in
between (`gguf-meta.txt`), so roughly 16 layers run FLASH_ATTN_EXT. The kernel test shape (4 KV heads, GQA 6,
D=256, 32K) is that model's attention shape, so the end-to-end change can be predicted from §3.1 as
upstream step time − 16 × kernel saving:

| test @32K | upstream step | 16 × saving | predicted p6cfg | measured p6cfg / conv8 |
| --- | --- | --- | --- | --- |
| pp4 | 53.7 ms | 3.3 ms | 79.3 t/s | 78.8 / 78.8 |
| pp8 | 76.9 ms | 6.2 ms | 113.1 t/s | 112.0 / 113.0 |
| pp16 | 110.5 ms | 19.2 ms | 175.2 t/s | 173.7 / 175.3 |

The match says the patch delivers its kernel gain and that FA is a minority of a small-batch step on this
model. Comparison with session 1's fork numbers (different host; upstream pp8/pp16 @32K there were
107.3 / 148.8 t/s vs 104.1 / 144.8 here): the fork reached 206.6 / 315.7 t/s. At pp16 @32K the whole upstream
FA time is ≈ 16 × 1.61 ms = 25.8 ms of a 110.5 ms step, so even zero-cost attention would give ≈ 189 t/s;
**most of the fork's small-batch gain on Qwen3.8-27B is outside FLASH_ATTN_EXT** (not identified here).

### 4.2 Quantised-KV decode and pp8 at 16K

| model | KV | test | upstream | p6cfg | qkv | conv8 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | f16 | tg64 | 65.51 | – | – | 65.50 (0.0%) |
| Llama-3.1-8B | q8_0 | tg64 | 49.18 | 49.18 | 57.39 (**+16.7%**) | 57.41 (**+16.7%**) |
| Llama-3.1-8B | q4_0 | tg64 | 40.06 | 40.09 | 54.54 (**+36.2%**) | 54.56 (**+36.2%**) |
| Llama-3.1-8B | q8_0 / q4_0 | pp8 | 261.6 / 229.7 | −0.3% / +0.2% | −1.7% / −1.6% | −1.7% / +0.1% |
| Qwen3.8-27B | f16 | tg64 | 29.96 | – | – | 29.96 (0.0%) |
| Qwen3.8-27B | q8_0 / q4_0 | tg64 | 29.38 / 29.06 | ±0.1% | ±0.1% | ±0.1% |
| Qwen3.8-27B | q8_0 / q4_0 | pp8 | 103.1 / 100.1 | +2.6% / +3.2% | +2.4% / +2.4% | +3.1% / +2.8% |

Llama-3.1-8B decode at 16K, cost of a quantised cache vs f16: upstream −25% (q8_0) / −39% (q4_0);
with the series −12% / −17%. The series recovers 50% (q8_0) and 57% (q4_0) of the gap. Qwen3.8-27B has no
measurable quantised-KV decode penalty, consistent with the unchanged GQA 6 single-token kernel in §3.2.

pp8 at 16K with a quantised cache does not gain on Llama-3.1-8B even though the kernel is ~2× faster
(§3.2, 8 tokens): not explained here.

## 5. Against the acceptance criteria

**P-6** (`docs/design/P6-small-batch-gqa.md`):
- 0 new failures: **pass**.
- GQA 6 at 8/16 tokens, 32K ≥ 1.3× upstream: **pass at D=256 (2.48× / 3.92×) and D=128 at 8 tokens (2.22×);
  fail at D=128, 16 tokens (0.86×).**
- GQA 4/8 within noise: **pass** (D=128 within 2%; D=256 faster, from 0003).
- SMALL_B decision data (padded `<8,8>` vs upstream `<16,2>` at 16 tokens): at D=256 padding wins
  (1609 → 411 µs at GQA 6, 32K, together with 0003); at **D=128 upstream's `<16,2>` wins** (206 vs 240 µs at
  32K, 44.5 vs 64.0 µs at 4K), and D=128 GQA 3 at 16 tokens and GQA 6 at 8 tokens (4K) also regress.
  B ≤ 4: 0.98–2.95×. Needs a design change in 0002 before upstreaming (owner decision): keep upstream's
  routing for the regressing D=128 shapes; the exact rule needs a routing trace for D=128 GQA 3/6 at 8–16 tokens.
- End to end on Qwen3.8-27B: +5.8% / +7.6% / +19.9% at pp4 / pp8 / pp16 @32K, as predicted from the kernel
  gain and the model's ~16 attention layers. Llama-3.1-8B: no change.

**P-8** (`docs/design/P8-quantised-kv-decode.md`):
- 0 new failures (FA and MUL_MAT): **pass**.
- `qkv` ≥ 2× `p6cfg` at 1 token, 32K, GQA 4: **pass** (D=128 3.31× / 2.86×; D=256 2.08× / 2.11×).
- `conv8` ≥ 1.5× `qkv` at 8 tokens, 32K: **pass at D=128 (1.95–2.21×), fail at D=256 (1.31–1.39×).**
- Llama-3.1-8B q8_0 tg64 at 16K ≥ +25% over upstream: **fail (+16.7%)**; q4_0 is +36.2%.
- No f16 case outside noise: **pass** (kernel within 2%; Llama and Qwen f16 tg64 0.0%).
- Microbench 0005/0006 bit-identical with V100 speedups: **pass** (1.67–6.92×).
- Change 4 gate: **triggered** (1.88–1.97 ≥ 1.6): design GQA-packed VEC.

## 6. Model predictions vs measurement

- `bin/qkv-decode-model.py` (4 KV heads, 32K): upstream D=128 522 µs predicted, 520 measured; D=256 606 vs
  612–617. TILE + conversion (+0006) vs `conv8` at 2 tokens: D=128 220 / 199 µs predicted vs 239 / 230 measured
  (q8_0 / q4_0), D=256 433 / 391 vs 452 / 417: within 4–16%. VEC (0004): q8_0 inside the predicted range but at
  its pessimistic end (D=128 157.5 µs in 45–178; D=256 296.6 in 89–357), i.e. K/V read about once per Q head,
  consistent with the Change 4 gate. **q4_0 VEC is outside the range** (D=128 180.7 µs vs ≤ 94; D=256 289.3 vs
  ≤ 189): q4_0 VEC is not bandwidth-bound as the model assumes. The model's "pessimistic VEC bound gives +38%"
  for Llama q8_0 tg64 was also optimistic (+16.7% measured).
- `bin/smallbatch-model.py`: wrong in direction at D=128 GQA 6, 16 tokens (predicts padding 191 µs vs upstream
  272 µs; measured 240 vs 206). Upstream's 3 K/V re-reads cost less than the model's 3 × HBM. At D=256 it
  underestimates upstream (544 predicted; 647 / 1610 µs measured at 8 / 16 tokens): spills are not modelled.
  Padded D=256 is within 7–36% of the model (191 / 382 predicted, 261 / 411 measured).
- Session 1's Llama decode numbers (f16 43.9, q8_0 28.1, q4_0 26.7 t/s tg128 @16K, ±2–4 t/s) were far below
  this host's (65.5 / 49.2 / 40.1, tg64, ranges < 0.1 t/s). Session 1's "−36 to −39%" quantised-KV cost was
  therefore host-affected; this session's −25% / −39% is the better reference.

## 7. Not measured here

Multi-GPU / PCIe P2P (single GPU), Nsight Compute metrics (unprivileged container), perplexity, prefill
(pp ≥ 512) with the series, D=128 models with GQA 3/6 end to end (no such model in the set), why
quantised-KV pp8 on Llama-3.1-8B does not follow the kernel gain, and where the fork's Qwen small-batch gain
comes from outside FLASH_ATTN_EXT.
