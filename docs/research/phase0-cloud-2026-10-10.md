# Phase 0 on a cloud V100 (2026-10-10)

Hardware: Vast.ai, **Tesla V100-PCIE-32GB** (same part as the home box), 1380 MHz max SM clock,
250 W, PCIe 3.0 x16, driver 580.173.02, CUDA 12.8.93, Ubuntu 24.04 container, AMD EPYC 7302P.
llama.cpp `79e2e74` (`bench/LLAMA_COMMIT`). Raw data under `bench/results/` (paths below).
Single-GPU instance: no P2P / multi-GPU results here.

## 1. PR #27997 and variants: D=256 prefill kernel (kernel level)

`bench/results/pr27997-20261010-025229-70a805e2950e/summary.md`. test-backend-ops perf, D=256,
nh=4, batch 512 (the ncols=64 path), KV 4K/16K/64K (+128K for GQA 6), GQA 2/4/6. Three interleaved
rounds; spread per variant < 1%; no throttling logged (SM clock 1380 MHz after the first warm-up
sample). Every variant passed all 149 `hsk=256` correctness cases.

| variant | config row change vs Ampere fallback | stack (spill) | blocks/SM | speedup vs upstream |
| --- | --- | --- | --- | --- |
| upstream | — | 432–640 B | 1 | 1.00× |
| `pr` (#27997) | V2 64, Q in smem | 48–112 B | 1 | **1.41–1.45×** |
| `pr_comb64` | + combine 64 | 48–112 B | 1 | 1.41–1.45× |
| `comb64` | combine 64 only | 432–640 B | 2 | 0.80–1.14× |
| **`pr_k64_c64`** | V2 64, K2 64, combine 64, Q in smem | 48–160 B | **2** | **2.26–2.65×** |

Reading: spill removal is worth ~1.43× (the PR); occupancy without spill removal mostly hurts
(`comb64`); both together (`pr_k64_c64`) is worth ~2.5×. Example, GQA 6 / 16K KV: upstream 11.62 ms
(≈17.8 TFLOPS) → `pr` 8.15 ms → `pr_k64_c64` 4.40 ms (≈47 TFLOPS), faster than upstream's own D=128
kernel at equal shape class (§2).

### 1.1 End to end: Qwen3.8-27B-UD-Q4_K_XL prefill at depth

`bench/results/pr27997-20261010-034851-70a805e2950e/`. `llama-bench -fa on -p 1024 -n 0 -d 4096,16384,65536 -r 5`,
2 interleaved rounds per variant (kernel-level rounds re-run in the same session agree with §1 within 1%).

| depth | upstream t/s | `pr` t/s | `pr_k64_c64` t/s | `pr` vs up | `pr_k64_c64` vs up |
| --- | --- | --- | --- | --- | --- |
| 4K | 766 (724–807) | 714 (681–748) | 694 (691–696) | 0.93× | 0.91× |
| 16K | 636 (622–651) | 665 (662–668) | 694 (686–701) | 1.04× | 1.09× |
| **64K** | **374** (374–374) | **456** (456–457) | **543** (543–543) | **1.22×** | **1.45×** |

- 64K reproduces the PR author's own number (+21.88% at 65K for Qwen3.8 27B; here +22%), and `pr_k64_c64`
  doubles the gain (+45%).
- 4K is **inconclusive**: per-run stddev there is ±31–60 t/s (4–9%); the high upstream round 1 (807) was the
  only run on a cool GPU (56 °C vs 75–77 °C for all others). In round 2 (all warm) the three are within one
  stddev (724 ± 32, 681 ± 60, 696 ± 32). Needs more reps before claiming either a gain or a regression at
  short depth.

## 2. Upstream FA kernel throughput by head dim (single run)

`bench/results/phase0-llama.cpp-20261010-023729-70a805e2950e/fa-perf.sql` (207 cases, 20 unsupported).
Prefill-class cases (batch 4096, KV 4K–64K):

| head dim | no GQA | GQA 4 | GQA 8 |
| --- | --- | --- | --- |
| 64 | 16 TFLOPS | 30 | 32–33 |
| 128 | 22 | 39–40 | 43 |
| 256 | **10** | **21** | **22** |
| 512 | — | 20–21 | 21 |
| 576/512 (MLA) | — | 21–22 | 21–22 |

Also: D=256 GQA 16 batch 512 = **8.9 TFLOPS**; D=72 (Qwen-VL) = 9.3 TFLOPS. Peak FP16 tensor
throughput of a V100 PCIe is ~112 TFLOPS; D=128 reaches ~38% of it, D=256 ~19%.

## 3. Quantised-KV decode (single run, same file)

| case (batch 1) | f16 K/V | quantised K/V | ratio |
| --- | --- | --- | --- |
| D=64, GQA 8, KV 7680 | 33 µs | q4_0 237 µs, q8_0 238 µs | **7.2×** |
| D=256, GQA 12, KV 16384 | 80 µs | q4_0 190 µs | 2.4× |
| D=256, GQA 16, KV 10000 | 145 µs | q8_0 220 µs | 1.5× |
| D=256, GQA 12, KV 16384 | 80 µs | bf16 204 µs | 2.6× (no BF16 on Volta) |

Consistent with the code reading (research gap 2): on Volta, GQA ≥ 4 decode goes to the TILE kernel,
which converts the whole quantised K/V cache to F16 per call.

## 4. Crash: test-backend-ops FLASH_ATTN_EXT aborts on Volta

The full correctness run aborted after 3964 passing cases (`fa-test.txt.gz`):
`fattn-mma-f16.cuh:1849: ERROR: CUDA kernel flash_attn_ext_f16 has no device code compatible with
CUDA arch 700`. Reproduces alone with case `hsk=128,hsv=128,nh=1,nr23=[8,1],kv=5003,nb=4,mask=1,…,
n_kv_max=512` (test-backend-ops.cpp:11696):

```
test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=128,hsv=128,nh=1,nr23=\[8,1\],kv=5003,nb=4,.*n_kv_max=512'
```

Cause (code reading, fattn.cu at 79e2e74):
1. Kernel selection (`fattn.cu:668-681`) computes the effective GQA ratio as if the GQA optimisation
   applies: 4 × 8 = 32 > 16 → MMA kernel.
2. Dispatch (`fattn.cu:203-216`) disables the GQA optimisation when K/V length is not a multiple of
   `FATTN_KQ_STRIDE` (5003 here), or there is no mask, ALiBi, or misaligned strides; on Volta it then
   uses `ncols2 = 1`.
3. `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1` (`fattn.cu:177-182`) takes the
   `Q->ne[1] <= 16/ncols2` branch → 16-column instance. Volta compiles every instance with
   `ncols1*ncols2 < 32` as `NO_DEVICE_CODE` (`fattn-mma-f16.cuh:1846-1851`). The 8-column branch just
   above is guarded with `turing_mma_available(cc)`; the 16-column branch is not.

Candidate one-line fix (being verified on the V100, §4.1): `if (!volta_mma_available(cc) && Q->ne[1] <= 16/ncols2)`
so Volta falls through to the 32-column instance. Not found among upstream issues on 2026-10-10
(#28661 is a different Volta FA crash: non-standard head dims).

### 4.1 Fix verification

`bench/results/volta-ncols-fix-20261010-034659/` (patch, logs). With the one-line change the crash case
passes, and the **full** `test-backend-ops test -o FLASH_ATTN_EXT` run completes: **3995 OK, 0 FAIL,
exit 0** (unpatched: aborts after 3964). Not timed: the fix only changes which instance a
previously-crashing configuration launches.

## 5. llama-superfast-v100 fork (`70df871`) vs upstream (kernel level, single run each)

`bench/results/phase0-llama-superfast-v100-20261010-031443-70a805e2950e/` (built with gcc-14,
`-DGGML_CUDA_CUB_3DOT2=ON`). `python3 bench/analyze.py tbo <upstream>/fa-perf.sql <fork>/fa-perf.sql`:
151 cases timed by both.

- **D=256 prefill: fork 2.45–2.71× faster** than upstream (e.g. GQA 6 / 16K / batch 512: 11.58 → 4.53 ms).
  That is the same class as the one-row `pr_k64_c64` config (4.40 ms on the same shape, §1; different
  runs, so a few-percent difference is not meaningful).
- **D=256 decode, GQA 6 (its target model): fork up to 2.91× faster** (128K KV: 2.00 → 0.69 ms).
- **D=256 decode, GQA 4 and 8: fork 0.68–0.70× (≈30% slower)** at every KV length tested.
- Median over all matched cases: 1.00×.
- **Correctness:** 3 failures, all D=256 GQA 6 at 64K KV (batch 64/200/1024), error 0.0013–0.0014 vs the
  0.0005 threshold, i.e. on the fork's own target shape. The run then **crashed** on a D=320/256 case
  (`CUDA error: invalid argument`, the #28661 class); upstream with the §4 fix handles that case without
  crashing.

## 6. Pending on this instance

- End-to-end: PR variants on Qwen3.8-27B-UD-Q4_K_XL (prefill at 4K/16K/64K depth, 2 rounds);
  upstream vs fork model sweeps on Qwen3.8-27B and Llama-3.1-8B-Instruct-Q8_0.
