# P-8 Change 4 (prototype): GQA-packed VEC kernel for quantised-KV decode

Status: **prototype, not part of the measured series.** Patch `third_party/patches/next/0007-*.patch` applies on
top of `0001..0006` (checked on `79e2e74`). Statically checked for sm_70; correctness and a relative timing on
the laptop's Maxwell GPU (below).
It goes into a V100 A/B only if the Change 4 gate in `P8-quantised-kv-decode.md` trips (`bench/patches/check.py`
reports it), or as an extra variant if a session has spare time.

## Why

With `0004`, single-token decode with a quantised cache uses VEC on Volta. VEC runs one block per Q head, so a
K/V head is read (and dequantised) once per Q head. GQA 4–8 means 4–8 reads, unless L2 absorbs them. The
cost model brackets 0004 between 1 read and GQA reads (Llama-like at 16K, q8_0: 45–178 µs per layer).
Packing the Q heads of a K/V head into one block makes it 1 read by construction.

## What the patch does (`fattn-vec.cuh`, plus tests)

- The kernel template gets `ncols2` (Q heads per block) next to the existing token count `ncols1`. Block
  column `jc` is token `jc / ncols2`, Q head `jc % ncols2`. The grid mapping is the MMA kernel's,
  `blockIdx.z = (sequence, K/V head, head group)`, with padded heads masked. For `ncols2 == 1` it reduces
  exactly to the old `(sequence, head)` indexing.
- Per column: the Q pointer, the validity check (token in the batch, head not padding), the mask row (shared
  by the heads of a token), the sink (guarded for padded heads), and the output and meta indices. ALiBi
  is excluded (one slope per block).
- Used only for **quantised K, one token, GQA ≥ 3, no softcap, no ALiBi**: `ncols2 = 4` for GQA 3–4 and 8
  for GQA ≥ 5, so GQA 12 uses two blocks per K/V head. Quantised K keeps Q as q8_1 (a few registers per
  column); F16/BF16 K keep Q as floats and would spill at 8 columns.
- Registers (sm_70, nvcc 12.8, q8_0 / q4_0 instances, all with 0 spill and 0 stack):

  | D | ncols2 = 4 | ncols2 = 8 |
  | --- | --- | --- |
  | 64 | 80 / 91 | 128 / 139 |
  | 128 | 85 / 92 | 130 / 142 |
  | 256 | 114 / 124 | 184 / 201 |

  Three changes were needed to get there:
  1. **The key loops are not unrolled in packed instances.** Fully unrolled, they spilled 24–52 B stores and
     168–212 B loads at 8 heads.
  2. **The sink loop indexes columns at compile time.** Indexing the per-column register arrays with
     `threadIdx.y` moved them to local memory once there were more than 2 columns (160–320 B of stack).
     The same rewrite also cuts upstream's existing 2-token instances from 48–80 B of stack to 0–16 B.
     The 1-token instances are unchanged.
  3. **The KQ shared buffer is sized for the softmax values** (`ncols × nthreads`), not just `ncols × D`.
     Upstream's ≤ 2 columns always fit inside the combine buffer. At D=64 with 8 columns they did not:
     the first build failed 14 tests (D=64, GQA 5/6/7/12) with inf/NaN, all of them that one shape.
- Unroll factor: packed instances with unroll 4 also have 0 spill, but use more registers (126–250). Unroll 1
  was chosen for occupancy; it is a one-constant change if V100 numbers favour unroll.
- Architecture scope: the packing is not Volta-specific. Any GPU whose selection picks VEC for quantised
  decode (Turing+, Pascal) would use it. That needs measurements there before upstreaming, or a
  Volta-only switch.

## Tests (60 new cases)

D 64/128/256 × q8_0/q4_0 × GQA 3/5/7/12 × sinks on/off, plus 2 sequences (GQA 6) and ALiBi (GQA 8, stays
unpacked). These are in addition to the P-8 cases at GQA 4/6/8 that `0004` added.

## Local results (Quadro M2200, sm_52, not a V100)

- Full `FLASH_ATTN_EXT`: **4260/4260 passed** (the 4200 earlier cases plus the 60 new ones).
  compute-sanitizer memcheck: 0 errors on 16 packed cases (D 64/256, GQA 3/5/6/12, 2 sequences, sinks).
- Timing, packed vs unpacked VEC (both with `0001..0006`), 1 token, 3 interleaved rounds, medians
  (`bench/results/m2200-vecpack-20261010/summary.md`):

  | D | GQA 1–2 | GQA 4 | GQA 6 (padded to 8) | GQA 8 |
  | --- | --- | --- | --- | --- |
  | 128, KV 4K/16K | 0.99–1.01× | 1.03–1.38× | **0.93–1.03×** | 1.20–1.36× |
  | 256, KV 4K/16K | 0.97–1.01× | 3.0–4.1× | 2.8–3.2× | 3.6–4.3× |

  Unpacked VEC time grows almost linearly with GQA on this GPU (2 MB L2), so the per-head re-reads are not
  absorbed. On the V100 (6 MB L2, far more bandwidth), whether they are is exactly what the Change 4 gate
  measures. **Decision (on V100 data):** for non-power-of-two GQA at D=128, padding to 8 did not help here.
  The alternative is the exact divisor (GQA 6 → 3 blocks of 2), i.e. P-6's question again in the VEC kernel.

## To measure on a V100

Copy the patch to `third_party/patches/0007-…` and add `vecpack) echo 7 ;;` to `npatch` in
`bench/patches/run.sh`. Then run `VARIANTS="upstream p6cfg qkv conv8 vecpack"`. Compare `vecpack` against
`conv8` at 1 token, GQA 4/6/8, KV 4K/32K, q8_0/q4_0. Expected: equal at GQA ≤ 2 (unpacked); faster as GQA
grows, if L2 was not already absorbing the re-reads.
