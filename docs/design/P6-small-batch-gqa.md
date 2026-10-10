# P-6 design: small-batch attention for non-power-of-two GQA on Volta

Status: **implemented as a 3-patch series (`third_party/patches/0001..0003`), measured on a V100
(`docs/research/patches-cloud-2026-10-10.md`).** 0 new failures; the crash fix holds. D=256: 1.0–3.9× at
kernel level, never slower (GQA 6, 32K: 2.48× / 3.92× at 8 / 16 tokens); Qwen3.8-27B +7.6% / +19.9% at
pp8 / pp16 @32K (FA is ~16 of its 65 layers). **Not accepted as is:** D=128 regresses where upstream's
unpadded tile was already good (GQA 6, 16 tokens: 0.86× at 32K, 0.69× at 4K; GQA 3, 16 tokens, 4K 0.83×;
GQA 6, 8 tokens, 4K 0.91×). 0002 needs a D=128 rule change (owner decision) and must ship with 0003.
Owner decision points are marked **Decision**.

## Problem (measured)

Small query batches at long context (speculative-decoding verify, short chunked prefill) are slow on
Volta for models whose GQA ratio is not a power of two. On Qwen3.8-27B (D=256, GQA 6) the
llama-superfast-v100 fork is +80–112% at 8–16 tokens over upstream
(`docs/research/phase0-cloud-2026-10-10.md` §6). On Llama-3.1-8B (GQA 4) there is nothing to win.

## Mechanism (code reading verified, `docs/research/small-batch-volta.md`)

- Upstream's Volta dispatch picks heads-per-tile `ncols2` as the largest of 8/4/2 that **divides** the
  GQA ratio (`fattn.cu:224-246`). GQA 6 → `ncols2 = 2`, so each KV head's K/V is streamed **3 times**
  (once per 2-head group).
- Kernel selection (`fattn.cu:667-681`) then sends GQA 6 at 2–8 tokens to the CUDA-core TILE kernel and
  16 tokens to MMA `<16,2>`; both read K/V 3×.
- Volta MMA tiles need ≥ 32 columns (four 8-column m8n8k4 slices per warp), so a tile must hold
  `B × ncols2 ≥ 32` query rows to use tensor cores.
- The MMA kernel **already supports padded "phantom" heads**: it rounds the head-group count up
  (`fattn-mma-f16.cuh:1892`), zero-fills missing Q heads (1300) and skips their writes (1742). Non-Volta
  NVIDIA GPUs already pad GQA 5–7 to 8 (`fattn.cu:266`). Volta alone uses exact divisors.

Cost model (`bin/smallbatch-model.py`, per layer, Qwen-like, 32K context):

| tokens | TILE (3× K/V) | MMA upstream ncols2=2 (3× K/V) | MMA padded ncols2=8 (1× K/V, 75% columns) |
| --- | --- | --- | --- |
| 4 | 181 µs | 544 µs | 181 µs |
| 8 | 322 µs | 544 µs | **191 µs** |
| 16 | 644 µs | 544 µs | **382 µs** |

(TILE throughput is assumed, not measured; TILE probably benefits from L2 on its re-reads, so the B=4
gain is the least certain.)

## Design

### Change 1: one source of truth for "is the GQA optimisation on" (also fixes the crash)

Today selection assumes the GQA optimisation and dispatch may then turn it off, which is how the
16-column `NO_DEVICE_CODE` crash happens (`docs/research/phase0-cloud-2026-10-10.md` §4). Factor the
`use_gqa_opt` computation (`fattn.cu:203-216`: mask present, no ALiBi, K length a multiple of
`FATTN_KQ_STRIDE`, 16-byte-aligned strides) into one helper used by **both** selection and dispatch.
When it is off, selection uses `gqa_ratio_eff = 1`, so small batches go to VEC/TILE instead of an
illegal MMA instance. Keep the one-line `!volta_mma_available(cc)` guard on the 16-column branch as
defence in depth.

### Change 2: padded heads-per-tile for small batches on Volta

In the Volta dispatch branch, before the exact-divisor `% 2` case:

```
if (use_gqa_opt && !is_pow2(gqa_ratio) && gqa_ratio < 8 && Q->ne[1] <= SMALL_B) -> ncols2 = next_pow2(gqa_ratio)   // 3->4, 5,6,7->8
```

and make selection compute `gqa_ratio_eff` the same way (one helper again), so GQA 6 at 4–16 tokens
selects MMA with `ncols2 = 8`.

- **Only small batches.** For prefill (hundreds of tokens) padding is worse on both axes: GQA 6, B=512
  with `ncols2=2` → 48 K/V streams at 100% columns; with `ncols2=8` → 64 streams at 75%. So the padded
  path is gated on `Q->ne[1] <= SMALL_B`. **Decision:** `SMALL_B = 16` (model: padded wins at 8 and 16,
  ties at 4). Confirm the 16 vs `<16,2>` crossover on hardware.
- **Power-of-two GQA unchanged** (4, 8, 16): byte-identical routing, so Llama-3.1-8B-like models
  cannot regress.
- **GQA > 8, not a power of two** (10, 12, 14): out of scope for now (rare; `ncols2` max is 8 on Volta).
- **Attention sinks:** guard the sink read for phantom heads (`fattn-mma-f16.cuh:1434`,
  `sinks_f[jc % ncols2]` reads past the group/array for padded heads). Latent on other architectures
  that already pad; fix applies to all.

### Change 3 (dependent): Volta D=256 tile configs

Small-batch MMA on D=256 uses ncols 32/64 instances whose Volta configs fall back to Ampere rows and
spill (static survey: 464–640 B stack at ncols=64). The ncols=64 row is PR #27997 territory
(`pr_k64_c64` measured 2.3–2.65×). For ncols=32 (B=4 at `ncols2=8`), run `bench/pr27997/static.sh`-style
variants before choosing a row. **Decision:** keep Change 3 separate from Changes 1–2 so each is
reviewable and measurable on its own.

## Why not port the fork's kernel

The fork's `verify_stream` is a new kernel for exactly D=256 + GQA 6 + F16 KV, with a fixed 48-row
tile (pads rows, so B=4 wastes half), an operand swap, and one-wave KV splitting. It wins on its
target but is a large, model-specific code path, and the fork fails 3 accuracy tests on that shape.
Changes 1–2 reuse a kernel that is already tested on other architectures with padded heads. If they
leave a measurable gap at B ≤ 4, the fork's operand swap (8-row granularity) is the structural
follow-up, with a batch-sized tile instead of a fixed 48.

## Verification plan

Static (no GPU, now): patch compiles for sm_70; `bench/pr27997/static.sh`-style register/spill report for
the newly used instances (`<4,8>`, `<8,8>` at D=128/256 and their `<1..2,4>` analogues for GQA 3).

On a V100 (next cloud session or the home box):
1. Correctness: full `test-backend-ops test -o FLASH_ATTN_EXT` (must stay 0 FAIL), plus new eval cases
   for GQA 3/5/6/7 at B ∈ {1,2,4,8,16}, with and without sinks, unpadded KV (the crash class).
2. Kernel A/B, interleaved rounds: upstream vs patched for D ∈ {128, 256}, GQA ∈ {3,4,5,6,7,8},
   B ∈ {1,2,4,8,16}, KV ∈ {4K, 32K}. Expect wins only for non-power-of-two GQA at B ≥ 4 (8+ most), and
   byte-identical timings for GQA 4/8.
3. End to end: `llama-bench -p 1,2,4,8,16 -n 0 -d 4096,32768` on Qwen3.8-27B; Llama-3.1-8B as the
   no-regression control; compare against the fork's numbers from 2026-10-10.
4. Accept if: 0 new failures; GQA 6 B=8/16 at 32K ≥ 1.3× upstream; GQA 4/8 within noise.

## Implementation status (2026-10-10)

`third_party/patches/` (git format-patch against llama.cpp `79e2e74`):

| patch | content | static checks |
| --- | --- | --- |
| 0001 | one Volta heads-per-tile helper (`ggml_cuda_fattn_mma_f16_volta_ncols2`) used by kernel selection **and** the Volta dispatch; never dispatch a < 32 column instance on Volta; 9 crash-class tests (GQA 4/6/8, KV 5003, 4–16 tokens) | applies alone; `fattn.cu` compiles alone |
| 0002 | padded heads per tile for GQA 3/5/6/7 at 3–16 tokens (D ≤ 256); sink guard for masked heads; 96 tests (D 128/256 × GQA 3/5/6/7 × 1–16 tokens × sinks) | full `ggml-cuda` compiles for sm_70 (nvcc 12.8.93, 0 errors); 115 non-D=256 MMA kernels have **identical** registers/stack to upstream (the sink guard is free) |
| 0003 | Volta rows `GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, {32,64}, 128, 2, 32, 64, 64, 64, 2, false)` | D=256 `<4,8>` stack 448 → 64 B (32 columns); ncols=64 row is the measured `pr_k64_c64` (2.3–2.65×) |

Routes after 0001+0002 (traced against the compiled instance set): GQA 5/6/7 → VEC at 1 token, TILE at 2 (both
unchanged), MMA `<4,8>` at 3–4 tokens, `<8,8>` at 5–16; GQA 3 → `<8,4>`/`<16,4>` from 5 tokens; GQA 4/8
unchanged; GQA optimisation off → TILE for ≤ 16 tokens (was: illegal 16-column MMA).

Without 0003, D=256 small-batch MMA instances spill 432–472 B (Ampere fallback rows), so 0002's gain on
Qwen-like models is expected to be smaller than with 0003. The harness measures upstream / fix / p6 / p6cfg.

**Upstream since the base (checked 2026-10-10, master `aa94f20`):** the series applies cleanly. The only
FA change is #30103, a missing `__syncthreads()` in the MMA combine step when `nbatch_combine != DKQ/2`
and `np == 1`. That would matter for the 0003 rows and `pr_k64_c64` (combine 64 at D=256), but on Volta
a warp covers 32 Q columns (`T_B_KQ = tile<32,4>`), so `np = 4·32/ncols` is 4 at ncols 32 and 2 at
ncols 64. The sync was already taken for every 128-thread Volta config, so the fix changes nothing there.

## Upstreaming

Changes 1 and the sink guard are bug fixes with tests (crash repro case, sink case). Change 2 is a
small, Volta-only dispatch change with measurements. Per llama.cpp's contribution rules the owner
writes and submits the issue/PR text and discloses AI use; this repo provides the patch, tests and data.
