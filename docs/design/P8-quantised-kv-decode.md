# P-8 design: decode with a quantised KV cache on Volta

Status: **implemented as patches `0004..0006` (on top of P-6's `0001..0003`), measured on a V100
(`docs/research/patches-cloud-2026-10-10.md`).** 0 new failures (FA and MUL_MAT); conversions bit-exact,
1.67–6.92× faster. Passed: `qkv` ≥ 2× at 1 token (2.08–3.31×), no f16 change, `conv8` ≥ 1.5× `qkv` at D=128
(1.95–2.21×). Missed: `conv8` vs `qkv` at D=256 (1.31–1.39×) and Llama-3.1-8B q8_0 tg64 @16K (+16.7% vs
+25%; q4_0 +36.2%). The series recovers 50–57% of the quantised-vs-f16 decode gap on that model. Change 4
gate triggered (GQA 8 ÷ GQA 4 = 1.88–1.97). **Change 4 is now patch `0007`** (`P8-change4-packed-vec.md`,
`docs/research/pad-rule-vecpack-v100-2026-10-10.md`): with it, Llama-3.1-8B q8_0 tg64 @16K is **+26.8%** over
upstream (target met; q4_0 +56.2%, f16 0.0%), and a quantised cache is 5% behind f16 on that model (was 25%).
Owner decision points are marked **Decision**.

## Problem (measured, `docs/research/phase0-cloud-2026-10-10.md` §3, §6)

| case (1 token) | f16 K/V | quantised K/V | ratio |
| --- | --- | --- | --- |
| D=64, GQA 8, KV 7680 | 33 µs | q4_0 237 µs, q8_0 238 µs | 7.2× |
| D=256, GQA 12, KV 16384 | 80 µs | q8_0 194, q4_0 190, bf16 204 µs | 2.4–2.6× |
| D=256, GQA 12, KV 32768 | 156 µs | q8_0 371, q4_0 364, bf16 392 µs | 2.3–2.5× |

End to end, Llama-3.1-8B tg128 at 16K: f16 KV 43.9 t/s, q8_0 28.1, q4_0 26.7 (−36 to −39%).
A quantised cache is how a 32 GB card fits long contexts, so this is the common long-context decode setup.

Two things in the numbers point at the cause. q4_0, q8_0 and bf16 cost about the same even though they
are 0.28×, 0.53× and 1× the bytes of f16. And D=64 is hit 3× harder than D=256.

## Mechanism (code reading at 79e2e74, calibrated against the rows above)

1. **Selection.** On Volta, decode with GQA ≥ 4 goes to the TILE kernel whatever the K/V type
   (`fattn.cu` Volta block: VEC only if `tokens × gqa_ratio_eff ≤ 2`). Turing+ sends one-token
   quantised decode to VEC, which reads quantised K/V directly (`fattn.cu:649-658`); Pascal and older
   do so for ≤ 2 tokens.
2. **Conversion.** TILE (and MMA) convert the whole K and V views to F16 before every call
   (`fattn-common.cuh` `launch_fattn` → `ggml_get_to_fp16_nc_cuda`). The K/V view of the cache is not
   contiguous (one row per token and KV head), so this goes through `dequantize_block_cuda`
   (`convert_unary_cuda` for BF16). Both use **256 threads along a row and one block per row**:
   with 2 values per thread a 64-value row keeps 32 of 256 threads busy, and the launch has one block per
   (token, head).
3. **Fit.** Modelling the conversion as ~1.05 µs per wave of 640 blocks (80 SMs × 8 resident blocks)
   plus ~8 µs fixed, on top of the measured F16 time, reproduces all 9 measured quantised/BF16 rows
   within ±4% (`python bin/qkv-decode-model.py` prints the check). It explains both observations: the
   cost follows the block count (rows), not the bytes, and D=64 has 4× the rows of D=256 per byte.
   End to end it predicts **+14.0 ms/token** for Llama-3.1-8B q8_0 at 16K (32 layers); measured:
   **+12.8 ms/token**.

So the slowdown is not dequantisation arithmetic or bandwidth. It is a full-cache conversion that
runs at block-launch rate, and on Volta it happens on every decode token.

## Design

### Change 1 (`0004`): single-token decode with quantised/BF16 K/V uses VEC on Volta

In the Volta block of kernel selection, before the existing VEC/TILE rules:

```cpp
const bool kv_converted = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                          (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32);
if (can_use_vector_kernel && Q->ne[1] == 1 && kv_converted &&
        ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) != nullptr) {
    return BEST_FATTN_KERNEL_VEC;
}
```

- This is Turing's rule, plus a guard: a K/V type pair without a compiled VEC instance (for example mixed
  q8_0/q4_0 without `GGML_CUDA_FA_ALL_QUANTS`) keeps TILE. VEC would otherwise fall back to converting
  to F16 *and* re-read the F16 per Q head, which is worse than TILE.
- BF16 is included: VEC handles it with integer shifts (no BF16 hardware involved), and the measured
  BF16 row is as slow as the quantised ones.
- VEC reads the quantised K/V once **per Q head**, while TILE packs a whole GQA group. How much of
  that re-reading the 6 MB L2 absorbs is unknown, so the model gives a range (below).
- **Decision:** one token only (Turing's rule). Ada and Pascal also use VEC at 2 tokens. With
  `0005/0006` the 2-token conversion is cheaper; revisit after measuring.
- Tests (102 new cases): D 128/256 × q8_0/q4_0/bf16 × GQA 4/6/8 × 1–2 tokens × sinks, plus unpadded
  KV (5003, keeps TILE), permuted K/V and softcap. Upstream's existing quantised FA cases cover only
  D=64/72 (and D=256 at GQA 16), which is why D=128, the Llama shape, gets its own cases.

### Change 2 (`0005`): launch geometry of the non-contiguous conversion

`dequantize_block_cuda` and `convert_unary_cuda` use only as many threads per row as the row needs
(a warp multiple) and put several rows in a block (row index `blockIdx.y*blockDim.y + threadIdx.y`).
Rows ≥ 512 values (≥ 256 for `convert_unary`) and the contiguous paths launch exactly as before.
This is architecture-independent. It helps every remaining conversion: 2–16 token batches (speculative
decoding) and prefill with a quantised cache on Volta, the MMA path with a quantised cache on all GPUs,
and `mul_mat` on non-contiguous quantised inputs.

### Change 3 (`0006`): vectorised q4_0/q8_0 conversion

A dedicated kernel for the two FA-default quantised types (`GGML_CUDA_FA_QUANTS` default: q4_0-q4_0,
q8_0-q8_0, f16-f16, bf16-bf16) converts 8 values per thread. The ggml blocks are only 2-byte aligned,
so it uses 2-byte aligned loads and 8/16-byte stores. Its arithmetic is the same as `dequantize_q4_0/q8_0`,
so the output is bit-identical. It falls back to the generic kernel if a row is not a whole number of
blocks or the plane exceeds `INT_MAX` threads. Registers: ≤ 32, no spills (sm_70, all dst types).

### Change 4 (gate triggered; now patch 0007, see `P8-change4-packed-vec.md`): GQA-packed VEC

If VEC's single-token time grows with the GQA ratio (L2 not absorbing the per-head re-reads), pack the
GQA group into VEC's column dimension, the same `jc → (token, head)` mapping TILE and MMA use. K/V are
then read and dequantised once per group. That is a real kernel change (`fattn-vec.cuh`, register
arrays scale with heads per block, new instances), so it waits for data.
**Gate:** at 32K, `qkv` time for GQA 8 ≥ 1.6× GQA 4 at equal KV bytes → design Change 4.

## Cost model (`bin/qkv-decode-model.py`, per layer, one token, µs)

| shape | depth | type | f16 | upstream | +0005 | +0006 | 0004 VEC (1 read – GQA reads) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-like D128, 8 KV heads, GQA 4 | 16K | q8_0 | 84 | 522 | 297 | 220 | 45 – 178 |
| | 16K | q4_0 | 84 | 522 | 264 | 199 | 24 – 94 |
| | 32K | q8_0 | 168 | 1036 | 587 | 433 | 89 – 357 |
| Qwen-like D256, 4 KV heads, GQA 6 | 32K | q8_0 | 168 | 606 | 587 | 433 | 89 – 535 |
| | 32K | q4_0 | 168 | 606 | 520 | 391 | 47 – 283 |

Only the "upstream" column is calibrated: 9 rows fit within ±4%. The others are assumptions:
- `+0005` is capped at the generic conversion body's measured rate (430–590 GB/s).
- `+0006` streams at 800 GB/s. That is assumed from the Maxwell result, where the kernel matched a
  device-to-device copy (below).
- VEC is bracketed between one K/V read and one read per Q head.

Even the pessimistic VEC bound beats upstream ~3× on Llama-like models. Its optimistic bound beats
f16, because q8_0/q4_0 is fewer bytes.

## Static and local verification (done, no V100)

- Patches apply in order on `79e2e74`, and on upstream master `aa94f20` (2026-10-10).
- Full `ggml-cuda` compiles for sm_70 with nvcc 12.8.93; new conversion kernels ≤ 32 registers, 0 spill.
- `bench/micro/dequant_nc.cu` replicates upstream's kernel, the `0005` geometry and the `0006` kernel
  byte for byte. On a **Quadro M2200 (Maxwell, not a V100)**, against upstream:
  - all outputs are bit-identical;
  - `0005` is 1.6× faster at D=64, 1.3× at D=128 and 1.1× at D=256;
  - `0006` is 2.6–5.2× faster and matches a device-to-device copy of the output size
    (`bench/results/micro-dequant-m2200-20261010/`). Relative only; V100 numbers come from the same
    binary built for sm_70.
- **Local GPU correctness** (patched `0001..0006`, sm_52 build, Quadro M2200). Maxwell routes quantised and BF16 FA at
  more than 2 tokens (and unpadded KV) through TILE plus the K/V conversion, so it exercises `0005/0006` for real.
  It cannot exercise the Volta-only selection in `0001/0002/0004`.
  - `FLASH_ATTN_EXT`: **4200/4200 passed**, 0 FAIL (1339 cases not supported on Maxwell, e.g. iq4_nl). This
    includes the 102 new P-8 cases and the 96 P-6 cases.
  - `MUL_MAT` for every type whose conversion changed (f32/f16/bf16/q4_0/q4_1/q5_0/q5_1/q8_0): **1297/1297
    passed**, 181 of them non-contiguous.
  - compute-sanitizer memcheck, 0 errors: D=64 at 32 tokens with bf16/q8_0/q4_0/q4_1/q5_0 (all three changed
    kernels), and the new D=128/256 unpadded-KV cases (18).
  - The unfiltered `MUL_MAT` run aborted at `MUL_MAT(type_a=q4_K,…,n=9,k=1024,k_v=1280)`:
    `GGML_ASSERT(convert_func != nullptr)` in the cuBLAS path, because there is no non-contiguous q4_K converter.
    Untouched by these patches: unpatched upstream (79e2e74, sm_52 build) aborts with the same assert on the same case (verified 2026-10-10).

## Verification plan (V100, `bench/patches/run.sh`, variants `upstream … p6cfg qkv conv conv8`)

1. Correctness: full `FLASH_ATTN_EXT` (0 FAIL for every patched variant); `MUL_MAT` on upstream and
   `conv8` (0005/0006 change its non-contiguous conversions).
2. Microbench on the V100: `0005`/`0006` bit-identical, and the V100 speedups.
3. Kernel A/B, interleaved rounds: q8_0/q4_0 × D 128/256 × GQA 4/6/8 × KV 4K/32K × 1/2/8 tokens.
   `qkv` vs `p6cfg` at 1 token; `conv`/`conv8` vs `qkv` at 2/8 tokens.
4. End to end: `llama-bench -ctk/-ctv {q8_0,q4_0} -p 8 -n 64 -d 16384` on Llama-3.1-8B and
   Qwen3.8-27B, variants `upstream p6cfg qkv conv8`.

**Accept if:**
- 0 new failures;
- `qkv` ≥ 2× `p6cfg` at 1 token, 32K, GQA 4 (q8_0 and q4_0);
- `conv8` ≥ 1.5× `qkv` at 8 tokens, 32K (quantised);
- Llama-3.1-8B q8_0 tg64 at 16K ≥ +25% over upstream (the model's pessimistic VEC bound gives +38%);
- no f16 case outside noise.

## Upstreaming

`0005` and `0006` are architecture-independent bug-class fixes with a microbenchmark and existing test
coverage (FA with quantised caches, `MUL_MAT`). `0004` is a one-condition Volta selection change with
tests and V100 numbers. Per llama.cpp's contribution rules the owner writes and submits the PR text and
discloses AI use; this repo provides the patches, tests and data.
