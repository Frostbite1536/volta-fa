# P-9 design: small-batch quantised matmul on Volta tensor cores

Status: **design; Phase 0 done on a V100** (2026-10-11): upstream baseline per shape, and the weight-layout
microbenchmark that settles D3 (native GGUF layout with shared-memory staging; no weight repacking). Phase 1 (the
kernel) not started. Data: `docs/research/qwen-smallbatch-and-vecpack-heads-v100-2026-10-11.md`,
`bench/results/p9-mmbase-v100-*`, `bench/results/p9-qload-v100-*`, `bench/results/p9-qload2-v100-*`.
Owner decision points are marked **Decision**.

## Problem (measured)

On a V100, Qwen3.8-27B-UD-Q4_K_XL at 16 tokens (speculative-verify and small-batch shape, depth 0):

| | upstream | llama-superfast-v100 fork | fork, `LLAMA_MMVQ_QPN=0` |
| --- | --- | --- | --- |
| t/s (median of samples) | 205.3 | 387.7 (+89%) | 133.7 (−35%) |
| quantised matmul per evaluation | **68.5 ms** (`mul_mat_q`, dp4a) | 32.8 ms (`qpn_mul*`) | 114.8 ms (`mul_mat_vec_q`) |
| everything else | 10.2 ms | 8.3 ms | 6.7 ms |

8 tokens: +85% for the fork, −2% with its QPN path off; 4 tokens +27%; 1 token −6% (QPN is slower than
upstream's `mul_mat_vec_q` at one token). The fork's whole small-batch gain on this model is its quantised matmul;
attention is ~1 ms of the step. Llama-3.1-8B-Q8_0 is the second model in the set (all Q8_0 weights).

Upstream per shape (`bench/results/p9-mmbase-v100-*`, `test-backend-ops perf -o MUL_MAT`, 2 rounds; tool:
`bin/qmatmul-table.py perf`), effective weight bandwidth in GB/s on the ffn gate/up shape (N 17408, K 5120) and the
ffn down shape (N 5120, K 17408):

| type | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
| --- | --- | --- | --- | --- | --- | --- |
| Q5_K gate/up / down | 717 / 650 | 672 / 670 | 476 / 498 | 318 / 309 | 266 / 266 | 264 / 265 |
| Q4_K | 680 / 686 | 616 / 639 | 413 / 440 | 333 / 324 | 248 / 248 | 247 / 247 |
| IQ4_XS | 653 / 729 | 623 / 703 | 505 / 544 | 341 / 308 | 238 / 228 | 236 / 223 |
| Q6_K | 652 (down) | 617 | 490 | 363 | 306 | 304 |
| Q8_0 (Llama ffn up, N 14336, K 4096) | 775 | 733 | 695 | 528 | 369 | 363 |

MMVQ streams at 570–775 GB/s at one token; by 4 tokens the K-quants are at 400–550 (Q8_0 still ~700), and MMQ at
9–16 tokens reaches 205–365 GB/s on every measured shape. Estimated matmul time per evaluation from these shapes (`qmatmul-table.py step`, 86% of Qwen's bytes
and 96% of Llama's measured exactly, the rest at the same type's median rate):

| model | n=1 | n=2 | n=4 | n=8 | n=16 | floor (748 GB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | 22.6 ms | 23.6 | 31.8 | 46.8 | 58.7 | 20.6 ms |
| Llama-3.1-8B | 10.2 ms | 10.7 | 11.3 | 14.5 | 21.7 | 9.9 ms |

The profiled Qwen step (23.6 ms at 1 token without the head, 68.5 ms at 16) is consistent; the 16-token profile also
contains the activation quantization, stream-k fixups and the 14% of bytes not measured here. **Both models have a
2–3× matmul gap at 8–16 tokens**; for Llama the fork does not help (it has no Q8_0 route for those shapes).

## Why upstream is slow here (code: `third_party/llama.cpp` @ `79e2e74`, `ggml/src/ggml-cuda/`)

Routing for a quantised weight and F32 activations at n = 1..16 on cc 700 (`ggml-cuda.cu:1906-1959`):
`mul_mat_vec_q` (MMVQ) up to a per-type column limit on Volta (`mmvq.cu:368-382`: Q4_K ≤ 5, Q5_K ≤ 6, Q6_K ≤ 7,
Q3_K ≤ 6, Q8_0 / IQ4_XS / IQ4_NL ≤ 8), then `mul_mat_q` (MMQ) for every n < 64 (`mmq.cu:423-425`,
`MMQ_DP4A_MAX_BATCH_SIZE`). cuBLAS is not used at these sizes.

- **MMVQ** decodes each weight block once *per column* (`mmvq.cu:747-758`), so its ALU work grows linearly with
  n while the bytes stay the same. It is bandwidth-bound at 1 token (the Q6_K output head, 1.04 GB, reads at
  748 GB/s) and compute-bound a few columns later, hence the per-type limits.
- **MMQ on Volta is the dp4a path**: `turing_mma_available` is false on sm_70 and there is no int8 tensor core,
  so it uses the Ampere config table with dp4a vec-dots (`mmq.cuh:189-201, 254-256`). At J=16 each 128×256 tile is
  decoded into shared memory and reused across only 16 columns (2 columns × 4 rows per thread), limited by shared
  loads, dp4a/IMAD and float scale work. Measured: 68.5 ms for 15.43 GB, i.e. ~225 GB/s effective.
- Volta does have FP16 tensor cores (`mma.sync.m8n8k4`), which neither path uses for quantised weights. Upstream's
  `mma.cuh` already has the Volta tile types (`mma.cuh:152, 281-429`), used by the FA kernels.

## Floor and budget

Weights read per evaluation (`bench/results/gguf-tensors-20261011/`, tool: `bin/qmatmul-table.py`):

| model | matmul weights (no output head) | by type | output head |
| --- | --- | --- | --- |
| Qwen3.8-27B-UD-Q4_K_XL | 15.43 GB, 496 tensors | Q5_K 7.93, IQ4_XS 3.13, Q4_K 2.34, Q6_K 1.48, IQ4_NL 0.28, Q8_0 0.12, Q3_K 0.11, IQ3_S 0.04 | Q6_K 1.04 GB |
| Llama-3.1-8B-Q8_0 | 7.42 GB, 224 tensors | Q8_0 7.42 | Q8_0 0.56 GB |

(Qwen's block 64, the next-token-prediction layer, is not run by a normal evaluation and is excluded.)

- **Bandwidth floor** at the 748 GB/s that MMVQ reaches on the Qwen output head: 20.6 ms (Qwen) and 9.9 ms
  (Llama) per evaluation, the same for every n ≤ 16. Upstream at 16 tokens is 3.3× above it, the fork 1.5×.
- **Tensor cores**: 2 × 24.35 G weights × 16 tokens = 0.78 TFLOP per Qwen evaluation; at the V100-PCIe peak
  (~112 TFLOPS) 7.0 ms. m8n8k4 pads tokens to 8, so 1–8 tokens cost the same as 8 (3.5 ms). Not the limit if it
  overlaps with the loads, but not negligible either.
- **Decode ALU**: turning codes into fp16 costs ~1–2 instructions per weight with the 0x6400 / HFMA2 trick (below),
  ~37 G lane-instructions per Qwen evaluation at 1.5 per weight, against ~14 T lane-instructions/s: ~2.6 ms,
  overlappable.
- **Target**: ≤ 25 ms of matmul per 16-token Qwen evaluation (≥ 620 GB/s, 83% of the measured stream rate), i.e.
  better than the fork's ~31 ms, with no loss at 1 token (MMVQ stays there unless beaten).

## Prior art: the fork's QPN (read, not copied)

`llama-superfast-v100` `70df871`, `ggml/src/ggml-cuda/mmvq-qpn.cu` (MIT; it credits NInfer's Apache-2.0
`nvfp4_volta_qpn_gemm.cuh` and "v100-skinny" and states no code was copied). Reference copy for reading:
`third_party/superfast-qpn-ref/` (gitignored). What it does:

- Weights are the A operand (`tile<32,4,half2>`: lane l holds 8 consecutive K values of row l, so a warp covers
  32 rows), tokens the B operand (8 columns, zero-padded), fp32 accumulators; two accumulators alternate for ILP.
  9–16 tokens: two 8-token B tiles share each decoded A fragment.
- Split-K across the warps of a block (shared-memory reduction), optionally across blocks with a deterministic
  last-block reduction.
- **Weights repacked once at load, in place and at the same size**, so each lane's 16-byte `ld.global.nc` is its
  fragment data and a warp reads 512 contiguous bytes; no shared-memory staging.
- Codes to fp16: or-ing 0x6400 into masked nibbles gives a half2 of 1024 + 16q; one HFMA2 with (sc/16, −64·sc)
  gives sc·q exactly for Q4_K/Q5_K (sc·(q−32) with one rounding for Q6_K); PRMT codebook lookups for IQ4_*;
  superblock d and Q4_K/Q5_K mins (one extra mma against per-32 activation sums) applied in fp32 per superblock.
- Activations: fp16, each 256-column slice of a token scaled by a power of two so max |x| < 2^15 (exact, undone in
  fp32); a non-finite slice routes that token to an fp32 fallback. Upstream MMQ/MMVQ quantise activations to q8_1
  (int8 per 32 values), so fp16 activations are likely no less accurate; no perplexity data is recorded.
- Measured: 32.8 vs 68.5 ms at 16 tokens; slower at 1 token (29.5 vs 25.0 ms: 7 of 8 B columns are padding).

Not upstreamable as it stands: every route, split and fusion is a table keyed to Qwen3.8-27B's exact (N, K) shapes
(other models get nothing unless forced); grouping relies on edits to `qwen35.cpp`; repacking needs a new tensor
flag, a load hook in `llama-model.cpp`, global state, and an abort if any other op reads a repacked weight.

## Design

### D1. Kernel family
m8n8k4 FP16 tensor-core products with the weights as A (32 rows per warp, 8 K per lane) and the tokens as B,
fp32 accumulation, split-K across warps; 1–8 tokens per B tile, 9–16 as two B tiles sharing each decoded A
fragment. Use upstream's `mma.cuh` Volta tiles. This is the QPN structure; it is the natural Volta mapping
(M = 8 is the token side) and there is no competing one worth exploring first.

### D2. Activations
fp16 with a power-of-two scale per 256 columns and an fp32 fallback for non-finite slices (needed for the HFMA2
decode path; q8_1 would need dp4a again). Correctness is judged by `test-backend-ops` MUL_MAT (NMSE against the CPU
reference, unchanged thresholds) and by perplexity against upstream on both models (Phase 1 exit).

### D3. Weight layout (**Decision**, settled by Phase 0 data)
- **(a) Native GGUF layout.** Each lane reads its own row's blocks. Q4_K (144 B = 9 × 16 B) and Q5_K
  (176 B = 11 × 16 B) blocks are whole 16-byte chunks with the same chunk structure QPN uses, so a lane can issue the
  same 16-byte loads; the difference is the access pattern: 32 lanes hit 32 rows (stride = row pitch), each load
  using half a 32-byte sector whose other half the same lane reads next. Q6_K (210 B), IQ4_XS (136 B) and Q8_0 (34 B)
  are not 16-byte multiples: 2/8-byte loads or pairing rows. **No ggml or llama.cpp changes**: one kernel file and a
  routing branch, which is what an upstream PR can carry.
- **(b) Repacked fragment order** (QPN). Best access pattern, any type. Needs a CUDA weight-layout mechanism that
  upstream does not have: no CUDA extra buffer type exists, and llama.cpp picks the first buffer type whose
  `supports_op` passes, with the device default first (`llama-model.cpp:1162-1206`,
  `llama-model-loader.cpp:1067-1077`), so a CUDA extra buffer type would never be selected without a change to that
  order or programmatic overrides. The CPU backend's repack buffer type (`ggml-cpu/repack.cpp:5143-5255`,
  `ggml-cpu.cpp:42-74`) is the pattern: `set_tensor` repacks, `get_tensor` is null, `supports_op` limits readers.
  Every other reader must be excluded (cuBLAS dequant fallback, fusion paths, scheduler copies).

**Phase 0 result** (`bench/micro/qload.cu`, V100, 2 rounds agreeing within 1–4%; best S × P geometry per variant;
GB/s = native-layout bytes / time). The `_mma` variants add the real kernel's per-weight work on every 16 bytes
(32 codes: 16 HFMA2 decodes and 8 m8n8k4 MMAs per lane, activations held in registers, so a lower bound):

| Q4_K / Q5_K, 8 Qwen shapes | load only | load + decode + mma |
| --- | --- | --- |
| stream (ideal contiguous read) | 617–787 | – |
| (b) repacked fragment order | 617–787 | **557–688** |
| (a) native, lane l reads row l (16 B per row per load) | 340–576 (0.44–0.80× of b) | 202–364 (0.29–0.64×) |
| (a') native, 128-byte row reads staged through shared memory | 566–721 (0.86–0.94×) | **422–617 (0.76–0.90×)** |

Q6_K, IQ4_XS and Q8_0 were measured load-only for (a) and (b): (a) 0.38–0.95× of (b). All variants pass a per-row
checksum against the CPU.

The naive native pattern fails (32 rows per load instruction, each a 16-byte piece of a different DRAM page). Staging
through shared memory recovers most of it: each load instruction reads 4 rows × 128 contiguous bytes, and a padded
(144-byte row stride, bank-conflict free) shared buffer turns them into fragment order. On the ffn shapes, which hold
most of the bytes, (a') reaches 0.81–0.90× of (b) with the decode and MMA work included, without software
pipelining (the staging loads of a warp do not overlap its own MMAs yet).

**Decision (recommended): (a') for Phase 1**, with double-buffered staging added in Phase 1. At ~570 GB/s Qwen's
16-token matmul would take ≈ 27 ms, already under the fork's ≈ 31 ms, with no ggml or llama.cpp changes. (b) stays
available as Phase 3 if the Phase 1 kernel lands more than ~15% below the repacked microbenchmark rate; it would buy
~15% at the cost of a cross-cutting ggml/llama change.

### D4. Routing: general, no shape tables
Volta, supported type, 2 ≤ n ≤ 16 (n = 1 stays MMVQ unless Phase 1 beats it), K a multiple of the superblock, N a
multiple of 32, contiguous 2-D weight. Split-K from a formula on (row tiles = N/32, superblocks = K/256, SM count),
not per-shape tables. Above 16 tokens, MMQ as today.

### D5. Types, in order
Q5_K, Q4_K, IQ4_XS, Q6_K (96% of Qwen's bytes), then Q8_0 (Llama), then IQ4_NL / Q3_K.

### D6. Determinism
Split-K reductions in a fixed order (shared memory within a block; cross-block only with a fixed-order final step),
so results do not depend on scheduling.

### D7. No fusion in Phase 1
QPN gains extra time from launching gate+up (and other pairs sharing an input) together; that is a later step and
needs a generic pairing rule (same src1, both routed), not graph edits.

## Plan

| phase | what | exit |
| --- | --- | --- |
| 0 (done) | V100: upstream MUL_MAT per shape at 1–16 tokens; layout microbenchmark (a) / (a') / (b) | D3 decided: (a') |
| 1 | kernel for Q5_K + Q4_K, then IQ4_XS and Q6_K, at 2–16 tokens, native layout with double-buffered shared-memory staging, as an upstream patch (one new file + a routing branch in `ggml_cuda_mul_mat`); `test-backend-ops` MUL_MAT cases for n = 2..16 on the Qwen shapes plus odd N/K and non-finite activations; static check (0 spill) | per shape ≥ 2× upstream at 16 tokens and ≥ 1.0× at 2–8; ≥ 85% of the (b) microbenchmark rate on the ffn shapes; Qwen pp8/pp16 at depth 0 ≥ +50% end to end; perplexity within noise of upstream; no change at n = 1 and n ≥ 17 |
| 2 | Q8_0 (Llama: 2.1× upstream headroom at 16 tokens), remaining types; generic gate+up pairing (same src1); 1 token if it beats MMVQ | Llama pp8/pp16 gain, no 1-token loss |
| 3 | only if Phase 1 misses the 85% rate: repacked layout (b) with a CUDA repacked-weight buffer type and the llama.cpp buffer-type order change | owner decision on upstreaming a ggml/llama change |

## Upstreaming

A Volta-only kernel plus a routing branch is a self-contained ggml-cuda change; (b) is a cross-cutting ggml/llama
change that maintainers would need to want. Per llama.cpp's rules the owner writes and submits any PR text and
discloses AI use; this repo provides patches, tests and data.

## Not measured

The real kernel (activation loads, scales/mins, split-K reduction, double buffering); (a') for Q6_K, IQ4_XS and Q8_0;
accuracy (perplexity) of fp16-scaled activations vs q8_1; the fork's memory or latency cost of repacking beyond the
177 ms of GPU time seen at load; models other than the two above.
