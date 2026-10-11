# P-9 design: small-batch quantised matmul on Volta tensor cores

Status: **Phases 1 and 2 (three parts) done: patch `third_party/patches/0008-*.patch`, harness variant `qmm`**,
measured on a V100 2026-10-11. Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 at 5–16 tokens (per-type minimum for N ≥ 4096;
N ≤ 128 wherever MMQ would run): Qwen3.8-27B +35.4% / +41.8% / +40.9% at 8 / 12 / 16 tokens and Llama-3.1-8B
+26.8% / +26.3% at 12 / 16 tokens end to end, unchanged at 1–4 tokens; against an FP32 reference 15× (Qwen) and 6×
(Llama) lower mean KL divergence than the current path; MUL_MAT 2628/2628 on sm_70. Not done: 2–4 tokens, matmul
pairing, IQ4_NL / Q3_K; the kernel's remaining limit is described in "Phase 2, third part".
Phase 0 data: `bench/results/p9-mmbase-v100-*`, `p9-qload-v100-*`, `p9-qload2-v100-*`.
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

## Phase 1 result (2026-10-11, V100)

Patch `0008` adds `ggml/src/ggml-cuda/mmq-volta.cu`: one kernel template for Q4_K, Q5_K and IQ4_XS (D1–D7 as
designed, layout (a')), a small activation-preparation kernel, and a routing branch in `ggml_cuda_mul_mat`
(Volta, 5–16 tokens, K % 256 == 0, contiguous 2-D weights; `GGML_CUDA_NO_MMQ_VOLTA` turns it off; the routing was
tightened in Phase 2, below).
All instances 0 spill, 163–224 registers. Every experiment below used one build with development switches (removed
from the patch) and the upstream path in the same binary as the baseline, interleaved; data in
`bench/results/p9-perf-v1-*`, `p9-sweep1..7-*`, `p9-e2e1/2-*`.

**How it got there** (GB/s on the Qwen ffn shapes, 2–8 tokens / 16 tokens):

| step | change | Q4_K/Q5_K GB/s | finding |
| --- | --- | --- | --- |
| v1 | 8 warps split K over one 32-row tile; B fragments loaded from global memory | 340–440 / 250–355 | correct first time (261/261), but slower than upstream MMVQ at 2–4 tokens |
| – | launch-bounds / warps-per-block sweep | ±10% | not occupancy |
| – | v1 with the activation loads removed (wrong results, timing only) | 510–550 / 437–497 | the 32 global B loads per superblock and warp cost 25–35% |
| v2 | warps split rows; the block shares a superblock's activations in shared memory (padded, conflict-free); K split over blocks with a deterministic last-block reduction | 468–497 / 373–401 | too few blocks (one 4-warp block per SM) collapses to ~305: target ~4 blocks per SM |
| v2 + prefetch | next superblock's weights and activations loaded into registers while the current one is computed | 480–547 / 381–416 | kept |
| v3 | activation preparation fused into the kernel (F32 read, converted per block) | ≤ v2, 10% slower at 16 tokens | rejected: twice the activation bytes per block, more registers |

**Final kernel vs the current path** (`p9-sweep7`, 2 interleaved rounds; upstream GB/s → P-9 GB/s):

| type, shape (N×K) | 4 tokens | 8 | 12 | 16 |
| --- | --- | --- | --- | --- |
| Q4_K 17408×5120 / 5120×17408 | upstream route | 1.32× / 1.40× (434 / 449) | 1.64× / 1.67× | 1.56× / 1.60× (385 / 396) |
| Q5_K | upstream route | 1.54× / 1.72× (494 / 528) | 1.61× / 1.59× | 1.56× / 1.54× (407 / 411) |
| IQ4_XS | upstream route | 1.12× / 1.34× (392 / 413) | 1.53× / 1.59× | 1.48× / 1.54× (345 / 344) |

At 2–4 tokens MMVQ streams at 610–675 GB/s, faster than this kernel (≈480–550), so routing starts at 5 tokens
(an earlier build routed from 2 tokens: Qwen −19% at 2 tokens, −6% at 4).

**End to end** (`p9-e2e2`, Qwen3.8-27B-UD-Q4_K_XL, llama-bench depth 0, 2 rounds × 5 reps, first repetition of each
process dropped; same binary with `GGML_CUDA_NO_MMQ_VOLTA=1` as the baseline): 1 / 2 / 4 tokens 0.0%,
**8 tokens 133.5 → 162.4 t/s (+21.6%), 16 tokens 206.9 → 251.9 t/s (+21.8%)**. The fork's QPN reaches +85–89% on
the same steps; the gap is Q6_K (10% of the bytes, still on MMQ), the lower streaming rate of the native layout
(≈400–530 GB/s vs QPN's ≈500 on its repacked layout at 16 tokens), and QPN's grouped gate+up launches.

**Correctness**: MUL_MAT 2549/2549 on sm_70 (2444 existing + 105 new: Q4_K/Q5_K/IQ4_XS × n = 2, 5, 7, 8, 9, 15, 16 ×
N 64/96/128/192/256 × K 256/512/2304/5120, covering 2- and 4-tile blocks, split and unsplit K, and the fallback
shapes). The new kernel was confirmed to run in the tests with nsys. Perplexity: not measured.

## Phase 2, first part (2026-10-11, V100): Q8_0, accuracy, routing thresholds

**Q8_0** (Llama-3.1-8B): 8 blocks of 32 (272 bytes = 17 16-byte units) are one superblock; each 34-byte block is read
with three 16-byte shared-memory loads and realigned with funnel shifts at compile-time offsets; codes + 128 under 0x64
minus 1152 give q exactly and one HMUL2 by d rounds d*q once, as ggml's FP16 dequantization does. Its weights are not
prefetched into registers (68 more registers spilled at 16 tokens), so it streams at ≈430–590 GB/s. MMVQ is strong on
Q8_0 up to 8 tokens (530–770 GB/s), so Q8_0 wins only from 9 tokens on the larger matrices (`p9-q8a-*`, `p9-final-*`):

| Q8_0 shape (N×K) | 2–6 tokens (forced) | 8 | 9 | 12 | 16 |
| --- | --- | --- | --- | --- | --- |
| 14336×4096 / 4096×14336 | 0.76–1.06× | 1.00–1.09× | 1.27 / 1.30× | 1.27 / 1.30× | 1.25 / 1.23× |
| 4096×4096 | 0.72–0.79× | 0.84× | 1.19× | 1.13× | 1.06× |
| 1024×4096 | 0.57–0.66× | 0.77× | – | 0.94× | 0.86× |

**Small matrices**: with N = 1024 the kernel has too few blocks for the GPU and loses for every type (K-quants
1024×5120: 0.66–0.90×); at 5 tokens IQ4_XS (0.76×) and Q5_K (0.93×) also lose to MMVQ on 6144×5120. **Routing now:**
N ≥ 4096, and from 5 tokens for Q4_K, 6 for Q5_K (MMVQ is used up to 6 on Volta), 8 for IQ4_XS, 9 for Q8_0, up to 16.

**Accuracy** (`p9-kld-*`; llama-perplexity KL divergence of the P-9 path against the upstream path in the same binary,
on 4096 tokens of llama.cpp documentation, batches of 16 and 8 so every matmul is routed to the new kernel; earlier
routing rule, which routed more shapes than the final one):

| model | batch | PPL(P-9)/PPL(upstream) | mean KLD | 99.9% KLD | same top token |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B Q8_0 | 16 / 8 | 1.0012 ± 0.0011 / 1.0008 ± 0.0011 | 0.00030 / 0.00029 | 0.0050 / 0.0037 | 99.0% / 99.2% |
| Qwen3.8-27B | 16 / 8 | 1.0011 ± 0.0015 / 0.9981 ± 0.0014 | 0.0017 / 0.0015 | 0.078 / 0.054 | 98.1% / 98.0% |

The perplexity changes are inside their error bars, and smaller than upstream's own difference between batch 8 and
batch 16 (0.04–0.3% PPL; different kernels). KLD here measures the difference from the upstream path, not error:
upstream quantizes activations to int8 per 32 values, P-9 converts them to FP16. An FP32 reference was not run.

**Final patch, end to end** (`p9-final2-*`, final routing; llama-bench depth 0, 2 rounds × 5 reps, first repetition
dropped, same binary with `GGML_CUDA_NO_MMQ_VOLTA=1` as the baseline): Qwen3.8-27B **+21.8% at 8 tokens (133.1 → 162.2
t/s) and +21.8% at 16 (207.1 → 252.1)**; Llama-3.1-8B-Q8_0 **+8.5% at 16 tokens (624.4 → 677.7)**; 1, 2, 4 tokens (and
Llama at 8) within ±0.3%. MUL_MAT 2540/2540 on sm_70 (96 new cases); the tested diff is byte-identical to patch 0008.

~~Next~~ (superseded by the second part below).

## Phase 2, second part (2026-10-11, V100): profile, K split, Q6_K, FP32-reference accuracy

**Where the time went** (`p9-prof-*`, nsys, Qwen3.8-27B, per 16-token evaluation, same binary): 78.6 ms of GPU time on
the current path, 65.8 ms with the first-part patch, of which Q5_K/IQ4_XS/Q4_K on the new kernel 24.7/10.2/7.9 ms,
Q6_K still on MMQ 4.7 ms (≈1.7 GB per evaluation without the output head, which only runs for the last token; the
earlier "10% of the bytes" was right but the in-loop share had been misread as 0.44 GB), the 96 Q8_0 `ssm_alpha` /
`ssm_beta` matrices (N = 48) 3.3 ms (MMQ 12 µs + stream-k fix-up 22 µs each), and the prep kernels 1.2 ms. Per launch
the new kernel streamed 316–475 GB/s on the large ffn matrices but 190–300 GB/s on N = 6144 / 10240.

**K split by occupancy.** At 9–16 tokens only 2 blocks of 4 warps fit per SM (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`;
4 of 2 warps at ≤ 8 tokens), so the old target of 4 blocks per SM ran in 2–3 waves of short blocks. The split now
takes the P that minimises waves × (superblocks per block + 2), the 2 standing for the unoverlapped first load and the
partial-sum reduction (`p9-q6b-*`, 2 interleaved rounds; costs 0, 2, 4 and 2 warps per block were tried: 2 and 4 are
equal, 0 loses on the largest N, 2 warps per block loses everywhere at 9–16 tokens):

| shape (N×K), vs current path | 8 tokens: old split → new | 12 | 16 |
| --- | --- | --- | --- |
| Q4_K 17408×5120 | 1.47× → 1.72× | 1.62× → 1.88× | 1.55× → 1.87× |
| Q5_K 17408×5120 | 1.55× → 1.73× | 1.59× → 1.78× | 1.54× → 1.80× |
| Q5_K 6144×5120 | 1.29× → 1.49× | 1.25× → 1.43× | 1.16× → 1.36× |
| IQ4_XS 6144×5120 | 1.01× → 1.17× | 1.18× → 1.31× | 1.09× → 1.24× |
| Q4_K 5120×6144 | 1.40× → 1.39× | 1.20× → 1.47× | 1.14× → 1.40× |

Over all 18 K-quant cases: 1.17–1.73× at 8 tokens and 1.24–1.88× at 12–16 (old split: 1.01–1.71× and 1.09–1.65×).
Q8_0 on the Llama shapes (`p9-q6c-*`): 1.21–1.41× at 9–16 tokens (was 1.06–1.30×); at 8 tokens 4096×4096 still loses
(0.89×), so Q8_0 stays from 9.

**Q6_K.** 210-byte superblocks are 2-byte aligned; with K % 1024 == 0 the row pitch is a multiple of 8, so a
superblock starts at the same offset o = 2·(sb mod 4) in every row. A warp loads the 27 8-byte units from the 8-byte
boundary below it (never past the row end), stores rows as 55 32-bit words (odd stride: conflict-free decode reads),
and realigns each word with a funnel shift by 8·(o mod 4). The Q4_K HFMA2 trick does not work: (s, −1056·s) is not
exact in FP16 for every int8 scale, so 1024 + q minus 1056 gives q − 32 exactly and one HMUL2 by the scale rounds
sc·(q − 32) (|·| ≤ 4096) once. 205–255 registers, 0 spill. Speed vs the current path (new split):

| Q6_K shape (N×K) | 4 tokens | 5 | 6 | 7 | 8 | 9–16 |
| --- | --- | --- | --- | --- | --- | --- |
| 17408×5120 / 5120×17408 (ffn) | 0.96 / 0.95× | 1.04 / 1.09× | 1.11 / 1.17× | 1.29 / 1.33× | 1.23 / 1.26× | 1.35–1.39× |
| 5120×6144 (ssm_out, attn_output) | 0.87× | 0.97× | 1.02× | 1.14× | 1.16× | 1.15–1.23× |
| 10240×5120 | 0.93× | 1.01× | 1.08× | 1.23× | 1.18× | 1.23–1.28× |
| 6144×5120 | 0.81× | 0.88× | 0.92× | 1.06× | 1.09× | 1.17–1.21× |

Routed from 7 tokens. It streams at ≈ 260–430 GB/s, below Q4_K/Q5_K (realignment, the extra HSUB2, 2× the shared-memory
instructions of the 16-byte types).

**Small matrices stay on the current path.** Rows past N are now clamped on load and skipped on store, so N need not
be a multiple of 64 (tested with N = 4100 and 4128), but N < 4096 still loses at 16 tokens (`p9-q6c-*`, forced): N =
1024 0.61–0.71×, N = 256 0.66–0.78×, N = 48 0.74–0.98× (Q4_K/Q5_K/Q8_0 1.03–1.23× only at 9–12 tokens). With one or two row groups
there are at most 10 blocks (2 superblocks each), and a warp's work per superblock is sequential: the N = 48 Q8_0 call
takes 23 µs (nsys) for 245 KB. Few-row matrices need a different split (inside a superblock, or across warps on K).

**Accuracy against an FP32 reference** (`p9-acc-*`). A development switch (`GGML_CUDA_F32_REF`, not in the patch)
dequantizes every quantized weight to FP32 and multiplies with cuBLAS SGEMM (Volta has no TF32): the same weights
in exact arithmetic. llama-perplexity KL divergence against it, 4096 tokens of llama.cpp documentation, batch 16
everywhere so the other kernels are identical:

| model | path | PPL ratio | mean KLD | max KLD | same top token |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B Q8_0 | current (q8_1 activations) | 1.0005 ± 0.0010 | 0.00030 | 0.0175 | 98.9% |
| | P-9, final routing | 1.0008 ± 0.0009 | 0.000053 | 0.0052 | 99.6% |
| | P-9 for every matmul | 1.0009 ± 0.0009 | 0.000001 | 0.0001 | 99.8% |
| Qwen3.8-27B | current | 0.9995 ± 0.0014 | 0.0017 | 0.43 | 98.2% |
| | P-9, final routing | 1.0001 ± 0.0003 | 0.00011 | 0.014 | 99.5% |
| | P-9 for every matmul | 1.0000 ± 0.0001 | 0.000013 | 0.0032 | 99.8% |

So the FP16 activations are much closer to exact than the current path's 8-bit ones; what remains with the final
routing comes from the matrices left on MMVQ/MMQ (N < 4096 and 1–4 tokens are not involved at batch 16).

**Final patch, end to end** (`p9-final3-*`, exact patch diff, llama-bench depth 0, 2 rounds × 5 reps, first
repetition dropped, same binary with `GGML_CUDA_NO_MMQ_VOLTA=1` as the baseline):

| model | 1 token | 2 | 4 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-UD-Q4_K_XL | 32.7 → 32.7 | 60.4 → 60.3 | 96.8 → 96.7 | 133.4 → **174.6 (+30.9%)** | 157.4 → **212.9 (+35.3%)** | 206.7 → **275.2 (+33.1%)** |
| Llama-3.1-8B-Q8_0 | 80.2 → 80.2 | 153.7 → 153.8 | 289.8 → 289.8 | 467.7 → 469.2 | 476.2 → **569.4 (+19.6%)** | 624.5 → **733.1 (+17.4%)** |

GPU time per 16-token Qwen evaluation (nsys): 78.6 ms current path, 59.1 ms now (quantised matmul 68.6 → 49.0 ms);
the fork's QPN is at 41.1 ms (32.8 ms of matmul). MUL_MAT 2598/2598 on sm_70; all 20 kernel instances 0 spill, 0
stack, 141–255 registers; the tested diff is byte-identical to patch 0008.

**Next** (superseded by the third part below): the N = 48 Q8_0 matrices (3.3 ms per Qwen evaluation on MMQ + fix-up; need a few-row split), 2–4 tokens
(MMVQ 610–675 GB/s; this kernel ≈ 480–550), pairing matmuls that share an input (one prep, larger grids; the prep kernels
take 1.2 ms per Qwen evaluation), IQ4_NL / Q3_K (1.9 ms per Qwen evaluation on MMQ), and a faster Q6_K decode.

## Phase 2, third part (2026-10-11, V100): split-K reduction, few-row matrices, where the time goes

**Split-K reduction.** The last block of a row group added the P partial sums of each of its elements in a loop of
dependent L2 loads, one chain per element: ntt·8·P round trips in sequence, ≈ 10–15 µs per launch. It showed as a
fixed cost on every split launch: N = 48 Q8_0 at 16 tokens took 26.2 µs with P = 10 and 20.8 µs with P = 20, N = 4096 with
P = 5 ≈ 15 µs more than its streaming time (`p9-few-*`, nsys). All partials of a step are now loaded together and
added in the same fixed order of q, so results are unchanged: N = 48 Q8_0 26.2 → 16.2 µs at 16 tokens, 18.1 → 12.0 µs
at 8 (`p9-red-*`, nsys). The split cost was re-checked with the new reduction (fixed cost 0, 1, 2; 2 interleaved
rounds over the Qwen K-quant, Q6_K and Llama Q8_0 shapes at 5–16 tokens): 2 stays best or tied everywhere; 0 is up to
20% slower on N = 6144 and 17408.

**Few-row matrices** (`p9-small-*`: N = 16–512, K = 4096 / 5120, 5–16 tokens, 3 interleaved rounds; small operations
jump between 20 and 70 µs between runs on both paths, so the minimum of the rounds is used). The kernel takes
17–31 µs at N ≤ 128, MMVQ 9–14 µs and MMQ with its stream-k fix-up 32–39 µs; so at N ≤ 128 it wins exactly where the
current path picks MMQ (per-type MMVQ limits on Volta: Q4_K 5, Q5_K 6, Q6_K 7, IQ4_XS and Q8_0 8 tokens) and loses
where it picks MMVQ:

| N (K = 4096 / 5120) | Q4_K | Q5_K | Q6_K | IQ4_XS | Q8_0 |
| --- | --- | --- | --- | --- | --- |
| 16–128, where MMQ runs | 1.35–2.90× | 1.21–1.86× | 1.07–1.51× | 1.29–1.55× | 1.38–1.97× |
| 16–128, where MMVQ runs | ≤ 0.68× | ≤ 0.70× | ≤ 0.64× | ≤ 0.62× | ≤ 0.78× |
| 256, where MMQ runs | 1.08–1.43× | 1.02–1.31× | 0.89–1.05× | 1.05–1.10× | 1.09–1.18× |
| 512, where MMQ runs | 0.94–1.25× | 0.89–1.17× | 0.77–0.93× | 0.87–0.98× | 0.94–1.02× |

Routing: N ≤ 128 when `ggml_cuda_should_use_mmvq` is false (N = 129–4095 stays on the current path). In the model the
gain is smaller than these numbers suggest: test-backend-ops repeats one 245 KB matrix, which stays in L2, while
Qwen's 96 `ssm_alpha` / `ssm_beta` launches per evaluation take 16.5–35.6 µs (median 23.0, `p9-few2-*/dist.txt`)
against 34.4 µs for MMQ + fix-up. Splitting K further (1 superblock per block, P = 20) changes nothing there (28.0 vs
28.2 µs average) or end to end; the cause of the in-model cost is not found.

**Q6_K decode** (from the fork's QPN comment): with the codes at bits 4–9 under 0x6400 (1024 + 16q), one HFMA2 with
sc/16 and −96·sc gives sc·(q − 32) with one rounding: both constants are exact in FP16 for every int8 sc, and the
result equals the previous HSUB2 + HMUL2 bit for bit (checked for all sc and q). About 100 fewer instructions per
superblock and warp (2221 → 2124 in the main loop, SASS); its speed was not measured separately.

**Where the time goes** (`p9-exp-*`, same binary, development switches; 2 rounds; times include the prep kernel):

| shape, 16 tokens | current path | P-9 | weights from L2 (compute only) | loads + staging only | 3 blocks/SM (`__launch_bounds__`) |
| --- | --- | --- | --- | --- | --- |
| Q4_K 17408×5120 | 205 µs | 109 µs | 86 µs | 86 µs | 131 µs |
| Q5_K 17408×5120 | 234 µs | 129 µs | 101 µs | 100 µs | 174 µs |
| IQ4_XS 17408×5120 | 202 µs | 121 µs | 97 µs | 80 µs | 140 µs |
| Q6_K 17408×5120 | 239 µs | 169 µs | 122 µs | 116 µs | 200 µs |
| Q8_0 4096×14336 (no prefetch) | 175 µs | 122 µs | 64 µs | 109 µs | 121 µs |

"Weights from L2": every row tile reads rows 0..31, so the weights come from L2 (wrong results, timing only). "Loads +
staging only": no decode or mma, shared memory padded so the same number of blocks fit per SM (at 8 tokens this
variant is slower than the full kernel, probably because the padding takes L1 capacity, so it is used at 16 only).
For the K-quants, compute alone and loads alone each take ≈ 80% of the full time: neither dominates, they already
overlap, and making one side faster gains little. Q8_0, which loaded its weights only when staging them, is
load-bound; prefetching 8 of its 17 units into registers (`p9-pf-*`, 0 / 8 / 17 units in three builds, each against
the current path in the same binary, 3 interleaved rounds) is 3–4% faster on the large shapes (e.g. 17408×5120 at 16
tokens 159.8 → 153.3 µs); all 17 fit (245 registers, 0 spill) but are no faster. More blocks per SM through
`__launch_bounds__(…, 3)` (≤ 168 registers) are slower everywhere, also with the weights in L2.

Static count (SASS, Q4_K, 9–16 tokens): 1430 instructions per superblock and warp in the main loop, 528 of them HMMA
(FP32 accumulation: 4 per m8n8k4). With 2 blocks of 4 warps per SM the schedulers issue in ≈ 35–40% of cycles, so
the compute side is latency-bound rather than issue-bound.

A load microbenchmark (`p9-exp-*/bwtest.cu`, `bwtest.txt`: 32 rows × 144 bytes per warp and step, registers →
shared memory, depth 1, 2 blocks of 4 warps per SM, 17408 × 5120 Q4_K-sized rows) reads 685 GB/s with the native
layout, 747 GB/s with 2 superblocks per row and step (288 contiguous bytes), and 801 GB/s with a tile-contiguous
layout like the fork's repacking. The fork's QPN pair kernel streams ≈ 590 GB/s on the Q5_K gate/up pair (207.8 µs for two
17408×5120 matrices; nsys, `ce-v100-*`) against 365–600 GB/s per launch here (`p9-final4-*/shapes-p9-pp16.txt`;
the 136-block launches of 17408-row matrices are the slow end). Closing that needs both a cheaper compute side (no
weight staging through shared memory, which in the native layout is what makes the loads coalesced) and longer
contiguous reads: a different kernel structure, probably with repacked weights, i.e. design option (b).

**Final patch, end to end** (`p9-final4-*`, exact patch diff, llama-bench depth 0, 2 rounds × 5 reps, first
repetition dropped, same binary with `GGML_CUDA_NO_MMQ_VOLTA=1` as the baseline):

| model | 1 token | 2 | 4 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-UD-Q4_K_XL | 32.7 → 32.7 | 60.3 → 60.4 | 96.8 → 96.8 | 133.3 → **180.4 (+35.4%)** | 157.6 → **223.6 (+41.8%)** | 207.0 → **291.7 (+40.9%)** |
| Llama-3.1-8B-Q8_0 | 80.2 → 80.3 | 153.8 → 153.8 | 289.9 → 289.9 | 469.3 → 469.6 | 476.1 → **603.7 (+26.8%)** | 624.8 → **789.2 (+26.3%)** |

(Second part: Qwen +30.9 / +35.3 / +33.1%, Llama +19.6 / +17.4%.) Most of the Llama gain is the reduction fix: its
4096-row matrices run with P = 5. GPU time per 16-token Qwen evaluation (nsys): 59.1 → 56.2 ms, quantised matmul
49.0 → 46.2 ms (fork: 32.8 ms). MUL_MAT 2628/2628 on sm_70; all 20 kernel instances 0 spill, 0 stack, 163–255
registers; the tested diff is byte-identical to patch 0008. Accuracy was not re-measured: the only routing change is
the N ≤ 128 rule, which moves matrices to the FP16 path.

**Next**: the kernel structure (above) is the remaining large item (46.2 ms of quantised matmul per Qwen evaluation
against the fork's 32.8 ms); it needs the weight-layout mechanism of design option (b), which is an owner decision. Smaller: 2–4 tokens (MMVQ
610–675 GB/s, this kernel ≈ 480–550), IQ4_NL / Q3_K (1.9 ms per Qwen evaluation on MMQ), pairing matmuls that share
an input (one prep; the fork's pair kernels), the in-model cost of the N = 48 matrices.

## Upstreaming

A Volta-only kernel plus a routing branch is a self-contained ggml-cuda change; (b) is a cross-cutting ggml/llama
change that maintainers would need to want. Per llama.cpp's rules the owner writes and submits any PR text and
discloses AI use; this repo provides patches, tests and data.

## Not measured

The real kernel (activation loads, scales/mins, split-K reduction, double buffering); (a') for Q6_K, IQ4_XS and Q8_0;
accuracy (perplexity) of fp16-scaled activations vs q8_1; the fork's memory or latency cost of repacking beyond the
177 ms of GPU time seen at load; models other than the two above.
