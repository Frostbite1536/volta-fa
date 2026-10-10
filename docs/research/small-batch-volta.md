# Small-batch FlashAttention on Volta: upstream vs llama-superfast-v100 (code study, 2026-10-10)

Code reading only; nothing built or run. Upstream anchors are `ggml/src/ggml-cuda/<file>:<line>` at llama.cpp
`79e2e74`. Fork anchors are `fork:<file>:<line>` at llama-superfast-v100 `70df871`. Notation: B = `Q->ne[1]`
(query tokens), G = GQA ratio, `<n1,n2>` = MMA/TILE instance with ncols1 = n1 tokens and ncols2 = n2 Q heads per
CTA (n1·n2 columns). "Passes" = how many times each KV head's K/V is streamed from DRAM per call, ignoring L2
and split-KV.

## 1. Upstream kernel choice on cc 7.0

**Selection** (`fattn.cu:667-681`): `gqa_ratio_eff` = largest power-of-two divisor of G, capped at 8
(`668-671`; it checks `max_bias` but not `gqa_opt_applies`). VEC if B·eff ≤ 2 (and `can_use_vector_kernel`,
`635`), TILE if B·eff ≤ 16 (`677-679`, "tensor cores are only faster for sufficiently large matrices"),
otherwise MMA. D=128 and D=256 take the same path.

**MMA dispatch**: Volta branch `fattn.cu:224-246`: ncols2 = 8 if G%8==0, 4 if G%4==0, 2 if G%2==0, else 1
(all need `use_gqa_opt`, `207-218`). Then `switch_ncols1` (`170-190`): the 8-column branch is Turing-only
(`171`); the 16-column branch (`178`) is unguarded; otherwise ncols1 = 32/ncols2 if B ≤ 32/ncols2, else 64/ncols2.
**TILE dispatch**: ncols2 uses the same divisibility rule (`fattn-tile.cuh:1298-1316`); ncols1 picks the
smallest block of {2,4,8,16,32} columns with B ≤ cols/ncols2 (`fattn-tile.cuh:1149-1231`).

| G (eff) | B=1 | B=2 | B=4 | B=8 | B=16 |
| --- | --- | --- | --- | --- | --- |
| 4 (4) | TILE `<1,4>` | TILE `<2,4>` | TILE `<4,4>` | MMA `<8,4>` | MMA `<16,4>` |
| 6 (2) | VEC (1 head/CTA, 6 passes) | TILE `<2,2>` | TILE `<4,2>` | TILE `<8,2>` | MMA `<16,2>` |
| 8 (8) | TILE `<1,8>` | TILE `<2,8>` | MMA `<4,8>` | MMA `<8,8>` | MMA `<8,8>` ×2 Q tiles |

Passes = ceil(G/ncols2)·ceil(B/ncols1): 1 for G=4 and G=8 (2 at G=8,B=16), **3 for every G=6 case**, and G for VEC
(`fattn-vec.cuh:106-111`: one CTA per Q head).

**Why G=6 stops at ncols2=2.** This is dispatch policy only: 6%8 and 6%4 fail at `fattn.cu:225/230`, 6%2
passes at `236` (comment `223`: "On Volta the GQA optimizations aren't as impactful vs. minimizing wasted
compute"). Selection mirrors this through `gqa_ratio_eff` = 2. The kernel does not need G to be divisible by ncols2:
`iter_z_gqa` is a ceiling (`fattn-mma-f16.cuh:1892`), the Q load zero-fills heads with
`zt_gqa*ncols2 + c >= gqa_ratio` (`1297-1314`), output writes skip them (`1739-1744`), and `launch_fattn` tiles
with `ntiles_z_gqa = ceil(G/ncols2)` (`fattn-common.cuh:1097`). Non-Volta NVIDIA already runs G=6 as ncols2=8
with 2 phantom heads (`fattn.cu:266-269`).

**Why Volta MMA needs ncols1·ncols2 ≥ 32.** Volta tiles (`fattn-mma-f16.cuh:1178-1186`): `T_B_KQ` =
`tile<32,4,half2>` (32 Q columns), `T_A_KQ` = `tile<8,4,half2,I_MAJOR_MIRRORED>` (8 K rows), `T_C_KQ` =
`tile<32,8,float>`. `m8n8k4` runs as four independent 8×8×4 products, one per quad-pair (`mma.cuh:21-22`);
ggml stacks four different 8-column Q slices as the PTX A operand and mirrors one K fragment across the
quad-pairs (`mma.cuh:1504-1520`; call site "swap A and B for CUDA", `fattn-mma-f16.cuh:672-683, 700-712`). PV
works the same way (V mirrored `T_A_VKQ`, 32-column probability `T_B_VKQ`, `T_C_VKQ` = `tile<32,4,half2>`, so
**VKQ accumulates in FP16**). The smallest tile a warp can work on is therefore 32 Q columns × 8 KV rows:
`cols_per_warp = T_B_KQ::I = 32` (`607, 1227`; host `340-347`), `cols_per_warp > ncols` traps (`1237-1240`), and
the entry guard compiles every `<32`-column instance as NO_DEVICE_CODE (`1847-1852`).

**Known bug:** selection ignores `gqa_opt_applies`, so a shape with K length not a multiple of 256, no mask,
or misaligned strides picks MMA. Dispatch then uses ncols2 = 1 and B ≤ 16 takes branch `178`, which launches the
`<16,1>` NO_DEVICE_CODE instance (phase0-cloud §4). **Config gap:** Volta D=256 falls through to the Ampere rows
(`fattn-mma-f16.cuh:126-127`) with `Q_in_reg=true` at 32/64 columns (`69-72`), which spill on sm_70 (phase0-cloud §1;
`fork:fattn-mma-f16.cuh:124-129` reports 568 B of spill).

## 2. TILE kernel on Volta

- **No tensor cores.** FAST_FP16_AVAILABLE is defined for sm_70 (`common.cuh:271-273`), so the
  `nvidia_fp16` configs apply (`fattn-tile.cuh:322-323`). KQ: K is staged in shared memory, and each step is a
  half2 multiply followed by two converts and two FP32 adds (`ggml_cuda_mad(float&, half2, half2)`,
  `common.cuh:764-777`; call `fattn-tile.cuh:546`). VKQ: half2 FMA into FP16 accumulators (`fattn-tile.cuh:753`,
  `common.cuh:780-782`). Softmax is FP32. The fork's measurement comment says this kernel is FMA-pipe bound at
  2.5-3.5× its byte floor at depth (`fork:fattn.cu:722-724`).
- **Configs (threads, occupancy, nbatch_fa, nbatch_K)**, same for D=128 (`fattn-tile.cuh:58-62`) and D=256
  (`70-74`): 2 columns → 64,2,64,64; 4 → 128,2,64,64; 8/16/32 → 256,2,64,64. Columns are spread over warps
  (`cpw`/`np`, `497-499`).
- **K/V traffic:** grid z = (Q heads / ncols2)·sequences. Each CTA reads KV head `head0/G` (`852-858`), so the
  KV head is streamed G/ncols2 times (once for G=4/8, **3× for G=6**), and once per CTA within its split-KV slice
  (stream-k off: `launch_fattn(..., stream_k=false)`, slicing at `fattn-common.cuh:1184-1214`). Whether the
  three G=6 re-reads hit L2 cannot be told from code. *(inferred: mostly misses, because the K+V of one head at
  32K depth and D=256 is 32 MB against 6 MB of L2, unless the three CTAs run in lockstep.)*

## 3. The fork's small-batch design

**Routing.** Every small-batch change is gated on D=256:
1. **`verify_stream`, a new hand-written kernel.** It replaces the MMA call (`fork:fattn.cu:837-841`; whole-cache
   decode graphs: `823-827`) when `fork:fattn-verify-stream.cu:33-75` holds: sm_70, F32 Q, F16 K/V and mask, no
   sinks, softcap or ALiBi, D=256, **1 ≤ B ≤ 16, Q heads == 6 × KV heads (G exactly 6)**, K ≥ 4096
   (`LLAMA_FATTN_VERIFY_STREAM`), K%256==0, 16-byte aligned. B ≤ 8 calls reach the MMA case only because selection
   was changed: `verify_to_mma` (`fork:fattn.cu:722-731`: 2 ≤ B ≤ 8, B·eff ≤ 16, D=256, F16/BF16, K ≥ 4096) and the
   one-row route (`711-718`: B=1, D=256, 2 < G ≤ 8).
2. **MMA fallback** when `verify_stream` does not apply (BF16 cache, or G ≠ 6): `fork:fattn.cu:218-228` sets
   ncols2 = 8 for D=256, 2 < G ≤ 8, B ≤ 8, K ≥ 4096 ("the padded columns are guarded"). It reuses upstream's
   phantom-head guards from §1. With the 16-column fix (`fork:fattn.cu:161`) this gives `<4,8>` for B ≤ 4 and
   `<8,8>` above.
3. `fattn-gqa6.cu` (CUTLASS) handles prompts only (≥ 256 rows, `fork:fattn-gqa6.cu:75,744`).

**How `verify_stream` handles G=6** (`fork:fattn-verify-stream.cuh`):
- **No head padding.** One CTA holds a fixed 48-row tile, 8 tokens × 6 heads, with row = token·6 + head
  (`29-31, 69, 126-139`). Rows ≥ B·6 get zero Q (`133`) and are skipped on output (`439`). The mask is per token
  (`s.M[(qrow/G)]`, `230`), and missing tokens get a 0 mask (`144`), so their softmax stays finite. **The full 48-row
  tile is computed for any B ≤ 8** (B=4 wastes 50%).
- **Operand roles are swapped relative to ggml.** In QK, the K fragments differ per quad-pair (`keyA`, `157`)
  and the Q fragment is shared (`204`), so one m8n8k4 covers 32 keys × 8 Q rows. In PV, V is the A operand with
  dims per quad-pair (`158, 438`) and Pᵀ is shared. Query granularity is 8 rows, not 32. O accumulates in **FP32**
  (`340`); upstream Volta MMA accumulates VKQ in FP16.
- **Warp specialisation**, 8 warps and 1 CTA/SM (`77`). Warps 0-3 do QK and the exp2 online softmax (K is staged
  in shared memory); warps 4-7 do PV one 64-key tile behind, with V loaded straight into registers and P passed
  through double-buffered shared memory (header `4-19`).
- **"Each KV head streamed once":** grid x = nsplit·nkv_heads·nhalf (`fork:fattn-verify-stream.cu:127`). One CTA
  per (KV head, split) owns all 6 heads, so each K/V element is read once per split, against 3× in upstream's
  `<·,2>` TILE/MMA.
- **Split-KV, not stream-k:** nsplit = max(1, min(occ·nsm/nkvh, ntiles)), one wave (`.cu:108`); partials are merged
  by `flash_attn_combine_results` (`.cu:134-137`). For 9-16 tokens, two CTAs per (head, split) stream the same tiles
  side by side, relying on L2 for the second read (`.cuh:70-72`, `.cu:117`).

**Llama-3.1-8B (D=128, G=4) regression: the code rules out the fork's FA routes.** None of the routes above can
see D=128: all are gated on `Q->ne[0]==256`, `DKQ==256`, or D=256 with G=6. The fork's Volta MMA config additions
are D=256 only (`fork:fattn-mma-f16.cuh:124-129`), and the D=128 MMA and TILE config rows match upstream. In both
trees B=8/16 runs MMA `<8,4>`/`<16,4>`, and B ≤ 4 runs TILE. The fork's Volta stream-k threshold
(`fork:fattn-common.cuh:1148-1154`) changes nothing here: Llama-8B has 8 KV heads, so there are 8 output tiles
for 160 CTA slots and both trees use stream-k. *(inferred)* The regression is most likely outside FA. Candidates:
the fork's ggml-cuda.cu is about 421 KB and rewrites matmul routing (MMVQ-QPN, the MMVQ-TC tensor-core path for
multi-token dense K-quant GEMVs, q8_1 sharing at `fork:ggml-cuda.cu:6715`); the Llama model is Q8_0, and pp8/pp16
sit at the MMVQ/MMQ boundary. A difference in base version is also possible (the fork was imported from
Unsloth's build of `6ba30d0`, which is not in our clone). The fork does show a real "route captures non-target
shapes" regression, but at D=256 B=1: the one-row route sends G=4/8 to `<4,8>` (4/32 or 8/32 columns real),
which matches phase0-cloud §5's 0.68-0.70× on D=256 G4/G8 decode.

## 4. Minimal upstream changes to reach tensor cores for G=6-like shapes

**A. Pad heads: ncols2 = 8 for 4 < G < 8 on Volta.** This copies the Turing rule, and it is what the fork's own
fallback does.
- *Dispatch:* `fattn.cu:224-246`: add `use_gqa_opt && gqa_ratio > 4 → switch_ncols1<…, 8>` (G=3 could similarly
  take 4).
- *Selection:* `fattn.cu:667-681`: derive Volta's ncols2 the same way, only when `gqa_opt_applies`. Choose VEC
  on real columns (B·G ≤ 2), and MMA once B·ncols2 ≥ 32, or a tuned threshold. Gating on `gqa_opt_applies` also
  fixes the NO_DEVICE_CODE bug; keep the `178` guard as well.
- *Kernel:* Q head indexing (`1917, 1300`), output writes (`1742`), tiling (`fattn-common.cuh:1097`) and the mask
  (shared per token, `j = jc/ncols2`, `748/814`) already handle phantom heads; zero Q yields finite, discarded
  results.
- *Must fix:* the sink lookup `sinks_f[jc % ncols2]` (`1434`) is unguarded. For phantom heads it reads the next
  group's sinks, or past the end of the array for the last KV head. That is out-of-bounds UB (*inferred*, and
  already latent on Turing+ for G=6 with sinks): guard it with −∞. Also add Volta D=256 32/64-column configs
  (`Q_in_reg=false`, K2/V2 64, as PR #27997 / the fork) or `<·,8>` D=256 will spill. Instances `<4,8>` and `<8,8>`
  already exist.
- *Risks:* 25% of tensor-core work and softmax is wasted on G=6. B ≤ 2 would be poor (19-38%), so keep VEC/TILE
  there. This changes behaviour for every Volta G=5-7 model at every D. At G=6, B=16, compare `<16,2>` (100%,
  3 passes) with `<8,8>`×2 (75%, 2 passes) by measurement.

- **B. ncols2 = 4 for G=6** (second head group half phantom): dispatch-only plus the sinks guard. Dominated by A.
- **C. Keep ncols2 = 2, allow MMA below 32 real columns** (`<16,2>` at B=4/8): selection-only; still 3 passes.
  Dominated.
- **D. G=4, B=4 on MMA `<8,4>`**: selection-only, 50% utilisation vs TILE's FMA-bound 100%; measure.
- **E (not minimal). Transposed operand roles** (mirror Q, stack K/V, as `verify_stream`): new `mma.cuh` tile
  types and rewritten KQ/softmax/VKQ/combine indexing, effectively a new kernel path; gives 8-column granularity.

## Implications for a design

Utilisation = real columns / computed columns (kernel), then passes. Upstream TILE cells are at 100% but use the
FP16 FMA pipe, not tensor cores.

| option | G4 B4 | G4 B8 | G4 B16 | G6 B4 | G6 B8 | G6 B16 | G8 B4 | G8 B8 | G8 B16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| upstream today | TILE 100/1 | 100/1 | 100/1 | TILE 100/3 | TILE 100/3 | 100/3 | 100/1 | 100/1 | 100/2 |
| fork `verify_stream` | — | — | — | 50/1 | 100/1 | 100/2 (L2 pair) | — | — | — |
| **A** (pad to 8) | as today | as today | as today | **75/1** | **75/1** | 75/2 | as today | as today | as today |
| A + D | 50/1 | 100/1 | 100/1 | 75/1 | 75/1 | 75/2 | 100/1 | 100/1 | 100/2 |
| B (ncols2=4) | as today | as today | as today | 37.5/2 | 75/2 | 75/2 | as today | as today | as today |
| C (`<16,2>`) | as today | as today | as today | 25/3 | 50/3 | 100/3 | as today | as today | as today |
| E (8-row tiles, ≤64 rows/CTA) | 100/1 | 100/1 | 100/1 | 100/1 | 100/1 | 100/2 | 100/1 | 100/1 | 100/2 |

Ranking:
1. **A**, together with the sinks guard, the Volta D=256 configs and the selection/NO_DEVICE_CODE fix. It is the
   smallest diff: G=6 reaches tensor cores at B ≥ 4, K/V is streamed once per Q tile instead of 3×, and G=4/8 keep
   today's 100% routing. *(inferred)* It should capture much of the fork's pp8/pp16 gain: the fork itself falls
   back to exactly this route, and the README credits the streaming kernel with a further 2.6 ms per round at 64K
   (`README-27B.md:235`).
2. **E**, if A leaves a gap at B ≤ 4 or against `verify_stream`. It is the structural fix for the 32-column floor.
   Size the row tile to ceil(B·G/8)·8 rather than the fork's fixed 48.
3. **D**, per measurement. **B and C**: dominated.

Design rule taken from the fork's failure mode: gate any new route on (D, G, B, KV length) explicitly, and never
widen a shape that upstream already runs at 100% utilisation (G=4/8).

## Open questions

- Do the TILE kernel's 3× G=6 re-reads hit L2 (ncu `lts__t_sector_hit_rate`, DRAM bytes)? This decides A's gain at
  B=4.
- At G=6, B=16, which is faster: `<16,2>` (3 passes) or `<8,8>`×2?
- Does `<8,8>` D=256 with `Q_in_reg=false` fit without spill (`bin/sm70-report.py`)? D=128 `<8,8>` uses the Ampere
  row unchanged: check it too.
- Llama-3.1-8B: run FLASH_ATTN_EXT perf (hsk=128, nh=8, nr23=[4,1], nb=8/16, kv 4K-32K) and MUL_MAT Q8_0 at n=8/16
  on both builds to confirm that the regression is not in FA.
- Precision: upstream Volta MMA accumulates VKQ in FP16 and the fork in FP32; how much error does this add at
  32K+ depth for padded G=6? The fork's own 3 FA test failures (D=256 G6 64K, batch 64/200/1024, phase0-cloud §5)
  come from non-`verify_stream` paths, because B > 16.
- What base does the fork really have (`6ba30d0` via Unsloth)? It is unresolved; the commit is absent from our
  clone and from the GitHub compare.
