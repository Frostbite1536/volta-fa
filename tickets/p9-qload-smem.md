# Ticket p9-qload-smem: add a shared-memory-staged native-layout variant to bench/micro/qload.cu

Your current directory is a git worktree of volta-fa (branch `p9-qload`). `bench/micro/qload.cu` exists (uncommitted,
from ticket p9-qload, then edited by the manager: `decode_mma` now decodes all 8 codes of a word, 2 mma per word).

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Volta sm_70 ONLY, CUDA 12.x. Tensor cores via PTX `mma.sync.aligned.m8n8k4` only. Do NOT use BF16, FP8, cp.async, ldmatrix, mma m16n8k*, TMA, clusters. If you think you need one, STOP and explain.
- Start every shell command with `export PATH=/home/p51/Github/volta-fa/.venv/bin:/home/p51/Github/volta-fa/tools/cuda-12.8/bin:$PATH;`. Never install anything.
- Spill must be 0 for every kernel (`nvcc -arch=sm_70 -Xptxas -v`); report registers, spill, stack and smem per kernel.
- Do NOT commit or run git commands that write. Never invent benchmark numbers (no V100 here; your sandbox cannot see the laptop GPU either: say "not run").
- Stay inside "Files you may touch".

## Background
Measured on a V100: Q4_K/Q5_K weights streamed in the native GGUF layout with lane l reading row l (one 16-byte load per
row per instruction) reach only 0.29-0.62x of the repacked fragment-order layout once decode + mma are added. The
question now: can the native layout reach the repacked rate if each load instruction reads full 128-byte lines of a row
and a shared-memory transpose puts the bytes into fragment order (lane l gets row l's 16-byte units)? If yes, P-9 needs
no weight repacking (no ggml plumbing).

## Design (manager decided, do not change)
Two new variants, both for Q4_K and Q5_K only, same grid/segment scheme as `native_mma` (grid = (N/32)*P, block = 32*S,
warp s of block p streams segment p*S+s of the tile's rows; segments are in 16-byte units; handle a short last segment):
1. `native_smem` (load + transpose only, XOR checksum like `native`, must PASS the same per-row XOR check) and
2. `native_smem_mma` (same staging, then exactly `layout_mma`'s per-16-byte work: 4 words, `decode_mma` each).

Staging step (per warp, private shared-memory region, no __syncthreads, use __syncwarp):
- A step covers U = 8 consecutive 16-byte units (128 bytes) of each of the 32 rows of the tile: 32 x 128 B = 4 KiB.
- Load phase: 8 load instructions per step; in instruction i (0..7), lane l reads row 4*i + l/8, unit (l % 8) of the
  step, with `ld.global.nc.v4.u32` (so every instruction reads 4 rows x 128 contiguous bytes = 4 full lines).
  Store each 16 B to shared memory at [row][unit] with a padded row stride of 8 units + 1 unit (144 B) so that the read
  phase below is bank-conflict free; state in a comment why the padding avoids conflicts.
- __syncwarp(), then read phase: lane l reads its row l's 8 units [l][0..7] from shared memory (8 x 16-byte LDS.128) and
  XORs them (native_smem) or runs decode_mma on each unit's 4 words (native_smem_mma). __syncwarp() before the next step
  overwrites the buffer.
- Optional software pipelining is NOT wanted in this version (keep it simple and measurable); note it as a follow-up.
- Rows of Q4_K/Q5_K have row_pitch a multiple of 16; a segment may not be a multiple of 8 units: the tail step loads only
  the remaining units (guard loads and the read loop).
Shared memory per block = S x 32 rows x 144 B (S=8: 36 KiB, under the 48 KiB default; no opt-in needed).

Add both variants to the S x P sweep, the results table, and the summary table (columns native_smem, native_smem_mma and
ratios native_smem/repacked and native_smem_mma/repacked_mma). Keep everything else unchanged.

## Files you may touch
- `bench/micro/qload.cu`
Nothing else.

## Must pass before you finish
- `nvcc -O3 -std=c++17 -arch=sm_70 -Xptxas -v bench/micro/qload.cu -o build/qload` (0 spill for every kernel; paste lines)
- `nvcc -O3 -std=c++17 -arch=sm_52 -DQLOAD_NO_MMA bench/micro/qload.cu -o build/qload52` (compiles; native_smem must be
  available without MMA so the manager can run the XOR check on the laptop GPU)
- `cuobjdump -sass build/qload | grep -c LDS` (> 0) and HMMA count
- `/home/p51/Github/volta-fa/bin/check-forbidden.sh`

## Done means
Both variants compiled, 0 spill, included in the tables. Final report: diff summary, each command with its result line,
the -Xptxas -v lines for the new kernels, what was NOT verified.
