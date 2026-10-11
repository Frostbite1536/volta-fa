# Ticket p9-qload: Phase 0 microbenchmark, native vs repacked quantised-weight streaming on Volta

Your current directory is a git worktree of volta-fa (branch `p9-qload`, based on `main` @ `8802adb`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Volta sm_70 ONLY, CUDA 12.x. Tensor cores via `nvcuda::wmma` (16x16x16 half) or PTX `mma.sync.aligned.m8n8k4`. Do NOT use BF16, FP8, cp.async, ldmatrix, mma m16n8k*, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout. Start every shell command with `export PATH=/home/p51/Github/volta-fa/.venv/bin:/home/p51/Github/volta-fa/tools/cuda-12.8/bin:$PATH;` (cmake, python, and a real nvcc 12.8). Never install anything.
- For every kernel file run `python /home/p51/Github/volta-fa/bin/sm70-report.py <file.cu>` (or `nvcc -arch=sm_70 -Xptxas -v`) and paste the registers/spill/stack per kernel. Spill must be 0.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Never invent benchmark numbers. This machine has a Quadro M2200 (sm_52), not a V100: V100 numbers are "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
P-9 (`docs/design/P9-small-batch-qmatmul.md` in the main checkout, read it: sections "Floor and budget" and D3) is a
Volta tensor-core matmul for quantised weights at 1-16 tokens. The open decision D3 is the weight layout: (a) the native
GGUF layout, where lane l of a warp streams row l of a 32-row tile (rows are `row_pitch` bytes apart), or (b) a
"fragment order" repack where a warp's 32 lanes read 512 contiguous bytes (lane l's 16 bytes belong to row l). This
benchmark measures, on the V100 later, how fast each layout can be streamed, load-only and with the decode + tensor-core
work of the real kernel, so the decision rests on data. It replicates no llama.cpp code.

## Design (manager decided, do not change)
One standalone file `bench/micro/qload.cu`, built like `bench/micro/dequant_nc.cu`
(`nvcc -O3 -std=c++17 -arch=sm_70 bench/micro/qload.cu -o build/qload && build/qload [iters]`). Every CUDA call checked
(a `CK()` macro as in dequant_nc.cu). Header comment: purpose, build line, what each variant means.

Types (block bytes / weights per block; static_assert the struct sizes): Q4_K 144/256, Q5_K 176/256, Q6_K 210/256,
IQ4_XS 136/256, Q8_0 34/32. Only the byte size matters for loads; fill buffers with random bytes (fixed seed).

Shapes (N rows x K columns): for Q4_K, Q5_K, Q6_K, IQ4_XS: (17408, 5120), (5120, 17408), (10240, 5120), (5120, 6144);
for Q8_0: (14336, 4096), (4096, 14336), (4096, 4096). row_pitch = K / weights_per_block * block_bytes. N is a multiple
of 32 in all shapes; a 32-row tile is rows 32t..32t+31.

Load width w for the native layout = the largest of 16, 8, 4, 2 that divides row_pitch (rows are then w-aligned;
cudaMalloc is 256-B aligned). Print w per (type, shape).

Variants (each a kernel; grid = (N/32) * P blocks, block = 32 * S threads; warp s of block p of tile t streams the
contiguous segment number p*S + s of its rows' bytes, segments = P*S equal parts of row_pitch rounded to the load unit,
the last one shorter):
1. `copy`: cudaMemcpyAsync device-to-device of the native buffer (reference; it also writes, report bytes/time anyway).
2. `stream`: ideal contiguous read, each thread `ld.global.nc.v4.u32` grid-stride over the whole native buffer, XOR into
   a register, one store per thread.
3. `native`: lane l reads row 32t+l of tile t, its segment, with loads of width w using `ld.global.nc` of the matching
   width (v4.u32 / v2.u32 / u32 / u16); XOR all loaded 32-bit words (u16 zero-extended) into a per-lane register.
4. `repacked`: a host-built copy where each tile's rows are interleaved in 16-byte units: tile chunk c holds 32 x 16 B,
   lane l's 16 B = bytes [16c, 16c+16) of row l; rows whose row_pitch is not a multiple of 16 are zero-padded to the next
   multiple of 16 in this copy (report the padded size). Lane l reads `ld.global.nc.v4.u32` at (c*32 + l)*16 within the
   tile, over its warp's segment of chunks; XOR the same way.
5. `native_mma` and 6. `repacked_mma`, **Q4_K and Q5_K only**: as 3 and 4 with w = 16, plus the arithmetic of the real
   kernel on every loaded 16-byte unit (4 words, 32 4-bit codes): for each word x, the two half2 values
   h_lo = (x & 0x000F000F) | 0x64006400 and h_hi = ((x >> 4) & 0x000F000F) | 0x64006400, reinterpret as half2, then
   `__hfma2(h, s, z)` with s, z half2 constants passed as kernel arguments (so it cannot be folded); that gives 8 half2
   per 16 B. Feed them as A fragments to `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` via inline PTX:
   `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};`
   (A = 2 x .b32 registers = 4 halves, B = 2 x .b32 registers held constant from a kernel argument, C = D = 8 fp32
   accumulators per lane). 8 half2 per 16 B = 4 mma instructions per 16-byte unit per lane. At the end fold the 8
   accumulators into the checksum output (sum as float, written to a float output array) so nothing is dead code.
   Correctness of the arithmetic is not the point; it must not be optimised away (check SASS contains HMMA and HFMA2:
   `cuobjdump -sass` on the build, grep counts, paste them).

Correctness check (runs every time, before timing): the per-row XOR of `native` and of `repacked` (combine the per-lane
partial XORs of all segments on the host) must equal a CPU XOR of each row's bytes as 32-bit words (u16 tails
zero-extended the same way; padding zeros do not change XOR). Print PASS/FAIL per (type, shape); on FAIL skip its timing
and say so. For the `_mma` variants check only that the float output is finite or NaN-free is NOT required (random
bytes); just print the float sum.

Sweep S in {4, 8} and P in {1, 2, 4} for native, repacked and the _mma variants. Timing: 3 warm-up launches, then
`iters` (default 20) launches each timed with cudaEvents, median reported. Matrix sizes exceed the 6 MB L2, so no flush.

Output: one Markdown table with columns type | N | K | MB (native bytes) | w | variant | S | P | median µs | GB/s
(native bytes / time, also for repacked so the numbers compare), then a summary table per (type, shape): best GB/s of
stream, native, repacked, native_mma, repacked_mma and the ratios native/repacked and native_mma/repacked_mma.

Local testing on the laptop GPU (sm_52, no fp16 arithmetic, no mma): support `-DQLOAD_NO_MMA` which compiles out the
_mma variants; build `nvcc -O3 -std=c++17 -arch=sm_52 -DQLOAD_NO_MMA ...` and run it with a small iters value to check
the PASS lines and that the tables print. Also build for sm_70 without the define (compile only) for the static report.

## Files you may touch
- `bench/micro/qload.cu`: new.
Nothing else.

## Must pass before you finish
- `nvcc -O3 -std=c++17 -arch=sm_70 bench/micro/qload.cu -o build/qload` (compiles, 0 spill per kernel in -Xptxas -v)
- `nvcc -O3 -std=c++17 -arch=sm_52 -DQLOAD_NO_MMA bench/micro/qload.cu -o build/qload52 && build/qload52 3` (all PASS)
- `cuobjdump -sass build/qload | grep -c HMMA` and `... | grep -c HFMA2` (both > 0)
- `/home/p51/Github/volta-fa/bin/check-forbidden.sh`

## Done means
The file exists, compiles for sm_70 with 0 spills, the sm_52 build runs and prints PASS for every (type, shape).
Final report: files changed; each command with its result line; the -Xptxas -v lines per kernel; the SASS counts; the
sm_52 output (it is a Maxwell GPU: its GB/s are not V100 numbers, label them so); what was NOT verified.
