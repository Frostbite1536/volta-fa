#!/usr/bin/env python3
"""Cost model for small-batch (1-16 token) attention at long context on a V100.

Per layer, B query tokens attend over L cached tokens. Each KV head's K and V must be read from
HBM; every query row that shares the KV head (B x GQA rows) reuses them. A kernel strategy
decides (a) how many times K/V are re-read and (b) how much of the tensor-core tile is padding.

  python bin/smallbatch-model.py                   # Qwen3.8-27B-like and Llama-3.1-8B-like tables
  python bin/smallbatch-model.py --d 256 --kv-heads 4 --gqa 6 --depth 32768

Calibration (measured on the cloud V100-PCIE-32GB, docs/research/phase0-cloud-2026-10-10.md):
  HBM ~740 GB/s achieved (D=256 GQA 8 decode at 16K reads 134 MB in 181 us);
  tensor-core FA ~45 TFLOPS achieved (D=128 GQA 8 prefill 43, D=256 pr_k64_c64 47);
  CUDA-core FP16 (TILE) assumed ~20 TFLOPS achieved (peak ~28). The TILE figure is NOT measured:
  treat TILE rows as a bound, and replace it once ncu or a TILE microbenchmark gives a number.
These are floors (max of compute and memory time); real kernels add launch/softmax/combine cost.
"""
import argparse
import math

BW = 740e9          # bytes/s, measured
TC = 45e12          # FLOP/s, measured (MMA FA kernels)
CC = 20e12          # FLOP/s, assumed (TILE, CUDA-core FP16)
COLS_MIN, COLS_MAX = 32, 64   # Volta MMA instances: ncols1*ncols2 must be >= 32 (sm_70 guard); max tile 64


def upstream_ncols2(gqa, gqa_opt=True):
    """Volta dispatch (fattn.cu Volta branch): largest of 8/4/2 dividing GQA, else 1."""
    if not gqa_opt:
        return 1
    for c in (8, 4, 2):
        if gqa % c == 0:
            return c
    return 1


def mma_cost(B, gqa, ncols2, L, D, kv_heads):
    """MMA strategy with a given ncols2 (heads packed per block). Returns (time, reads, util)."""
    groups_per_kv = math.ceil(gqa / ncols2)            # blocks needed to cover one KV head's Q heads
    cols_needed = B * ncols2
    tiles = max(1, math.ceil(cols_needed / COLS_MAX))  # Q-batch splits per head group
    ncols = min(COLS_MAX, max(COLS_MIN, 2 ** math.ceil(math.log2(max(1, math.ceil(cols_needed / tiles))))))
    reads = groups_per_kv * tiles                      # times each KV head's K/V is streamed
    useful = B * gqa                                   # real query rows per KV head
    issued = reads * ncols                             # tile columns computed per KV head
    util = useful / issued
    kv_bytes = 2 * L * D * 2 * kv_heads                # K + V, fp16
    flops_issued = 2 * 2 * issued * L * D * kv_heads   # QK^T and PV
    t = max(flops_issued / TC, reads * kv_bytes / BW)
    return t, reads, util, ncols


def tile_cost(B, gqa, L, D, kv_heads):
    """TILE strategy (CUDA cores): assume K/V streamed once per KV head (optimistic)."""
    kv_bytes = 2 * L * D * 2 * kv_heads
    flops = 2 * 2 * B * gqa * L * D * kv_heads
    return max(flops / CC, kv_bytes / BW)


def table(name, D, kv_heads, gqa, depth):
    print(f"\n{name}: D={D}, {kv_heads} KV heads x GQA {gqa}, depth {depth}  (per layer, floors in us)")
    print(f"{'B':>3} {'B*GQA':>6} | {'TILE':>7} | {'MMA upstream':>22} | {'MMA pad pow2':>22} | {'MMA exact GQA':>22}")
    up = upstream_ncols2(gqa)
    pad = min(8, 2 ** math.ceil(math.log2(gqa)))
    for B in (1, 2, 4, 8, 16):
        t_tile = tile_cost(B, gqa, depth, D, kv_heads)
        cells = []
        for nc2 in (up, pad, gqa):
            t, reads, util, ncols = mma_cost(B, gqa, nc2, depth, D, kv_heads)
            cells.append(f"{t*1e6:7.0f} r{reads} u{util:4.0%} n{ncols:<2}")
        print(f"{B:>3} {B*gqa:>6} | {t_tile*1e6:7.0f} | " + " | ".join(f"{c:>22}" for c in cells))
    print("  rN = K/V reads per KV head, uN = useful fraction of tensor-core columns, nN = ncols per block")
    print(f"  upstream ncols2={up}; pad pow2 ncols2={pad}; exact ncols2={gqa} (needs a non-power-of-two or masked instance)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--gqa", type=int)
    ap.add_argument("--depth", type=int, default=32768)
    a = ap.parse_args()
    if a.d:
        table("custom", a.d, a.kv_heads, a.gqa, a.depth)
        return
    for depth in (4096, 32768):
        table("Qwen3.8-27B-like", 256, 4, 6, depth)
        table("Llama-3.1-8B-like", 128, 8, 4, depth)


if __name__ == "__main__":
    main()
