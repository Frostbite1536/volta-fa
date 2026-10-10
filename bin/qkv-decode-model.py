#!/usr/bin/env python3
"""Cost model for single-token decode attention with a quantised (or BF16) KV cache on a V100.

Upstream llama.cpp sends Volta decode with GQA >= 4 to the TILE kernel, which first converts the whole
K and V cache views to F16 (convert.cu dequantize_block_cuda / convert_unary_cuda). That launcher puts
256 threads along a row and one block per (token, KV head) row, so a 64-256 value row leaves most
threads idle and the conversion is bound by the number of blocks, not by bytes.

  python bin/qkv-decode-model.py            # check against the measured rows, then model shapes
  python bin/qkv-decode-model.py --d 128 --kv-heads 8 --gqa 4 --depth 16384 --type q8_0

Calibration (cloud V100-PCIE-32GB, docs/research/phase0-cloud-2026-10-10.md §3, fa-perf.sql):
  F16 decode streams K/V at ~800 GB/s at >= 16K (measured 720-855 GB/s, TILE and VEC).
  Conversion: ~1.05 us per "wave" of 640 blocks (80 SMs x 8 resident 256-thread blocks) + ~8 us fixed.
  Fitted to 7 rows (q8_0/q4_0/bf16, D 64/256, KV 4K-32K): see the check table this prints.
  Generic conversion body: 430-590 GB/s read+write (upstream rows at D=256, 32K, after the fixed cost).
Not measured (model only): patch 0005/0006 conversion times on a V100 (bench/micro/dequant_nc.cu
measures them; 0006 is assumed to stream at 800 GB/s as it reached copy bandwidth on a Maxwell GPU),
and VEC with a quantised cache at GQA >= 4 on Volta. VEC reads the quantised K/V
once per Q head; how much of that L2 absorbs is unknown, so VEC is shown as a range
[1 read (perfect L2 reuse), GQA reads (none)].
"""
import argparse
import math

BW = 800e9              # B/s, F16 decode streaming at >= 16K (measured)
WAVE_US = 1.05          # us per wave of conversion blocks (fitted)
FIXED_US = 8.0          # us per FA call for the two conversion launches (fitted)
BW_GENERIC = 500e9      # B/s read+write, generic conversion body: measured 430-590 GB/s upstream at D=256, 32K
SM, BLOCKS_PER_SM, BLOCK = 80, 8, 256
BYTES = {"f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}


def kv_bytes(t, n_kv, d, nh):
    return 2 * n_kv * d * nh * BYTES[t]          # K + V


def conv_waves(t, n_kv, d, nh, patched):
    """Blocks launched for converting K and V (2 tensors) / blocks resident per wave."""
    per_thread = 1 if t == "bf16" else 2          # convert_unary: 1 value/thread; dequantize_block: 2
    if not patched:                               # upstream: 256 threads along the row, one block per row
        rows_per_block = 1
        x_blocks = math.ceil(d / (per_thread * BLOCK))
    else:                                         # patch 0005: nth_x = smallest warp multiple covering the row
        nth_x = 32
        while nth_x < BLOCK and per_thread * nth_x < d:
            nth_x *= 2
        rows_per_block = BLOCK // nth_x
        x_blocks = math.ceil(d / (per_thread * nth_x))
    blocks = 2 * x_blocks * math.ceil(n_kv * nh / rows_per_block)
    return blocks / (SM * BLOCKS_PER_SM)


def t_conv_upstream(t, n_kv, d, nh):
    return FIXED_US + WAVE_US * conv_waves(t, n_kv, d, nh, False)


def t_conv_0005(t, n_kv, d, nh):
    """Fewer blocks, same body: bound by the waves or by the generic body's measured rate."""
    bytes_rw = kv_bytes(t, n_kv, d, nh) + kv_bytes("f16", n_kv, d, nh)
    return FIXED_US + max(WAVE_US * conv_waves(t, n_kv, d, nh, True), bytes_rw / BW_GENERIC * 1e6)


def t_conv_0006(t, n_kv, d, nh):
    """q4_0/q8_0 only: 8 values/thread, assumed to run at the streaming bandwidth (as on the M2200)."""
    if t == "bf16":
        return t_conv_0005(t, n_kv, d, nh)
    bytes_rw = kv_bytes(t, n_kv, d, nh) + kv_bytes("f16", n_kv, d, nh)
    return FIXED_US + bytes_rw / BW * 1e6


def t_f16(n_kv, d, nh):
    return kv_bytes("f16", n_kv, d, nh) / BW * 1e6


def t_vec(t, n_kv, d, nh, gqa):
    one = kv_bytes(t, n_kv, d, nh) / BW * 1e6
    return one, one * gqa


# (D, KV heads, GQA, KV length, type, measured us, measured f16 us) from fa-perf.sql, batch 1
MEASURED = [
    (64, 8, 8, 7680, "q8_0", 237.6, 33.2), (64, 8, 8, 7680, "q4_0", 237.1, 33.2),
    (256, 2, 12, 4096, "q8_0", 66.9, 31.5), (256, 2, 12, 16384, "q8_0", 194.1, 80.0),
    (256, 2, 12, 16384, "q4_0", 190.0, 80.0), (256, 2, 12, 16384, "bf16", 204.4, 80.0),
    (256, 2, 12, 32768, "q8_0", 370.6, 155.5), (256, 2, 12, 32768, "q4_0", 364.0, 155.5),
    (256, 2, 12, 32768, "bf16", 391.8, 155.5),
]


def check():
    print("Check: upstream = measured F16 time + modelled conversion (us)")
    print(f"{'case':<34} {'measured':>9} {'model':>7} {'error':>6}")
    for d, nh, g, n, t, meas, f16 in MEASURED:
        model = f16 + t_conv_upstream(t, n, d, nh)
        print(f"D={d:<3} kv_heads={nh} GQA={g:<2} KV={n:<6} {t:<5} {meas:9.1f} {model:7.1f} {100*(model-meas)/meas:+5.0f}%")


def table(name, d, nh, gqa, depths, types):
    print(f"\n{name}: D={d}, {nh} KV heads x GQA {gqa}, one decode token, per layer (us)")
    print(f"{'depth':>6} {'type':<5} | {'f16 ref':>7} | {'upstream':>8} | {'+0005':>7} | {'+0006':>7} | {'0004 VEC range':>15}")
    for n in depths:
        f16 = t_f16(n, d, nh)
        for t in types:
            up = f16 + t_conv_upstream(t, n, d, nh)
            c5 = f16 + t_conv_0005(t, n, d, nh)
            c6 = f16 + t_conv_0006(t, n, d, nh)
            lo, hi = t_vec(t, n, d, nh, gqa)
            print(f"{n:>6} {t:<5} | {f16:7.0f} | {up:8.0f} | {c5:7.0f} | {c6:7.0f} | {lo:6.0f} - {hi:6.0f}")
    print("  +0005/+0006: TILE kept, conversion faster (these also apply to 2-16 token batches and prefill).")
    print("  0004 VEC: no conversion; low = K/V read once (L2 reuse across the GQA heads), high = once per Q head.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--gqa", type=int)
    ap.add_argument("--depth", type=int, default=16384)
    ap.add_argument("--type", default="q8_0", choices=sorted(BYTES))
    ap.add_argument("--layers", type=int, default=0, help="also print the per-token total for this many layers")
    a = ap.parse_args()
    if a.d:
        table("custom", a.d, a.kv_heads, a.gqa, [a.depth], [a.type])
        if a.layers:
            extra = a.layers * t_conv_upstream(a.type, a.depth, a.d, a.kv_heads) / 1e3
            print(f"  upstream conversion over {a.layers} layers: +{extra:.1f} ms per token")
        return
    check()
    table("Llama-3.1-8B-like", 128, 8, 4, (4096, 16384, 32768), ("q8_0", "q4_0"))
    table("Qwen3.8-27B-like (per full-attention layer)", 256, 4, 6, (4096, 32768), ("q8_0", "q4_0"))
    extra = 32 * t_conv_upstream("q8_0", 16384, 128, 8) / 1e3
    print(f"\nEnd-to-end check, Llama-3.1-8B q8_0 KV at 16K (32 layers): model +{extra:.1f} ms/token of conversion;"
          f"\nmeasured tg128 f16 43.9 t/s -> q8_0 28.1 t/s = +{1e3/28.1 - 1e3/43.9:.1f} ms/token (phase0-cloud §6).")


if __name__ == "__main__":
    main()
