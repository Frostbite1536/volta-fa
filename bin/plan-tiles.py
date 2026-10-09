#!/usr/bin/env python3
"""Planning calculator for fused-attention tile configs on V100 (sm_70). No GPU needed.

For a candidate (BR, BC, D, warps, buffering) it reports shared memory per block, the
occupancy limit from smem / registers / threads, and a roofline floor for one prefill
attention call. Everything is static arithmetic: real register counts come from
bin/sm70-report.py, real timing from the V100 box.

  python bin/plan-tiles.py --d 128 --br 64 --bc 64 --warps 4 --regs 128
  python bin/plan-tiles.py --d 128 --sweep          # table over common configs
"""
import argparse
import math

# Volta GV100 limits (CUDA C Programming Guide, compute capability 7.0 table;
# Volta tuning guide). V100 PCIe clocks; the SXM2 part boosts higher.
SM = dict(
    n_sm=80,
    regs_per_sm=65536,
    max_regs_per_thread=255,
    reg_alloc_unit=256,          # registers are allocated per warp in units of 256
    max_warps_per_sm=64,
    max_blocks_per_sm=32,
    smem_per_sm_max=96 * 1024,   # carveout options: 0, 8, 16, 32, 64, 96 KB
    smem_per_block_max=96 * 1024,  # > 48 KB needs cudaFuncAttributeMaxDynamicSharedMemorySize
    smem_carveouts=[0, 8, 16, 32, 64, 96],
)
PEAK = dict(
    tc_fp16_tflops_pcie=112.0,   # V100 PCIe 32GB datasheet, 1380 MHz boost
    hbm_gbs=900.0,
)


def smem_bytes(br, bc, dqk, dv, bufs, wmma, pad_halves):
    h = 2
    q = br * (dqk + pad_halves) * h
    k = bufs * bc * (dqk + pad_halves) * h
    v = bufs * bc * (dv + pad_halves) * h
    # WMMA fragments have an unspecified layout, so S (fp32) and P (fp16) go through smem
    # for the row-wise softmax. With m8n8k4 they can stay in registers.
    s = br * bc * 4 if wmma else 0
    p = br * (bc + pad_halves) * h if wmma else 0
    return dict(q=q, k=k, v=v, s=s, p=p, total=q + k + v + s + p)


def est_regs(br, bc, dv, warps, wmma):
    """Very rough: O accumulator + S tile in fp32 registers + fixed overhead."""
    threads = warps * 32
    o_acc = br * dv / threads
    s_tile = 0 if wmma else br * bc / threads
    return int(min(255, o_acc + s_tile + 48))


def occupancy(smem, regs, warps):
    threads = warps * 32
    regs_per_warp = math.ceil(regs * 32 / SM["reg_alloc_unit"]) * SM["reg_alloc_unit"]
    by_regs = SM["regs_per_sm"] // (regs_per_warp * warps) if regs else SM["max_blocks_per_sm"]
    by_smem = SM["smem_per_sm_max"] // smem if smem else SM["max_blocks_per_sm"]
    by_warps = SM["max_warps_per_sm"] // warps
    blocks = max(0, min(by_regs, by_smem, by_warps, SM["max_blocks_per_sm"]))
    limiter = min(("regs", by_regs), ("smem", by_smem), ("warps", by_warps), key=lambda t: t[1])[0]
    return dict(blocks=blocks, warps=blocks * warps, occ=blocks * warps / SM["max_warps_per_sm"],
                limiter=limiter, threads=threads)


def roofline(sq, skv, dqk, dv, heads, br, causal):
    frac = 0.5 if causal else 1.0
    flops = 2 * sq * skv * (dqk + dv) * heads * frac
    qo_bytes = sq * (dqk + dv) * 2 * heads
    # DRAM traffic is bracketed: worst case re-reads K/V from DRAM for every Q block (no L2
    # reuse); best case reads each head's K/V once (perfect L2 reuse across Q blocks of a head).
    worst = math.ceil(sq / br) * skv * (dqk + dv) * 2 * heads * frac + qo_bytes
    best = skv * (dqk + dv) * 2 * heads + qo_bytes
    t_c = flops / (PEAK["tc_fp16_tflops_pcie"] * 1e12) * 1e3
    t_worst = worst / (PEAK["hbm_gbs"] * 1e9) * 1e3
    t_best = best / (PEAK["hbm_gbs"] * 1e9) * 1e3
    return dict(flops=flops, worst=worst, best=best, t_c=t_c, t_worst=t_worst, t_best=t_best)


def report(a, br, bc, warps, regs=None, quiet=False):
    s = smem_bytes(br, bc, a.d, a.dv or a.d, a.bufs, a.wmma, a.pad)
    r = regs if regs is not None else est_regs(br, bc, a.dv or a.d, warps, a.wmma)
    o = occupancy(s["total"], r, warps)
    rf = roofline(a.sq, a.skv, a.d, a.dv or a.d, a.heads, br, a.causal)
    ok = s["total"] <= SM["smem_per_block_max"] and r <= SM["max_regs_per_thread"] and o["blocks"] > 0
    grid = math.ceil(a.sq / br) * a.heads
    waves = grid / (o["blocks"] * SM["n_sm"]) if o["blocks"] else float("inf")
    if quiet:
        return (br, bc, warps, s["total"], r, o["blocks"], o["occ"], o["limiter"], waves, ok)
    print(f"config  BR={br} BC={bc} D={a.d} Dv={a.dv or a.d} warps={warps} bufs={a.bufs} "
          f"{'wmma(S,P via smem)' if a.wmma else 'm8n8k4(S,P in regs)'} pad={a.pad} halves")
    print(f"smem    Q {s['q']}  K {s['k']}  V {s['v']}  S {s['s']}  P {s['p']}  = {s['total']} B "
          f"({s['total']/1024:.1f} KB){'  > 48 KB: needs opt-in attribute' if s['total'] > 48*1024 else ''}")
    print(f"regs    {r}/thread {'(given)' if regs is not None else '(ROUGH estimate; use sm70-report.py)'}")
    print(f"occ     {o['blocks']} blocks/SM, {o['warps']}/64 warps = {o['occ']:.0%} (limiter: {o['limiter']})")
    print(f"grid    {grid} blocks over {SM['n_sm']} SMs = {waves:.2f} waves (tail effect if fractional and small)")
    print(f"roof    sq={a.sq} skv={a.skv} heads={a.heads} causal={a.causal}: {rf['flops']/1e9:.1f} GFLOP")
    print(f"        tensor-core floor {rf['t_c']:.3f} ms at {PEAK['tc_fp16_tflops_pcie']:.0f} TFLOPS; DRAM floor "
          f"{rf['t_best']:.3f} ms (perfect L2 reuse, {rf['best']/1e6:.0f} MB) .. {rf['t_worst']:.3f} ms "
          f"(no reuse, {rf['worst']/1e6:.0f} MB) at {PEAK['hbm_gbs']:.0f} GB/s")
    print(f"        -> {'compute-bound unless L2 reuse collapses' if rf['t_c'] > rf['t_best'] else 'memory-bound'}; "
          f"real kernels on V100 reach a fraction of the TC peak, so treat {rf['t_c']:.3f} ms as unreachable")
    print("valid" if ok else "INVALID: exceeds a hardware limit")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=128, help="head dim of Q/K")
    ap.add_argument("--dv", type=int, default=0, help="head dim of V (default = d)")
    ap.add_argument("--br", type=int, default=64)
    ap.add_argument("--bc", type=int, default=64)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--regs", type=int, help="registers/thread from sm70-report.py")
    ap.add_argument("--bufs", type=int, default=1, help="K/V smem buffers (2 = double buffering)")
    ap.add_argument("--pad", type=int, default=8, help="row padding in halves (bank-conflict avoidance)")
    ap.add_argument("--wmma", action="store_true", help="model WMMA (S/P through smem) instead of m8n8k4")
    ap.add_argument("--sq", type=int, default=8192)
    ap.add_argument("--skv", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=32, help="q heads x batch")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if not a.sweep:
        report(a, a.br, a.bc, a.warps, a.regs)
        return
    print(f"sweep D={a.d} bufs={a.bufs} {'wmma' if a.wmma else 'm8n8k4'}; regs are ROUGH estimates")
    print(f"{'BR':>4} {'BC':>4} {'warps':>5} {'smem':>7} {'regs':>4} {'blk/SM':>6} {'occ':>5} {'limit':>6} {'waves':>6}")
    for br in (16, 32, 64, 128):
        for bc in (32, 64, 128):
            for warps in (2, 4, 8):
                if br < 16 * warps // 4:  # each warp needs >= one 16-row slab... rough feasibility
                    continue
                row = report(a, br, bc, warps, quiet=True)
                if not row[-1]:
                    continue
                print(f"{row[0]:>4} {row[1]:>4} {row[2]:>5} {row[3]:>7} {row[4]:>4} {row[5]:>6} {row[6]:>5.0%} {row[7]:>6} {row[8]:>6.2f}")


if __name__ == "__main__":
    main()
