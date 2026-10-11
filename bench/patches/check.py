#!/usr/bin/env python3
"""Check bench/patches/run.sh results against the P-6 and P-8 acceptance criteria and decision gates.

  python3 bench/patches/check.py bench/results/patches-<kernel-run> [bench/results/patches-<models-run> ...]

Reads, from any of the given directories: run.log (correctness lines), <variant>.perf.r<i>.sql (kernel
rounds), dequant_nc.txt (conversion microbench), <variant>.<model>.<tag>.r<i>.jsonl (llama-bench rounds).
Prints a Markdown report: one row per criterion with the measured value, the threshold and
PASS / FAIL / NOT RUN. Criteria come from docs/design/P6-small-batch-gqa.md and
docs/design/P8-quantised-kv-decode.md ("Accept if"); a FAIL here is a prompt to look at the data, not a
substitute for reading it.
"""
import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import analyze  # noqa: E402  (bench/analyze.py: SQL and llama-bench loaders)

NOISE = 0.03  # relative difference treated as noise when there is only one round


def load_kernel(dirs):
    files = {}
    for d in dirs:
        for f in sorted(d.glob("*.perf.r*.sql")):
            files.setdefault(f.name.split(".perf.")[0], []).append(str(f))
    return {v: analyze.load_tbo(",".join(fs)) for v, fs in files.items()}


def find(data, v, hs, gqa, kv, nb, t):
    """Median/min/max of the harness perf case (nh=4, mask, no sinks/softcap) or None."""
    want = {"hsk": str(hs), "nh": "4", "nr23": f"[{gqa},1]", "kv": str(kv), "nb": str(nb), "type_K": t,
            "type_V": t, "sinks": "0", "logit_softcap": "0.000000", "max_bias": "0.000000", "n_kv_max": "0",
            "permute": "[0,1,2,3]"}
    for key, r in data.get(v, {}).items():
        p = dict(key[1])
        if key[0] == "FLASH_ATTN_EXT" and all(p.get(k) == val for k, val in want.items()) and r[0] and r[1]:
            return r[1], r[2], r[3], r[4]
    return None


def speedup(data, base, new, *case):
    a, b = find(data, base, *case), find(data, new, *case)
    if not a or not b:
        return None
    noise = (b[1] <= a[2] and a[1] <= b[2]) if (a[3] > 1 and b[3] > 1) else abs(a[0] / b[0] - 1) <= NOISE
    return a[0] / b[0], noise, a, b


def correctness(dirs):
    out = []
    for d in dirs:
        log = d / "run.log"
        if not log.exists():
            continue
        cur = None
        for line in log.read_text(errors="replace").splitlines():
            m = re.search(r"correctness (\S+) \((full FLASH_ATTN_EXT|MUL_MAT)", line)
            if m:
                cur = (m.group(1), "FA" if m.group(2).startswith("full") else "MUL_MAT")
                continue
            m = re.search(r"rc=(\d+)\s+OK=(\d+)\s+FAIL=(\d+)", line)
            if m and cur:
                out.append((cur[0], cur[1], int(m.group(1)), int(m.group(2)), int(m.group(3))))
                cur = None
    return out


def microbench(dirs):
    rows = []
    for d in dirs:
        f = d / "dequant_nc.txt"
        if f.exists():
            for line in f.read_text(errors="replace").splitlines():
                m = re.search(r"^(q\d_\d)\s+(.*?)\s+D=.*?\|\s*0005 rows/block\s+[\d.]+ us\s+([\d.]+)x\s*\|\s*0006 vec8\s+[\d.]+ us\s+"
                              r"([\d.]+)x.*\|\s*(.*)$", line)
                if m:
                    rows.append((m.group(1), m.group(2), float(m.group(3)), float(m.group(4)), m.group(5).strip()))
    return rows


def load_lb(dirs, tag, model_pat):
    files = {}
    for d in dirs:
        for f in sorted(d.glob(f"*.{tag}.r*.jsonl")):
            if f.stat().st_size == 0:
                continue
            v, model = f.name.split(".")[0], ".".join(f.name.split(".")[1:-3])
            if re.search(model_pat, model, re.I):
                files.setdefault(v, []).append(str(f))
    return {v: analyze.load_lb(",".join(fs)) for v, fs in files.items()}


def lb_find(data, v, n_prompt, n_gen, depth):
    for key, r in data.get(v, {}).items():
        k = dict(zip(analyze.LB_KEYS, key))
        if k["n_prompt"] == str(n_prompt) and k["n_gen"] == str(n_gen) and k["n_depth"] == str(depth):
            return r
    return None


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, measured, threshold, verdict):
        self.rows.append((name, measured, threshold, verdict))

    def ratio(self, name, s, threshold, op=">="):
        if s is None:
            self.add(name, "–", f"{op} {threshold}×", "NOT RUN")
            return
        r, noise, a, b = s
        ok = r >= threshold if op == ">=" else (r <= threshold)
        self.add(name, f"{r:.2f}× ({analyze.fmt_us(a[0])} → {analyze.fmt_us(b[0])})", f"{op} {threshold}×",
                 "PASS" if ok else "FAIL")

    def print(self):
        print("| criterion | measured | threshold | verdict |")
        print("|---|---|---|---|")
        for r in self.rows:
            print("| " + " | ".join(r) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    dirs = ap.parse_args().dirs
    k = load_kernel(dirs)
    rep = Report()

    # --- correctness
    corr = correctness(dirs)
    for v, suite, rc, ok, fail in corr:
        if v == "upstream" and suite == "FA":
            rep.add("upstream FA suite (expected: aborts on the Volta crash)", f"rc={rc} OK={ok} FAIL={fail}", "abort",
                    "as expected" if rc != 0 else "UNEXPECTED (no abort)")
        else:
            rep.add(f"{v} {suite} correctness", f"rc={rc} OK={ok} FAIL={fail}", "rc=0, FAIL=0",
                    "PASS" if rc == 0 and fail == 0 and ok > 0 else "FAIL")
    if not corr:
        rep.add("correctness", "–", "rc=0, FAIL=0", "NOT RUN")

    # --- P-6
    for nb in (8, 16):
        rep.ratio(f"P-6: GQA 6, D=256 (Qwen-like), {nb} tokens, 32K, p6cfg vs upstream",
                  speedup(k, "upstream", "p6cfg", 256, 6, 32768, nb, "f16"), 1.3)
    for g in (3, 5, 6, 7):
        for nb in (8, 16):
            s = speedup(k, "upstream", "p6cfg", 128, g, 32768, nb, "f16")
            rep.add(f"P-6 (info): GQA {g}, D=128, {nb} tokens, 32K, p6cfg vs upstream",
                    f"{s[0]:.2f}×" if s else "–", "info", "info" if s else "NOT RUN")
    # "Within noise" guards against regressions: 0003's D=256 configs make GQA 4/8 faster, which is reported but
    # does not fail the criterion.
    slower, faster = [], []
    for hs in (128, 256):
        for g in (4, 8):
            for kv in (4096, 32768):
                for nb in (1, 2, 4, 8, 16):
                    s = speedup(k, "upstream", "p6cfg", hs, g, kv, nb, "f16")
                    if s and not s[1] and abs(s[0] - 1) > NOISE:
                        (slower if s[0] < 1 else faster).append(f"D{hs} G{g} kv{kv} nb{nb}: {s[0]:.2f}×")
    have = speedup(k, "upstream", "p6cfg", 128, 4, 4096, 1, "f16") is not None
    rep.add("P-6: GQA 4/8 (f16) p6cfg not slower than upstream", "; ".join(slower) or ("none slower" if have else "–"),
            "no case slower beyond noise", ("PASS" if not slower else "FAIL") if have else "NOT RUN")
    if faster:
        rep.add("P-6 (info): GQA 4/8 cases faster beyond noise", f"{len(faster)}: " + "; ".join(faster), "info", "info")
    slower = []
    for hs in (128, 256):
        for g in (3, 4, 5, 6, 7, 8):
            for kv in (4096, 32768):
                for nb in (1, 2, 4, 8, 16):
                    s = speedup(k, "upstream", "p6cfg", hs, g, kv, nb, "f16")
                    if s and not s[1] and s[0] < 1 - NOISE:
                        slower.append(f"D{hs} G{g} kv{kv} nb{nb}: {s[0]:.2f}×")
    rep.add("P-6: no f16 shape slower than upstream (GQA 3-8)", "; ".join(slower) or ("none slower" if have else "–"),
            "no case slower beyond noise", ("PASS" if not slower else "FAIL") if have else "NOT RUN")
    for nb in (16, 4, 2):
        s = speedup(k, "upstream", "p6cfg", 256, 6, 32768, nb, "f16")
        rep.add(f"P-6 decision data: D=256 GQA 6, {nb} tokens, 32K (p6cfg vs upstream)",
                f"{s[0]:.2f}×" if s else "–", "SMALL_B / B ≤ 4 decision", "info" if s else "NOT RUN")

    # --- P-8
    for t in ("q8_0", "q4_0"):
        rep.ratio(f"P-8: qkv vs p6cfg, 1 token, 32K, GQA 4, D=128, {t}", speedup(k, "p6cfg", "qkv", 128, 4, 32768, 1, t), 2.0)
    worst = None
    for hs in (128, 256):
        for g in (4, 6, 8):
            for t in ("q8_0", "q4_0"):
                s = speedup(k, "qkv", "conv8", hs, g, 32768, 8, t)
                if s and (worst is None or s[0] < worst[0][0]):
                    worst = (s, f"D{hs} G{g} {t}")
    rep.ratio("P-8: conv8 vs qkv, 8 tokens, 32K, quantised (worst case" + (f": {worst[1]})" if worst else ")"),
              worst[0] if worst else None, 1.5)
    out, have = [], False
    for hs in (128, 256):
        for g in (3, 4, 5, 6, 7, 8):
            for kv in (4096, 32768):
                for nb in (1, 2, 4, 8, 16):
                    s = speedup(k, "p6cfg", "conv8", hs, g, kv, nb, "f16")
                    have |= s is not None
                    if s and not s[1] and abs(s[0] - 1) > NOISE:
                        out.append(f"D{hs} G{g} kv{kv} nb{nb}: {s[0]:.2f}×")
    rep.add("P-8: f16 cases conv8 vs p6cfg within noise", "; ".join(out) or ("all within noise" if have else "–"),
            "no case outside noise", ("PASS" if not out else "FAIL") if have else "NOT RUN")
    for t in ("q8_0", "q4_0"):
        for hs in (128, 256):
            a, b = find(k, "qkv", hs, 4, 32768, 1, t), find(k, "qkv", hs, 8, 32768, 1, t)
            if a and b:
                r = b[0] / a[0]
                rep.add(f"P-8 Change 4 gate: qkv 1 token 32K, GQA 8 / GQA 4 time, D={hs}, {t}", f"{r:.2f}×",
                        "≥ 1.6× → design GQA-packed VEC", "TRIPPED" if r >= 1.6 else "not tripped")
            else:
                rep.add(f"P-8 Change 4 gate: D={hs}, {t}", "–", "≥ 1.6×", "NOT RUN")

    mb = microbench(dirs)
    if mb:
        bad = [r for r in mb if "bit-identical" not in r[4]]
        rep.add("P-8 microbench: 0005/0006 outputs bit-identical", f"{len(mb) - len(bad)}/{len(mb)} rows", "all",
                "PASS" if not bad else "FAIL")
        rep.add("P-8 microbench: 0006 speedup over upstream (min–max)",
                f"{min(r[3] for r in mb):.2f}–{max(r[3] for r in mb):.2f}× (0005: {min(r[2] for r in mb):.2f}–"
                f"{max(r[2] for r in mb):.2f}×)", "info", "info")
    else:
        rep.add("P-8 microbench", "–", "bit-identical", "NOT RUN")

    lb = load_lb(dirs, "kvq8_0", "llama")
    base = lb_find(lb, "upstream", 0, 64, 16384)
    for v in ("qkv", "conv8"):
        new = lb_find(lb, v, 0, 64, 16384)
        if base and new:
            pct = (new[0] / base[0] - 1) * 100
            rep.add(f"P-8 end to end: Llama q8_0 tg64 @16K, {v} vs upstream",
                    f"{pct:+.1f}% ({base[0]:.1f} → {new[0]:.1f} t/s)", "≥ +25%", "PASS" if pct >= 25 else "FAIL")
        else:
            rep.add(f"P-8 end to end: Llama q8_0 tg64 @16K, {v} vs upstream", "–", "≥ +25%", "NOT RUN")
    small = load_lb(dirs, "small", "qwen")
    for nb in (4, 8, 16):
        for d in (4096, 32768):
            a, b = lb_find(small, "upstream", nb, 0, d), lb_find(small, "p6cfg", nb, 0, d)
            if a and b:
                rep.add(f"P-6 end to end (info): Qwen pp{nb} @{d // 1024}K, p6cfg vs upstream",
                        f"{(b[0] / a[0] - 1) * 100:+.1f}% ({a[0]:.1f} → {b[0]:.1f} t/s)",
                        "fork measured +80–112% (pp8/16 @depth)", "info")
    rep.print()


if __name__ == "__main__":
    main()
