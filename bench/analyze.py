#!/usr/bin/env python3
"""Compare benchmark outputs from different builds as Markdown tables.

  python bench/analyze.py tbo upstream.perf.sql pr.perf.sql comb64.perf.sql [--labels a,b,c]
      test-backend-ops `perf --output sql` files (the CSV printer omits timings).
      Rows are matched on op_params; speedup is relative to the first file.
  python bench/analyze.py lb upstream.jsonl fork.jsonl [--labels a,b]
      llama-bench `-o jsonl` files. Rows are matched on the test configuration;
      a change is marked "~" when it is within the two runs' combined stddev.

Only the Python standard library is used, so it runs on a fresh cloud instance.
"""
import argparse
import json
import pathlib
import re
import sqlite3
import sys

TBO_KEYS = ("hsk", "hsv", "nh", "nr23", "kv", "nb", "mask", "logit_softcap", "type_K", "type_V", "n_kv_max")
LB_KEYS = ("model_type", "n_prompt", "n_gen", "n_depth", "n_ubatch", "type_k", "type_v", "flash_attn",
           "split_mode", "devices")


def load_tbo(path):
    db = sqlite3.connect(":memory:")
    text = pathlib.Path(path).read_text(errors="replace")
    # Keep only the SQL statements; stderr noise or progress lines may be interleaved.
    stmts = [s for s in re.findall(r"(CREATE TABLE.*?\);|INSERT INTO.*?\);)\s*$", text, re.S | re.M)]
    if not stmts:
        sys.exit(f"{path}: no SQL found (was it produced with --output sql?)")
    db.executescript("\n".join(stmts))
    rows = {}
    for op, params, supported, time_us, flops in db.execute(
            "SELECT op_name, op_params, supported, time_us, flops FROM test_backend_ops WHERE test_mode='perf'"):
        rows[(op, params)] = (bool(supported), time_us, flops)
    return rows


def short_params(params):
    kv = dict(re.findall(r"(\w+)=(\[[^\]]*\]|[^,]+)", params))
    out = []
    for k in TBO_KEYS:
        if k not in kv:
            continue
        v = kv[k]
        if k in ("mask",) and v == "1":
            continue
        if k in ("logit_softcap",) and float(v) == 0:
            continue
        if k == "n_kv_max" and v == "0":
            continue
        if k == "hsv" and v == kv.get("hsk"):
            continue
        if k == "type_V" and v == kv.get("type_K"):
            continue
        out.append(f"{k}={v}")
    return ",".join(out)


def fmt_us(t):
    return f"{t/1000:.3f} ms" if t >= 1000 else f"{t:.1f} µs"


def cmd_tbo(files, labels):
    data = [load_tbo(f) for f in files]
    keys = sorted(set().union(*data), key=lambda k: (k[0], k[1]))
    print("| op | case | " + " | ".join(labels) + " | " + " | ".join(f"{l} vs {labels[0]}" for l in labels[1:]) + " |")
    print("|---|---|" + "---:|" * (2 * len(labels) - 1))
    for k in keys:
        cells, base = [], None
        for i, d in enumerate(data):
            r = d.get(k)
            if r is None:
                cells.append("absent")
            elif not r[0]:
                cells.append("unsupported")
            else:
                cells.append(fmt_us(r[1]))
                if i == 0:
                    base = r[1]
        sp = []
        for d in data[1:]:
            r = d.get(k)
            sp.append(f"{base / r[1]:.2f}×" if base and r and r[0] and r[1] else "–")
        print(f"| {k[0]} | {short_params(k[1])} | " + " | ".join(cells) + " | " + " | ".join(sp) + " |")


def load_lb(path):
    rows = {}
    for n, line in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"{path}:{n}: {e}")
        key = tuple(str(r.get(k, "")) for k in LB_KEYS)
        rows[key] = (float(r["avg_ts"]), float(r.get("stddev_ts", 0.0)))
    if not rows:
        sys.exit(f"{path}: no llama-bench JSON lines")
    return rows


def lb_test_name(key):
    k = dict(zip(LB_KEYS, key))
    t = f"pp{k['n_prompt']}" if k["n_prompt"] != "0" else f"tg{k['n_gen']}"
    extra = [f"d{k['n_depth']}" if k["n_depth"] != "0" else ""]
    if k["type_k"] != "f16" or k["type_v"] != "f16":
        extra.append(f"kv={k['type_k']}/{k['type_v']}")
    extra.append(f"ub{k['n_ubatch']}")
    extra.append(f"fa={k['flash_attn']}")
    if k["split_mode"] not in ("", "layer"):
        extra.append(f"sm={k['split_mode']}")
    return k["model_type"], " ".join([t] + [e for e in extra if e])


def cmd_lb(files, labels):
    data = [load_lb(f) for f in files]
    keys = sorted(set().union(*data), key=lambda k: lb_test_name(k))
    print("| model | test | " + " | ".join(f"{l} t/s" for l in labels) + " | " +
          " | ".join(f"{l} vs {labels[0]}" for l in labels[1:]) + " |")
    print("|---|---|" + "---:|" * (2 * len(labels) - 1))
    for k in keys:
        model, test = lb_test_name(k)
        base = data[0].get(k)
        cells = [f"{r[0]:.2f} ± {r[1]:.2f}" if r else "absent" for r in (d.get(k) for d in data)]
        deltas = []
        for d in data[1:]:
            r = d.get(k)
            if not (base and r and base[0]):
                deltas.append("–")
                continue
            pct = (r[0] / base[0] - 1) * 100
            noise = abs(r[0] - base[0]) <= (r[1] + base[1])
            deltas.append(f"{pct:+.1f}%{' ~' if noise else ''}")
        print(f"| {model} | {test} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["tbo", "lb"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--labels", help="comma-separated, default: file stems")
    a = ap.parse_args()
    labels = a.labels.split(",") if a.labels else [pathlib.Path(f).name.split(".")[0] for f in a.files]
    if len(labels) != len(a.files):
        sys.exit("--labels must have one label per file")
    (cmd_tbo if a.kind == "tbo" else cmd_lb)(a.files, labels)


if __name__ == "__main__":
    main()
