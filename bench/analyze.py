#!/usr/bin/env python3
"""Compare benchmark outputs from different builds as Markdown tables.

  python bench/analyze.py tbo upstream.perf.sql pr.perf.sql comb64.perf.sql [--labels a,b,c]
  python bench/analyze.py tbo up.r1.sql,up.r2.sql,up.r3.sql pr.r1.sql,pr.r2.sql,pr.r3.sql
      test-backend-ops `perf --output sql` files (the CSV printer omits timings). A comma-joined
      group is repeated rounds of one build: the median time is shown with the min-max spread.
      Rows are matched on the parsed op parameters, with defaults filled in for fields that older
      trees do not print (so upstream and a fork with an older base still line up). Speedup is
      relative to the first group.
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
import statistics
import sys

TBO_KEYS = ("hsk", "hsv", "nh", "nr23", "kv", "nb", "mask", "logit_softcap", "type_K", "type_V", "n_kv_max")
# Fields newer test-backend-ops prints for FLASH_ATTN_EXT, with the value an older tree implies.
TBO_DEFAULTS = {"sinks": "0", "max_bias": "0.000000", "logit_softcap": "0.000000", "prec": "f32",
                "permute": "[0,1,2,3]", "kv_view": "0", "v_is_view_of_k": "0", "n_kv_max": "0"}
LB_KEYS = ("model_type", "n_prompt", "n_gen", "n_depth", "n_ubatch", "type_k", "type_v", "flash_attn",
           "split_mode", "devices")


def parse_params(params):
    kv = dict(re.findall(r"(\w+)=(\[[^\]]*\]|[^,]+)", params))
    for k, v in TBO_DEFAULTS.items():
        kv.setdefault(k, v)
    return kv


def load_tbo_one(path):
    db = sqlite3.connect(":memory:")
    text = pathlib.Path(path).read_text(errors="replace")
    # Keep only the SQL statements; stderr noise or progress lines may be interleaved.
    stmts = re.findall(r"(CREATE TABLE.*?\);|INSERT INTO.*?\);)\s*$", text, re.S | re.M)
    if not stmts:
        sys.exit(f"{path}: no SQL found (was it produced with --output sql?)")
    db.executescript("\n".join(stmts))
    rows = {}
    for op, params, supported, time_us in db.execute(
            "SELECT op_name, op_params, supported, time_us FROM test_backend_ops WHERE test_mode='perf'"):
        key = (op, tuple(sorted(parse_params(params).items())))
        rows[key] = (bool(supported), time_us)
    return rows


def load_tbo(group):
    """A comma-joined group of files = repeated rounds of one build -> (supported, median, min, max, n)."""
    rounds = [load_tbo_one(f) for f in group.split(",")]
    out = {}
    for key in set().union(*rounds):
        vals = [r[key] for r in rounds if key in r]
        times = sorted(t for ok, t in vals if ok and t)
        if not times:
            out[key] = (False, None, None, None, 0)
            continue
        out[key] = (True, statistics.median(times), times[0], times[-1], len(times))
    return out


def short_params(key, all_keys):
    """Case label: the main shape fields, plus any other field that varies between rows of this op."""
    kv = dict(key[1])
    varying = set()
    same_op = [dict(k[1]) for k in all_keys if k[0] == key[0]]
    for f in kv:
        if len({d.get(f) for d in same_op}) > 1:
            varying.add(f)
    out = []
    for k in TBO_KEYS + tuple(sorted(set(kv) - set(TBO_KEYS))):
        if k not in kv:
            continue
        v = kv[k]
        if k in TBO_KEYS:
            if k == "mask" and v == "1" and k not in varying:
                continue
            if k in TBO_DEFAULTS and v == TBO_DEFAULTS[k] and k not in varying:
                continue
            if k == "hsv" and v == kv.get("hsk") and k not in varying:
                continue
            if k == "type_V" and v == kv.get("type_K") and k not in varying:
                continue
        elif k not in varying:
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
                spread = f" ({fmt_us(r[2])}–{fmt_us(r[3])}, n={r[4]})" if r[4] > 1 else ""
                cells.append(fmt_us(r[1]) + spread)
                if i == 0:
                    base = r[1]
        sp = []
        for d in data[1:]:
            r = d.get(k)
            sp.append(f"{base / r[1]:.2f}×" if base and r and r[0] and r[1] else "–")
        print(f"| {k[0]} | {short_params(k, keys)} | " + " | ".join(cells) + " | " + " | ".join(sp) + " |")


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
    labels = a.labels.split(",") if a.labels else [pathlib.Path(f.split(",")[0]).name.split(".")[0] for f in a.files]
    if len(labels) != len(a.files):
        sys.exit("--labels must have one label per file")
    (cmd_tbo if a.kind == "tbo" else cmd_lb)(a.files, labels)


if __name__ == "__main__":
    main()
