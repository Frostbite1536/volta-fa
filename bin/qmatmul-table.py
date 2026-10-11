#!/usr/bin/env python3
"""Tabulate GGUF matmul weights, test-backend-ops timings, and evaluation estimates.

SQL_GROUP is one perf SQL file or comma-separated repeated rounds. Timings are
microseconds; bandwidth uses decimal GB/s, and weight sizes use binary MiB.
Only the Python standard library and the repository's bench/analyze.py are used.
"""
import argparse
from collections import Counter, defaultdict
import math
import pathlib
import re
import sqlite3
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bench"))
import analyze  # noqa: E402 (reuse the SQL parser and round summarizer)

# ggml storage: (bytes per block, elements per block).
BLOCKS = {
    "Q4_0": (18, 32), "Q4_1": (20, 32), "Q5_0": (22, 32),
    "Q5_1": (24, 32), "Q8_0": (34, 32), "Q2_K": (84, 256),
    "Q3_K": (110, 256), "Q4_K": (144, 256), "Q5_K": (176, 256),
    "Q6_K": (210, 256), "IQ4_NL": (18, 32), "IQ4_XS": (136, 256),
    "IQ3_S": (110, 256), "F16": (2, 1), "BF16": (2, 1),
}
MIB = 1024 ** 2
TENSOR = re.compile(
    r"^\s*\d+:\s*(\d+)\s*\|\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)"
    r"\s*\|\s*(\w+)\s*\|\s*(\S+)\s*$"
)


def weight_bytes(kind, elements):
    block = BLOCKS.get(kind)
    if block is None:
        return None
    size, length = block
    if elements % length:
        raise ValueError(f"{kind}: {elements} elements is not a multiple of block size {length}")
    return elements // length * size


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(cell) for cell in row) + " |")
    print()


def load_weights(path, exclude=None):
    """Return counts keyed by (type, N, K, role), retaining unknown types. exclude: regex on the tensor name."""
    groups = Counter()
    found = False
    for number, line in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        match = TENSOR.match(line)
        if not match:
            continue
        found = True
        elements, k, n, ne2, ne3 = map(int, match.groups()[:5])
        kind, name = match.groups()[5:]
        kind = kind.upper()
        if min(k, n, ne2, ne3) <= 0 or elements != k * n * ne2 * ne3:
            raise ValueError(f"{path}:{number}: invalid tensor dimensions or element count")
        if n <= 1 or ne2 != 1 or ne3 != 1 or kind == "F32" or name.startswith("token_embd"):
            continue
        if exclude and re.search(exclude, name):
            continue
        role = re.sub(r"^blk\.\d+\.", "", name)
        groups[kind, n, k, role] += 1
    if not found:
        raise ValueError(f"{path}: no tensor lines found")
    return groups


def unknown_types(groups):
    counts = Counter()
    for (kind, n, k, role), count in groups.items():
        if kind not in BLOCKS:
            counts[kind] += count
    for kind, count in sorted(counts.items()):
        print(f"unknown type: {kind} ({count} tensors); excluded from byte totals.")
    if counts:
        print()


def cmd_shapes(args):
    for path in args.dumps:
        groups = load_weights(path, args.exclude)
        print(f"### {pathlib.Path(path).name}\n")
        rows, elements_by_type, bytes_by_type = [], Counter(), Counter()
        output_bytes = 0
        for (kind, n, k, role), count in sorted(groups.items(), key=lambda item: (item[0][3], item[0][:3])):
            elements = count * n * k
            size = weight_bytes(kind, elements)
            rows.append((role, kind, k, n, count, "unknown" if size is None else f"{size / MIB:.6f}"))
            elements_by_type[kind] += elements
            if size is not None:
                bytes_by_type[kind] += size
                if role == "output.weight":
                    output_bytes += size
        table(["role", "type", "K", "N", "count", "MiB (total)"], rows)
        table(["type", "elements", "MiB"], [
            (kind, elements, "unknown" if kind not in BLOCKS else f"{bytes_by_type[kind] / MIB:.6f}")
            for kind, elements in sorted(elements_by_type.items())
        ])
        total = sum(bytes_by_type.values())
        print(f"Total matmul weights: {total} bytes ({total / MIB:.6f} MiB), including output.weight.")
        print(f"output.weight: {output_bytes} bytes ({output_bytes / MIB:.6f} MiB), known types only.\n")
        unknown_types(groups)


def load_perf(group):
    """Project summarized MUL_MAT cases to (type_a, m, k, n), without merging variants."""
    data = {}
    for (op, params), summary in analyze.load_tbo(group).items():
        if op != "MUL_MAT":
            continue
        p = dict(params)  # analyze.load_tbo already used analyze.parse_params.
        try:
            key = (p["type_a"].upper(), int(p["m"]), int(p["k"]), int(p["n"]))
        except (KeyError, ValueError) as error:
            raise ValueError(f"{group}: invalid MUL_MAT shape parameters: {p}") from error
        if min(key[1:]) <= 0:
            raise ValueError(f"{group}: nonpositive MUL_MAT shape: {key}")
        if key in data:
            raise ValueError(f"{group}: ambiguous MUL_MAT variants for {key}; supply one case per shape/token count")
        if summary[0] and any(not math.isfinite(t) or t <= 0 for t in summary[1:4]):
            raise ValueError(f"{group}: invalid timing for {key}")
        data[key] = summary
    return data


def bandwidth(key, summary):
    size = weight_bytes(key[0], key[1] * key[2])
    return None if size is None or not summary[0] else size / (summary[1] * 1000)


def cmd_perf(args):
    labels = args.labels.split(",") if args.labels else [pathlib.Path(g.split(",")[0]).name for g in args.groups]
    if len(labels) != len(args.groups):
        raise ValueError("--labels must have one label per input group")
    for label, group in zip(labels, args.groups):
        data = load_perf(group)
        shapes = sorted({key[:3] for key in data})
        tokens = sorted({key[3] for key in data})
        print(f"### {label}\n")
        for metric in ("Median time (µs)", "Effective weight bandwidth (GB/s)"):
            print(f"{metric}\n")
            rows = []
            for shape in shapes:
                cells = []
                for n in tokens:
                    key = (*shape, n)
                    summary = data.get(key)
                    if summary is None:
                        cell = "not measured"
                    elif not summary[0]:
                        cell = "unsupported"
                    elif metric.startswith("Median"):
                        cell = f"{summary[1]:.3f}"
                    else:
                        gbps = bandwidth(key, summary)
                        cell = "unknown type" if gbps is None else f"{gbps:.3f}"
                    cells.append(cell)
                rows.append((*shape, *cells))
            table(["type_a", "m (N)", "k (K)", *(f"n={n}" for n in tokens)], rows)
        if "," in group:
            spreads = [(r[3] - r[2]) / r[1] for r in data.values() if r[0] and r[4] > 1]
            spread = f"{max(spreads) * 100:.2f}%" if spreads else "not measured (no case has multiple samples)"
            print(f"Max round spread (max-min)/median: {spread}.\n")
        for kind in sorted({key[0] for key in data} - BLOCKS.keys()):
            print(f"unknown type: {kind}; excluded from bandwidth calculations.\n")


def parse_tokens(value):
    try:
        tokens = sorted(set(int(n) for n in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("tokens must be comma-separated positive integers") from error
    if not tokens or tokens[0] <= 0:
        raise argparse.ArgumentTypeError("tokens must be comma-separated positive integers")
    return tokens


def cmd_step(args):
    weights = load_weights(args.dump, args.exclude)
    groups = Counter()
    for (kind, n, k, role), count in weights.items():
        if args.with_output or role != "output.weight":
            groups[kind, n, k] += count
    data = load_perf(args.group)
    rates = defaultdict(list)
    for key, summary in data.items():
        rate = bandwidth(key, summary)
        if rate is not None:
            rates[key[0], key[3]].append(rate)
    rates = {key: statistics.median(values) for key, values in rates.items()}
    sizes = {key: weight_bytes(key[0], key[1] * key[2] * count) for key, count in groups.items()}
    total_bytes = sum(size for size in sizes.values() if size is not None)
    print(f"### {pathlib.Path(args.dump).name}: estimated matmul time per evaluation\n")
    print(f"output.weight is {'included (--with-output)' if args.with_output else 'excluded (default)'}. "
          "token_embd and F32 tensors are excluded.\n")
    print("Missing shapes use the same type's median effective weight GB/s over measured shapes at that token count. "
          "Types without usable measurements at that token count are not measured and excluded from time. "
          "Totals are partial when groups are not measured.\n")
    print(f"Coverage denominator: {total_bytes} bytes of included weights with known storage sizes.\n")
    unknown_types({(*key, ""): count for key, count in groups.items()})
    rows, missing = [], []
    for tokens in args.tokens:
        measured_us = estimated_us = 0.0
        covered_bytes = excluded_count = 0
        for shape, count in sorted(groups.items()):
            summary = data.get((*shape, tokens))
            size = sizes[shape]
            if summary and summary[0]:
                measured_us += count * summary[1]
                if size is not None:
                    covered_bytes += size
            else:
                rate = rates.get((shape[0], tokens))
                if rate is not None and size is not None:
                    time_us = size / (rate * 1000)
                    estimated_us += time_us
                    method, rate_cell, time_cell = "estimated", f"{rate:.3f}", f"{time_us:.3f}"
                else:
                    excluded_count += count
                    method, rate_cell, time_cell = "not measured", "—", "excluded"
                missing.append((tokens, *shape, count, method, rate_cell, time_cell))
        coverage = f"{100 * covered_bytes / total_bytes:.2f}%" if total_bytes else "n/a"
        rows.append((tokens, f"{measured_us:.3f}", f"{estimated_us:.3f}",
                     f"{measured_us + estimated_us:.3f}", f"{covered_bytes}/{total_bytes}", coverage, excluded_count))
    table(["tokens", "measured µs", "estimated µs", "total µs", "exact bytes/total bytes", "coverage", "not measured tensors"], rows)
    if missing:
        print("Groups without an exact measured shape (time includes count):\n")
        table(["tokens", "type", "N", "K", "count", "method", "median GB/s", "total µs"], missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shapes = commands.add_parser("shapes", help="matmul shapes and storage per GGUF dump")
    shapes.add_argument("dumps", nargs="+", metavar="DUMP")
    shapes.add_argument("--exclude", metavar="REGEX", help="skip tensors whose name matches (e.g. layers a run does not use)")
    shapes.set_defaults(run=cmd_shapes)
    perf = commands.add_parser("perf", help="median timings and effective weight bandwidth")
    perf.add_argument("groups", nargs="+", metavar="SQL_GROUP")
    perf.add_argument("--labels", help="comma-separated labels, one per input group")
    perf.set_defaults(run=cmd_perf)
    step = commands.add_parser("step", help="estimated evaluation matmul time from shapes and timings")
    step.add_argument("dump", metavar="DUMP")
    step.add_argument("group", metavar="SQL_GROUP")
    step.add_argument("--tokens", type=parse_tokens, default=parse_tokens("1,2,4,8,16"))
    step.add_argument("--with-output", action="store_true", help="include output.weight in time and coverage")
    step.add_argument("--exclude", metavar="REGEX", help="skip tensors whose name matches (e.g. layers a run does not use)")
    step.set_defaults(run=cmd_step)
    args = parser.parse_args()
    try:
        args.run(args)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
