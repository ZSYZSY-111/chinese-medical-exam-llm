#!/usr/bin/env python3
"""Paired comparison of two per-question record files from eval_validation (headroom_records.jsonl), joined on sample_id.

Output: the 2×2 table (A right/wrong × B right/wrong), the accuracy difference with a paired bootstrap 95%
interval, the exact McNemar p-value, the minimum detectable effect at the observed discordance, the same per slice,
and each side's format-failure rate (no answer parsed).

  python scripts/compare_headroom_records.py --a direct.jsonl --b rationale.jsonl --label-a direct --label-b rationale [--json out.json]
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

try:
    from .score_predictions import mcnemar_exact, mde_paired, paired_bootstrap_delta, wilson
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_predictions import mcnemar_exact, mde_paired, paired_bootstrap_delta, wilson

SLICE_NAMES = ("multi_choice", "negation", "case", "calc", "long_stem")


def load_records(path):
    records = OrderedDict()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["sample_id"]] = record
    return records


def cells_of(pairs):
    """pairs: [(a_correct, b_correct)] → (both_right, only_a, only_b, both_wrong)。"""
    both = only_a = only_b = neither = 0
    for a, b in pairs:
        if a and b:
            both += 1
        elif a:
            only_a += 1
        elif b:
            only_b += 1
        else:
            neither += 1
    return (both, only_a, only_b, neither)


def summarize(pairs, resamples=10000, seed=42):
    n = len(pairs)
    cells = cells_of(pairs)
    acc_a = sum(a for a, _ in pairs) / n if n else 0.0
    acc_b = sum(b for _, b in pairs) / n if n else 0.0
    delta, lo, hi = paired_bootstrap_delta(cells, resamples, seed)
    discordance = (cells[1] + cells[2]) / n if n else 0.0
    return {
        "n": n, "acc_a": round(100 * acc_a, 2), "acc_b": round(100 * acc_b, 2),
        "ci_a": [round(100 * v, 2) for v in wilson(sum(a for a, _ in pairs), n)[1:]],
        "ci_b": [round(100 * v, 2) for v in wilson(sum(b for _, b in pairs), n)[1:]],
        "cells": {"both_right": cells[0], "only_a": cells[1], "only_b": cells[2], "both_wrong": cells[3]},
        "delta_b_minus_a": round(-delta, 2), "delta_ci": [round(-hi, 2), round(-lo, 2)],
        "mcnemar_p": mcnemar_exact(cells[1], cells[2]), "discordance": round(discordance, 4),
        "mde_points": round(mde_paired(n, discordance), 2) if n else None,
    }


def compare(records_a, records_b, resamples=10000, seed=42):
    common = [sid for sid in records_a if sid in records_b]
    pairs = [(bool(records_a[s]["greedy_correct"]), bool(records_b[s]["greedy_correct"])) for s in common]
    report = {"common": len(common), "only_in_a": len(records_a) - len(common), "only_in_b": len(records_b) - len(common),
              "overall": summarize(pairs, resamples, seed), "slices": {},
              "format_failure_a": round(100 * sum(1 for s in common if records_a[s].get("greedy_answer") is None) / max(len(common), 1), 2),
              "format_failure_b": round(100 * sum(1 for s in common if records_b[s].get("greedy_answer") is None) / max(len(common), 1), 2)}
    for name in SLICE_NAMES:
        subset = [(bool(records_a[s]["greedy_correct"]), bool(records_b[s]["greedy_correct"])) for s in common
                  if (records_b[s].get("slices") or {}).get(name)]
        if subset:
            report["slices"][name] = summarize(subset, resamples, seed)
    return report


def render(report, label_a, label_b):
    o = report["overall"]
    c = o["cells"]
    lines = [f"shared questions {report['common']} (only in A {report['only_in_a']}, only in B {report['only_in_b']})",
             f"{label_a}: {o['acc_a']}%（{o['ci_a'][0]}–{o['ci_a'][1]}）  {label_b}: {o['acc_b']}%（{o['ci_b'][0]}–{o['ci_b'][1]}）",
             f"format failures: {label_a} {report['format_failure_a']}%, {label_b} {report['format_failure_b']}%", "",
             f"|  | {label_b} right | {label_b} wrong |", "|---|---:|---:|",
             f"| {label_a} right | {c['both_right']} | {c['only_a']} |", f"| {label_a} wrong | {c['only_b']} | {c['both_wrong']} |", "",
             f"{label_b} − {label_a} = {o['delta_b_minus_a']:+.2f}（bootstrap 95% {o['delta_ci'][0]:+.2f}～{o['delta_ci'][1]:+.2f}），"
             f"McNemar p = {o['mcnemar_p']:.2g}, discordance {100 * o['discordance']:.1f}%, MDE ≈ {o['mde_points']} points", "",
             f"| slice | n | {label_a} | {label_b} | delta | 95% CI | p |", "|---|---:|---:|---:|---:|---|---:|"]
    for name, s in report["slices"].items():
        lines.append(f"| {name} | {s['n']} | {s['acc_a']} | {s['acc_b']} | {s['delta_b_minus_a']:+.2f} | "
                     f"{s['delta_ci'][0]:+.2f}～{s['delta_ci'][1]:+.2f} | {s['mcnemar_p']:.2g} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    report = compare(load_records(args.a), load_records(args.b), args.resamples, args.seed)
    if report["common"] == 0:
        print("the two record files share no sample_id", file=sys.stderr)
        return 1
    print(render(report, args.label_a, args.label_b))
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
