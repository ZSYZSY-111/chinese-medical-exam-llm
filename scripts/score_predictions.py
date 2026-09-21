#!/usr/bin/env python3
"""Score prediction files with uncertainty, and quantify how much difference an evaluation can detect.

Usage:
  one file:        python scripts/score_predictions.py preds.jsonl [--slices exam_type,question_type]
  paired files:    python scripts/score_predictions.py a.jsonl --compare b.jsonl
  detectable-effect table: python scripts/score_predictions.py --mde --n 11200,3000 --discordance 0.02,0.064,0.12

Fields are detected automatically (override with --id-field / --gold-field / --pred-field / --correct-field):
  id: id, question_id, qid, sample_id     gold: gold, answer, label, gold_answer, reference
  pred: prediction, pred, predicted, model_answer, predicted_answer, choice
  correct: correct, is_correct
Correctness follows eval_cmb.py: the predicted letter set must equal the reference set.
"""
import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict

ID_FIELDS = ("id", "question_id", "qid", "sample_id")
GOLD_FIELDS = ("gold", "answer", "label", "gold_answer", "reference")
PRED_FIELDS = ("prediction", "pred", "predicted", "model_answer", "predicted_answer", "choice")
CORRECT_FIELDS = ("correct", "is_correct")
DEFAULT_SLICES = ("exam_type", "question_type", "exam_class")
Z_975 = 1.959963984540054
Z_80 = 0.8416212335729143


def letter_set(value):
    return frozenset(re.findall(r"[A-Z]", str(value or "").upper()))


def pick_field(record, candidates, override=None):
    if override:
        return override if override in record else None
    for name in candidates:
        if name in record:
            return name
    return None


def load_answers(path):
    """Official answer file: a JSON list [{id, answer}] or JSONL. Returns {str(id): answer}."""
    if path.endswith(".jsonl"):
        records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    else:
        records = json.load(open(path, encoding="utf-8"))
    return {str(r["id"]): r["answer"] for r in records}


def load_predictions(path, id_field=None, gold_field=None, pred_field=None, correct_field=None, answers=None):
    """Returns [(id, correct, record)]; the line number is the id when no id field exists. With `answers`, the reference answer is joined by id."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if answers is not None:
                rid0 = record.get(id_field) if id_field else next((record[k] for k in ID_FIELDS if k in record), None)
                if rid0 is not None and str(rid0) in answers:
                    record["gold"] = answers[str(rid0)]
            if line_no == 0:
                idf = pick_field(record, ID_FIELDS, id_field)
                gf = pick_field(record, GOLD_FIELDS, gold_field)
                pf = pick_field(record, PRED_FIELDS, pred_field)
                cf = pick_field(record, CORRECT_FIELDS, correct_field)
                if cf is None and (gf is None or pf is None):
                    raise ValueError(f"{path}: neither a correct field nor gold + prediction fields found; first-row keys: {sorted(record)}")
            rid = record[idf] if idf else line_no
            if cf is not None:
                correct = bool(record[cf]) if not isinstance(record[cf], str) else record[cf].lower() in ("1", "true", "yes")
            else:
                correct = letter_set(record[gf]) == letter_set(record[pf]) and bool(letter_set(record[gf]))
            rows.append((rid, correct, record))
    return rows


def wilson(k, n, z=Z_975):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_accuracy(correct_flags, resamples=10000, seed=42):
    """Bootstrapping a Bernoulli mean is equivalent to drawing Binomial(n, acc) / n, which is instant."""
    n = len(correct_flags)
    if n == 0:
        return (0.0, 0.0)
    acc = sum(correct_flags) / n
    rng = random.Random(seed)
    draws = sorted(_binomial(rng, n, acc) / n for _ in range(resamples))
    return (draws[int(0.025 * resamples)], draws[min(resamples - 1, int(0.975 * resamples))])


def _binomial(rng, n, p):
    if hasattr(rng, "binomialvariate"):
        return rng.binomialvariate(n, p)
    if n <= 400:
        return sum(1 for _ in range(n) if rng.random() < p)
    # Python < 3.12 and large n: normal approximation, clipped to [0, n]
    mean, sd = n * p, math.sqrt(max(n * p * (1 - p), 1e-12))
    return min(n, max(0, round(rng.gauss(mean, sd))))


def mcnemar_exact(b, c):
    """Exact two-sided binomial McNemar test. b = only A correct, c = only B correct."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar_chi2(b, c):
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    stat = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2))
    return (stat, p)


def paired_bootstrap_delta(cells, resamples=10000, seed=42):
    """cells = (both_right, only_a, only_b, both_wrong); returns (delta, lo, hi) in percentage points."""
    n = sum(cells)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    probs = [cnt / n for cnt in cells]
    deltas = []
    for _ in range(resamples):
        draw = _multinomial(rng, n, probs)
        deltas.append((draw[1] - draw[2]) / n * 100)
    deltas.sort()
    point = (cells[1] - cells[2]) / n * 100
    return (point, deltas[int(0.025 * resamples)], deltas[min(resamples - 1, int(0.975 * resamples))])


def _multinomial(rng, n, probs):
    out, remaining, rest = [], n, 1.0
    for p in probs[:-1]:
        share = 0 if rest <= 0 else min(1.0, p / rest)
        k = _binomial(rng, remaining, share)
        out.append(k)
        remaining -= k
        rest -= p
    out.append(remaining)
    return out


def mde_paired(n, discordance, alpha_z=Z_975, power_z=Z_80):
    """Smallest accuracy difference (points) a paired McNemar test detects with 80% power at the given discordance rate."""
    n_d = n * discordance
    if n_d <= 0:
        return float("inf")
    return (alpha_z + power_z) * math.sqrt(n_d) / n * 100


def ci_half_width(n, acc=0.84, z=Z_975):
    return z * math.sqrt(acc * (1 - acc) / n) * 100


def pred_of(record):
    for name in PRED_FIELDS + ("greedy_answer", "model_answer", "sft_answer"):
        if name in record:
            return letter_set(record[name])
    return None


def slice_values(record, name):
    """Dotted paths are supported, e.g. slices.multi_choice in the records written by eval_validation.py."""
    value = record
    for part in name.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return None if value in (None, "") else str(value)


def score_single(rows, slices, resamples, seed):
    flags = [c for _, c, _ in rows]
    acc, lo, hi = wilson(sum(flags), len(flags))
    blo, bhi = bootstrap_accuracy(flags, resamples, seed)
    result = {"n": len(flags), "accuracy": acc * 100, "wilson95": [lo * 100, hi * 100],
              "bootstrap95": [blo * 100, bhi * 100], "slices": {}}
    for name in slices:
        groups = defaultdict(list)
        for _, correct, record in rows:
            value = slice_values(record, name)
            if value is not None:
                groups[value].append(correct)
        if groups:
            result["slices"][name] = {
                value: {"n": len(v), "accuracy": wilson(sum(v), len(v))[0] * 100,
                        "wilson95": [x * 100 for x in wilson(sum(v), len(v))[1:]]}
                for value, v in sorted(groups.items())}
    return result


def compare(rows_a, rows_b, slices, resamples, seed):
    a = {rid: (c, r) for rid, c, r in rows_a}
    b = {rid: (c, r) for rid, c, r in rows_b}
    common = [rid for rid in a if rid in b]
    if len(common) != len(a) or len(common) != len(b):
        print(f"warning: the two files do not share all ids (A {len(a)}, B {len(b)}, common {len(common)}); comparing the common part only", file=sys.stderr)
    cells = Counter()
    per_slice = defaultdict(Counter)
    answer_pairs = answer_changed = 0
    for rid in common:
        ca, ra = a[rid]
        cb, _ = b[rid]
        key = "both_right" if ca and cb else "only_a" if ca else "only_b" if cb else "both_wrong"
        cells[key] += 1
        pa, pb = pred_of(ra), pred_of(b[rid][1])
        if pa is not None and pb is not None:
            answer_pairs += 1
            if pa != pb:
                answer_changed += 1
        for name in slices:
            value = slice_values(ra, name)
            if value is not None:
                per_slice[(name, value)][key] += 1
    tup = (cells["both_right"], cells["only_a"], cells["only_b"], cells["both_wrong"])
    delta, lo, hi = paired_bootstrap_delta(tup, resamples, seed)
    n = sum(tup)
    out = {"n": n, "acc_a": (tup[0] + tup[1]) / n * 100 if n else 0, "acc_b": (tup[0] + tup[2]) / n * 100 if n else 0,
           "delta_a_minus_b": delta, "delta_bootstrap95": [lo, hi], "only_a": tup[1], "only_b": tup[2],
           "discordance": (tup[1] + tup[2]) / n if n else 0,
           "mcnemar_exact_p": mcnemar_exact(tup[1], tup[2]), "mcnemar_chi2_p": mcnemar_chi2(tup[1], tup[2])[1],
           "mde_pp_at_this_discordance": mde_paired(n, (tup[1] + tup[2]) / n) if n else None,
           "answer_changed": answer_changed, "answer_change_rate": (answer_changed / answer_pairs) if answer_pairs else None, "slices": {}}
    for (name, value), cnt in sorted(per_slice.items()):
        m = cnt["both_right"] + cnt["only_a"] + cnt["only_b"] + cnt["both_wrong"]
        out["slices"].setdefault(name, {})[value] = {
            "n": m, "acc_a": (cnt["both_right"] + cnt["only_a"]) / m * 100, "acc_b": (cnt["both_right"] + cnt["only_b"]) / m * 100,
            "delta": (cnt["only_a"] - cnt["only_b"]) / m * 100, "only_a": cnt["only_a"], "only_b": cnt["only_b"],
            "mcnemar_exact_p": mcnemar_exact(cnt["only_a"], cnt["only_b"])}
    return out


def mde_table(ns, discordances):
    rows = []
    for n in ns:
        for d in discordances:
            rows.append({"n": n, "discordance": d, "mde_pp": mde_paired(n, d), "single_acc_ci_half_width_pp": ci_half_width(n)})
    return rows


def format_markdown(payload):
    lines = []
    if "mde_table" in payload:
        lines.append("| n | discordance | minimum detectable difference at 80% power (points) | 95% CI half-width of one accuracy (points) |")
        lines.append("|---:|---:|---:|---:|")
        for r in payload["mde_table"]:
            lines.append(f"| {r['n']} | {r['discordance']*100:.1f}% | {r['mde_pp']:.2f} | {r['single_acc_ci_half_width_pp']:.2f} |")
    if "single" in payload:
        s = payload["single"]
        lines.append(f"accuracy {s['accuracy']:.2f} (Wilson 95% {s['wilson95'][0]:.2f}–{s['wilson95'][1]:.2f}; bootstrap {s['bootstrap95'][0]:.2f}–{s['bootstrap95'][1]:.2f}), n={s['n']}")
        for name, values in s["slices"].items():
            lines.append(f"\n| {name} | n | accuracy | 95% CI |")
            lines.append("|---|---:|---:|---|")
            for value, v in values.items():
                lines.append(f"| {value} | {v['n']} | {v['accuracy']:.2f} | {v['wilson95'][0]:.2f}–{v['wilson95'][1]:.2f} |")
    if "compare" in payload:
        c = payload["compare"]
        lines.append(f"A {c['acc_a']:.2f} vs B {c['acc_b']:.2f}: difference {c['delta_a_minus_b']:+.2f} (bootstrap 95% {c['delta_bootstrap95'][0]:+.2f} to {c['delta_bootstrap95'][1]:+.2f}), "
                     f"only A correct {c['only_a']} / only B correct {c['only_b']}, discordance {c['discordance']*100:.1f}%, exact McNemar p={c['mcnemar_exact_p']:.2e}, "
                     f"minimum detectable difference at this discordance {c['mde_pp_at_this_discordance']:.2f} points"
                     + (f"; answers changed {c['answer_changed']} ({c['answer_change_rate']*100:.2f}%)" if c.get('answer_change_rate') is not None else ""))
        for name, values in c["slices"].items():
            lines.append(f"\n| {name} | n | A | B | diff | p |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for value, v in values.items():
                lines.append(f"| {value} | {v['n']} | {v['acc_a']:.2f} | {v['acc_b']:.2f} | {v['delta']:+.2f} | {v['mcnemar_exact_p']:.3f} |")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="?")
    parser.add_argument("--compare", help="second prediction file; may equal the first one when used with --compare-correct-field")
    parser.add_argument("--compare-correct-field", help="correct field of model B inside the same file, e.g. base_correct")
    parser.add_argument("--slices", default=",".join(DEFAULT_SLICES))
    parser.add_argument("--id-field"), parser.add_argument("--gold-field"), parser.add_argument("--pred-field"), parser.add_argument("--correct-field")
    parser.add_argument("--answers-file", help="official answer file (json/jsonl with id and answer), joined onto predictions that only carry model_answer")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mde", action="store_true")
    parser.add_argument("--n", default="11200,9400,2784")
    parser.add_argument("--discordance", default="0.02,0.04,0.064,0.08,0.12")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    slices = [s for s in args.slices.split(",") if s]
    payload = {}
    if args.mde:
        payload["mde_table"] = mde_table([int(x) for x in args.n.split(",")], [float(x) for x in args.discordance.split(",")])
    if args.predictions:
        answers = load_answers(args.answers_file) if args.answers_file else None
        rows_a = load_predictions(args.predictions, args.id_field, args.gold_field, args.pred_field, args.correct_field, answers)
        payload["single"] = score_single(rows_a, slices, args.resamples, args.seed)
        if args.compare or args.compare_correct_field:
            path_b = args.compare or args.predictions
            rows_b = load_predictions(path_b, args.id_field, args.gold_field, args.pred_field, args.compare_correct_field or args.correct_field, answers)
            payload["compare"] = compare(rows_a, rows_b, slices, args.resamples, args.seed)
    if not payload:
        print("nothing to do: pass a prediction file or --mde", file=sys.stderr)
        return 1
    print(format_markdown(payload))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
