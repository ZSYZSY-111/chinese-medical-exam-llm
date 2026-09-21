#!/usr/bin/env python3
"""Build ablation subsets from the Stage 3 training file: data-scale curve, shuffled-copy dose, cleaned vs raw.
Every subset is written with a manifest.

The unit is a question (the original row plus its shuffled copies, grouped by stem + option-set hash); the CMExam
component is kept unchanged in every variant. Subsets are stratified by (exam_type, question_type) and cut in
stable-hash order, so they are nested: 25% is a subset of 50%, which is a subset of 100%.

  python scripts/build_ablation_sets.py --train-file data/cmb_sft_v4/cmb_sft_train.jsonl \
    --metadata data/cmb_sft_v4/cmb_sft_metadata.jsonl --audit-dir data/label_audit --output-dir data/ablation
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .build_cmb_train_sft import normalize_stem, stem_option_hash
    from .cmexam_prompts import (MODE_DIRECT, build_prompt_messages, is_shuffle_safe, make_permutation, parse_user_content,
                                 permute_options, remap_answer, stable_fraction, type_hint_for)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_cmb_train_sft import normalize_stem, stem_option_hash
    from cmexam_prompts import (MODE_DIRECT, build_prompt_messages, is_shuffle_safe, make_permutation, parse_user_content,
                                permute_options, remap_answer, stable_fraction, type_hint_for)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(train_file, metadata_file):
    meta = {}
    with open(metadata_file, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                meta[record["sample_id"]] = record
    rows = []
    with open(train_file, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            users = [m["content"] for m in record["messages"] if m["role"] == "user"]
            sample_id = sha256_text("\n".join(users))
            parsed = parse_user_content(users[0])
            extra = meta.get(sample_id, {})
            unit = stem_option_hash(normalize_stem(parsed["question"]), parsed["options"])
            rows.append({"sample_id": sample_id, "record": record, "question": parsed["question"], "options": parsed["options"],
                         "gold": record["messages"][-1]["content"].strip(), "unit": unit,
                         "source": extra.get("source", "unknown"), "variant": extra.get("variant", "original"),
                         "exam_type": extra.get("exam_type") or "", "question_type": extra.get("question_type") or ""})
    return rows


def content_answer(row):
    return tuple(sorted(normalize_stem(text) for letter, text in row["options"] if letter in row["gold"]))


def select_units(units, scale, seed):
    """units: {unit: [rows]} (CMB only). Stratified, stable-hash ordered; returns the set of units in the first `scale` fraction."""
    if scale >= 1.0:
        return set(units)
    if scale <= 0.0:
        return set()
    strata = defaultdict(list)
    for unit, rows in units.items():
        original = next((r for r in rows if r["variant"] == "original"), rows[0])
        strata[(original["exam_type"], original["question_type"])].append(unit)
    chosen = set()
    for key, members in strata.items():
        members.sort(key=lambda u: stable_fraction(seed, "ablation_scale", u))
        chosen.update(members[:int(math.ceil(len(members) * scale))])
    return chosen


def extra_shuffle_copies(row, copies_total, seed):
    """Add copies copy_index 1..copies_total-1 on top of the existing copy (copy_index 0), continuing the permutation sequence of build_cmb_train_sft."""
    options = row["options"]
    if len(row["gold"]) < 2 or any(not text for _, text in options) or not is_shuffle_safe(options):
        return []
    key = normalize_stem(row["question"])
    used = [make_permutation(len(options), seed, key, 0)]
    new_rows = []
    for copy_index in range(1, copies_total):
        permutation = make_permutation(len(options), seed, key, copy_index, forbidden=used)
        used.append(permutation)
        messages = build_prompt_messages(row["question"], permute_options(options, permutation), MODE_DIRECT)
        messages.append({"role": "assistant", "content": remap_answer(row["gold"], permutation)})
        sample_id = sha256_text("\n".join(m["content"] for m in messages if m["role"] == "user"))
        new_rows.append({**row, "sample_id": sample_id, "record": {"messages": messages}, "variant": f"shuffled_extra{copy_index}"})
    return new_rows


def load_audit(audit_dir):
    """Returns (tie_hashes, majority_keep: {hash: content_tuple}, structural_hashes: {hash: reason})."""
    audit = Path(audit_dir)
    ties, keep, structural = set(), {}, {}
    with open(audit / "resolutions.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record["resolution"] == "tie":
                    ties.add(record["hash"])
                else:
                    keep[record["hash"]] = tuple(record["keep_content"])
    with open(audit / "structural_flags.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("hash") and record["reason"] in ("figure_reference", "truncated_stem"):
                    structural[record["hash"]] = record["reason"]
    return ties, keep, structural


def apply_clean(rows, audit):
    ties, keep, structural = audit
    kept, dropped = [], Counter()
    for row in rows:
        if row["source"] != "cmb_train":
            kept.append(row)
            continue
        if row["unit"] in ties:
            dropped["tie_conflict"] += 1
            continue
        if row["unit"] in keep and content_answer(row) != keep[row["unit"]]:
            dropped["minority_label"] += 1
            continue
        if row["unit"] in structural:
            dropped[structural[row["unit"]]] += 1
            continue
        kept.append(row)
    return kept, dropped


def with_type_hint(row):
    """Re-render the user content with a question-type hint aligned with the official evaluation prompt; sample_id changes accordingly."""
    hint = type_hint_for(row["gold"], row.get("question_type") or None)
    messages = build_prompt_messages(row["question"], row["options"], MODE_DIRECT, hint=hint)
    messages.append({"role": "assistant", "content": row["gold"]})
    sample_id = sha256_text("\n".join(m["content"] for m in messages if m["role"] == "user"))
    return {**row, "sample_id": sample_id, "record": {"messages": messages}}


def build_variant(rows, cmb_units, scale, copies, clean_audit, seed, type_hint=False):
    chosen = select_units(cmb_units, scale, seed)
    out = []
    for row in rows:
        if row["source"] != "cmb_train":
            out.append(row)
            continue
        if row["unit"] not in chosen:
            continue
        if row["variant"] == "shuffled" and copies == 0:
            continue
        out.append(row)
    if copies > 1:
        extras = []
        for row in out:
            if row["source"] == "cmb_train" and row["variant"] == "original":
                extras.extend(extra_shuffle_copies(row, copies, seed))
        out.extend(extras)
    dropped = Counter()
    if clean_audit is not None:
        out, dropped = apply_clean(out, clean_audit)
    if type_hint:
        out = [with_type_hint(row) for row in out]
    seen, unique = set(), []
    for row in out:
        if row["sample_id"] in seen:
            continue
        seen.add(row["sample_id"])
        unique.append(row)
    unique.sort(key=lambda r: stable_fraction(seed, "order", r["sample_id"]))
    return unique, chosen, dropped


def summarize(rows, chosen_units):
    return {"rows": len(rows), "by_source": dict(Counter(r["source"] for r in rows)),
            "by_variant": dict(Counter(r["variant"] for r in rows)),
            "cmb_units": len(chosen_units),
            "multi_rows": sum(1 for r in rows if len(r["gold"]) > 1),
            "by_exam_type": dict(Counter(r["exam_type"] for r in rows if r["source"] == "cmb_train"))}


def write_variant(out_dir, name, rows, chosen, dropped, spec, input_hashes, seed):
    target = Path(out_dir) / name
    target.mkdir(parents=True, exist_ok=True)
    with open(target / "train.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row["record"], ensure_ascii=False) + "\n")
    manifest = {"name": name, **spec, "seed": seed, "inputs": input_hashes, "dropped_by_clean": dict(dropped), **summarize(rows, chosen)}
    with open(target / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--audit-dir", help="label-audit output directory; when given, a cleaned variant is built")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scales", default="0,0.25,0.5", help="CMB-train fractions for the scale curve (100%% is the Stage 3 model itself)")
    parser.add_argument("--dose-scale", type=float, default=0.5)
    parser.add_argument("--dose-copies", default="0,3", help="shuffled copies per multi-answer question (1 copy is the scale-curve point at --dose-scale)")
    parser.add_argument("--clean-scale", type=float, default=0.5)
    parser.add_argument("--skip-defaults", action="store_true", help="skip the default scale / dose / clean variants; build only --extra-variant")
    parser.add_argument("--extra-variant", action="append", default=[],
                        help="custom variant name:scale:copies:clean:hint, e.g. tagged_100:1.0:0:1:1 (100%% CMB, no shuffled copies, cleaned, type hint)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_rows(args.train_file, args.metadata)
    cmb_units = defaultdict(list)
    for row in rows:
        if row["source"] == "cmb_train":
            cmb_units[row["unit"]].append(row)
    input_hashes = {args.train_file: sha256_file(args.train_file), args.metadata: sha256_file(args.metadata)}
    audit = load_audit(args.audit_dir) if args.audit_dir else None
    plan = []
    for spec in args.extra_variant:
        name, scale, copies, clean, hint = spec.split(":")
        scale, copies, clean, hint = float(scale), int(copies), clean == "1", hint == "1"
        if clean and audit is None:
            raise ValueError(f"{name}: clean=1 requires --audit-dir")
        built, chosen, dropped = build_variant(rows, cmb_units, scale, copies, audit if clean else None, args.seed, type_hint=hint)
        plan.append(write_variant(args.output_dir, name, built, chosen, dropped,
                                  {"scale": scale, "copies": copies, "clean": clean, "type_hint": hint}, input_hashes, args.seed))
    if args.skip_defaults:
        with open(Path(args.output_dir) / f"plan_extra.json", "w", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
        for m in plan:
            print(f"{m['name']}: rows {m['rows']} | cmb units {m['cmb_units']} | multi rows {m['multi_rows']} | variants {m['by_variant']} | dropped {m['dropped_by_clean']}")
        return 0
    for scale in [float(x) for x in args.scales.split(",") if x]:
        name = f"scale_{int(round(scale * 100)):03d}"
        built, chosen, dropped = build_variant(rows, cmb_units, scale, 1, None, args.seed)
        plan.append(write_variant(args.output_dir, name, built, chosen, dropped, {"scale": scale, "copies": 1, "clean": False}, input_hashes, args.seed))
    for copies in [int(x) for x in args.dose_copies.split(",") if x]:
        name = f"dose_{int(round(args.dose_scale * 100)):03d}_copies{copies}"
        built, chosen, dropped = build_variant(rows, cmb_units, args.dose_scale, copies, None, args.seed)
        plan.append(write_variant(args.output_dir, name, built, chosen, dropped, {"scale": args.dose_scale, "copies": copies, "clean": False}, input_hashes, args.seed))
    if audit is not None:
        name = f"clean_{int(round(args.clean_scale * 100)):03d}"
        built, chosen, dropped = build_variant(rows, cmb_units, args.clean_scale, 1, audit, args.seed)
        plan.append(write_variant(args.output_dir, name, built, chosen, dropped, {"scale": args.clean_scale, "copies": 1, "clean": True}, input_hashes, args.seed))
    with open(Path(args.output_dir) / "plan.json", "w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    for m in plan:
        print(f"{m['name']}: rows {m['rows']} | cmb units {m['cmb_units']} | multi rows {m['multi_rows']} | variants {m['by_variant']} | dropped {m['dropped_by_clean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
