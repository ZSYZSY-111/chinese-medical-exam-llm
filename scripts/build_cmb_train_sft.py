"""CMB-Exam train split -> direct-answer SFT data, strictly decontaminated against CMB-test, CMB-val and the CMExam
validation and test splits.

Decontamination rules (every decision is written to the report and the audit files):
  1. exact match of the normalised stem (NFKC, lower-case, whitespace and punctuation removed) -> removed
  2. match of the stem + option-set hash -> removed
  3. near duplicate by character 5-gram MinHash (128 permutations): Jaccard >= --near-threshold with any reference
     stem -> removed; rows in [--borderline-threshold, --near-threshold) are kept and written to a borderline file
     so they can be excluded at evaluation time
  4. rows are deduplicated inside the training split by stem + option set, then a validation split is cut by stable hash
Output is messages-only JSONL in the same template as the CMExam direct-answer data; multi-answer questions can get
shuffled copies, and the CMExam training file can be merged in.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

try:
    from .cmexam_prompts import (
        MODE_DIRECT,
        build_prompt_messages,
        is_shuffle_safe,
        make_permutation,
        permute_options,
        remap_answer,
        stable_fraction,
    )
except ImportError:
    from cmexam_prompts import (
        MODE_DIRECT,
        build_prompt_messages,
        is_shuffle_safe,
        make_permutation,
        permute_options,
        remap_answer,
        stable_fraction,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="CMB-train -> decontaminated direct-answer SFT data.")
    parser.add_argument("--cmb-train", required=True)
    parser.add_argument("--cmb-test", required=True)
    parser.add_argument("--cmb-val", required=True)
    parser.add_argument("--cmexam-csv", action="append", default=[], help="CMExam test/val CSV; only the Question column is read, as a reference.")
    parser.add_argument("--merge-direct-file", default=None, help="CMExam direct-answer messages JSONL to merge into the training file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--internal-val-size", type=int, default=3000)
    parser.add_argument("--near-threshold", type=float, default=0.7)
    parser.add_argument("--borderline-threshold", type=float, default=0.5)
    parser.add_argument("--min-ngram-chars", type=int, default=8, help="stems shorter than this are only matched exactly.")
    parser.add_argument("--multi-shuffle-copies", type=int, default=1)
    parser.add_argument("--merge-multi-shuffle-copies", type=int, default=2, help="shuffled copies for multi-answer questions of the merged CMExam file.")
    parser.add_argument("--drop-question-types", default="C型选择题")
    parser.add_argument("--max-chars", type=int, default=900, help="skip rows whose stem + options exceed this many characters.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def normalize_stem(text):
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(ch for ch in normalized if not (ch.isspace() or unicodedata.category(ch).startswith("P")))


def option_signature(options):
    return "|".join(sorted(normalize_stem(text) for _, text in options))


def stem_option_hash(stem_key, options):
    return hashlib.sha256((stem_key + "||" + option_signature(options)).encode("utf-8")).hexdigest()


def shingles(text, n=5):
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def load_cmb_rows(path):
    return json.load(open(path, encoding="utf-8"))


def parse_cmb_row(row):
    """Validate and convert one CMB record; raises ValueError(reason) when it is malformed."""
    question = " ".join((row.get("question") or "").split())
    if not question:
        raise ValueError("empty_question")
    option_map = row.get("option") or {}
    if not isinstance(option_map, dict) or not option_map:
        raise ValueError("empty_options")
    letters = sorted(option_map)
    expected = [chr(ord("A") + i) for i in range(len(letters))]
    if letters != expected or len(letters) < 3:
        raise ValueError("invalid_option_letters")
    options = [(letter, " ".join(str(option_map[letter]).split())) for letter in letters]
    # CMB stores many four-option questions with an empty "E"; the evaluator renders it as "E. ", so trailing empty options are kept and rendered identically.
    trailing_empty = []
    while options and not options[-1][1]:
        trailing_empty.append(options.pop())
    if any(not text for _, text in options):
        raise ValueError("empty_option_text")
    if len(options) < 3:
        raise ValueError("too_few_options")
    if len({text for _, text in options}) != len(options):
        raise ValueError("duplicate_option_text")
    answer = re.sub(r"[\s,，、/]", "", str(row.get("answer") or "")).upper()
    if not answer or not re.fullmatch(r"[A-Z]+", answer) or len(set(answer)) != len(answer):
        raise ValueError("invalid_answer")
    non_empty_letters = {letter for letter, _ in options}
    if any(letter not in option_map for letter in answer):
        raise ValueError("answer_not_in_options")
    if any(letter not in non_empty_letters for letter in answer):
        raise ValueError("answer_on_empty_option")
    options = options + list(reversed(trailing_empty))
    return {
        "question": question,
        "options": options,
        "has_empty_option": bool(trailing_empty),
        "answer": "".join(sorted(answer)),
        "exam_type": row.get("exam_type", ""),
        "exam_class": row.get("exam_class", ""),
        "question_type": row.get("question_type", ""),
    }


def load_reference_stems(args):
    """Reference stems from CMB-test, CMB-val and the CMExam CSVs. Returns {stem_key: [(source, id)]} and the set of option hashes."""
    refs = {}
    ref_hashes = set()
    def add(source, index, question, options=None):
        key = normalize_stem(question)
        if not key:
            return
        refs.setdefault(key, []).append((source, index))
        if options:
            ref_hashes.add(stem_option_hash(key, options))
    for source, path in (("cmb_test", args.cmb_test), ("cmb_val", args.cmb_val)):
        for index, row in enumerate(load_cmb_rows(path)):
            option_map = row.get("option") or {}
            options = [(k, str(v)) for k, v in sorted(option_map.items())] if isinstance(option_map, dict) else None
            add(source, row.get("id", index), row.get("question", ""), options)
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    for path in args.cmexam_csv:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                add("cmexam_" + Path(path).stem, index, row.get("Question", ""))
    return refs, ref_hashes


def near_duplicate_scores(train_keys, ref_keys, threshold, min_chars, num_perm=128):
    """Returns {train_index: (best_jaccard, ref_key)}; the exact Jaccard is computed only for LSH candidate pairs."""
    ref_list = [k for k in ref_keys if len(k) >= min_chars]
    ref_shingles = {k: shingles(k) for k in ref_list}
    scores = {}
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        for index, key in enumerate(train_keys):
            if len(key) < min_chars:
                continue
            sh = shingles(key)
            best = max(((jaccard(sh, ref_shingles[r]), r) for r in ref_list), default=(0.0, None))
            if best[0] >= threshold:
                scores[index] = best
        return scores, "bruteforce"
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for key in ref_list:
        m = MinHash(num_perm=num_perm)
        for s in ref_shingles[key]:
            m.update(s.encode("utf-8"))
        lsh.insert(key, m)
    for index, key in enumerate(train_keys):
        if len(key) < min_chars:
            continue
        m = MinHash(num_perm=num_perm)
        sh = shingles(key)
        for s in sh:
            m.update(s.encode("utf-8"))
        candidates = lsh.query(m)
        if not candidates:
            continue
        best = max((jaccard(sh, ref_shingles[c]), c) for c in candidates)
        if best[0] >= threshold:
            scores[index] = best
    return scores, "minhash_lsh"


def build(args):
    refs, ref_hashes = load_reference_stems(args)
    drop_types = {t for t in args.drop_question_types.split(",") if t}
    raw = load_cmb_rows(args.cmb_train)
    if args.limit:
        raw = raw[:args.limit]
    stats = Counter()
    removed = []
    parsed_rows = []
    seen_hashes = set()
    for index, row in enumerate(raw):
        stats["input_rows"] += 1
        if row.get("question_type") in drop_types:
            stats["dropped_question_type"] += 1
            continue
        try:
            parsed = parse_cmb_row(row)
        except ValueError as error:
            stats[f"invalid_{error}"] += 1
            continue
        if len(parsed["question"]) + sum(len(t) for _, t in parsed["options"]) > args.max_chars:
            stats["dropped_too_long"] += 1
            continue
        key = normalize_stem(parsed["question"])
        h = stem_option_hash(key, parsed["options"])
        if h in seen_hashes:
            stats["duplicate_within_train"] += 1
            continue
        seen_hashes.add(h)
        if key in refs:
            stats["removed_exact_stem"] += 1
            removed.append({"index": index, "reason": "exact_stem", "refs": refs[key][:3], "question": parsed["question"][:80]})
            continue
        if h in ref_hashes:
            stats["removed_stem_option_hash"] += 1
            removed.append({"index": index, "reason": "stem_option_hash", "question": parsed["question"][:80]})
            continue
        parsed["index"] = index
        parsed["key"] = key
        parsed_rows.append(parsed)

    keys = [p["key"] for p in parsed_rows]
    scores, method = near_duplicate_scores(keys, set(refs), args.borderline_threshold, args.min_ngram_chars)
    kept, borderline = [], []
    for position, parsed in enumerate(parsed_rows):
        score = scores.get(position)
        if score and score[0] >= args.near_threshold:
            stats["removed_near_duplicate"] += 1
            removed.append({"index": parsed["index"], "reason": "near_duplicate", "jaccard": round(score[0], 3),
                            "ref": refs.get(score[1], [])[:2], "question": parsed["question"][:80]})
            continue
        if score:
            stats["borderline_kept"] += 1
            borderline.append({"index": parsed["index"], "jaccard": round(score[0], 3), "ref": refs.get(score[1], [])[:2],
                               "question": parsed["question"][:80]})
        kept.append(parsed)

    # validation split: after deduplication, by stable hash
    kept.sort(key=lambda p: stable_fraction(args.seed, "cmb_split", p["key"]))
    internal_val = kept[:args.internal_val_size]
    train_rows = kept[args.internal_val_size:]
    stats["internal_val"] = len(internal_val)
    stats["cmb_train_kept"] = len(train_rows)

    def to_record(parsed, options, answer):
        messages = build_prompt_messages(parsed["question"], options, MODE_DIRECT)
        messages.append({"role": "assistant", "content": answer})
        return {"messages": messages}

    outputs, metadata = [], []
    def emit(record, meta):
        sample_id = hashlib.sha256("\n".join(m["content"] for m in record["messages"][:-1] if m["role"] == "user").encode("utf-8")).hexdigest()
        outputs.append((stable_fraction(args.seed, "order", sample_id), record, {"sample_id": sample_id, **meta}))

    type_counts = Counter()
    for parsed in train_rows:
        type_counts[(parsed["exam_type"], parsed["question_type"])] += 1
        emit(to_record(parsed, parsed["options"], parsed["answer"]),
             {"source": "cmb_train", "variant": "original", "exam_type": parsed["exam_type"], "question_type": parsed["question_type"]})
        if (len(parsed["answer"]) > 1 and args.multi_shuffle_copies and not parsed["has_empty_option"]
                and is_shuffle_safe(parsed["options"])):
            used = []
            for copy_index in range(args.multi_shuffle_copies):
                permutation = make_permutation(len(parsed["options"]), args.seed, parsed["key"], copy_index, forbidden=used)
                used.append(permutation)
                emit(to_record(parsed, permute_options(parsed["options"], permutation), remap_answer(parsed["answer"], permutation)),
                     {"source": "cmb_train", "variant": "shuffled", "exam_type": parsed["exam_type"], "question_type": parsed["question_type"]})
                stats["cmb_multi_shuffled"] += 1

    if args.merge_direct_file:
        try:
            from .cmexam_prompts import parse_user_content
        except ImportError:
            from cmexam_prompts import parse_user_content
        with open(args.merge_direct_file, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                stats["merged_direct_rows"] += 1
                emit(record, {"source": "cmexam", "variant": "original"})
                answer = record["messages"][-1]["content"].strip()
                if len(answer) > 1 and args.merge_multi_shuffle_copies:
                    parsed_user = parse_user_content(record["messages"][1]["content"])
                    if not is_shuffle_safe(parsed_user["options"]):
                        continue
                    used = []
                    key = normalize_stem(parsed_user["question"])
                    for copy_index in range(args.merge_multi_shuffle_copies):
                        permutation = make_permutation(len(parsed_user["options"]), args.seed, key, copy_index, forbidden=used)
                        used.append(permutation)
                        messages = build_prompt_messages(parsed_user["question"], permute_options(parsed_user["options"], permutation), MODE_DIRECT)
                        messages.append({"role": "assistant", "content": remap_answer(answer, permutation)})
                        emit({"messages": messages}, {"source": "cmexam", "variant": "shuffled"})
                        stats["cmexam_multi_shuffled"] += 1

    seen_ids, final = set(), []
    for order, record, meta in sorted(outputs, key=lambda item: item[0]):
        if meta["sample_id"] in seen_ids:
            stats["duplicate_prompt_skipped"] += 1
            continue
        seen_ids.add(meta["sample_id"])
        final.append((record, meta))

    val_records = [to_record(p, p["options"], p["answer"]) for p in internal_val]
    stats["cmb_train_kept_with_empty_option"] = sum(1 for p in train_rows if p["has_empty_option"])
    report = {
        "stats": dict(stats),
        "near_duplicate_method": method,
        "thresholds": {"near": args.near_threshold, "borderline": args.borderline_threshold, "min_ngram_chars": args.min_ngram_chars},
        "reference_sets": {"cmb_test": args.cmb_test, "cmb_val": args.cmb_val, "cmexam_csv": args.cmexam_csv, "reference_stems": len(refs)},
        "cmb_train_kept_by_exam_type": dict(Counter(p["exam_type"] for p in train_rows)),
        "cmb_train_kept_by_question_type": dict(Counter(p["question_type"] for p in train_rows)),
        "internal_val_by_exam_type": dict(Counter(p["exam_type"] for p in internal_val)),
        "final_rows": len(final),
        "final_by_source": dict(Counter(m["source"] + "/" + m["variant"] for _, m in final)),
        "removed_examples": removed[:30],
        "borderline_count": len(borderline),
        "config": vars(args),
    }
    return final, val_records, report, removed, borderline


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    final, val_records, report, removed, borderline = build(args)
    print(json.dumps({k: v for k, v in report.items() if k != "removed_examples"}, ensure_ascii=False, indent=2))
    if args.check_only:
        print("check-only: no files written.")
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / name for name in ("cmb_sft_train.jsonl", "cmb_internal_val.jsonl", "cmb_sft_metadata.jsonl",
                                             "cmb_sft_report.json", "removed_audit.jsonl", "borderline_audit.jsonl")}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("output exists; choose another directory or pass --overwrite:\n" + "\n".join(existing))
    write_jsonl(paths["cmb_sft_train.jsonl"], [r for r, _ in final])
    write_jsonl(paths["cmb_sft_metadata.jsonl"], [m for _, m in final])
    write_jsonl(paths["cmb_internal_val.jsonl"], val_records)
    write_jsonl(paths["removed_audit.jsonl"], removed)
    write_jsonl(paths["borderline_audit.jsonl"], borderline)
    with paths["cmb_sft_report.json"].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"wrote {paths['cmb_sft_train.jsonl']} ({len(final)} rows); validation split {len(val_records)} rows")


if __name__ == "__main__":
    main()
