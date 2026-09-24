#!/usr/bin/env python3
"""Split the harness's segment-score export (data_sentence_score.jsonl) into a training set and a held-out set, with a report.

The held-out set only measures how well the auxiliary head predicts the scores; it is never trained on. The split
is a stable hash of sample_id, so it is reproducible. Teacher scores are an auxiliary regression target, not answer
labels: the answer stays the official one.

  python scripts/build_sentence_score_sft.py --input data/sentence_score/run_10k/data_sentence_score.jsonl \
    --output-dir data/sentence_score/sft --dev-size 500
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def stable_fraction(*parts):
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def validate_row(row):
    """An export row must be self-consistent: contiguous spans covering the whole rationale, scores 1–5, and the rationale text present in the assistant turn."""
    explanation, sentences = row["explanation"], row["sentences"]
    if not sentences:
        return "no_sentences"
    if sentences[0]["start"] != 0 or sentences[-1]["end"] != len(explanation):
        return "spans_do_not_cover"
    for left, right in zip(sentences, sentences[1:]):
        if left["end"] != right["start"]:
            return "spans_not_contiguous"
    if any(not 1 <= sentence["score"] <= 5 for sentence in sentences):
        return "score_out_of_range"
    if explanation not in row["messages"][-1]["content"]:
        return "explanation_not_in_target"
    if "\ufffd" in explanation or any("\ufffd" in m["content"] for m in row["messages"]):
        return "garbled_text"   # U+FFFD in the source text; present in the original CMExam train.csv, not introduced here
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rows, problems, seen = [], Counter(), set()
    with open(args.input, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            problem = validate_row(row)
            if problem is None and row["sample_id"] in seen:
                problem = "duplicate_sample_id"
            if problem:
                problems[problem] += 1
                continue
            seen.add(row["sample_id"])
            rows.append(row)
    if len(rows) <= args.dev_size:
        print(f"{len(rows)} valid rows are not enough to hold out {args.dev_size}", file=sys.stderr)
        return 1
    rows.sort(key=lambda row: stable_fraction(args.seed, "sentence_score_split", row["sample_id"]))
    dev, train = rows[:args.dev_size], rows[args.dev_size:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("dev", dev)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for row in part:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    scores = Counter(sentence["score"] for row in rows for sentence in row["sentences"])
    per_item = Counter(len(row["sentences"]) for row in rows)
    flat = sum(1 for row in rows if len({sentence["score"] for sentence in row["sentences"]}) == 1)
    report = {"input": args.input, "valid_rows": len(rows), "dropped": dict(problems), "train": len(train), "dev": len(dev),
              "sentences": sum(scores.values()), "score_distribution": {str(k): scores[k] for k in sorted(scores)},
              "sentences_per_item": {str(k): per_item[k] for k in sorted(per_item)},
              "rows_with_identical_scores": flat, "seed": args.seed}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
