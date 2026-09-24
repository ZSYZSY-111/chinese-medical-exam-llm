#!/usr/bin/env python3
"""Sample an RL prompt pool from the rationale-style SFT data (legacy rationale prompt + "解析：…\\n答案：X").

RL uses only the prompt and the answer (train_grpo strips the assistant turn), so rows are copied unchanged.
Any sample_id (= sha256 of the user content, the same id the annotation harness uses) found in the --exclude files
is kept out of the pool, so RL sees questions the SFT never trained on. --limit rows are drawn by stable hash.
  python scripts/build_cot_rl_pool.py --input <rationale SFT train file> --exclude data/sentence_score/sft/train.jsonl data/sentence_score/sft/dev.jsonl \\
    --limit 4000 --output data/rl_cot_pool/train.jsonl
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_fraction(*parts):
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def sample_id_of(messages):
    return sha256_text("\n".join(m["content"] for m in messages if m.get("role") == "user"))


def load_ids(paths):
    ids = set()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    ids.add(row.get("sample_id") or sample_id_of(row["messages"]))
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--max-user-chars", type=int, default=0, help="drop questions whose user content exceeds this many characters (one long stem pads the whole generation batch and spikes memory); 0 = no limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    excluded = load_ids(args.exclude)
    rows, counts = [], {"loaded": 0, "excluded": 0, "duplicate": 0}
    seen = set()
    with open(args.input, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts["loaded"] += 1
            if args.max_user_chars and sum(len(m["content"]) for m in row["messages"] if m.get("role") == "user") > args.max_user_chars:
                counts["too_long"] = counts.get("too_long", 0) + 1
                continue
            sid = sample_id_of(row["messages"])
            if sid in excluded:
                counts["excluded"] += 1
                continue
            if sid in seen:
                counts["duplicate"] += 1
                continue
            seen.add(sid)
            rows.append((stable_fraction(args.seed, "cot_rl_pool", sid), row))
    rows.sort(key=lambda pair: pair[0])
    chosen = [row for _, row in rows[:args.limit]] if args.limit else [row for _, row in rows]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
    counts.update(candidates=len(rows), written=len(chosen), seed=args.seed, excluded_ids=len(excluded))
    (out.parent / (out.stem + "_manifest.json")).write_text(json.dumps(counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
