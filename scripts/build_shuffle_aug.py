#!/usr/bin/env python3
"""Option-shuffle augmentation for direct-answer SFT data.

For every question whose options do not refer to each other ("all of the above", "A and B", ...),
add copies with the options permuted and the answer letters remapped. The permutation is a stable
hash of (seed, sample_id, copy_index), so the output is fully reproducible. Questions that are not
shuffle-safe are kept once, unpermuted.

    python scripts/build_shuffle_aug.py --source-file data/cmexam_sft_train.jsonl \
        --output-file data/cmexam_sft_train_shuffle_aug.jsonl --copies 1 --seed 42
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from .cmexam_prompts import (MODE_DIRECT, build_prompt_messages, detect_slices, make_permutation, parse_user_content,
                                 permute_options, remap_answer, stable_fraction)
    from .data_utils import parse_messages_record
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cmexam_prompts import (MODE_DIRECT, build_prompt_messages, detect_slices, make_permutation, parse_user_content,
                                permute_options, remap_answer, stable_fraction)
    from data_utils import parse_messages_record


def make_record(question, options, answer):
    messages = build_prompt_messages(question, options, MODE_DIRECT)
    messages.append({"role": "assistant", "content": answer})
    record = {"messages": messages}
    converted = parse_messages_record(record)
    if converted["answer"] != answer:
        raise ValueError("rebuilt record does not round-trip to the expected answer")
    return record, converted["sample_id"]


def build(source_file, copies=1, seed=42):
    stats, outputs, seen = Counter(), [], set()
    with open(source_file, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            example = parse_messages_record(json.loads(line), str(source_file), line_number)
            stats["source_rows"] += 1
            user_content = "\n".join(m["content"] for m in example["prompt"] if m["role"] == "user")
            try:
                parsed = parse_user_content(user_content)
            except ValueError:
                stats["unparsed_user_content"] += 1
                continue
            question, options, answer = parsed["question"], parsed["options"], example["answer"]
            slices = detect_slices(question, options, answer)
            if not slices["shuffle_safe"]:
                stats["shuffle_unsafe_questions"] += 1
            variants, used = [None], []
            for copy_index in range(copies if slices["shuffle_safe"] else 0):
                permutation = make_permutation(len(options), seed, example["sample_id"], copy_index, forbidden=used)
                used.append(permutation)
                variants.append(permutation)
            for permutation in variants:
                if permutation is None:
                    variant_options, variant_answer = options, answer
                else:
                    variant_options, variant_answer = permute_options(options, permutation), remap_answer(answer, permutation)
                record, sample_id = make_record(question, variant_options, variant_answer)
                if sample_id in seen:
                    stats["duplicate_prompt_skipped"] += 1
                    continue
                seen.add(sample_id)
                stats["shuffled" if permutation is not None else "original"] += 1
                outputs.append((stable_fraction(seed, "order", sample_id), record))
    outputs.sort(key=lambda item: item[0])
    return [record for _, record in outputs], dict(stats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--copies", type=int, default=1, help="shuffled copies per shuffle-safe question")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    records, stats = build(args.source_file, args.copies, args.seed)
    target = Path(args.output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(records), **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
