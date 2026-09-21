#!/usr/bin/env python3
"""CMExam official splits -> direct-answer SFT data (chat `messages` JSONL, target = answer letters only).

Cleaning rules: options must be a contiguous A.. sequence with distinct non-empty texts; the answer must be
letters that exist among the options; a question is written once (NFKC + whitespace-insensitive key);
train questions that also occur in val or test, and val questions that also occur in test, are dropped so
the three splits stay disjoint. Overlap with CMB-test is removed afterwards by
`decontaminate_cmexam_against_cmb.py`.

    python scripts/build_cmexam_direct_sft.py --input-dir data/CMExam/data --output-dir data
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
    from .cmexam_prompts import MODE_DIRECT, build_prompt_messages
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cmexam_prompts import MODE_DIRECT, build_prompt_messages

TRAIN_FILENAME = "cmexam_sft_train.jsonl"
VALIDATION_FILENAME = "cmexam_sft_validation.jsonl"
REPORT_FILENAME = "cmexam_sft_report.json"


OPTION_LINE_PATTERN = re.compile(
    r"^\s*([A-Z])(?:[.、．:：]\s*|\s+)(\S.*?)\s*$"
)


ANSWER_PATTERN = re.compile(r"^[A-Z]+$")


VALID_OPTION_SEQUENCES = {
    tuple("ABC"),
    tuple("ABCD"),
    tuple("ABCDE"),
}


MAX_REJECTION_EXAMPLES = 5


def configure_csv_field_limit():
    """Raise the CSV field-size limit as far as the platform's C long allows."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalize_question(text):
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", "", normalized)


def normalize_text(text):
    return " ".join((text or "").split())


def parse_options(raw_options):
    if not isinstance(raw_options, str) or not raw_options.strip():
        raise ValueError("empty_options")

    options = []
    for raw_line in raw_options.splitlines():
        if not raw_line.strip():
            continue
        match = OPTION_LINE_PATTERN.fullmatch(raw_line)
        if match is None:
            raise ValueError("invalid_option_line")
        letter, content = match.groups()
        options.append((letter, normalize_text(content)))

    labels = tuple(letter for letter, _ in options)
    if labels not in VALID_OPTION_SEQUENCES:
        raise ValueError("invalid_option_sequence")
    if any(not content for _, content in options):
        raise ValueError("empty_option_content")
    if len({content for _, content in options}) != len(options):
        raise ValueError("duplicate_option_content")
    return options


def normalize_answer(raw_answer, option_labels):
    answer = normalize_text(raw_answer).upper().replace(" ", "")
    if not answer or ANSWER_PATTERN.fullmatch(answer) is None:
        raise ValueError("invalid_answer_format")
    if any(letter not in option_labels for letter in answer):
        raise ValueError("answer_not_in_options")
    return "".join(sorted(set(answer)))


def validate_question_options_answer(row):
    question = normalize_text(row.get("Question"))
    if not question:
        raise ValueError("empty_question")

    options = parse_options(row.get("Options"))
    option_labels = {letter for letter, _ in options}
    answer = normalize_answer(row.get("Answer"), option_labels)
    return question, options, answer


def load_normalized_questions(path):
    questions = set()
    input_rows = 0
    empty_questions = 0
    with path.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        require_columns(reader, path)
        for row in reader:
            input_rows += 1
            question_key = normalize_question(row.get("Question"))
            if question_key:
                questions.add(question_key)
            else:
                empty_questions += 1
    return questions, input_rows, empty_questions


def load_eligible_question_keys(path, blocked_questions):
    """Keys of structurally valid questions in a split; used to keep the splits disjoint."""
    questions = set()
    with path.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        require_columns(reader, path)
        for row in reader:
            question_key = normalize_question(row.get("Question"))
            if not question_key or question_key in blocked_questions:
                continue
            if question_key in questions:
                continue
            try:
                validate_question_options_answer(row)
            except ValueError:
                continue
            questions.add(question_key)
    return questions


def require_columns(reader, path):
    required = {"Question", "Options", "Answer", "Explanation"}
    actual = set(reader.fieldnames or [])
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def record_rejection(stats, examples, reason, line_number, row):
    stats[reason] += 1
    if len(examples) < MAX_REJECTION_EXAMPLES:
        examples.append(
            {
                "line_number": line_number,
                "reason": reason,
                "question": normalize_text(row.get("Question"))[:200],
            }
        )


def convert_split(input_path, output_path, blocked_questions, max_samples=0):
    stats, examples, accepted = Counter(), [], set()
    hasher = hashlib.sha256()
    with input_path.open("r", encoding="utf-8-sig", newline="") as fin, output_path.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        require_columns(reader, input_path)
        for line_number, row in enumerate(reader, start=2):
            if max_samples and stats["written"] >= max_samples:
                break
            stats["input_rows_read"] += 1
            key = normalize_question(row.get("Question"))
            if not key:
                record_rejection(stats, examples, "empty_question", line_number, row)
                continue
            if key in blocked_questions:
                record_rejection(stats, examples, "overlap_with_later_split", line_number, row)
                continue
            if key in accepted:
                record_rejection(stats, examples, "duplicate_question", line_number, row)
                continue
            try:
                question, options, answer = validate_question_options_answer(row)
            except ValueError as error:
                record_rejection(stats, examples, str(error), line_number, row)
                continue
            messages = build_prompt_messages(question, options, MODE_DIRECT)
            messages.append({"role": "assistant", "content": answer})
            line = json.dumps({"messages": messages}, ensure_ascii=False) + "\n"
            fout.write(line)
            hasher.update(line.encode("utf-8"))
            accepted.add(key)
            stats["written"] += 1
    rejection_counts = {k: v for k, v in sorted(stats.items()) if k not in ("input_rows_read", "written", "rejected")}
    return {"input_rows_read": stats["input_rows_read"], "written": stats["written"], "rejected": stats["rejected"],
            "rejection_counts": rejection_counts, "rejection_examples": examples, "sha256": hasher.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="directory with CMExam train.csv, val.csv, test_with_annotations.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    args = parser.parse_args(argv)
    configure_csv_field_limit()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_questions, test_rows, test_empty = load_normalized_questions(input_dir / "test_with_annotations.csv")
    validation_keys = load_eligible_question_keys(input_dir / "val.csv", blocked_questions=test_questions)
    validation = convert_split(input_dir / "val.csv", output_dir / VALIDATION_FILENAME, test_questions, args.max_validation_samples)
    train = convert_split(input_dir / "train.csv", output_dir / TRAIN_FILENAME, test_questions | validation_keys, args.max_train_samples)
    report = {"format": "messages_only", "test_reference": {"input_rows": test_rows, "unique_normalized_questions": len(test_questions), "empty_questions": test_empty},
              "counts": {"train": train, "validation": validation}}
    with open(output_dir / REPORT_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"train_written": train["written"], "validation_written": validation["written"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
