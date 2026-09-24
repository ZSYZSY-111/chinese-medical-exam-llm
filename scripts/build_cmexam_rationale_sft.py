#!/usr/bin/env python3
"""Build rationale-then-answer SFT data from the official CMExam splits (question, options, answer, explanation).

The prompt is the one all rationale-style experiments in this repository use (cmexam_prompts.LEGACY_EXPLAIN_SYSTEM_PROMPT);
the assistant target is "解析：<official explanation>\n答案：<letters>". Questions without a usable explanation are
skipped, validation questions are kept out of train, and only normalised question text is read from the test split,
for decontamination.
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


DEFAULT_INPUT_DIR = "data/CMExam/data"
DEFAULT_OUTPUT_DIR = "data/cmexam_rationale"
TRAIN_FILENAME = "cmexam_rationale_sft_train.jsonl"
VALIDATION_FILENAME = "cmexam_rationale_sft_validation.jsonl"
REPORT_FILENAME = "cmexam_rationale_sft_report.json"

SYSTEM_PROMPT = (
    "你是一名医学考试答题助手。回答中国医学考试选择题。"
    "请先给出医学解析，再在最后给出正确选项字母；"
    "多选题按字母顺序连续输出。严格使用以下格式：\n"
    "解析：具体解析\n答案：X"
)

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
PLACEHOLDER_EXPLANATIONS = {"请等待更新"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clean the official CMExam splits and convert them to rationale-then-answer "
            "messages JSONL (rationale: the official explanation, target: 解析：…\\n答案：X)."
        )
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="cap on training rows; 0 = all",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=0,
        help="cap on validation rows; 0 = all",
    )
    return parser.parse_args()


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


def normalize_explanation(raw_explanation):
    if not isinstance(raw_explanation, str):
        raise ValueError("empty_explanation")

    explanation = raw_explanation.replace("\r\n", "\n").replace("\r", "\n")
    explanation = "\n".join(
        line.rstrip() for line in explanation.splitlines()
    ).strip()
    if not explanation:
        raise ValueError("empty_explanation")
    if explanation in PLACEHOLDER_EXPLANATIONS:
        raise ValueError("placeholder_explanation")
    return explanation


def validate_and_convert(row):
    question, options, answer = validate_question_options_answer(row)
    explanation = normalize_explanation(row.get("Explanation"))

    option_text = "\n".join(
        f"{letter}. {content}" for letter, content in options
    )
    user_content = (
        f"题目：\n{question}\n\n"
        f"选项：\n{option_text}\n\n"
        "请先给出医学解析，再在最后输出正确选项字母。"
    )
    converted = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": f"解析：{explanation}\n答案：{answer}",
            },
        ]
    }
    return converted, len(explanation)


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
    """Load the keys of structurally valid questions; used to keep splits apart even when the explanation is missing."""
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


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0
    index = int((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def build_explanation_length_stats(lengths):
    lengths.sort()
    return {
        "samples": len(lengths),
        "p50": percentile(lengths, 0.50),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": lengths[-1] if lengths else 0,
    }


def convert_split(
    input_path,
    output_path,
    blocked_questions,
    max_samples,
):
    stats = Counter()
    examples = []
    accepted_questions = set()
    explanation_lengths = []
    output_hasher = hashlib.sha256()

    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with (
            input_path.open("r", encoding="utf-8-sig", newline="") as fin,
            temporary_path.open("w", encoding="utf-8") as fout,
        ):
            reader = csv.DictReader(fin)
            require_columns(reader, input_path)

            for line_number, row in enumerate(reader, start=2):
                if max_samples and stats["written"] >= max_samples:
                    break
                stats["input_rows_read"] += 1

                question_key = normalize_question(row.get("Question"))
                if not question_key:
                    record_rejection(
                        stats,
                        examples,
                        "empty_question",
                        line_number,
                        row,
                    )
                    continue
                if question_key in blocked_questions:
                    record_rejection(
                        stats,
                        examples,
                        "overlap_with_later_split",
                        line_number,
                        row,
                    )
                    continue
                if question_key in accepted_questions:
                    record_rejection(
                        stats,
                        examples,
                        "duplicate_question",
                        line_number,
                        row,
                    )
                    continue

                try:
                    converted, explanation_length = validate_and_convert(row)
                except ValueError as error:
                    record_rejection(
                        stats,
                        examples,
                        str(error),
                        line_number,
                        row,
                    )
                    continue

                serialized = json.dumps(converted, ensure_ascii=False) + "\n"
                fout.write(serialized)
                output_hasher.update(serialized.encode("utf-8"))
                accepted_questions.add(question_key)
                explanation_lengths.append(explanation_length)
                stats["written"] += 1

        if stats["written"] == 0:
            raise ValueError(f"{input_path} produced no valid rows")
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    rejected = stats["input_rows_read"] - stats["written"]
    rejection_counts = {
        key: value
        for key, value in sorted(stats.items())
        if key not in {"input_rows_read", "written"}
    }
    result = {
        "input_rows_read": stats["input_rows_read"],
        "written": stats["written"],
        "rejected": rejected,
        "rejection_counts": rejection_counts,
        "rejection_examples": examples,
        "explanation_length_chars": build_explanation_length_stats(
            explanation_lengths
        ),
        "sha256": output_hasher.hexdigest(),
    }
    return result, accepted_questions


def validate_args(args):
    if args.max_train_samples < 0:
        raise ValueError("max-train-samples must not be negative")
    if args.max_validation_samples < 0:
        raise ValueError("max-validation-samples must not be negative")


def ensure_paths(input_paths, output_paths):
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"input file not found: {path}")

    resolved_inputs = {path.resolve() for path in input_paths}
    resolved_outputs = {path.resolve() for path in output_paths}
    if len(resolved_outputs) != len(output_paths):
        raise ValueError("output paths must be distinct")
    if resolved_inputs & resolved_outputs:
        raise ValueError("input and output paths must differ")


def main():
    args = parse_args()
    validate_args(args)
    configure_csv_field_limit()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    train_input = input_dir / "train.csv"
    validation_input = input_dir / "val.csv"
    test_input = input_dir / "test_with_annotations.csv"
    train_output = output_dir / TRAIN_FILENAME
    validation_output = output_dir / VALIDATION_FILENAME
    report_output = output_dir / REPORT_FILENAME

    ensure_paths(
        (train_input, validation_input, test_input),
        (train_output, validation_output, report_output),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    test_questions, test_rows, test_empty_questions = (
        load_normalized_questions(test_input)
    )
    validation_split_questions = load_eligible_question_keys(
        validation_input,
        blocked_questions=test_questions,
    )

    validation_stats, _ = convert_split(
        input_path=validation_input,
        output_path=validation_output,
        blocked_questions=test_questions,
        max_samples=args.max_validation_samples,
    )
    train_stats, _ = convert_split(
        input_path=train_input,
        output_path=train_output,
        blocked_questions=test_questions | validation_split_questions,
        max_samples=args.max_train_samples,
    )

    report = {
        "format": "messages_only_explanation_with_final_answer",
        "system_prompt": SYSTEM_PROMPT,
        "source": {
            "train": str(train_input),
            "validation": str(validation_input),
            "test_for_decontamination_only": str(test_input),
        },
        "output": {
            "train": str(train_output),
            "validation": str(validation_output),
        },
        "limits": {
            "max_train_samples": args.max_train_samples,
            "max_validation_samples": args.max_validation_samples,
        },
        "test_reference": {
            "input_rows": test_rows,
            "unique_normalized_questions": len(test_questions),
            "empty_questions": test_empty_questions,
        },
        "validation_split_reference": {
            "eligible_normalized_questions": len(
                validation_split_questions
            ),
            "note": (
                "validation questions with a valid structure block the same question "
                "from entering train even when their explanation is missing."
            ),
        },
        "counts": {
            "train": train_stats,
            "validation": validation_stats,
        },
        "cleaning_policy": {
            "test_usage": "only normalised questions are read, for exact decontamination; no answers are used as labels.",
            "split_priority": "test > validation > train",
            "duplicate_key": "NFKC + lowercase + remove whitespace(question)",
            "invalid_rows": (
                "rows with an empty question, malformed options, an empty answer, an answer outside the options, "
                "or an empty / placeholder explanation are skipped."
            ),
            "assistant_target": (
                "the target is exactly 解析：<official explanation>\\n答案：<letters>."
            ),
            "explanation": (
                "the explanation appears only in the assistant turn, never in the system or user prompt."
            ),
        },
    }

    report_temporary = report_output.with_name(report_output.name + ".tmp")
    try:
        with report_temporary.open("w", encoding="utf-8") as fout:
            json.dump(report, fout, ensure_ascii=False, indent=2)
            fout.write("\n")
        report_temporary.replace(report_output)
    except Exception:
        if report_temporary.exists():
            report_temporary.unlink()
        raise

    print(f"train: {train_stats['written']} -> {train_output}")
    print(
        "validation: "
        f"{validation_stats['written']} -> {validation_output}"
    )
    print(f"report: {report_output}")


if __name__ == "__main__":
    main()
