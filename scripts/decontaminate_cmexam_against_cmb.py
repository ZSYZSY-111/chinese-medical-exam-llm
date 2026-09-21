import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


QUESTION_PATTERN = re.compile(
    r"题目：\s*\n(.*?)\n\s*\n选项：",
    re.DOTALL,
)
MAX_REMOVAL_EXAMPLES = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "删除 CMExam 仅答案 SFT 中与 CMB-test 规范化题干重合的样本，"
            "并更新数据报告。"
        )
    )
    parser.add_argument("--cmb-test-file", required=True)
    parser.add_argument(
        "--train-file",
        default="cmexam_data/no_explanation/cmexam_sft_train.jsonl",
    )
    parser.add_argument(
        "--validation-file",
        default="cmexam_data/no_explanation/cmexam_sft_validation.jsonl",
    )
    parser.add_argument(
        "--report-file",
        default="cmexam_data/no_explanation/cmexam_sft_report.json",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只统计将被删除的样本，不改写任何文件。",
    )
    return parser.parse_args()


def normalize_question(text):
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(
        character
        for character in normalized
        if not (
            character.isspace()
            or unicodedata.category(character).startswith("P")
        )
    )


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_cmb_questions(path):
    with path.open("r", encoding="utf-8") as input_file:
        rows = json.load(input_file)
    if not isinstance(rows, list):
        raise ValueError("CMB-test 根节点必须是 JSON 数组")

    key_to_ids = {}
    empty_questions = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"CMB-test 第 {index + 1} 条不是对象")
        question_key = normalize_question(row.get("question"))
        if not question_key:
            empty_questions += 1
            continue
        cmb_id = row.get("id", index + 1)
        key_to_ids.setdefault(question_key, []).append(cmb_id)

    return rows, key_to_ids, empty_questions


def extract_question(record, path, line_number):
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{path}:{line_number} 缺少 messages 数组")
    user_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(user_messages) != 1:
        raise ValueError(
            f"{path}:{line_number} user message 数量不是 1"
        )
    content = user_messages[0].get("content")
    if not isinstance(content, str):
        raise ValueError(f"{path}:{line_number} user content 不是字符串")
    match = QUESTION_PATTERN.search(content)
    if match is None:
        raise ValueError(f"{path}:{line_number} 无法提取题干")
    question = match.group(1).strip()
    if not question:
        raise ValueError(f"{path}:{line_number} 题干为空")
    return question


def inspect_split(path, blocked_questions, cmb_ids_by_question):
    input_hasher = hashlib.sha256()
    output_hasher = hashlib.sha256()
    kept_lines = []
    removed_examples = []
    removed_question_keys = set()
    matched_cmb_ids = set()
    input_rows = 0
    removed_rows = 0

    with path.open("rb") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            input_rows += 1
            input_hasher.update(raw_line)
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number} 不是有效 JSON") from error

            question = extract_question(record, path, line_number)
            question_key = normalize_question(question)
            if question_key in blocked_questions:
                removed_rows += 1
                removed_question_keys.add(question_key)
                matched_cmb_ids.update(cmb_ids_by_question[question_key])
                if len(removed_examples) < MAX_REMOVAL_EXAMPLES:
                    removed_examples.append(
                        {
                            "line_number": line_number,
                            "question": question,
                            "cmb_test_ids": cmb_ids_by_question[question_key],
                        }
                    )
                continue

            if not raw_line.endswith(b"\n"):
                raw_line += b"\n"
            kept_lines.append(raw_line)
            output_hasher.update(raw_line)

    if input_rows == 0:
        raise ValueError(f"{path} 是空文件")

    return {
        "path": str(path),
        "input_rows": input_rows,
        "removed_rows": removed_rows,
        "removed_unique_normalized_questions": len(removed_question_keys),
        "output_rows": input_rows - removed_rows,
        "matched_cmb_test_items": len(matched_cmb_ids),
        "matched_cmb_test_ids": sorted(matched_cmb_ids),
        "removed_examples": removed_examples,
        "input_sha256": input_hasher.hexdigest(),
        "output_sha256": output_hasher.hexdigest(),
        "kept_lines": kept_lines,
    }


def stage_split(path, kept_lines):
    temporary_path = path.with_name(path.name + ".cmb-decontam.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    with temporary_path.open("wb") as output_file:
        output_file.writelines(kept_lines)
    return temporary_path


def public_split_stats(stats):
    return {key: value for key, value in stats.items() if key != "kept_lines"}


def update_existing_counts(report, split_name, stats):
    split_counts = report.get("counts", {}).get(split_name)
    if not isinstance(split_counts, dict):
        return

    previous_overlap = split_counts.get("rejection_counts", {}).get(
        "overlap_with_cmb_test",
        0,
    )
    split_counts["written"] = stats["output_rows"]
    input_rows_read = split_counts.get("input_rows_read")
    if isinstance(input_rows_read, int):
        split_counts["rejected"] = input_rows_read - stats["output_rows"]
    split_counts.setdefault("rejection_counts", {})[
        "overlap_with_cmb_test"
    ] = previous_overlap + stats["removed_rows"]
    split_counts["sha256"] = stats["output_sha256"]


def build_report(
    report_path,
    cmb_path,
    cmb_rows,
    cmb_questions,
    empty_cmb_questions,
    split_stats,
):
    with report_path.open("r", encoding="utf-8") as input_file:
        report = json.load(input_file)

    report.setdefault("source", {})[
        "cmb_test_for_decontamination_only"
    ] = str(cmb_path)
    report["output"] = {
        "train": split_stats["train"]["path"],
        "validation": split_stats["validation"]["path"],
    }
    report["cmb_test_reference"] = {
        "input_rows": len(cmb_rows),
        "unique_normalized_questions": len(cmb_questions),
        "empty_questions": empty_cmb_questions,
        "sha256": sha256_file(cmb_path),
        "usage": "只读取题干做排除，不读取或使用 CMB-test 答案。",
    }
    report["cmb_decontamination"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "NFKC + lowercase + remove whitespace and Unicode punctuation "
            "from question"
        ),
        "scope": "exact normalized-question overlap only",
        "splits": {
            name: public_split_stats(stats)
            for name, stats in split_stats.items()
        },
    }
    report.setdefault("cleaning_policy", {})[
        "cmb_duplicate_key"
    ] = "NFKC + lowercase + remove whitespace and Unicode punctuation(question)"
    report["cleaning_policy"]["cmb_near_duplicates"] = (
        "本次不自动删除；字符级 MinHash 阈值校准和人工复核后另行处理。"
    )

    for split_name, stats in split_stats.items():
        update_existing_counts(report, split_name, stats)
    return report


def main():
    args = parse_args()
    cmb_path = Path(args.cmb_test_file)
    train_path = Path(args.train_file)
    validation_path = Path(args.validation_file)
    report_path = Path(args.report_file)
    for path in (cmb_path, train_path, validation_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(f"找不到文件: {path}")

    cmb_rows, cmb_questions, empty_cmb_questions = load_cmb_questions(
        cmb_path
    )
    blocked_questions = set(cmb_questions)
    split_stats = {
        "train": inspect_split(
            train_path,
            blocked_questions,
            cmb_questions,
        ),
        "validation": inspect_split(
            validation_path,
            blocked_questions,
            cmb_questions,
        ),
    }

    for split_name, stats in split_stats.items():
        print(
            f"{split_name}: {stats['input_rows']} -> "
            f"{stats['output_rows']} (removed {stats['removed_rows']})"
        )
    if args.check_only:
        return

    staged_splits = {
        "train": stage_split(train_path, split_stats["train"]["kept_lines"]),
        "validation": stage_split(
            validation_path,
            split_stats["validation"]["kept_lines"],
        ),
    }
    report = build_report(
        report_path,
        cmb_path,
        cmb_rows,
        cmb_questions,
        empty_cmb_questions,
        split_stats,
    )
    staged_report = report_path.with_name(report_path.name + ".tmp")
    try:
        with staged_report.open("w", encoding="utf-8") as output_file:
            json.dump(report, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")

        staged_splits["train"].replace(train_path)
        staged_splits["validation"].replace(validation_path)
        staged_report.replace(report_path)
    finally:
        for temporary_path in (*staged_splits.values(), staged_report):
            if temporary_path.exists():
                temporary_path.unlink()


if __name__ == "__main__":
    main()
