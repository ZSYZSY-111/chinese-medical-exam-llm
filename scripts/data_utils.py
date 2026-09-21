"""Record validation and tokenizer loading shared by data builders and evaluators.

Every SFT row is `{"messages": [system, user, assistant]}`. `parse_messages_record` validates the
structure, derives the valid option letters from the prompt, parses the reference answer and
computes `sample_id` (SHA-256 of the user content) used to join rows across the whole pipeline.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

try:
    from .answer_utils import parse_reference_target
except ImportError:
    from answer_utils import parse_reference_target


OPTION_PATTERN = re.compile(r"(?m)^\s*([A-Z])[.、．:：]\s+\S")


VALID_OPTION_SEQUENCES = {
    "ABC",
    "ABCD",
    "ABCDE",
    "ABCDEF",  # CMB-Exam 含少量六选项题
}


def extract_valid_letters(messages, path="dataset", line_number=0):
    user_contents = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not user_contents or not all(
        isinstance(content, str) for content in user_contents
    ):
        raise ValueError(f"{path}:{line_number} 缺少合法 user content")

    labels = []
    for content in user_contents:
        labels.extend(OPTION_PATTERN.findall(content))
    valid_letters = "".join(dict.fromkeys(labels))
    if valid_letters not in VALID_OPTION_SEQUENCES:
        raise ValueError(
            f"{path}:{line_number} 选项序列不合法: {valid_letters!r}"
        )
    return valid_letters


def parse_messages_record(record, path="dataset", line_number=0):
    if not isinstance(record, dict) or set(record) != {"messages"}:
        raise ValueError(
            f"{path}:{line_number} 每行必须且只能包含 messages"
        )
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{path}:{line_number} messages 不合法")

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{path}:{line_number} message 不是对象")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"{path}:{line_number} role 不合法")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{path}:{line_number} 存在空 content")

    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{path}:{line_number} 最后一条必须是 assistant")
    if any(message.get("role") == "assistant" for message in messages[:-1]):
        raise ValueError(f"{path}:{line_number} prompt 中不能包含 assistant")

    valid_letters = extract_valid_letters(messages[:-1], path, line_number)
    target = parse_reference_target(
        messages[-1]["content"],
        valid_letters,
    )
    user_content = "\n".join(
        message["content"]
        for message in messages[:-1]
        if message["role"] == "user"
    )
    sample_id = hashlib.sha256(user_content.encode("utf-8")).hexdigest()
    return {
        "prompt": messages[:-1],
        "answer": target["answer"],
        "valid_letters": valid_letters,
        "answer_format": target["answer_format"],
        "reference_explanation": target["reference_explanation"],
        "is_multi_choice": len(target["answer"]) > 1,
        "sample_id": sample_id,
    }


def validate_messages_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    stats = Counter()
    answer_counts = Counter()
    seen_ids = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} 不是合法 JSON"
                ) from error
            converted = parse_messages_record(record, str(path), line_number)
            sample_id = converted["sample_id"]
            if sample_id in seen_ids:
                raise ValueError(f"{path}:{line_number} 出现重复 prompt")
            seen_ids.add(sample_id)

            stats["samples"] += 1
            stats[converted["answer_format"]] += 1
            choice_key = (
                "multi_choice" if converted["is_multi_choice"] else "single_choice"
            )
            stats[choice_key] += 1
            answer_counts[converted["answer"]] += 1

    if not stats["samples"]:
        raise ValueError(f"数据文件为空: {path}")
    return {
        **dict(stats),
        "answer_counts": dict(sorted(answer_counts.items())),
    }


def load_tokenizer(AutoTokenizer, args):
    sources = []
    if args.tokenizer:
        sources.append(args.tokenizer)
    elif args.adapter:
        sources.append(args.adapter)
    if args.model not in sources:
        sources.append(args.model)

    errors = []
    for source in sources:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                use_fast=True,
                local_files_only=args.local_files_only,
            )
            print(f"Tokenizer: {source}")
            break
        except (OSError, ValueError) as error:
            errors.append(f"{source}: {error}")
    else:
        raise RuntimeError(
            "无法从 adapter 或 base model 加载 tokenizer:\n"
            + "\n".join(errors)
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer
