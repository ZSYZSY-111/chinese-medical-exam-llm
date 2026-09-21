import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("data", "test", "questions", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"无法识别数据结构：{path}")


def get_question_id(item: dict[str, Any], index: int) -> Any:
    return item.get("id", index + 1)


def get_options(item: dict[str, Any]) -> list[tuple[str, str]]:
    options = item.get("option") or item.get("options")

    if isinstance(options, dict):
        return [(str(k).upper(), str(v)) for k, v in options.items()]

    if isinstance(options, list):
        return [
            (chr(ord("A") + i), str(value))
            for i, value in enumerate(options)
        ]

    raise ValueError(f"题目缺少 option/options 字段：{item}")


def build_prompt(item: dict[str, Any]) -> str:
    options = get_options(item)

    exam_type = item.get("exam_type", "医学")
    exam_class = item.get("exam_class", "")
    question_type = item.get("question_type", "选择题")
    question = item.get("question", "")

    option_text = "\n".join(
        f"{letter}. {content}"
        for letter, content in options
    )

    return (
        f"以下是中国{exam_type}中{exam_class}考试的一道{question_type}，"
        "不需要做任何分析和解释，直接输出答案选项。\n"
        f"{question}\n"
        f"{option_text}\n"
        "只能输出选项字母。单选题例如：A；多选题例如：ABC。"
    )


def extract_choice(
    generated_text: str,
    valid_letters: list[str],
    is_multiple: bool,
) -> str:
    translation = str.maketrans(
        "ａｂｃｄｅｆｇＡＢＣＤＥＦＧ",
        "abcdefgABCDEFG",
    )
    text = generated_text.translate(translation).upper()

    patterns = [
        r"(?:答案|选项|选择)\s*(?:是|为|：|:)?\s*"
        r"([A-Z](?:[\s,，、/;；和与及]*[A-Z])*)",
        r"^\s*([A-Z](?:[\s,，、/;；和与及]*[A-Z])*)"
        r"\s*[。.!！]?\s*$",
    ]

    letters: list[str] = []

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            letters = [
                char
                for char in match.group(1)
                if char in valid_letters
            ]
            if letters:
                break

    if not letters:
        letters = [
            char
            for char in text
            if char in valid_letters
        ]

    if not letters:
        return ""

    if not is_multiple:
        return letters[0]

    selected = set(letters)

    # 多选答案统一整理为 ABC 这种顺序，避免输出 CBA 导致严格匹配失败。
    return "".join(
        letter
        for letter in valid_letters
        if letter in selected
    )


def save_predictions(
    output_path: Path,
    predictions: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用 CUDA GPU。")

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    tokenizer_path = args.adapter or args.base_model

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载基础模型：{args.base_model}")
    print(f"计算精度：{dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if args.adapter:
        print(f"加载 LoRA Adapter：{args.adapter}")
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            local_files_only=True,
        )
    else:
        print("未加载 Adapter，将评测原始模型。")

    model.eval()

    records = load_json_records(args.questions)
    output_path = Path(args.output)
    detail_path = output_path.with_suffix(".details.jsonl")

    predictions: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        completed_ids = {
            str(item["id"])
            for item in predictions
        }

        print(f"断点续跑：已存在 {len(predictions)} 条结果。")
    elif detail_path.exists():
        detail_path.unlink()

    pending: list[tuple[int, dict[str, Any]]] = []

    for index, item in enumerate(records):
        question_id = get_question_id(item, index)

        if str(question_id) not in completed_ids:
            pending.append((index, item))

    if args.limit > 0:
        pending = pending[:args.limit]

    print(f"数据总量：{len(records)}")
    print(f"本次待评测：{len(pending)}")
    print(f"batch size：{args.batch_size}")

    input_device = next(model.parameters()).device

    for start in tqdm(
        range(0, len(pending), args.batch_size),
        desc="CMB evaluation",
    ):
        batch_items = pending[start:start + args.batch_size]

        prompts = [
            build_prompt(item)
            for _, item in batch_items
        ]

        chat_texts = []

        for prompt in prompts:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个人工智能助手。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            chat_texts.append(chat_text)

        encoded = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )

        encoded = {
            key: value.to(input_device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_only = generated[
            :,
            encoded["input_ids"].shape[1]:,
        ]

        raw_answers = tokenizer.batch_decode(
            generated_only,
            skip_special_tokens=True,
        )

        detail_rows = []

        for (record_index, item), raw_answer in zip(
            batch_items,
            raw_answers,
        ):
            question_id = get_question_id(item, record_index)
            option_pairs = get_options(item)
            valid_letters = [letter for letter, _ in option_pairs]

            question_type = str(item.get("question_type", ""))
            is_multiple = "多项" in question_type

            model_answer = extract_choice(
                raw_answer,
                valid_letters,
                is_multiple,
            )

            predictions.append(
                {
                    "id": question_id,
                    "model_answer": model_answer,
                }
            )

            detail_rows.append(
                {
                    "id": question_id,
                    "question_type": question_type,
                    "exam_type": item.get("exam_type"),
                    "exam_class": item.get("exam_class"),
                    "question": item.get("question"),
                    "raw_output": raw_answer,
                    "model_answer": model_answer,
                }
            )

        save_predictions(output_path, predictions)

        with open(detail_path, "a", encoding="utf-8") as f:
            for row in detail_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    empty_count = sum(
        1
        for item in predictions
        if not item.get("model_answer")
    )

    print(f"完成：{len(predictions)} 条")
    print(f"无法抽取答案：{empty_count} 条")
    print(f"答案文件：{output_path}")
    print(f"详细输出：{detail_path}")


if __name__ == "__main__":
    main()
