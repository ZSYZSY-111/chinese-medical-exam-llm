"""Validation diagnostics for a model on a messages-format JSONL file.

Four measurements, each optional:
  1. greedy accuracy (the main number; exact letter-set match)
  2. k sampled answers per question: pass@k (any correct), SC@k (majority vote correct), mean sampled accuracy
     -> pass@k - pass@1 bounds what sharpening the current policy could still gain
  3. constrained scoring (direct mode, single-answer questions): compare the logits of the option-letter tokens
     -> constrained accuracy - generated accuracy = the loss caused by format and decoding
  4. option-permutation consistency: is the greedy answer the same, after mapping back, under P option orders?
     -> low consistency means position dependence that shuffle augmentation can remove

Writes headroom_records.jsonl (one row per question) and headroom_report.json (overall and per-slice summary).
Use --skip-sampling --skip-shuffle --skip-constrained for a plain greedy validation score.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from .cmexam_prompts import (
        MODE_DIRECT,
        MODES,
        build_prompt_messages,
        type_hint_for,
        detect_slices,
        make_permutation,
        parse_user_content,
        permute_options,
        stable_fraction,
        unmap_answer,
    )
    from .answer_utils import parse_completion
    from .data_utils import load_tokenizer, parse_messages_record
except ImportError:
    from cmexam_prompts import (
        MODE_DIRECT,
        MODES,
        build_prompt_messages,
        type_hint_for,
        detect_slices,
        make_permutation,
        parse_user_content,
        permute_options,
        stable_fraction,
        unmap_answer,
    )
    from answer_utils import parse_completion
    from data_utils import load_tokenizer, parse_messages_record


DEFAULT_INPUT_FILE = "cmexam_data/no_explanation/cmexam_sft_validation.jsonl"
DEFAULT_OUTPUT_DIR = "training_outputs/rl_headroom"
RECORDS_FILENAME = "headroom_records.jsonl"
REPORT_FILENAME = "headroom_report.json"
SLICE_NAMES = ("multi_choice", "negation", "case", "calc", "long_stem")


def parse_args():
    parser = argparse.ArgumentParser(description="Validation accuracy and diagnostics.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=MODES, default=MODE_DIRECT)
    parser.add_argument("--type-hint", action="store_true", help="add a question-type hint to the prompt, aligned with the official evaluation prompt; inferred from the number of reference letters by default")
    parser.add_argument("--type-hint-file", default=None, help="official-layout JSON (id = sample_id, with question_type); when given, the dataset question type is used")
    parser.add_argument("--prompt-style", choices=["legacy_explain"], default=None, help="legacy_explain: render with the prompt of the rationale-style SFT data (cot mode only), so adapters trained on it are evaluated in their own format")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="shuffle by stable hash before taking the first `limit` rows; file order otherwise.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=None, help="default: 32 in direct mode, 384 otherwise")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--permutations", type=int, default=4)
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-constrained", action="store_true")
    parser.add_argument("--skip-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 32 if args.mode == MODE_DIRECT else 384
    return args


def validate_args(args):
    for name in ("batch_size", "num_samples", "max_new_tokens", "max_prompt_length"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be greater than 0")
    if args.limit < 0 or args.permutations < 0:
        raise ValueError("limit and permutations must not be negative")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be positive and top_p in (0, 1]")
    if args.mode != MODE_DIRECT and not args.skip_constrained:
        print("note: constrained letter scoring only applies to direct mode and is skipped otherwise.")
        args.skip_constrained = True


def load_examples(path, mode, limit, shuffle_seed=None, type_hint=False, type_hint_map=None, prompt_style=None):
    """Read a messages JSONL file and re-render each prompt for `mode` (identical to the source in direct mode).

    With shuffle_seed set, rows are ordered by stable hash before `limit` is applied, so a small sample is not
    just the first exam section of the file.
    """
    examples = []
    stats = Counter()
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if limit and shuffle_seed is None and len(examples) >= limit:
                break
            if not line.strip():
                continue
            converted = parse_messages_record(json.loads(line), str(path), line_number)
            user_content = next(m["content"] for m in converted["prompt"] if m["role"] == "user")
            try:
                parsed = parse_user_content(user_content)
            except ValueError:
                stats["unparsed"] += 1
                continue
            slices = detect_slices(parsed["question"], parsed["options"], converted["answer"])
            examples.append(
                {
                    "sample_id": converted["sample_id"],
                    "line_number": line_number,
                    "question": parsed["question"],
                    "options": parsed["options"],
                    "answer": converted["answer"],
                    "valid_letters": converted["valid_letters"],
                    "slices": {name: slices[name] for name in SLICE_NAMES},
                    "shuffle_safe": slices["shuffle_safe"],
                    "type_hint": type_hint_for(converted["answer"], (type_hint_map or {}).get(converted["sample_id"])) if type_hint else None,
                    "messages": build_prompt_messages(parsed["question"], parsed["options"], mode,
                                                      hint=type_hint_for(converted["answer"], (type_hint_map or {}).get(converted["sample_id"])) if type_hint else None,
                                                      style=prompt_style),
                }
            )
            stats["loaded"] += 1
    if shuffle_seed is not None:
        examples.sort(key=lambda e: stable_fraction(shuffle_seed, "diagnose", e["sample_id"]))
        if limit:
            examples = examples[:limit]
        stats["loaded"] = len(examples)
        stats["shuffle_seed"] = shuffle_seed
    if not examples:
        raise ValueError(f"no usable rows in {path}")
    return examples, dict(stats)


def majority_vote(answers):
    """Majority vote; None is ignored; ties go to the answer seen first."""
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts:
        return None
    best = max(counts.values())
    for answer in answers:
        if answer is not None and counts[answer] == best:
            return answer
    return None


def bucket_of(correct_count, num_samples):
    if correct_count == num_samples:
        return "easy"
    if correct_count == 0:
        return "zero"
    if correct_count * 2 > num_samples:
        return "medium"
    return "hard"


def _rate(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarize_records(records, num_samples):
    def metrics(subset):
        result = {
            "questions": len(subset),
            "pass1_greedy": _rate([r["greedy_correct"] for r in subset]),
            "greedy_strict_format_rate": _rate([r["greedy_strict"] for r in subset]),
            "greedy_cot_rate": _rate([r["greedy_used_cot"] for r in subset]),
        }
        sampled = [r for r in subset if r.get("correct_count") is not None]
        if sampled:
            result.update(
                {
                    "pass_at_k": _rate([r["correct_count"] > 0 for r in sampled]),
                    "sc_at_k": _rate([r["majority_correct"] for r in sampled]),
                    "sampled_accuracy_mean": _rate(
                        [r["correct_count"] / num_samples for r in sampled]
                    ),
                }
            )
            result["sharpening_headroom"] = round(
                result["pass_at_k"] - result["pass1_greedy"], 4
            )
            result["self_consistency_gap"] = round(result["sc_at_k"] - result["pass1_greedy"], 4)
        constrained = [r for r in subset if r.get("constrained_correct") is not None]
        if constrained:
            result["constrained_accuracy_single"] = _rate(
                [r["constrained_correct"] for r in constrained]
            )
            result["free_accuracy_single"] = _rate([r["greedy_correct"] for r in constrained])
            result["format_gap"] = round(
                result["constrained_accuracy_single"] - result["free_accuracy_single"], 4
            )
        shuffled = [r for r in subset if r.get("shuffle_consistent") is not None]
        if shuffled:
            result["shuffle_consistency_rate"] = _rate(
                [r["shuffle_consistent"] for r in shuffled]
            )
            result["shuffle_accuracy"] = _rate(
                [
                    sum(p["correct"] for p in r["permutations"]) / len(r["permutations"])
                    for r in shuffled
                ]
            )
        return result

    report = {"overall": metrics(records), "slices": {}}
    for name in SLICE_NAMES:
        subset = [r for r in records if r["slices"].get(name)]
        if subset:
            report["slices"][name] = metrics(subset)
    buckets = Counter(r["bucket"] for r in records if r.get("bucket"))
    if buckets:
        report["sampled_bucket_histogram"] = dict(sorted(buckets.items()))
    histogram = Counter(r["correct_count"] for r in records if r.get("correct_count") is not None)
    if histogram:
        report["correct_count_histogram"] = {str(k): v for k, v in sorted(histogram.items())}
    return report


def finalize_shuffle_consistency(records, record_indexes):
    """For every question with shuffled variants: do all variants, mapped back to the original letters, equal the greedy answer?"""
    for record_index in sorted(set(record_indexes)):
        record = records[record_index]
        answers = [item["answer_original_letters"] for item in record.get("permutations", [])]
        record["shuffle_consistent"] = bool(answers) and all(
            answer == record["greedy_answer"] for answer in answers
        )
    return records


def letter_token_ids(tokenizer, letters):
    ids = []
    for letter in letters:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"letter {letter!r} is not a single token; constrained scoring is impossible: {encoded}")
        ids.append(encoded[0])
    return ids


def load_model(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": 0},
        local_files_only=args.local_files_only,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(
            model, args.adapter, is_trainable=False, local_files_only=args.local_files_only
        )
    model.eval()
    model.config.use_cache = True
    tokenizer = load_tokenizer(AutoTokenizer, args)
    return model, tokenizer, torch


def encode_prompts(tokenizer, messages_list, max_prompt_length, device):
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def generate(model, tokenizer, torch, encoded, do_sample, temperature, top_p, n, max_new_tokens):
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": n,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})
    with torch.inference_mode():
        output = model.generate(**encoded, **generation_kwargs)
    completions = output[:, encoded["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
    return [texts[index:index + n] for index in range(0, len(texts), n)]


def constrained_scores(model, torch, encoded, token_ids):
    with torch.inference_mode():
        logits = model(**encoded).logits[:, -1, :]
    scores = logits[:, token_ids].float().cpu()
    return scores.argmax(dim=1).tolist()


def run(args):
    validate_args(args)
    type_hint_map = None
    if args.type_hint_file:
        with open(args.type_hint_file, encoding="utf-8") as handle:
            type_hint_map = {str(item["id"]): item.get("question_type") for item in json.load(handle)}
    examples, load_stats = load_examples(
        args.input_file, args.mode, args.limit, args.shuffle_seed, type_hint=args.type_hint, type_hint_map=type_hint_map,
        prompt_style=args.prompt_style,
    )
    load_stats["type_hint"] = bool(args.type_hint)
    load_stats["prompt_style"] = args.prompt_style
    print("input check passed:", json.dumps(load_stats, ensure_ascii=False))
    if args.check_only:
        print("check-only: the model is not loaded.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / RECORDS_FILENAME
    report_path = output_dir / REPORT_FILENAME
    if (records_path.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError(f"output exists; choose another directory or pass --overwrite: {output_dir}")

    model, tokenizer, torch = load_model(args)
    torch.manual_seed(args.seed)
    device = model.device
    print(f"GPU: {torch.cuda.get_device_name(0)}; questions: {len(examples)}")

    records = []
    with records_path.open("w", encoding="utf-8") as records_file:
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            encoded = encode_prompts(
                tokenizer, [e["messages"] for e in batch], args.max_prompt_length, device
            )
            greedy = generate(model, tokenizer, torch, encoded, False, 1.0, 1.0, 1, args.max_new_tokens)
            sampled = None
            if not args.skip_sampling:
                sampled = generate(
                    model, tokenizer, torch, encoded, True, args.temperature, args.top_p,
                    args.num_samples, args.max_new_tokens,
                )
            constrained = None
            if not args.skip_constrained:
                letters = batch[0]["valid_letters"]
                if all(e["valid_letters"] == letters for e in batch):
                    constrained = constrained_scores(
                        model, torch, encoded, letter_token_ids(tokenizer, letters)
                    )
                else:
                    constrained = []
                    for example in batch:
                        single = encode_prompts(tokenizer, [example["messages"]], args.max_prompt_length, device)
                        constrained.extend(
                            constrained_scores(model, torch, single, letter_token_ids(tokenizer, example["valid_letters"]))
                        )

            for index, example in enumerate(batch):
                greedy_parsed = parse_completion(greedy[index][0], args.mode, example["valid_letters"])
                record = {
                    "sample_id": example["sample_id"],
                    "line_number": example["line_number"],
                    "gold": example["answer"],
                    "slices": example["slices"],
                    "greedy_text": greedy_parsed["text"],
                    "greedy_answer": greedy_parsed["answer"],
                    "greedy_correct": greedy_parsed["answer"] == example["answer"],
                    "greedy_strict": greedy_parsed["strict"],
                    "greedy_used_cot": greedy_parsed["used_cot"],
                }
                if sampled is not None:
                    answers = [
                        parse_completion(text, args.mode, example["valid_letters"])["answer"]
                        for text in sampled[index]
                    ]
                    correct_count = sum(answer == example["answer"] for answer in answers)
                    majority = majority_vote(answers)
                    record.update(
                        {
                            "sampled_answers": answers,
                            "correct_count": correct_count,
                            "majority_answer": majority,
                            "majority_correct": majority == example["answer"],
                            "bucket": bucket_of(correct_count, args.num_samples),
                        }
                    )
                if constrained is not None and len(example["answer"]) == 1:
                    predicted = chr(ord("A") + constrained[index])
                    record["constrained_answer"] = predicted
                    record["constrained_correct"] = predicted == example["answer"]
                records.append(record)

            if not args.skip_shuffle and args.permutations > 0:
                shuffle_batch, shuffle_meta = [], []
                for index, example in enumerate(batch):
                    if not example["shuffle_safe"]:
                        continue
                    used = []
                    for copy_index in range(args.permutations):
                        permutation = make_permutation(
                            len(example["options"]), args.seed, example["sample_id"], copy_index, forbidden=used
                        )
                        used.append(permutation)
                        shuffle_batch.append(
                            build_prompt_messages(
                                example["question"], permute_options(example["options"], permutation), args.mode,
                                hint=example.get("type_hint"), style=args.prompt_style
                            )
                        )
                        shuffle_meta.append((start + index, permutation))
                if shuffle_batch:
                    outputs = []
                    for sub_start in range(0, len(shuffle_batch), args.batch_size):
                        sub_encoded = encode_prompts(
                            tokenizer, shuffle_batch[sub_start:sub_start + args.batch_size], args.max_prompt_length, device
                        )
                        outputs.extend(
                            generate(model, tokenizer, torch, sub_encoded, False, 1.0, 1.0, 1, args.max_new_tokens)
                        )
                    for (record_index, permutation), output in zip(shuffle_meta, outputs):
                        record = records[record_index]
                        example = examples[record_index]
                        parsed = parse_completion(output[0], args.mode, example["valid_letters"])
                        mapped = unmap_answer(parsed["answer"], permutation)
                        record.setdefault("permutations", []).append(
                            {
                                "permutation": permutation,
                                "answer_original_letters": mapped,
                                "correct": mapped == example["answer"],
                            }
                        )
                    finalize_shuffle_consistency(records, [index for index, _ in shuffle_meta])

            for record in records[start:start + len(batch)]:
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"done {min(start + len(batch), len(examples))}/{len(examples)}")

    report = summarize_records(records, args.num_samples)
    report["config"] = vars(args)
    report["input_stats"] = load_stats
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
        report_file.write("\n")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print(f"report: {report_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
