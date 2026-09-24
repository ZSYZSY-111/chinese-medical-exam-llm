#!/usr/bin/env python3
"""Rationale-style SFT with segment-level supervision from teacher scores.

Data: rows exported by the harness's sentence_score task (messages + explanation + sentences[{start, end, score}]).
Training: the usual rationale-then-answer LM loss (the same region-weighted loss as the rationale SFT:
0.2 × mean over rationale tokens + 0.8 × mean over answer tokens), plus a linear head on the hidden state of each
segment's last token that predicts the teacher's score for that segment (1–5, normalised to 0–1):
    total = lm_loss + score_loss_weight × MSE(sigmoid(head(h)), target)
--score-loss-weight 0 is the control arm (same data and hyper-parameters, no auxiliary loss).
--sentence-weighting score is the other use of the same scores: no prediction, the scores become per-token weights
of the rationale LM loss.

This does not go through TRL's SFTTrainer: segment ends must be aligned to tokens exactly via offset mappings,
so tokenisation, collation and the loss are done here. The head is not part of the LoRA adapter and is saved
separately as final_adapter/score_head.pt; evaluating answer accuracy needs only the adapter.

Check the data first (no model is loaded):
  python scripts/train_sft_score_head.py --check-only --model Models/Qwen2.5-7B-Instruct \
    --train-file data/sentence_score/sft/train.jsonl --dev-file data/sentence_score/sft/dev.jsonl
"""
import argparse
import json
import math
import sys
from pathlib import Path

try:
    from .train_sft import DEFAULT_ANSWER_MARKER, build_cot_answer_masks
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_sft import DEFAULT_ANSWER_MARKER, build_cot_answer_masks

SCORE_MIN, SCORE_MAX = 1, 5
WEIGHTING_NONE, WEIGHTING_SCORE = "none", "score"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", default=None, help="held-out rows with teacher scores, used only to measure how well the head predicts them; never trained on")
    parser.add_argument("--output-dir", default="training_outputs/score_head_run")
    parser.add_argument("--score-loss-weight", type=float, default=1.0, help="0 = control arm without the score-prediction loss")
    parser.add_argument("--sentence-weighting", choices=(WEIGHTING_NONE, WEIGHTING_SCORE), default=WEIGHTING_NONE,
                        help="score: weight each rationale token's LM loss by its segment's teacher score (normalised within the example)")
    parser.add_argument("--cot-loss-weight", type=float, default=0.2)
    parser.add_argument("--answer-loss-weight", type=float, default=0.8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3, help="the score head starts from scratch and gets its own, larger learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2560, help="longer examples are dropped (truncation would remove the final answer line)")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--loss-logit-chunk-size", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="only tokenise and check the segment-end alignment; do not load the model")
    args = parser.parse_args(argv)
    if args.score_loss_weight < 0 or args.cot_loss_weight < 0 or args.answer_loss_weight <= 0:
        parser.error("loss weights must be non-negative and the answer weight positive")
    return args


# ---------------------------------------------------------------- data: segment-end characters -> token positions

def normalize_score(score):
    if not SCORE_MIN <= score <= SCORE_MAX:
        raise ValueError(f"score out of range: {score}")
    return (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


def sentence_char_ends(content, explanation, sentences):
    """Return [(index in `content` of the segment's last non-space character, index of its first character, raw score)]."""
    offset = content.find(explanation)
    if offset < 0:
        raise ValueError("the rationale text was not found in the assistant turn")
    ends = []
    for sentence in sentences:
        start, end = sentence["start"], sentence["end"]
        while end > start and explanation[end - 1].isspace():
            end -= 1
        if end <= start:
            raise ValueError("empty segment")
        ends.append((offset + end - 1, offset + start, sentence["score"]))
    return ends


def encode_example(tokenizer, row, max_length):
    """Tokenise and align one row. Returns a dict, or None when the example is too long."""
    messages = row["messages"]
    content = messages[-1]["content"]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    if not full_text.startswith(prompt_text):
        raise ValueError("the rendered prompt is not a prefix of the rendered conversation")
    if full_text[len(prompt_text):len(prompt_text) + len(content)] != content:
        raise ValueError("the assistant content does not directly follow the prompt")
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    if len(input_ids) > max_length:
        return None
    base = len(prompt_text)
    labels = [token if start >= base else -100 for token, (start, _) in zip(input_ids, offsets)]

    def token_at(char_index):
        for index, (start, end) in enumerate(offsets):
            if start <= char_index < end:
                return index
        raise ValueError(f"character {char_index} is not covered by any token")

    positions, targets = [], []
    weights = [1.0] * len(input_ids)
    for end_char, start_char, score in sentence_char_ends(content, row["explanation"], row["sentences"]):
        last = token_at(base + end_char)
        first = token_at(base + start_char)
        positions.append(last)
        targets.append(normalize_score(score))
        for index in range(first, last + 1):
            weights[index] = float(score)
    if positions != sorted(set(positions)):
        raise ValueError("two segments end on the same token; the split is too fine")
    return {"input_ids": input_ids, "labels": labels, "score_positions": positions, "score_targets": targets,
            "sentence_weights": weights}


def load_rows(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def encode_rows(tokenizer, rows, max_length):
    """Return (examples, dropped). Over-long or misaligned rows are dropped and counted rather than aborting the run."""
    examples, dropped = [], 0
    for row in rows:
        try:
            example = encode_example(tokenizer, row, max_length)
        except ValueError as error:
            print(f"skipping {row.get('sample_id', '?')[:12]}: {error}", file=sys.stderr)
            example = None
        if example is None:
            dropped += 1
        else:
            examples.append(example)
    return examples, dropped


def collate(batch, pad_token_id, torch):
    length = max(len(example["input_ids"]) for example in batch)
    sentences = max(len(example["score_positions"]) for example in batch)

    def pad(values, fill, size):
        return list(values) + [fill] * (size - len(values))

    return {
        "input_ids": torch.tensor([pad(e["input_ids"], pad_token_id, length) for e in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad([1] * len(e["input_ids"]), 0, length) for e in batch], dtype=torch.long),
        "labels": torch.tensor([pad(e["labels"], -100, length) for e in batch], dtype=torch.long),
        "sentence_weights": torch.tensor([pad(e["sentence_weights"], 1.0, length) for e in batch], dtype=torch.float32),
        "score_positions": torch.tensor([pad(e["score_positions"], -1, sentences) for e in batch], dtype=torch.long),
        "score_targets": torch.tensor([pad(e["score_targets"], 0.0, sentences) for e in batch], dtype=torch.float32),
    }


# ---------------------------------------------------------------- losses

def weighted_lm_loss(hidden_states, lm_head, labels, cot_mask, answer_mask, sentence_weights, cot_weight, answer_weight,
                     chunk_size, torch, use_sentence_weights=False):
    """Per example: cot_weight × (weighted mean CE over rationale tokens) + answer_weight × (mean CE over answer tokens), then averaged over the batch."""
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    cot = cot_mask[:, 1:].float()
    answer = answer_mask[:, 1:].float()
    if use_sentence_weights:
        cot = cot * sentence_weights[:, 1:]
    cot_total = cot.sum(dim=1, keepdim=True)
    answer_total = answer.sum(dim=1, keepdim=True)
    if not bool((cot_total > 0).all()) or not bool((answer_total > 0).all()):
        raise ValueError("at least one example has no rationale tokens or no answer tokens")
    token_weights = cot * (cot_weight / cot_total) + answer * (answer_weight / answer_total)
    supervised = shift_labels != -100
    selected_hidden = shift_hidden[supervised]
    selected_labels = shift_labels[supervised]
    selected_weights = token_weights[supervised]

    def chunk_loss(hidden_chunk, label_chunk, weight_chunk):
        logits = lm_head(hidden_chunk).float()
        losses = torch.nn.functional.cross_entropy(logits, label_chunk, reduction="none")
        return (losses * weight_chunk.float()).sum()

    total = hidden_states.sum() * 0.0
    for start in range(0, len(selected_labels), chunk_size):
        stop = start + chunk_size
        total = total + torch.utils.checkpoint.checkpoint(chunk_loss, selected_hidden[start:stop], selected_labels[start:stop],
                                                          selected_weights[start:stop], use_reentrant=False)
    return total / labels.shape[0]


def predict_scores(hidden_states, score_head, score_positions, torch):
    """Run the segment-end hidden states through the head; returns (predictions in 0–1, valid-position mask)."""
    valid = score_positions >= 0
    gather_index = score_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    picked = hidden_states.gather(1, gather_index).float()
    return torch.sigmoid(score_head(picked).squeeze(-1)), valid


def score_loss(predicted, targets, valid, torch):
    """Distance between predicted and teacher scores: mean squared error over valid segments."""
    count = valid.sum().clamp(min=1)
    return (((predicted - targets) ** 2) * valid.float()).sum() / count


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        for k in range(index, stop + 1):
            result[order[k]] = (index + stop) / 2 + 1
        index = stop + 1
    return result


def summarize_score_predictions(predicted, targets, groups):
    """predicted/targets are in 0–1; groups gives each segment's example index. Returns errors and correlations on the 1–5 scale."""
    if not predicted:
        return {"sentences": 0}
    scale = SCORE_MAX - SCORE_MIN
    errors = [abs(p - t) * scale for p, t in zip(predicted, targets)]
    by_group = {}
    for p, t, g in zip(predicted, targets, groups):
        by_group.setdefault(g, []).append((p, t))
    hits = [max(pairs, key=lambda pt: pt[0])[1] == max(t for _, t in pairs) for pairs in by_group.values() if len(pairs) > 1]
    constant = sum(targets) / len(targets)
    return {
        "sentences": len(predicted), "examples": len(by_group),
        "mae_points": round(sum(errors) / len(errors), 4),
        "mae_points_constant_baseline": round(sum(abs(constant - t) * scale for t in targets) / len(targets), 4),
        "pearson": pearson(predicted, targets), "spearman": pearson(ranks(predicted), ranks(targets)),
        "top_sentence_hit_rate": round(sum(hits) / len(hits), 4) if hits else None,
    }


# ---------------------------------------------------------------- training

def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = load_rows(args.train_file, args.max_train_samples)
    dev_rows = load_rows(args.dev_file) if args.dev_file else []
    train_examples, train_dropped = encode_rows(tokenizer, train_rows, args.max_length)
    dev_examples, dev_dropped = encode_rows(tokenizer, dev_rows, args.max_length)
    lengths = sorted(len(e["input_ids"]) for e in train_examples)
    sentence_total = sum(len(e["score_positions"]) for e in train_examples)
    print(f"train {len(train_examples)} rows ({train_dropped} dropped as too long), held-out {len(dev_examples)} rows ({dev_dropped} dropped); "
          f"{sentence_total} segments; length p50 {lengths[len(lengths) // 2]} / max {lengths[-1]}")
    if args.check_only:
        sample = train_examples[0]
        for position, target in zip(sample["score_positions"], sample["score_targets"]):
            tail = tokenizer.decode(sample["input_ids"][max(0, position - 5):position + 1])
            print(f"  segment-end token {position} target {target:.2f} …{tail!r}")
        print("check-only: the model is not loaded.")
        return 0

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    transformers.set_seed(args.seed)
    marker_ids = tokenizer.encode(DEFAULT_ANSWER_MARKER, add_special_tokens=False)
    special_ids = list(tokenizer.all_special_ids)

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, use_cache=False, local_files_only=args.local_files_only)
    model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
                                            task_type="CAUSAL_LM", target_modules="all-linear"))
    model.enable_input_require_grads()
    hidden_size = base.config.hidden_size
    torch.manual_seed(args.seed)
    model.score_head = torch.nn.Linear(hidden_size, 1).to(dtype=torch.float32)
    torch.nn.init.normal_(model.score_head.weight, std=0.02)
    torch.nn.init.zeros_(model.score_head.bias)
    model.print_trainable_parameters()

    use_sentence_weights = args.sentence_weighting == WEIGHTING_SCORE
    running = {"lm": 0.0, "score": 0.0, "n": 0}

    def forward_losses(peft_model, inputs):
        causal_lm = peft_model.get_base_model()
        decoder_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "use_cache": False, "return_dict": True}
        hidden = causal_lm.model(**decoder_inputs).last_hidden_state
        cot_mask, answer_mask = build_cot_answer_masks(inputs["input_ids"], inputs["labels"], marker_ids, special_ids, torch)
        lm = weighted_lm_loss(hidden, causal_lm.get_output_embeddings(), inputs["labels"], cot_mask, answer_mask,
                              inputs["sentence_weights"], args.cot_loss_weight, args.answer_loss_weight, args.loss_logit_chunk_size,
                              torch, use_sentence_weights)
        predicted, valid = predict_scores(hidden, peft_model.score_head, inputs["score_positions"], torch)
        return lm, score_loss(predicted, inputs["score_targets"], valid, torch), predicted, valid

    class ScoreHeadTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            del num_items_in_batch
            lm, score, _, _ = forward_losses(model, inputs)
            # the control arm (weight 0) still computes the score loss for logging; detached, so neither the head nor the backbone is affected
            total = lm + args.score_loss_weight * score if args.score_loss_weight > 0 else lm + 0.0 * score.detach()
            running["lm"] += float(lm.detach())
            running["score"] += float(score.detach())
            running["n"] += 1
            return (total, {"loss": total.detach()}) if return_outputs else total

        def log(self, logs, *log_args, **log_kwargs):
            if running["n"]:
                logs["lm_loss"] = round(running["lm"] / running["n"], 5)
                logs["score_mse"] = round(running["score"] / running["n"], 5)
                running.update(lm=0.0, score=0.0, n=0)
            return super().log(logs, *log_args, **log_kwargs)

        def create_optimizer(self):
            if self.optimizer is None:
                head = [p for n, p in self.model.named_parameters() if p.requires_grad and "score_head" in n]
                rest = [p for n, p in self.model.named_parameters() if p.requires_grad and "score_head" not in n]
                self.optimizer = torch.optim.AdamW([
                    {"params": rest, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
                    {"params": head, "lr": args.head_learning_rate, "weight_decay": 0.0},
                ], betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon, fused=True)
            return self.optimizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=args.num_train_epochs, learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, max_grad_norm=1.0,
        bf16=True, tf32=True, logging_strategy="steps", logging_steps=args.logging_steps, eval_strategy="no", save_strategy="no",
        remove_unused_columns=False, dataloader_num_workers=0, report_to="none", seed=args.seed, data_seed=args.seed)
    run_config = {"arguments": vars(args), "resolved": {"train_samples": len(train_examples), "train_dropped_overlength": train_dropped,
                                                        "dev_samples": len(dev_examples), "train_sentences": sentence_total,
                                                        "gpu": torch.cuda.get_device_name(0)},
                  "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    trainer = ScoreHeadTrainer(model=model, args=training_args, train_dataset=train_examples,
                               data_collator=lambda batch: collate(batch, tokenizer.pad_token_id, torch))
    result = trainer.train()
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    report = {"train_metrics": result.metrics}
    if dev_examples:
        model.eval()
        predicted, targets, groups, lm_values = [], [], [], []
        with torch.no_grad():
            for start in range(0, len(dev_examples), args.per_device_train_batch_size):
                chunk = dev_examples[start:start + args.per_device_train_batch_size]
                inputs = {k: v.to(model.device) for k, v in collate(chunk, tokenizer.pad_token_id, torch).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lm, _, pred, valid = forward_losses(model, inputs)
                lm_values.append(float(lm) * len(chunk))
                for row in range(len(chunk)):
                    for col in range(valid.shape[1]):
                        if bool(valid[row, col]):
                            predicted.append(float(pred[row, col]))
                            targets.append(float(inputs["score_targets"][row, col]))
                            groups.append(start + row)
        report["dev_lm_loss"] = round(sum(lm_values) / len(dev_examples), 5)
        report["dev_score_prediction"] = summarize_score_predictions(predicted, targets, groups)
    (output_dir / "score_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    final_dir = output_dir / "final_adapter"
    head = model.score_head
    del model.score_head  # the head is not part of the adapter; save it separately so it never ends up in the adapter weights
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    torch.save(head.state_dict(), str(final_dir / "score_head.pt"))
    print(f"done: {final_dir} (score_head.pt is the score head)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
