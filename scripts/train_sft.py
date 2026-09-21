import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TRAIN_FILE = "data/huatuo_sft_train.jsonl"
DEFAULT_VALIDATION_FILE = "data/huatuo_sft_validation.jsonl"
DEFAULT_OUTPUT_DIR = "training_outputs/qwen2.5-3b-huatuo-lora"
DEFAULT_ANSWER_MARKER = "答案："
LOSS_MODE_COMPLETION = "completion"
LOSS_MODE_COT_ANSWER_WEIGHTED = "cot_answer_weighted"


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA supervised fine-tuning of a chat model with TRL + PEFT (bf16 base, completion-only loss)."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation-file", default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=1000)
    parser.add_argument("--length-check-samples", type=int, default=2000)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--init-adapter",
        default=None,
        help="continue training from an existing LoRA adapter (fresh optimizer and schedule); --lora-r/alpha/dropout are ignored.",
    )
    parser.add_argument("--report-to", default="none")
    parser.add_argument(
        "--loss-mode",
        choices=(LOSS_MODE_COMPLETION, LOSS_MODE_COT_ANSWER_WEIGHTED),
        default=LOSS_MODE_COMPLETION,
        help=(
            "completion: mean loss over completion tokens; "
            "cot_answer_weighted: separate means for the explanation and the answer span."
        ),
    )
    parser.add_argument("--cot-loss-weight", type=float, default=0.2)
    parser.add_argument("--answer-loss-weight", type=float, default=0.8)
    parser.add_argument(
        "--loss-logit-chunk-size",
        type=int,
        default=256,
        help="supervised tokens projected to the vocabulary per chunk in weighted-loss mode.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the data files only; do not load the model or train.",
    )
    return parser.parse_args()


def validate_positive_args(args):
    positive_int_names = (
        "max_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
        "lora_r",
        "lora_alpha",
        "dataset_num_proc",
        "loss_logit_chunk_size",
    )
    for name in positive_int_names:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be greater than 0")

    nonnegative_int_names = (
        "max_train_samples",
        "max_eval_samples",
        "length_check_samples",
        "dataloader_num_workers",
    )
    for name in nonnegative_int_names:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must not be negative")

    if args.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be greater than 0")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if args.init_adapter and args.resume_from_checkpoint:
        raise ValueError("--init-adapter and --resume-from-checkpoint are mutually exclusive")
    if args.loss_mode == LOSS_MODE_COT_ANSWER_WEIGHTED:
        if args.packing:
            raise ValueError("the weighted explanation/answer loss does not support --packing")
        if args.cot_loss_weight < 0 or args.answer_loss_weight < 0:
            raise ValueError("loss weights must not be negative")
        if not math.isclose(
            args.cot_loss_weight + args.answer_loss_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("the explanation and answer loss weights must sum to 1")
        if args.answer_loss_weight == 0:
            raise ValueError("the answer loss weight must be greater than 0")


def validate_messages_file(path, require_weighted_answer=False):
    """Validate the JSONL schema of a whole file; returns the number of rows."""
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")

    count = 0
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number} is not valid JSON") from exc

            if set(record) != {"messages"}:
                raise ValueError(
                    f"{path} line {line_number} must contain exactly one key, messages"
                )

            messages = record["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path} line {line_number} has invalid messages")
            if messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"{path} line {line_number}: the last message must be from the assistant"
                )
            if not any(message.get("role") == "user" for message in messages[:-1]):
                raise ValueError(f"{path} line {line_number} has no user message")

            for message in messages:
                if message.get("role") not in {"system", "user", "assistant"}:
                    raise ValueError(
                        f"{path} line {line_number} contains an unsupported role"
                    )
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        f"{path} line {line_number} has empty content"
                    )
            if require_weighted_answer:
                assistant = messages[-1]["content"]
                if not re.search(r"\n答案：[A-Z]+\s*$", assistant):
                    raise ValueError(
                        f"{path} line {line_number}: the assistant message must end with "
                        "a final line of the form 答案：<letters> to use the weighted loss"
                    )
            count += 1

    if count == 0:
        raise ValueError(f"data file is empty: {path}")
    return count


def messages_to_prompt_completion(example):
    """Split messages into prompt and completion so that TRL computes the loss on the assistant completion only."""
    messages = example["messages"]
    return {
        "prompt": messages[:-1],
        "completion": [messages[-1]],
    }


def select_subset(dataset, max_samples, seed):
    if not max_samples or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0
    index = int((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def report_length_stats(dataset, tokenizer, max_length, sample_count, seed):
    if sample_count == 0:
        return None

    sample_count = min(sample_count, len(dataset))
    sample = dataset.shuffle(seed=seed).select(range(sample_count))
    lengths = []
    for example in sample:
        encoded = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
        token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        lengths.append(len(token_ids))

    lengths.sort()
    truncated = sum(length > max_length for length in lengths)
    stats = {
        "samples": sample_count,
        "p50": percentile(lengths, 0.50),
        "p95": percentile(lengths, 0.95),
        "max": lengths[-1],
        "over_max_length": truncated,
        "over_max_length_ratio": truncated / sample_count,
    }
    print("token length sample:", json.dumps(stats, ensure_ascii=False))
    return stats


def build_cot_answer_masks(
    input_ids,
    labels,
    marker_ids,
    special_token_ids,
    torch,
):
    """Split supervised tokens into non-answer and answer spans at the last supervised answer marker."""
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("input_ids and labels must be 2-D tensors of the same shape")

    marker = torch.as_tensor(
        marker_ids,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    marker_length = int(marker.numel())
    sequence_length = int(input_ids.shape[1])
    if marker_length == 0 or marker_length > sequence_length:
        raise ValueError("the answer-marker token sequence is empty or longer than the batch")

    input_windows = input_ids.unfold(1, marker_length, 1)
    label_windows = labels.unfold(1, marker_length, 1)
    marker_matches = (input_windows == marker).all(dim=-1)
    marker_matches &= (label_windows != -100).all(dim=-1)

    has_marker = marker_matches.any(dim=1)
    if not bool(has_marker.all()):
        missing_rows = (
            (~has_marker).nonzero(as_tuple=False).flatten().tolist()
        )
        raise ValueError(
            "no supervised answer marker found in the assistant completion of this batch: "
            f"{DEFAULT_ANSWER_MARKER!r}, batch rows: {missing_rows}. "
            "The sample may have been truncated by max_length, or the target format is invalid."
        )

    window_positions = torch.arange(
        marker_matches.shape[1],
        device=input_ids.device,
    ).unsqueeze(0)
    marker_starts = torch.where(
        marker_matches,
        window_positions,
        torch.full_like(window_positions, -1),
    ).max(dim=1).values
    answer_starts = marker_starts + marker_length

    token_positions = torch.arange(
        sequence_length,
        device=input_ids.device,
    ).unsqueeze(0)
    supervised = labels != -100
    answer_mask = supervised & (token_positions >= answer_starts.unsqueeze(1))

    if special_token_ids:
        special_ids = torch.as_tensor(
            sorted(set(special_token_ids)),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        special_after_answer = (
            torch.isin(input_ids, special_ids)
            & (token_positions >= answer_starts.unsqueeze(1))
        )
        first_special = torch.where(
            special_after_answer,
            token_positions,
            torch.full_like(token_positions, sequence_length),
        ).min(dim=1).values
        answer_mask &= token_positions < first_special.unsqueeze(1)

    cot_mask = supervised & ~answer_mask
    if not bool(answer_mask.any(dim=1).all()):
        missing_rows = (
            (~answer_mask.any(dim=1))
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        raise ValueError(
            "no trainable answer tokens after the answer marker, batch rows: "
            f"{missing_rows}"
        )
    if not bool(cot_mask.any(dim=1).all()):
        raise ValueError("at least one sample has no trainable explanation tokens")
    return cot_mask, answer_mask


def make_weighted_sft_trainer_class(
    sft_trainer_class,
    torch,
    marker_ids,
    special_token_ids,
    cot_loss_weight,
    answer_loss_weight,
    logit_chunk_size,
):
    """Build an SFTTrainer that normalises the explanation and answer losses separately per sample."""

    class CotAnswerWeightedSFTTrainer(sft_trainer_class):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            del num_items_in_batch
            if "labels" not in inputs or "input_ids" not in inputs:
                raise ValueError("the weighted loss needs input_ids and labels")

            labels = inputs["labels"]
            input_ids = inputs["input_ids"]

            cot_mask, answer_mask = build_cot_answer_masks(
                input_ids,
                labels,
                marker_ids,
                special_token_ids,
                torch,
            )

            causal_lm = (
                model.get_base_model()
                if hasattr(model, "get_base_model")
                else model
            )
            decoder = getattr(causal_lm, "model", None)
            lm_head = causal_lm.get_output_embeddings()
            if decoder is None or lm_head is None:
                raise TypeError(
                    "the weighted loss requires a Hugging Face CausalLM that exposes "
                    "its inner decoder and output embeddings"
                )

            decoder_inputs = {
                "input_ids": input_ids,
                "use_cache": False,
                "return_dict": True,
            }
            for key in ("attention_mask", "position_ids"):
                if key in inputs:
                    decoder_inputs[key] = inputs[key]
            decoder_outputs = decoder(**decoder_inputs)
            hidden_states = decoder_outputs.last_hidden_state

            shift_hidden_states = hidden_states[:, :-1, :]
            shift_labels = labels[:, 1:].contiguous()
            shifted_cot_mask = cot_mask[:, 1:]
            shifted_answer_mask = answer_mask[:, 1:]
            cot_counts = shifted_cot_mask.sum(dim=1)
            answer_counts = shifted_answer_mask.sum(dim=1)
            if not bool((cot_counts > 0).all()):
                raise ValueError("after the causal shift at least one sample has no explanation tokens")
            if not bool((answer_counts > 0).all()):
                raise ValueError("after the causal shift at least one sample has no answer tokens")

            token_weights = torch.zeros_like(
                shift_labels,
                dtype=hidden_states.dtype,
            )
            token_weights += shifted_cot_mask * (
                cot_loss_weight / cot_counts.unsqueeze(1)
            )
            token_weights += shifted_answer_mask * (
                answer_loss_weight / answer_counts.unsqueeze(1)
            )

            supervised = shift_labels != -100
            selected_hidden = shift_hidden_states[supervised]
            selected_labels = shift_labels[supervised]
            selected_weights = token_weights[supervised]

            def chunk_loss(hidden_chunk, label_chunk, weight_chunk):
                chunk_logits = lm_head(hidden_chunk).float()
                losses = torch.nn.functional.cross_entropy(
                    chunk_logits,
                    label_chunk,
                    reduction="none",
                )
                return (losses * weight_chunk.float()).sum()

            loss_sum = hidden_states.sum() * 0.0
            for start in range(0, len(selected_labels), logit_chunk_size):
                stop = start + logit_chunk_size
                loss_sum = loss_sum + torch.utils.checkpoint.checkpoint(
                    chunk_loss,
                    selected_hidden[start:stop],
                    selected_labels[start:stop],
                    selected_weights[start:stop],
                    use_reentrant=False,
                )
            loss = loss_sum / labels.shape[0]
            if return_outputs:
                return loss, {"loss": loss.detach()}
            return loss

    return CotAnswerWeightedSFTTrainer


def main():
    args = parse_args()
    validate_positive_args(args)

    train_path = Path(args.train_file)
    validation_path = Path(args.validation_file)
    use_weighted_loss = args.loss_mode == LOSS_MODE_COT_ANSWER_WEIGHTED
    train_count = validate_messages_file(
        train_path,
        require_weighted_answer=use_weighted_loss,
    )
    validation_count = validate_messages_file(
        validation_path,
        require_weighted_answer=use_weighted_loss,
    )
    print(f"training file validated: {train_count} rows")
    print(f"validation file validated: {validation_count} rows")

    if args.check_only:
        print("check-only: the model is not loaded and no training starts.")
        return

    try:
        import accelerate
        import datasets
        import peft
        import torch
        import transformers
        import trl
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "missing training dependencies; run: pip install -r requirements.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires a CUDA GPU")

    use_bf16 = bool(torch.cuda.is_bf16_supported())
    use_fp16 = not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    major_capability = torch.cuda.get_device_capability()[0]
    use_tf32 = major_capability >= 8

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("method: LoRA on an unquantised base model")
    print(f"compute dtype: {'bf16' if use_bf16 else 'fp16'}")
    if use_weighted_loss:
        print(
            "training loss: "
            f"{args.cot_loss_weight:.3f} x mean(explanation token loss) + "
            f"{args.answer_loss_weight:.3f} x mean(answer token loss)"
        )
    else:
        print("training loss: mean over completion tokens")
    print(
        "effective batch size on one GPU: "
        f"{args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )

    data_files = {
        "train": str(train_path),
        "validation": str(validation_path),
    }
    raw_datasets = load_dataset("json", data_files=data_files)
    train_dataset = select_subset(
        raw_datasets["train"],
        args.max_train_samples,
        args.seed,
    )
    eval_dataset = select_subset(
        raw_datasets["validation"],
        args.max_eval_samples,
        args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    marker_ids = tokenizer.encode(
        DEFAULT_ANSWER_MARKER,
        add_special_tokens=False,
    )
    if use_weighted_loss and not marker_ids:
        raise ValueError("the tokenizer cannot encode the answer marker")

    if use_weighted_loss:
        length_stats = {
            "train": report_length_stats(
                train_dataset,
                tokenizer,
                args.max_length,
                len(train_dataset),
                args.seed,
            ),
            "validation": report_length_stats(
                eval_dataset,
                tokenizer,
                args.max_length,
                len(eval_dataset),
                args.seed,
            ),
        }
        overlength_splits = {
            name: stats["over_max_length"]
            for name, stats in length_stats.items()
            if stats["over_max_length"]
        }
        if overlength_splits:
            raise ValueError(
                "the weighted loss needs the final answer line untruncated; raise --max-length. "
                f"over-length samples: {overlength_splits}"
            )
    else:
        length_stats = report_length_stats(
            train_dataset,
            tokenizer,
            args.max_length,
            args.length_check_samples,
            args.seed,
        )

    map_kwargs = {
        "remove_columns": ["messages"],
        "desc": "splitting prompt / completion",
    }
    if args.dataset_num_proc > 1:
        map_kwargs["num_proc"] = args.dataset_num_proc

    train_dataset = train_dataset.map(
        messages_to_prompt_completion,
        **map_kwargs,
    )
    eval_dataset = eval_dataset.map(
        messages_to_prompt_completion,
        **map_kwargs,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        model_init_kwargs={
            "dtype": compute_dtype,
            "use_cache": False,
            "local_files_only": args.local_files_only,
        },
        max_length=args.max_length,
        completion_only_loss=True,
        assistant_only_loss=False,
        eos_token=tokenizer.eos_token,
        packing=args.packing,
        eval_packing=False,
        shuffle_dataset=True,
        dataset_num_proc=args.dataset_num_proc,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_tf32,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        seed=args.seed,
        data_seed=args.seed,
    )

    run_config = {
        "arguments": vars(args),
        "resolved": {
            "training_method": "lora_continue" if args.init_adapter else "lora",
            "init_adapter": args.init_adapter,
            "base_model_quantized": False,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "compute_dtype": str(compute_dtype),
            "gpu": torch.cuda.get_device_name(0),
            "length_stats": length_stats,
            "loss_mode": args.loss_mode,
            "loss_formula": (
                {
                    "cot_mean_weight": args.cot_loss_weight,
                    "answer_body_mean_weight": args.answer_loss_weight,
                    "answer_marker": DEFAULT_ANSWER_MARKER,
                    "logit_chunk_size": args.loss_logit_chunk_size,
                }
                if use_weighted_loss
                else None
            ),
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as fout:
        json.dump(run_config, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    trainer_class = SFTTrainer
    if use_weighted_loss:
        trainer_class = make_weighted_sft_trainer_class(
            SFTTrainer,
            torch,
            marker_ids,
            tokenizer.all_special_ids,
            args.cot_loss_weight,
            args.answer_loss_weight,
            args.loss_logit_chunk_size,
        )

    if args.init_adapter:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        print(f"continuing from adapter: {args.init_adapter} (--lora-r/alpha/dropout ignored)")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=compute_dtype,
            use_cache=False,
            local_files_only=args.local_files_only,
        )
        model_for_trainer = PeftModel.from_pretrained(
            base_model,
            args.init_adapter,
            is_trainable=True,
            local_files_only=args.local_files_only,
        )
        peft_config_for_trainer = None
    else:
        model_for_trainer = args.model
        peft_config_for_trainer = peft_config

    trainer = trainer_class(
        model=model_for_trainer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config_for_trainer,
    )

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)

    final_adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"training finished; final LoRA adapter: {final_adapter_dir}")


if __name__ == "__main__":
    main()
