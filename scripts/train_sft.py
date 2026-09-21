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
        description="使用 TRL + PEFT 对 Qwen2.5-3B-Instruct 进行普通 LoRA SFT。"
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
        help="从已有 LoRA adapter 继续训练（新的优化器与学习率调度）；设置后忽略 --lora-r/alpha/dropout。",
    )
    parser.add_argument("--report-to", default="none")
    parser.add_argument(
        "--loss-mode",
        choices=(LOSS_MODE_COMPLETION, LOSS_MODE_COT_ANSWER_WEIGHTED),
        default=LOSS_MODE_COMPLETION,
        help=(
            "completion 使用原始 completion token 平均 loss；"
            "cot_answer_weighted 分别计算解析与答案区域平均 loss。"
        ),
    )
    parser.add_argument("--cot-loss-weight", type=float, default=0.2)
    parser.add_argument("--answer-loss-weight", type=float, default=0.8)
    parser.add_argument(
        "--loss-logit-chunk-size",
        type=int,
        default=256,
        help="加权 loss 每次投影到词表的监督 token 数。",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只校验数据格式，不加载模型、不启动训练。",
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
            raise ValueError(f"{name} 必须大于 0")

    nonnegative_int_names = (
        "max_train_samples",
        "max_eval_samples",
        "length_check_samples",
        "dataloader_num_workers",
    )
    for name in nonnegative_int_names:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} 不能小于 0")

    if args.num_train_epochs <= 0:
        raise ValueError("num_train_epochs 必须大于 0")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio 必须在 [0, 1) 之间")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout 必须在 [0, 1) 之间")
    if args.init_adapter and args.resume_from_checkpoint:
        raise ValueError("--init-adapter 与 --resume-from-checkpoint 不能同时使用")
    if args.loss_mode == LOSS_MODE_COT_ANSWER_WEIGHTED:
        if args.packing:
            raise ValueError("加权解析/答案 loss 不支持 --packing")
        if args.cot_loss_weight < 0 or args.answer_loss_weight < 0:
            raise ValueError("解析和答案 loss 权重不能为负数")
        if not math.isclose(
            args.cot_loss_weight + args.answer_loss_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("解析和答案 loss 权重之和必须等于 1")
        if args.answer_loss_weight == 0:
            raise ValueError("答案 loss 权重必须大于 0")


def validate_messages_file(path, require_weighted_answer=False):
    """完整检查 JSONL schema；返回样本数。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    count = 0
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from exc

            if set(record) != {"messages"}:
                raise ValueError(
                    f"{path} 第 {line_number} 行必须且只能包含 messages"
                )

            messages = record["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path} 第 {line_number} 行 messages 不合法")
            if messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"{path} 第 {line_number} 行最后一条消息必须是 assistant"
                )
            if not any(message.get("role") == "user" for message in messages[:-1]):
                raise ValueError(f"{path} 第 {line_number} 行缺少 user 消息")

            for message in messages:
                if message.get("role") not in {"system", "user", "assistant"}:
                    raise ValueError(
                        f"{path} 第 {line_number} 行包含不支持的 role"
                    )
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        f"{path} 第 {line_number} 行存在空 content"
                    )
            if require_weighted_answer:
                assistant = messages[-1]["content"]
                if not re.search(r"\n答案：[A-Z]+\s*$", assistant):
                    raise ValueError(
                        f"{path} 第 {line_number} 行 assistant 必须以"
                        "换行后的‘答案：字母’结尾，才能使用加权 loss"
                    )
            count += 1

    if count == 0:
        raise ValueError(f"数据文件为空: {path}")
    return count


def messages_to_prompt_completion(example):
    """把 messages 拆开，使 TRL 只对 assistant completion 计算 loss。"""
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
    print("Token 长度抽样:", json.dumps(stats, ensure_ascii=False))
    return stats


def build_cot_answer_masks(
    input_ids,
    labels,
    marker_ids,
    special_token_ids,
    torch,
):
    """按最后一个已监督答案标记划分非答案与答案正文 token。"""
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("input_ids 和 labels 必须是形状相同的二维张量")

    marker = torch.as_tensor(
        marker_ids,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    marker_length = int(marker.numel())
    sequence_length = int(input_ids.shape[1])
    if marker_length == 0 or marker_length > sequence_length:
        raise ValueError("答案标记 token 序列为空或长于当前 batch")

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
            "当前 batch 的 assistant completion 中找不到已监督答案标记 "
            f"{DEFAULT_ANSWER_MARKER!r}，batch 行号: {missing_rows}。"
            "样本可能被 max_length 截断，或标签格式不合法。"
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
            "答案标记后没有可训练的答案正文 token，batch 行号: "
            f"{missing_rows}"
        )
    if not bool(cot_mask.any(dim=1).all()):
        raise ValueError("至少一条样本没有可训练的解析 token")
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
    """创建按样本分别归一化解析与答案 loss 的 SFTTrainer。"""

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
                raise ValueError("加权 loss 需要 input_ids 和 labels")

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
                    "加权 loss 当前要求 Qwen/Hugging Face CausalLM 提供"
                    "内部 decoder 和 output embeddings"
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
                raise ValueError("因 causal shift 导致至少一条样本没有解析 token")
            if not bool((answer_counts > 0).all()):
                raise ValueError("因 causal shift 导致至少一条样本没有答案 token")

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
    print(f"训练集格式检查通过: {train_count} 条")
    print(f"验证集格式检查通过: {validation_count} 条")

    if args.check_only:
        print("check-only 已启用，不加载模型，不启动训练。")
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
            "缺少 SFT 依赖，请先执行: pip install -r requirements-sft.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA 训练需要可用的 NVIDIA CUDA GPU")

    use_bf16 = bool(torch.cuda.is_bf16_supported())
    use_fp16 = not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    major_capability = torch.cuda.get_device_capability()[0]
    use_tf32 = major_capability >= 8

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("训练方式: 普通 LoRA（基础模型不量化）")
    print(f"计算精度: {'bf16' if use_bf16 else 'fp16'}")
    if use_weighted_loss:
        print(
            "训练 loss: "
            f"{args.cot_loss_weight:.3f} × mean(解析/非答案 token loss) + "
            f"{args.answer_loss_weight:.3f} × mean(答案正文 token loss)"
        )
    else:
        print("训练 loss: completion token 平均 loss")
    print(
        "单卡有效 batch size: "
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
        raise ValueError("tokenizer 无法编码答案标记")

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
                "加权 loss 要求末行答案不可被截断；请提高 --max-length。"
                f"超长样本: {overlength_splits}"
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
        "desc": "拆分 prompt/completion",
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

        print(f"从已有 adapter 继续训练: {args.init_adapter}（忽略 --lora-r/alpha/dropout）")
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
    print(f"训练完成，最终 LoRA adapter: {final_adapter_dir}")


if __name__ == "__main__":
    main()
