"""GRPO on a LoRA adapter: bf16 base model, many prompts per step, DAPO/Dr.GRPO options, three output-mode rewards,
and optional segment credit (--segment-credit-lambda, see grpo_segment_credit.py).

Design choices, each independently switchable:
  * bf16 base model by default rather than 4-bit (policy log-probs carry no quantisation noise)
  * --prompts-per-step controls how many questions each optimizer step sees
  * clip-higher（epsilon / epsilon_high）、scale_rewards=none（Dr.GRPO）、KL-free
  * rewards assembled per --reward-mode direct|cot|adaptive (see grpo_rewards.py)
  * optional entropy regularisation
  * every GRPOConfig field is checked against the installed TRL before construction; a missing field is an error, not a silent drop

Data: messages-format JSONL such as the pool written by build_cot_rl_pool.py (cot mode) or the direct-answer
training files (direct mode).
"""

import argparse
import dataclasses
import json
from pathlib import Path

try:
    from .cmexam_prompts import MODE_ADAPTIVE, MODE_COT, MODE_DIRECT, MODES
    from .data_utils import load_tokenizer, parse_messages_record, validate_messages_file
    from .grpo_rewards import RewardConfig, build_reward_functions, describe_reward, self_test
    from .grpo_segment_credit import make_segment_credit_trainer
except ImportError:
    from cmexam_prompts import MODE_ADAPTIVE, MODE_COT, MODE_DIRECT, MODES
    from data_utils import load_tokenizer, parse_messages_record, validate_messages_file
    from grpo_rewards import RewardConfig, build_reward_functions, describe_reward, self_test
    from grpo_segment_credit import make_segment_credit_trainer


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TRAIN_FILE = "training_outputs/rl_pool_v3/rl_pool_train.jsonl"
DEFAULT_VALIDATION_FILE = "training_outputs/rl_pool_v3/rl_pool_validation.jsonl"
DEFAULT_OUTPUT_DIR = "training_outputs/qwen2.5-cmexam-grpo-v3"
MODE_DEFAULTS = {
    MODE_DIRECT: {"max_completion_length": 32, "temperature": 1.2, "think_cost": 0.0},
    MODE_COT: {"max_completion_length": 384, "temperature": 1.0, "think_cost": 0.0},
    MODE_ADAPTIVE: {"max_completion_length": 384, "temperature": 1.0, "think_cost": 0.05},
}


def convert_dataset_example(example):
    # datasets.Dataset.map may pass a LazyRow-like Mapping rather than a dict; materialise it first,
    # parse_messages_record still validates the top-level fields strictly.
    return parse_messages_record(dict(example))


def select_subset(dataset, max_samples, seed):
    if not max_samples or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def prepare_datasets(load_dataset, args):
    raw = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.validation_file,
        },
    )
    converted = {}
    for split_name, split_dataset in raw.items():
        map_kwargs = {
            "remove_columns": split_dataset.column_names,
            "desc": f"converting {split_name} to GRPO prompts",
        }
        if args.dataset_num_proc > 1:
            map_kwargs["num_proc"] = args.dataset_num_proc
        converted[split_name] = split_dataset.map(
            convert_dataset_example,
            **map_kwargs,
        )

    train_dataset = select_subset(
        converted["train"],
        args.max_train_samples,
        args.seed,
    )
    eval_dataset = select_subset(
        converted["validation"],
        args.max_eval_samples,
        args.seed,
    )
    return train_dataset, eval_dataset


def load_policy_model(AutoModelForCausalLM, PeftModel, torch, args, compute_dtype):
    """Load the bf16 base model and the LoRA adapter to continue training (is_trainable=True)."""
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=compute_dtype, use_cache=False, local_files_only=args.local_files_only,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=True, local_files_only=args.local_files_only)
    adapter_config = model.peft_config["default"]
    configured_base = getattr(adapter_config, "base_model_name_or_path", None)
    if configured_base and configured_base != args.model:
        print(
            "note: the adapter_config base model is "
            f"{configured_base!r}; loading {args.model!r} as requested."
        )
    return model, adapter_config


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO on a LoRA adapter (bf16), with optional segment credit.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory of the SFT (or previous GRPO) model to continue from.")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation-file", default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=256)

    # rewards
    parser.add_argument("--reward-mode", choices=MODES, default=MODE_DIRECT)
    parser.add_argument("--answer-reward-weight", type=float, default=0.95)
    parser.add_argument("--format-reward-weight", type=float, default=0.05)
    parser.add_argument("--format-warmup-steps", type=int, default=50)
    parser.add_argument("--think-cost", type=float, default=None, help="default: direct 0, cot 0, adaptive 0.05")
    parser.add_argument("--think-cost-chars", type=int, default=150)
    parser.add_argument("--think-cost-max-multiplier", type=float, default=3.0)
    parser.add_argument("--multi-partial-weight", type=float, default=0.5)
    parser.add_argument("--multi-partial-anneal-steps", type=int, default=100)
    parser.add_argument("--overlong-weight", type=float, default=0.2)
    parser.add_argument("--overlong-cache-tokens", type=int, default=64)

    # batch geometry
    parser.add_argument("--prompts-per-step", type=int, default=32, help="questions per optimizer step.")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=16, help="completions per forward pass.")
    parser.add_argument("--generation-batch-size", type=int, default=None, help="TRL generation_batch_size (default: the whole step). TRL 1.8 still hands all completions of a step to generate at once; --generation-chunk is what bounds memory.")
    parser.add_argument("--generation-chunk", type=int, default=32, help="completions actually passed to generate per call (chunked generation saves memory; a chunk that still OOMs is split in half).")
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--num-generations-eval", type=int, default=1)

    # sampling and length
    parser.add_argument("--max-completion-length", type=int, default=None, help="default: direct 32, otherwise 384")
    parser.add_argument("--temperature", type=float, default=None, help="default: direct 1.2, otherwise 1.0")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--mask-truncated-completions", action=argparse.BooleanOptionalAction, default=False)

    # optimisation
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=0.28)
    parser.add_argument("--scale-rewards", choices=("group", "batch", "none"), default="none")
    parser.add_argument("--loss-type", choices=("grpo", "dr_grpo", "dapo", "bnpo", "cispo"), default="dapo")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--adaptive-entropy", action="store_true")
    parser.add_argument("--entropy-target", type=float, default=0.2)

    # generation backend

    # misc
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=6)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-completions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    # segment credit (grpo_segment_credit.py): enabled when λ>0; segment credit = change in probed confidence, token advantage = sequence advantage + λ×credit
    parser.add_argument("--segment-credit-lambda", type=float, default=0.0, help="0 = standard GRPO; 2.0 recommended (+0.5 confidence earns +1)")
    parser.add_argument("--segment-max-probes", type=int, default=8, help="maximum probes per completion; extra segments are merged evenly")
    parser.add_argument("--segment-probe-batch", type=int, default=64, help="batch size of the probe forward pass")
    return parser.parse_args()


def resolve_mode_defaults(args):
    defaults = MODE_DEFAULTS[args.reward_mode]
    if args.max_completion_length is None:
        args.max_completion_length = defaults["max_completion_length"]
    if args.temperature is None:
        args.temperature = defaults["temperature"]
    if args.think_cost is None:
        args.think_cost = defaults["think_cost"]
    return args


def derive_batch_geometry(prompts_per_step, num_generations, micro_batch, generation_batch_size=None):
    """Convert "prompts per step × generations per prompt" into TRL batch sizes and accumulation steps.

    generation_batch_size: completions generated per call (default: the whole step). Smaller values reduce
    generation memory (KV cache and prefill activations scale with it) while each optimizer step still covers
    the same questions.
    """
    if prompts_per_step < 1 or num_generations < 2 or micro_batch < 1:
        raise ValueError("prompts_per_step ≥ 1、num_generations ≥ 2、micro_batch ≥ 1")
    completions_per_step = prompts_per_step * num_generations
    if completions_per_step % micro_batch != 0:
        raise ValueError(
            f"prompts_per_step × num_generations = {completions_per_step} "
            f"must be divisible by micro_batch = {micro_batch}"
        )
    if micro_batch > completions_per_step:
        raise ValueError("micro_batch cannot exceed the completions per step")
    if generation_batch_size is None:
        generation_batch_size = completions_per_step
    if (generation_batch_size < micro_batch or generation_batch_size % micro_batch != 0
            or generation_batch_size % num_generations != 0 or completions_per_step % generation_batch_size != 0):
        raise ValueError(
            f"generation_batch_size = {generation_batch_size} must be a common multiple of micro_batch and num_generations "
            f"and divide the completions per step ({completions_per_step})"
        )
    gradient_accumulation_steps = completions_per_step // micro_batch
    return {
        "prompts_per_step": prompts_per_step,
        "num_generations": num_generations,
        "completions_per_step": completions_per_step,
        "per_device_train_batch_size": micro_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "generation_batch_size": generation_batch_size,
    }


def validate_args(args):
    credit_lambda = getattr(args, "segment_credit_lambda", 0.0)
    if credit_lambda < 0:
        raise ValueError("segment_credit_lambda must not be negative")
    if credit_lambda > 0 and args.reward_mode == MODE_DIRECT:
        raise ValueError("segment credit needs cot or adaptive mode (the output must contain a rationale)")
    resolve_mode_defaults(args)
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    if args.learning_rate <= 0 or args.temperature <= 0:
        raise ValueError("learning_rate and temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if args.epsilon <= 0 or args.epsilon_high < args.epsilon:
        raise ValueError("need 0 < epsilon <= epsilon_high")
    if args.beta < 0 or args.entropy_coef < 0:
        raise ValueError("beta and entropy_coef must not be negative")
    if args.per_device_eval_batch_size % args.num_generations_eval != 0:
        raise ValueError("per_device_eval_batch_size must be divisible by num_generations_eval")
    if args.max_eval_samples and args.max_eval_samples % args.per_device_eval_batch_size != 0:
        print(
            "note: max_eval_samples is not a multiple of per_device_eval_batch_size; "
            "the last evaluation batch will be smaller."
        )
    geometry = derive_batch_geometry(
        args.prompts_per_step, args.num_generations, args.micro_batch, getattr(args, "generation_batch_size", None)
    )
    return geometry


def build_reward_config(args):
    return RewardConfig(
        mode=args.reward_mode,
        answer_weight=args.answer_reward_weight,
        format_weight=args.format_reward_weight,
        format_warmup_steps=args.format_warmup_steps,
        think_cost=args.think_cost,
        think_cost_chars=args.think_cost_chars,
        think_cost_max_multiplier=args.think_cost_max_multiplier,
        multi_partial_weight=args.multi_partial_weight,
        multi_partial_anneal_steps=args.multi_partial_anneal_steps,
        overlong_weight=args.overlong_weight,
        max_completion_length=args.max_completion_length,
        overlong_cache_tokens=min(args.overlong_cache_tokens, args.max_completion_length),
    ).validate()


def build_grpo_config_kwargs(args, geometry, output_dir, use_bf16, reward_weights):
    kwargs = {
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        # transformers 5.x expresses the warm-up ratio as a fractional warmup_steps; older versions have warmup_ratio.
        # adapt_warmup_field switches to whichever the installed version supports.
        "warmup_steps": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "per_device_train_batch_size": geometry["per_device_train_batch_size"],
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": geometry["gradient_accumulation_steps"],
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_grad_norm": args.max_grad_norm,
        "optim": "adamw_torch_fused",
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": args.report_to,
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": False,
        "shuffle_dataset": True,
        "disable_dropout": True,
        "num_generations": args.num_generations,
        "num_generations_eval": args.num_generations_eval,
        "num_iterations": args.num_iterations,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "epsilon_high": args.epsilon_high,
        "reward_weights": list(reward_weights),
        "multi_objective_aggregation": "sum_then_normalize",
        "scale_rewards": args.scale_rewards,
        "loss_type": args.loss_type,
        "mask_truncated_completions": args.mask_truncated_completions,
        "log_completions": args.log_completions,
        "num_completions_to_print": 4 if args.log_completions else None,
    }
    if args.entropy_coef > 0 or args.adaptive_entropy:
        kwargs["entropy_coef"] = args.entropy_coef
        kwargs["use_adaptive_entropy"] = args.adaptive_entropy
        kwargs["entropy_target"] = args.entropy_target
    return kwargs


def adapt_warmup_field(config_cls, kwargs):
    """Choose warmup_ratio or a fractional warmup_steps depending on the installed TrainingArguments."""
    known = {field.name for field in dataclasses.fields(config_cls)}
    ratio = kwargs.pop("warmup_steps", None)
    if ratio is None:
        return kwargs
    if "warmup_ratio" in known and "warmup_steps" in known:
        kwargs["warmup_ratio"] = ratio
    elif "warmup_steps" in known:
        kwargs["warmup_steps"] = ratio
    else:
        raise ValueError("the installed TrainingArguments has neither warmup_ratio nor warmup_steps")
    return kwargs


def validate_config_fields(config_cls, kwargs):
    """Check that every field exists in the installed GRPOConfig, so version drift cannot silently drop one."""
    known = {field.name for field in dataclasses.fields(config_cls)}
    unknown = sorted(key for key in kwargs if key not in known)
    if unknown:
        raise ValueError(
            "the installed GRPOConfig does not support these fields; upgrade TRL or drop the feature: "
            + ", ".join(unknown)
        )
    return kwargs


def split_half(value):
    """Split a list, a dict of lists or None into two halves (used to halve a generation batch after an OOM)."""
    if value is None:
        return None, None
    if isinstance(value, dict):
        halves = {key: split_half(item) for key, item in value.items()}
        return {k: v[0] for k, v in halves.items()}, {k: v[1] for k, v in halves.items()}
    middle = len(value) // 2
    return value[:middle], value[middle:]


def with_memory_logging(trainer_cls, torch, generation_chunk=32):
    """Wrap a trainer class with two additions:
    1. Memory logging: before each log, record the step's peak / current / reserved GPU memory (GB), the generation
       peak and the longest prompt, then reset the peak counter. A subclass is used rather than a callback because
       on_log fires after the entry is already in log_history.
    2. Chunked generation with OOM recovery: TRL 1.8 hands every completion of a step (prompts × samples, 128 here)
       to generate at once and GRPOConfig's generation_batch_size does not reach that call, so completions are
       generated generation_chunk at a time and concatenated. If a chunk still runs out of memory, leave the except
       block first (the exception object would otherwise keep the failed forward pass's tensors alive), free the
       cache, and retry in two halves. The result equals a single call (samples are independent); splits are
       counted in gen/oom_splits."""

    class MemoryLoggingTrainer(trainer_cls):
        generation_chunk_size = generation_chunk

        def _generate_chunk(self, prompt_ids, images, multimodal_fields):
            oom = False
            try:
                return super()._generate_single_turn(prompt_ids, images, multimodal_fields)
            except torch.OutOfMemoryError:
                if len(prompt_ids) < 2:
                    raise
                oom = True
            # the exception is released here; free what the failed forward pass left behind, then retry in halves
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            self._metrics["train"]["gen/oom_splits"].append(1.0)
            print(f"generation ran out of memory: retrying {len(prompt_ids)} prompts in two halves", flush=True)
            left_ids, right_ids = split_half(prompt_ids)
            left_images, right_images = split_half(images)
            left_fields, right_fields = split_half(multimodal_fields or {})
            return self._merge_generations(self._generate_chunk(left_ids, left_images, left_fields),
                                           self._generate_chunk(right_ids, right_images, right_fields))

        @staticmethod
        def _merge_generations(left, right):
            completions = list(left[0]) + list(right[0])
            logprobs = None if left[1] is None or right[1] is None else list(left[1]) + list(right[1])
            return completions, logprobs

        def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
            if torch.cuda.is_available():
                self._metrics["train"]["gen/prompts"].append(float(len(prompt_ids)))
                self._metrics["train"]["gen/max_prompt_len"].append(float(max(len(ids) for ids in prompt_ids)))
            size = max(1, int(self.generation_chunk_size))
            result = None
            for start in range(0, len(prompt_ids), size):
                chunk_ids = prompt_ids[start:start + size]
                chunk_images = None if images is None else images[start:start + size]
                chunk_fields = {k: v[start:start + size] for k, v in (multimodal_fields or {}).items()}
                piece = self._generate_chunk(chunk_ids, chunk_images, chunk_fields)
                result = piece if result is None else self._merge_generations(result, piece)
            if torch.cuda.is_available():
                self._metrics["train"]["gen/peak_gb"].append(round(torch.cuda.max_memory_allocated() / 1e9, 2))
            return result

        def log(self, logs, *log_args, **log_kwargs):
            if torch.cuda.is_available():
                logs["mem/peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                logs["mem/alloc_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
                logs["mem/reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
                torch.cuda.reset_peak_memory_stats()
            return super().log(logs, *log_args, **log_kwargs)

    return MemoryLoggingTrainer


def print_reward_self_test(reward_config):
    rows = self_test(reward_config)
    print("reward self-test (synthetic outputs):")
    for row in rows:
        print("  " + json.dumps(row, ensure_ascii=False))


def main():
    args = parse_args()
    geometry = validate_args(args)
    reward_config = build_reward_config(args)
    reward_description = describe_reward(reward_config)

    train_stats = validate_messages_file(Path(args.train_file))
    validation_stats = validate_messages_file(Path(args.validation_file))
    print("train data check passed:", json.dumps(train_stats, ensure_ascii=False))
    print("validation data check passed:", json.dumps(validation_stats, ensure_ascii=False))
    print("batch geometry:", json.dumps(geometry, ensure_ascii=False))
    print("reward:", json.dumps(reward_description, ensure_ascii=False))
    print_reward_self_test(reward_config)

    if args.check_only:
        print("check-only: the model is not loaded and GRPO is not started.")
        return
    if not args.adapter:
        raise ValueError("training requires an SFT adapter via --adapter")

    try:
        import accelerate
        import datasets
        import peft
        import torch
        import transformers
        import trl
        from datasets import load_dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise RuntimeError("GRPO dependencies are missing: pip install -r requirements.txt") from error

    if not torch.cuda.is_available():
        raise RuntimeError("GRPO needs an NVIDIA CUDA GPU")
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"precision: {'bf16' if use_bf16 else 'fp16'}")

    train_dataset, eval_dataset = prepare_datasets(load_dataset, args)
    tokenizer = load_tokenizer(AutoTokenizer, args)
    model, adapter_config = load_policy_model(AutoModelForCausalLM, PeftModel, torch, args, compute_dtype)
    model.print_trainable_parameters()

    reward_functions, reward_weights = build_reward_functions(reward_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_kwargs = validate_config_fields(
        GRPOConfig,
        adapt_warmup_field(
            GRPOConfig,
            build_grpo_config_kwargs(args, geometry, output_dir, use_bf16, reward_weights),
        ),
    )
    if use_bf16 and torch.cuda.get_device_capability()[0] >= 8:
        config_kwargs["tf32"] = True
    training_args = GRPOConfig(**config_kwargs)

    run_config = {
        "arguments": vars(args),
        "batch_geometry": geometry,
        "reward": reward_description,
        "grpo_config": {
            key: value for key, value in config_kwargs.items() if key != "output_dir"
        },
        "resolved": {
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "compute_dtype": str(compute_dtype),
            "gpu": torch.cuda.get_device_name(0),
            "train_source_stats": train_stats,
            "validation_source_stats": validation_stats,
            "adapter_base_model": getattr(adapter_config, "base_model_name_or_path", None),
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
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as output_file:
        json.dump(run_config, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    trainer_cls, trainer_extra = GRPOTrainer, {}
    if args.segment_credit_lambda > 0:
        trainer_cls = make_segment_credit_trainer(GRPOTrainer, torch)
        trainer_extra = {"credit_lambda": args.segment_credit_lambda, "max_probes": args.segment_max_probes,
                         "probe_batch_size": args.segment_probe_batch}
        with (output_dir / "segment_credit.json").open("w", encoding="utf-8") as output_file:
            json.dump(trainer_extra, output_file, ensure_ascii=False, indent=2)
        print(f"segment credit enabled: {trainer_extra}")
    trainer = with_memory_logging(trainer_cls, torch, generation_chunk=args.generation_chunk)(
        model=model,
        args=training_args,
        reward_funcs=reward_functions,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        **trainer_extra,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)

    final_adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"GRPO finished; final adapter: {final_adapter_dir}")


if __name__ == "__main__":
    main()
