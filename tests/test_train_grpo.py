import dataclasses
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.cmexam_prompts import MODE_ADAPTIVE, MODE_COT, MODE_DIRECT
from scripts.train_grpo import (
    adapt_warmup_field,
    build_grpo_config_kwargs,
    build_reward_config,
    derive_batch_geometry,
    resolve_mode_defaults,
    validate_args,
    validate_config_fields,
)


# 从 trl==1.8.0 的 GRPOConfig 与 transformers==5.17.0 的 TrainingArguments 源码中抽取的字段名，
# 用于在本地（无 GPU、无 TRL）就能发现拼写或版本漂移问题。
TRL_GRPO_CONFIG_FIELDS = set(
    """learning_rate model_init_kwargs trust_remote_code router_aux_loss_coef disable_dropout
    remove_unused_columns num_generations num_generations_eval max_completion_length shuffle_dataset
    pad_to_multiple_of generation_batch_size steps_per_generation temperature top_p top_k min_p
    generation_kwargs chat_template_kwargs repetition_penalty cache_implementation use_vllm vllm_mode
    vllm_model_impl vllm_enable_sleep_mode vllm_structured_outputs_regex vllm_server_base_url
    vllm_server_host vllm_server_port vllm_server_timeout vllm_group_port vllm_gpu_memory_utilization
    vllm_max_model_length vllm_tensor_parallel_size beta num_iterations epsilon delta epsilon_high
    sapo_temperature_neg sapo_temperature_pos vespo_k_pos vespo_lambda_pos vespo_k_neg vespo_lambda_neg
    importance_sampling_level reward_weights multi_objective_aggregation scale_rewards loss_type
    mask_truncated_completions sync_ref_model ref_model_mixup_alpha ref_model_sync_steps
    top_entropy_quantile entropy_coef use_adaptive_entropy entropy_coef_min entropy_coef_max
    entropy_coef_delta entropy_target max_tool_calling_iterations vllm_importance_sampling_correction
    vllm_importance_sampling_mode vllm_importance_sampling_clip_max vllm_importance_sampling_clip_min
    off_policy_mask_threshold use_bias_correction_kl log_completions num_completions_to_print
    log_unique_prompts log_completions_hub_repo use_transformers_continuous_batching
    transformers_continuous_batching_config use_transformers_paged vllm_importance_sampling_cap""".split()
)
TRAINING_ARGUMENTS_FIELDS = set(
    """accelerator_config adam_beta1 adam_beta2 adam_epsilon auto_find_batch_size
    average_tokens_across_devices batch_eval_metrics bf16 bf16_full_eval data_seed dataloader_drop_last
    dataloader_in_order dataloader_multiprocessing_context dataloader_num_workers
    dataloader_persistent_workers dataloader_pin_memory dataloader_prefetch_factor ddp_backend
    ddp_broadcast_buffers ddp_bucket_cap_mb ddp_find_unused_parameters ddp_static_graph ddp_timeout debug
    deepspeed disable_tqdm do_eval do_predict do_train enable_jit_checkpoint eval_accumulation_steps
    eval_delay eval_do_concat_batches eval_on_start eval_steps eval_strategy eval_use_gather_object fp16
    fp16_full_eval fsdp fsdp_config full_determinism gradient_accumulation_steps gradient_checkpointing
    gradient_checkpointing_kwargs greater_is_better hub_always_push hub_model_id hub_private_repo
    hub_revision hub_strategy hub_token ignore_data_skip include_for_metrics include_num_input_tokens_seen
    label_names label_smoothing_factor learning_rate length_column_name liger_kernel_config
    load_best_model_at_end local_rank log_level log_level_replica log_on_each_node logging_first_step
    logging_nan_inf_filter logging_steps logging_strategy lr_scheduler_kwargs lr_scheduler_type
    max_grad_norm max_steps metric_for_best_model neftune_noise_alpha num_train_epochs optim optim_args
    optim_target_modules output_dir parallelism_config per_device_eval_batch_size
    per_device_train_batch_size prediction_loss_only project push_to_hub remove_unused_columns report_to
    restore_callback_states_from_checkpoint resume_from_checkpoint run_name save_on_each_node
    save_only_model save_steps save_strategy save_total_limit seed skip_memory_metrics tf32 torch_compile
    torch_compile_backend torch_compile_mode torch_empty_cache_steps trackio_bucket_id trackio_space_id
    trackio_static_space_id train_sampling_strategy use_cache use_cpu use_liger_kernel warmup_steps
    weight_decay""".split()
)
KNOWN_FIELDS = TRL_GRPO_CONFIG_FIELDS | TRAINING_ARGUMENTS_FIELDS


def make_args(**overrides):
    values = {
        "reward_mode": MODE_DIRECT,
        "answer_reward_weight": 0.95,
        "format_reward_weight": 0.05,
        "format_warmup_steps": 50,
        "think_cost": None,
        "think_cost_chars": 150,
        "think_cost_max_multiplier": 3.0,
        "multi_partial_weight": 0.5,
        "multi_partial_anneal_steps": 100,
        "overlong_weight": 0.2,
        "overlong_cache_tokens": 64,
        "prompts_per_step": 32,
        "num_generations": 8,
        "micro_batch": 16,
        "num_iterations": 1,
        "per_device_eval_batch_size": 16,
        "num_generations_eval": 1,
        "max_completion_length": None,
        "temperature": None,
        "top_p": 1.0,
        "mask_truncated_completions": False,
        "max_steps": -1,
        "num_train_epochs": 1.0,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "scale_rewards": "none",
        "loss_type": "dapo",
        "beta": 0.0,
        "entropy_coef": 0.0,
        "adaptive_entropy": False,
        "entropy_target": 0.2,
        "use_vllm": False,
        "vllm_gpu_memory_utilization": 0.3,
        "vllm_sleep_mode": False,
        "vllm_max_model_length": 2048,
        "vllm_is_correction": True,
        "transformers_continuous_batching": False,
        "logging_steps": 1,
        "eval_steps": 50,
        "save_steps": 50,
        "save_total_limit": 6,
        "dataset_num_proc": 4,
        "dataloader_num_workers": 2,
        "seed": 42,
        "report_to": "none",
        "resume_from_checkpoint": None,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "log_completions": True,
        "local_files_only": False,
        "check_only": True,
        "max_eval_samples": 256,
    }
    values.update(overrides)
    return Namespace(**values)


class GeometryTests(unittest.TestCase):
    def test_prompts_per_step_translates_to_accumulation(self):
        geometry = derive_batch_geometry(32, 8, 16)
        self.assertEqual(geometry["gradient_accumulation_steps"], 16)
        self.assertEqual(geometry["generation_batch_size"], 256)
        self.assertEqual(geometry["per_device_train_batch_size"], 16)
        self.assertEqual(derive_batch_geometry(1, 8, 8)["gradient_accumulation_steps"], 1)
        halves = derive_batch_geometry(16, 8, 8, generation_batch_size=64)
        self.assertEqual((halves["generation_batch_size"], halves["gradient_accumulation_steps"]), (64, 16))
        self.assertEqual(derive_batch_geometry(16, 8, 8)["generation_batch_size"], 128)
        for bad in (48, 4, 256, 96):   # 不是 8 的公倍数 / 小于 micro_batch / 超过每步总数 / 不能整除 128
            with self.assertRaises(ValueError):
                derive_batch_geometry(16, 8, 8, generation_batch_size=bad)
        with self.assertRaises(ValueError):
            derive_batch_geometry(3, 8, 16)
        with self.assertRaises(ValueError):
            derive_batch_geometry(1, 1, 1)

    def test_mode_defaults(self):
        direct = resolve_mode_defaults(make_args())
        self.assertEqual((direct.max_completion_length, direct.temperature, direct.think_cost), (32, 1.2, 0.0))
        adaptive = resolve_mode_defaults(make_args(reward_mode=MODE_ADAPTIVE))
        self.assertEqual((adaptive.max_completion_length, adaptive.temperature, adaptive.think_cost), (384, 1.0, 0.05))
        explicit = resolve_mode_defaults(make_args(reward_mode=MODE_COT, temperature=0.8, think_cost=0.02))
        self.assertEqual((explicit.max_completion_length, explicit.temperature, explicit.think_cost), (384, 0.8, 0.02))

    def test_validate_args_rejects_bad_combinations(self):
        with self.assertRaises(ValueError):
            validate_args(make_args(epsilon_high=0.1))
        with self.assertRaises(ValueError):
            validate_args(make_args(use_vllm=True, transformers_continuous_batching=True))
        with self.assertRaises(ValueError):
            validate_args(make_args(use_vllm=True, load_in_4bit=True))
        with self.assertRaises(ValueError):
            validate_args(make_args(per_device_eval_batch_size=3, num_generations_eval=2))
        self.assertEqual(validate_args(make_args())["completions_per_step"], 256)


class ConfigTests(unittest.TestCase):
    def test_reward_config_caps_overlong_cache(self):
        args = resolve_mode_defaults(make_args())
        config = build_reward_config(args)
        self.assertEqual(config.overlong_cache_tokens, 32)
        adaptive = build_reward_config(resolve_mode_defaults(make_args(reward_mode=MODE_ADAPTIVE)))
        self.assertEqual(adaptive.overlong_cache_tokens, 64)
        self.assertEqual(adaptive.think_cost, 0.05)

    def test_all_config_keys_exist_in_real_dataclasses(self):
        for overrides in (
            {},
            {"reward_mode": MODE_ADAPTIVE, "use_vllm": True, "vllm_sleep_mode": True},
            {"reward_mode": MODE_COT, "entropy_coef": 0.001, "adaptive_entropy": True},
            {"transformers_continuous_batching": True},
        ):
            args = make_args(**overrides)
            geometry = validate_args(args)
            kwargs = build_grpo_config_kwargs(args, geometry, Path("out"), True, [0.95, 0.05])
            unknown = sorted(key for key in kwargs if key not in KNOWN_FIELDS)
            self.assertEqual(unknown, [], f"未知字段: {unknown}")
            self.assertEqual(kwargs["epsilon_high"], 0.28)
            self.assertEqual(kwargs["scale_rewards"], "none")
            self.assertEqual(kwargs["beta"], 0.0)
            self.assertEqual(kwargs["gradient_accumulation_steps"], 16)
            self.assertEqual("use_vllm" in kwargs, bool(overrides.get("use_vllm")))
            self.assertEqual("entropy_coef" in kwargs, "entropy_coef" in overrides)

    def test_warmup_field_adapts_to_installed_transformers(self):
        @dataclasses.dataclass
        class Modern:
            warmup_steps: float = 0

        @dataclasses.dataclass
        class Legacy:
            warmup_steps: float = 0
            warmup_ratio: float = 0

        self.assertEqual(adapt_warmup_field(Modern, {"warmup_steps": 0.03}), {"warmup_steps": 0.03})
        self.assertEqual(adapt_warmup_field(Legacy, {"warmup_steps": 0.03}), {"warmup_ratio": 0.03})
        self.assertEqual(adapt_warmup_field(Modern, {"x": 1}), {"x": 1})

    def test_validate_config_fields_reports_unknown(self):
        @dataclasses.dataclass
        class Dummy:
            known: int = 0

        self.assertEqual(validate_config_fields(Dummy, {"known": 1}), {"known": 1})
        with self.assertRaisesRegex(ValueError, "typo_field"):
            validate_config_fields(Dummy, {"known": 1, "typo_field": 2})


if __name__ == "__main__":
    unittest.main()


class SplitHalfTests(unittest.TestCase):
    def test_split_half_handles_lists_dicts_and_none(self):
        from scripts.train_grpo import split_half
        self.assertEqual(split_half([1, 2, 3, 4, 5]), ([1, 2], [3, 4, 5]))
        self.assertEqual(split_half(None), (None, None))
        self.assertEqual(split_half({"a": [1, 2], "b": [3, 4]}), ({"a": [1], "b": [3]}, {"a": [2], "b": [4]}))
        self.assertEqual(split_half({}), ({}, {}))


try:
    import torch as _torch
except ImportError:
    _torch = None


@unittest.skipIf(_torch is None, "需要 torch")
class RobustGenerationTests(unittest.TestCase):
    def test_generation_splits_on_oom_and_logs(self):
        from collections import defaultdict
        from scripts.train_grpo import with_memory_logging

        class FakeBase:
            calls = []
            fail_sizes = {4}

            def __init__(self):
                self._metrics = {"train": defaultdict(list)}

            def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
                FakeBase.calls.append(len(prompt_ids))
                if len(prompt_ids) in FakeBase.fail_sizes:
                    raise _torch.OutOfMemoryError("fake oom")
                return [[7] * len(ids) for ids in prompt_ids], None

            def log(self, logs):
                return logs

        # 分块 4 条：整批 4 条会 OOM，拆成 2+2 成功
        trainer = with_memory_logging(FakeBase, _torch, generation_chunk=4)()
        completions, logprobs = trainer._generate_single_turn([[1, 2], [3], [4, 5, 6], [8]], None, {})
        self.assertEqual(completions, [[7, 7], [7], [7, 7, 7], [7]])
        self.assertIsNone(logprobs)
        self.assertEqual(FakeBase.calls, [4, 2, 2])
        self.assertEqual(trainer._metrics["train"]["gen/oom_splits"], [1.0])
        # 分块 2 条：6 条 prompt 分成 2+2+2，不触发 OOM 路径
        FakeBase.calls.clear()
        trainer = with_memory_logging(FakeBase, _torch, generation_chunk=2)()
        completions, _ = trainer._generate_single_turn([[1]] * 6, None, {"k": list(range(6))})
        self.assertEqual(len(completions), 6)
        self.assertEqual(FakeBase.calls, [2, 2, 2])
        self.assertNotIn("gen/oom_splits", trainer._metrics["train"])
