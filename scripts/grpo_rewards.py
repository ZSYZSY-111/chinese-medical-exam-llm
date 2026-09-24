"""Verifiable GRPO rewards for the direct / cot / adaptive output modes.

Every reward depends only on the official answer letters and the output text: no model is called and the
reference rationale is never used to score "medical correctness". Total reward (TRL sums with reward_weights):

    R = w_a · correct
      − w_a · λ · correct · used_cot · min(L / L0, m)      (thinking cost, cot/adaptive only)
      + w_f · strict_format
      + w_a · w_m · anneal(step) · jaccard · [multi-answer and not an exact match]
      + w_o · overlong                                     (DAPO-style soft penalty, cot/adaptive only)

where correct is the exact answer-set match, L the rationale length in characters, L0 the reference length
(150 characters) and m the cap on the cost multiplier. Right without a rationale → full reward; right with a
rationale → slightly less; wrong → 0.
"""

import re
from dataclasses import asdict, dataclass
from types import SimpleNamespace

try:
    from .cmexam_prompts import MODE_ADAPTIVE, MODE_COT, MODE_DIRECT, MODES
    from .answer_utils import (
        completion_to_text,
        extract_predicted_answer,
        normalize_answer,
    )
except ImportError:
    from cmexam_prompts import MODE_ADAPTIVE, MODE_COT, MODE_DIRECT, MODES
    from answer_utils import (
        completion_to_text,
        extract_predicted_answer,
        normalize_answer,
    )


_COT_PATTERN = re.compile(
    r"^解析：(?P<analysis>\S[\s\S]*?)\n答案：(?P<answer>[A-Z]+)$"
)
_ADAPTIVE_PATTERN = re.compile(
    r"^(?:解析：(?P<analysis>\S[\s\S]*?)\n)?答案：(?P<answer>[A-Z]+)$"
)
_LABELED_ANSWER_LINE = re.compile(
    r"(?im)^[ \t]*(?:最终答案|答案)[ \t]*[:：][ \t]*([A-Za-z](?:[\s,，、/]*[A-Za-z])*)[ \t]*$"
)
_ANY_ANSWER_LABEL = re.compile(r"(?m)^\s*(?:最终答案|答案)\s*[:：]")


@dataclass
class RewardConfig:
    mode: str = MODE_DIRECT
    answer_weight: float = 0.95
    format_weight: float = 0.05
    format_warmup_steps: int = 0
    think_cost: float = 0.0
    think_cost_chars: int = 150
    think_cost_max_multiplier: float = 3.0
    multi_partial_weight: float = 0.5
    multi_partial_anneal_steps: int = 100
    overlong_weight: float = 0.2
    max_completion_length: int = 32
    overlong_cache_tokens: int = 64

    def validate(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown reward mode: {self.mode!r}")
        if self.answer_weight <= 0:
            raise ValueError("answer_weight must be positive")
        for name in (
            "format_weight",
            "think_cost",
            "multi_partial_weight",
            "overlong_weight",
            "think_cost_max_multiplier",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        for name in (
            "format_warmup_steps",
            "think_cost_chars",
            "multi_partial_anneal_steps",
            "overlong_cache_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.max_completion_length < 1:
            raise ValueError("max_completion_length must be positive")
        if self.mode == MODE_DIRECT and self.think_cost > 0:
            raise ValueError("direct mode has no rationale, so think_cost cannot be set")
        if (
            self.mode != MODE_DIRECT
            and self.overlong_weight > 0
            and self.overlong_cache_tokens > self.max_completion_length
        ):
            raise ValueError("overlong_cache_tokens cannot exceed max_completion_length")
        return self


def global_step(trainer_state):
    step = getattr(trainer_state, "global_step", None)
    if step is None and isinstance(trainer_state, dict):
        step = trainer_state.get("global_step")
    return int(step) if isinstance(step, (int, float)) else 0


def _extract_labeled_answer(text, valid_letters):
    matches = list(_LABELED_ANSWER_LINE.finditer(text))
    if len(matches) != 1:
        return None
    return normalize_answer(matches[0].group(1), valid_letters)


def parse_completion(completion, mode, valid_letters, lenient_answer=False):
    """Parse one completion; returns the answer, strict-format flag, whether a rationale was written and its length.

    direct: the answer may be bare letters (a single "答案：X" line is tolerated); strict format means bare letters only.
    cot / adaptive: the answer must be a single "答案：X" line; with lenient_answer=True (format warm-up) bare
    letters also count as an answer, but the format reward stays 0.
    """
    if mode not in MODES:
        raise ValueError(f"unknown output mode: {mode!r}")
    text = completion_to_text(completion).replace("\r\n", "\n").strip()
    result = {
        "text": text,
        "answer": None,
        "strict": False,
        "used_cot": False,
        "analysis_chars": 0,
    }
    if not text:
        return result

    if mode == MODE_DIRECT:
        result["answer"] = extract_predicted_answer(text, valid_letters)
        normalized = normalize_answer(text, valid_letters)
        result["strict"] = normalized is not None and text == normalized
        return result

    labeled = _extract_labeled_answer(text, valid_letters)
    if labeled is not None:
        result["answer"] = labeled
    elif lenient_answer:
        result["answer"] = normalize_answer(text, valid_letters)

    pattern = _COT_PATTERN if mode == MODE_COT else _ADAPTIVE_PATTERN
    match = pattern.fullmatch(text)
    if match is None:
        result["used_cot"] = text.startswith("解析：")
        return result

    analysis = match.group("analysis")
    raw_answer = match.group("answer")
    normalized = normalize_answer(raw_answer, valid_letters)
    strict = normalized is not None and raw_answer == normalized
    if analysis and _ANY_ANSWER_LABEL.search(analysis):
        strict = False
    result["strict"] = strict
    result["used_cot"] = analysis is not None
    result["analysis_chars"] = len(analysis.strip()) if analysis else 0
    return result


def jaccard(left, right):
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


class _ModeReward:
    def __init__(self, config):
        self.config = config

    def parse_all(self, completions, valid_letters, trainer_state):
        lenient = (
            self.config.mode != MODE_DIRECT
            and global_step(trainer_state) < self.config.format_warmup_steps
        )
        return [
            parse_completion(completion, self.config.mode, letters, lenient)
            for completion, letters in zip(completions, valid_letters)
        ]


class AnswerReward(_ModeReward):
    """Exact match with the official answer set: 1 / 0."""

    __name__ = "answer_reward"

    def __call__(
        self,
        completions,
        answer,
        valid_letters,
        sample_id=None,
        trainer_state=None,
        log_extra=None,
        log_metric=None,
        **kwargs,
    ):
        parsed = self.parse_all(completions, valid_letters, trainer_state)
        rewards = [
            1.0 if item["answer"] == gold else 0.0
            for item, gold in zip(parsed, answer)
        ]
        if log_extra is not None:
            if sample_id is not None:
                log_extra("sample_id", list(sample_id))
            log_extra("gold_answer", list(answer))
            log_extra(
                "parsed_answer",
                [item["answer"] if item["answer"] is not None else "[invalid]" for item in parsed],
            )
            log_extra("answer_correct", [bool(reward) for reward in rewards])
        if log_metric is not None and rewards:
            log_metric("answer_accuracy", _mean(rewards))
            log_metric(
                "answer_parse_rate",
                _mean([1.0 if item["answer"] is not None else 0.0 for item in parsed]),
            )
        return rewards


class FormatReward(_ModeReward):
    """Whether the output strictly follows the mode's format: 1 / 0."""

    __name__ = "format_reward"

    def __call__(
        self,
        completions,
        valid_letters,
        trainer_state=None,
        log_extra=None,
        log_metric=None,
        **kwargs,
    ):
        parsed = self.parse_all(completions, valid_letters, trainer_state)
        rewards = [1.0 if item["strict"] else 0.0 for item in parsed]
        if log_extra is not None:
            log_extra("strict_format_ok", [bool(reward) for reward in rewards])
            log_extra("used_cot", [bool(item["used_cot"]) for item in parsed])
            log_extra("analysis_chars", [int(item["analysis_chars"]) for item in parsed])
        if log_metric is not None and rewards:
            log_metric("format_rate", _mean(rewards))
            log_metric("cot_rate", _mean([1.0 if item["used_cot"] else 0.0 for item in parsed]))
            log_metric("analysis_chars_mean", _mean([item["analysis_chars"] for item in parsed]))
        return rewards


class ThinkCostReward(_ModeReward):
    """Charge λ·min(L/L0, m) when the answer is right and a rationale was written; 0 otherwise. Its weight should equal answer_weight."""

    __name__ = "think_cost_reward"

    def cost_for(self, item):
        if not (item["strict"] and item["used_cot"]):
            return 0.0
        if self.config.think_cost_chars <= 0:
            multiplier = 1.0
        else:
            multiplier = min(
                item["analysis_chars"] / self.config.think_cost_chars,
                self.config.think_cost_max_multiplier,
            )
        return self.config.think_cost * multiplier

    def __call__(
        self,
        completions,
        answer,
        valid_letters,
        trainer_state=None,
        log_metric=None,
        **kwargs,
    ):
        parsed = self.parse_all(completions, valid_letters, trainer_state)
        rewards = []
        cot_when_correct, cot_when_wrong = [], []
        for item, gold in zip(parsed, answer):
            correct = item["answer"] == gold
            (cot_when_correct if correct else cot_when_wrong).append(
                1.0 if item["used_cot"] else 0.0
            )
            rewards.append(-self.cost_for(item) if correct else 0.0)
        if log_metric is not None and rewards:
            log_metric("think_cost_mean", _mean(rewards))
            if cot_when_correct:
                log_metric("cot_rate_when_correct", _mean(cot_when_correct))
            if cot_when_wrong:
                log_metric("cot_rate_when_wrong", _mean(cot_when_wrong))
        return rewards


class MultiChoicePartialReward(_ModeReward):
    """Jaccard partial credit for multi-answer questions, annealed linearly to 0 over training; 0 on exact matches (already covered by the answer reward)."""

    __name__ = "multi_choice_partial_reward"

    def anneal_factor(self, trainer_state):
        steps = self.config.multi_partial_anneal_steps
        if steps <= 0:
            return 1.0
        return max(0.0, 1.0 - global_step(trainer_state) / steps)

    def __call__(
        self,
        completions,
        answer,
        valid_letters,
        trainer_state=None,
        log_metric=None,
        **kwargs,
    ):
        parsed = self.parse_all(completions, valid_letters, trainer_state)
        factor = self.anneal_factor(trainer_state)
        rewards = []
        multi_jaccard, multi_exact = [], []
        for item, gold in zip(parsed, answer):
            if len(gold) < 2:
                rewards.append(0.0)
                continue
            predicted = item["answer"]
            score = jaccard(predicted, gold) if predicted is not None else 0.0
            multi_jaccard.append(score)
            multi_exact.append(1.0 if predicted == gold else 0.0)
            if predicted is None or predicted == gold:
                rewards.append(0.0)
            else:
                rewards.append(score * factor)
        if log_metric is not None and multi_jaccard:
            log_metric("multi_choice_jaccard", _mean(multi_jaccard))
            log_metric("multi_choice_exact_rate", _mean(multi_exact))
            log_metric("multi_partial_anneal_factor", factor)
        return rewards


class OverlongReward:
    """DAPO-style soft over-length penalty: 0 up to L_max−cache, linear to −1 at L_max, −1 beyond."""

    __name__ = "overlong_reward"

    def __init__(self, config):
        self.config = config

    def penalty_for(self, length):
        limit = self.config.max_completion_length
        cache = self.config.overlong_cache_tokens
        if length <= limit - cache:
            return 0.0
        if length <= limit:
            if cache <= 0:
                return -1.0
            return (limit - cache - length) / cache
        return -1.0

    def __call__(self, completions, completion_ids=None, log_metric=None, **kwargs):
        if completion_ids is None:
            return [0.0 for _ in completions]
        rewards = [self.penalty_for(len(ids)) for ids in completion_ids]
        if log_metric is not None and rewards:
            log_metric("overlong_rate", _mean([1.0 if r < 0 else 0.0 for r in rewards]))
            log_metric("completion_tokens_mean", _mean([len(ids) for ids in completion_ids]))
        return rewards


def build_reward_functions(config):
    """Assemble TRL reward_funcs and reward_weights from the config."""
    config.validate()
    functions = [AnswerReward(config), FormatReward(config)]
    weights = [config.answer_weight, config.format_weight]
    if config.mode != MODE_DIRECT and config.think_cost > 0:
        functions.append(ThinkCostReward(config))
        weights.append(config.answer_weight)
    if config.multi_partial_weight > 0:
        functions.append(MultiChoicePartialReward(config))
        weights.append(config.answer_weight * config.multi_partial_weight)
    if config.mode != MODE_DIRECT and config.overlong_weight > 0:
        functions.append(OverlongReward(config))
        weights.append(config.overlong_weight)
    return functions, weights


def describe_reward(config):
    functions, weights = build_reward_functions(config)
    terms = []
    for function, weight in zip(functions, weights):
        terms.append(f"{weight:g} × {function.__name__}")
    return {
        "config": asdict(config),
        "functions": [function.__name__ for function in functions],
        "weights": weights,
        "formula": " + ".join(terms),
        "reference_explanation_used": False,
        "gold_answer_in_prompt": False,
    }


def self_test(config, valid_letters="ABCDE", gold="D", gold_multi="BDE"):
    """Run every reward function on synthetic outputs and return printable table rows, for --check-only.

    The self-test runs at the step where format warm-up ends (global_step = format_warmup_steps), so it shows the
    rewards of regular training rather than the lenient warm-up values. Case labels are data and stay as written.
    """
    functions, weights = build_reward_functions(config)
    trainer_state = SimpleNamespace(global_step=config.format_warmup_steps)
    if config.mode == MODE_DIRECT:
        cases = [
            ("正确纯字母", "D", gold),
            ("错误纯字母", "B", gold),
            ("正确但带标签", "答案：D", gold),
            ("重复标签", "答案：D\n最终答案：B", gold),
            ("多选完全匹配", "BDE", gold_multi),
            ("多选部分匹配", "BD", gold_multi),
            ("非法字母", "F", gold),
        ]
    else:
        long_analysis = "解析：" + "要点。" * 80 + "\n答案：D"
        cases = [
            ("直答正确", "答案：D", gold),
            ("解析后正确", "解析：逐项排除，D 符合题意。\n答案：D", gold),
            ("解析后错误", "解析：逐项排除。\n答案：B", gold),
            ("纯字母无标签", "D", gold),
            ("超长解析正确", long_analysis, gold),
            ("解析中混入答案行", "解析：先猜。\n答案：B\n答案：D", gold),
            ("多选部分匹配", "答案：BD", gold_multi),
        ]
    completions = [text for _, text, _ in cases]
    answers = [answer for _, _, answer in cases]
    letters = [valid_letters] * len(cases)
    completion_ids = [[1] * max(1, len(text)) for text in completions]
    per_function = []
    for function, weight in zip(functions, weights):
        values = function(
            prompts=[None] * len(cases),
            completions=completions,
            completion_ids=completion_ids,
            answer=answers,
            valid_letters=letters,
            trainer_state=trainer_state,
        )
        per_function.append((function.__name__, weight, values))
    rows = []
    for index, (label, text, _) in enumerate(cases):
        total = sum(weight * values[index] for _, weight, values in per_function)
        rows.append(
            {
                "case": label,
                "completion": text if len(text) <= 40 else text[:37] + "...",
                **{name: round(values[index], 4) for name, _, values in per_function},
                "total": round(total, 4),
            }
        )
    return rows
