import unittest
from types import SimpleNamespace

from scripts.cmexam_prompts import MODE_ADAPTIVE, MODE_COT, MODE_DIRECT
from scripts.grpo_rewards import (
    AnswerReward,
    FormatReward,
    MultiChoicePartialReward,
    OverlongReward,
    RewardConfig,
    ThinkCostReward,
    build_reward_functions,
    describe_reward,
    jaccard,
    parse_completion,
    self_test,
)


LETTERS = "ABCDE"


class ParseCompletionTests(unittest.TestCase):
    def test_direct_mode(self):
        self.assertEqual(parse_completion("D", MODE_DIRECT, LETTERS)["answer"], "D")
        self.assertTrue(parse_completion("D", MODE_DIRECT, LETTERS)["strict"])
        labeled = parse_completion("答案：D", MODE_DIRECT, LETTERS)
        self.assertEqual(labeled["answer"], "D")
        self.assertFalse(labeled["strict"])
        self.assertIsNone(parse_completion("F", MODE_DIRECT, LETTERS)["answer"])
        self.assertEqual(parse_completion("", MODE_DIRECT, LETTERS)["answer"], None)

    def test_cot_mode_requires_analysis_and_label(self):
        strict = parse_completion("解析：逐项排除。\n答案：D", MODE_COT, LETTERS)
        self.assertEqual(strict["answer"], "D")
        self.assertTrue(strict["strict"])
        self.assertTrue(strict["used_cot"])
        self.assertEqual(strict["analysis_chars"], 5)
        no_analysis = parse_completion("答案：D", MODE_COT, LETTERS)
        self.assertEqual(no_analysis["answer"], "D")
        self.assertFalse(no_analysis["strict"])
        pure = parse_completion("D", MODE_COT, LETTERS)
        self.assertIsNone(pure["answer"])
        lenient = parse_completion("D", MODE_COT, LETTERS, lenient_answer=True)
        self.assertEqual(lenient["answer"], "D")
        self.assertFalse(lenient["strict"])

    def test_adaptive_mode(self):
        direct = parse_completion("答案：D", MODE_ADAPTIVE, LETTERS)
        self.assertTrue(direct["strict"])
        self.assertFalse(direct["used_cot"])
        cot = parse_completion("解析：需要鉴别。\n答案：ACD", MODE_ADAPTIVE, LETTERS)
        self.assertTrue(cot["strict"])
        self.assertTrue(cot["used_cot"])
        self.assertEqual(cot["answer"], "ACD")
        duplicate = parse_completion("解析：先猜。\n答案：B\n答案：D", MODE_ADAPTIVE, LETTERS)
        self.assertIsNone(duplicate["answer"])
        self.assertFalse(duplicate["strict"])
        trailing = parse_completion("答案：D\n希望有帮助", MODE_ADAPTIVE, LETTERS)
        self.assertEqual(trailing["answer"], "D")
        self.assertFalse(trailing["strict"])
        unsorted = parse_completion("答案：DA", MODE_ADAPTIVE, LETTERS)
        self.assertEqual(unsorted["answer"], "AD")
        self.assertFalse(unsorted["strict"])
        conversational = parse_completion(
            [{"role": "assistant", "content": "答案：B"}], MODE_ADAPTIVE, LETTERS
        )
        self.assertEqual(conversational["answer"], "B")

    def test_jaccard(self):
        self.assertAlmostEqual(jaccard("BD", "BDE"), 2 / 3)
        self.assertEqual(jaccard("", ""), 1.0)


class RewardFunctionTests(unittest.TestCase):
    def call(self, function, completions, answers, step=None, completion_ids=None):
        state = SimpleNamespace(global_step=step) if step is not None else None
        return function(
            prompts=[None] * len(completions),
            completions=completions,
            completion_ids=completion_ids,
            answer=answers,
            valid_letters=[LETTERS] * len(completions),
            trainer_state=state,
        )

    def test_direct_answer_and_format(self):
        config = RewardConfig(mode=MODE_DIRECT).validate()
        answer, form = AnswerReward(config), FormatReward(config)
        completions = ["D", "答案：D", "B", "F"]
        self.assertEqual(self.call(answer, completions, ["D"] * 4), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(self.call(form, completions, ["D"] * 4), [1.0, 0.0, 1.0, 0.0])

    def test_format_warmup_only_affects_answer_extraction(self):
        config = RewardConfig(mode=MODE_ADAPTIVE, format_warmup_steps=50, max_completion_length=384).validate()
        answer, form = AnswerReward(config), FormatReward(config)
        self.assertEqual(self.call(answer, ["D"], ["D"], step=10), [1.0])
        self.assertEqual(self.call(answer, ["D"], ["D"], step=50), [0.0])
        self.assertEqual(self.call(form, ["D"], ["D"], step=10), [0.0])

    def test_think_cost_only_when_correct_strict_and_cot(self):
        config = RewardConfig(
            mode=MODE_ADAPTIVE, think_cost=0.05, think_cost_chars=10, think_cost_max_multiplier=3.0,
            max_completion_length=384,
        ).validate()
        cost = ThinkCostReward(config)
        completions = [
            "答案：D",
            "解析：" + "字" * 10 + "\n答案：D",
            "解析：" + "字" * 50 + "\n答案：D",
            "解析：" + "字" * 10 + "\n答案：B",
            "解析：" + "字" * 10 + "\n答案：D\n答案：D",
        ]
        rewards = self.call(cost, completions, ["D"] * 5)
        self.assertEqual(rewards[0], 0.0)
        self.assertAlmostEqual(rewards[1], -0.05)
        self.assertAlmostEqual(rewards[2], -0.15)
        self.assertEqual(rewards[3], 0.0)
        self.assertEqual(rewards[4], 0.0)
        flat = ThinkCostReward(
            RewardConfig(mode=MODE_COT, think_cost=0.1, think_cost_chars=0, max_completion_length=384).validate()
        )
        self.assertAlmostEqual(self.call(flat, [completions[2]], ["D"])[0], -0.1)

    def test_multi_choice_partial_anneals(self):
        config = RewardConfig(mode=MODE_DIRECT, multi_partial_anneal_steps=100).validate()
        partial = MultiChoicePartialReward(config)
        completions = ["BD", "BDE", "A", "B", "F"]
        answers = ["BDE", "BDE", "BDE", "B", "BDE"]
        rewards = self.call(partial, completions, answers, step=0)
        self.assertAlmostEqual(rewards[0], 2 / 3)
        self.assertEqual(rewards[1], 0.0)
        self.assertEqual(rewards[2], 0.0)
        self.assertEqual(rewards[3], 0.0)
        self.assertEqual(rewards[4], 0.0)
        half = self.call(partial, completions, answers, step=50)
        self.assertAlmostEqual(half[0], 1 / 3)
        self.assertEqual(self.call(partial, completions, answers, step=100)[0], 0.0)
        no_anneal = MultiChoicePartialReward(
            RewardConfig(mode=MODE_DIRECT, multi_partial_anneal_steps=0).validate()
        )
        self.assertAlmostEqual(self.call(no_anneal, completions, answers, step=999)[0], 2 / 3)

    def test_overlong_soft_penalty(self):
        config = RewardConfig(
            mode=MODE_COT, max_completion_length=100, overlong_cache_tokens=20
        ).validate()
        overlong = OverlongReward(config)
        ids = [[1] * 80, [1] * 90, [1] * 100, [1] * 101]
        rewards = self.call(overlong, ["x"] * 4, ["D"] * 4, completion_ids=ids)
        self.assertEqual(rewards[0], 0.0)
        self.assertAlmostEqual(rewards[1], -0.5)
        self.assertAlmostEqual(rewards[2], -1.0)
        self.assertEqual(rewards[3], -1.0)
        self.assertEqual(self.call(overlong, ["x"], ["D"]), [0.0])

    def test_build_and_describe(self):
        direct_functions, direct_weights = build_reward_functions(RewardConfig(mode=MODE_DIRECT))
        self.assertEqual(
            [f.__name__ for f in direct_functions],
            ["answer_reward", "format_reward", "multi_choice_partial_reward"],
        )
        self.assertEqual(direct_weights, [0.95, 0.05, 0.95 * 0.5])
        adaptive = RewardConfig(mode=MODE_ADAPTIVE, think_cost=0.05, max_completion_length=384)
        functions, weights = build_reward_functions(adaptive)
        self.assertEqual(
            [f.__name__ for f in functions],
            [
                "answer_reward",
                "format_reward",
                "think_cost_reward",
                "multi_choice_partial_reward",
                "overlong_reward",
            ],
        )
        self.assertEqual(weights[2], 0.95)
        description = describe_reward(adaptive)
        self.assertFalse(description["gold_answer_in_prompt"])
        self.assertIn("think_cost_reward", description["formula"])

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            RewardConfig(mode=MODE_DIRECT, think_cost=0.1).validate()
        with self.assertRaises(ValueError):
            RewardConfig(mode=MODE_COT, max_completion_length=32, overlong_cache_tokens=64).validate()
        with self.assertRaises(ValueError):
            RewardConfig(mode="other").validate()

    def test_self_test_totals(self):
        rows = {row["case"]: row for row in self_test(RewardConfig(mode=MODE_ADAPTIVE, think_cost=0.05, max_completion_length=384))}
        self.assertAlmostEqual(rows["直答正确"]["total"], 1.0)
        self.assertAlmostEqual(rows["纯字母无标签"]["total"], 0.0)
        self.assertAlmostEqual(rows["解析后错误"]["total"], 0.05)
        self.assertLess(rows["解析后正确"]["total"], 1.0)
        self.assertGreater(rows["解析后正确"]["total"], 0.9)
        self.assertLess(rows["超长解析正确"]["total"], rows["解析后正确"]["total"])
        direct_rows = {row["case"]: row for row in self_test(RewardConfig(mode=MODE_DIRECT))}
        self.assertAlmostEqual(direct_rows["正确纯字母"]["total"], 1.0)
        self.assertAlmostEqual(direct_rows["正确但带标签"]["total"], 0.95)
        self.assertGreater(direct_rows["多选部分匹配"]["total"], 0.0)


if __name__ == "__main__":
    unittest.main()
