import unittest

from scripts.answer_utils import (ANALYSIS_WITH_FINAL_ANSWER, ANSWER_ONLY, ANSWER_WITH_ANALYSIS, EXPLANATION_WITH_ANSWER,
                                  extract_predicted_answer, normalize_answer, parse_completion, parse_reference_target)
from scripts.cmexam_prompts import MODE_ADAPTIVE, MODE_COT, MODE_DIRECT

LETTERS = "ABCDE"


class AnswerNormalisationTests(unittest.TestCase):
    def test_normalize_multi_choice_answer(self):
        self.assertEqual(normalize_answer("D, A", "ABCDE"), "AD")
        self.assertIsNone(normalize_answer("AA", "ABCDE"))
        self.assertIsNone(normalize_answer("F", "ABCDE"))

    def test_extracts_supported_answer_styles(self):
        self.assertEqual(extract_predicted_answer("C", "ABCDE"), "C")
        self.assertEqual(
            extract_predicted_answer("答案：C\n解析：理由", "ABCDE"),
            "C",
        )
        self.assertEqual(
            extract_predicted_answer(
                "分析：理由\n最终答案：C",
                "ABCDE",
            ),
            "C",
        )
        self.assertEqual(
            extract_predicted_answer("解析：理由\n答案：C", "ABCDE"),
            "C",
        )

    def test_rejects_multiple_answer_markers(self):
        output = "答案：A\n解析：先猜测。\n最终答案：B"
        self.assertIsNone(extract_predicted_answer(output, "ABCDE"))

    def test_conversational_completion(self):
        completion = [{"role": "assistant", "content": "答案：B\n解析：理由"}]
        self.assertEqual(extract_predicted_answer(completion, "ABCDE"), "B")

    def test_reference_target_formats(self):
        answer_only = parse_reference_target("ACD", "ABCDE")
        self.assertEqual(answer_only["answer"], "ACD")
        self.assertEqual(answer_only["answer_format"], ANSWER_ONLY)

        answer_first = parse_reference_target(
            "答案：B\n解析：医学解释",
            "ABCDE",
        )
        self.assertEqual(answer_first["answer"], "B")
        self.assertEqual(answer_first["answer_format"], ANSWER_WITH_ANALYSIS)
        self.assertEqual(answer_first["reference_explanation"], "医学解释")

        analysis_first = parse_reference_target(
            "分析：医学解释\n最终答案：D",
            "ABCDE",
        )
        self.assertEqual(
            analysis_first["answer_format"],
            ANALYSIS_WITH_FINAL_ANSWER,
        )

        explanation_first = parse_reference_target(
            "解析：医学解释\n答案：A",
            "ABCDE",
        )
        self.assertEqual(explanation_first["answer"], "A")
        self.assertEqual(
            explanation_first["answer_format"],
            EXPLANATION_WITH_ANSWER,
        )
        self.assertEqual(
            explanation_first["reference_explanation"],
            "医学解释",
        )


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


if __name__ == "__main__":
    unittest.main()
