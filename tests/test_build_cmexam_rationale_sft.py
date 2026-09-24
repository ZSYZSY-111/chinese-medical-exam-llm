import unittest

from scripts import build_cmexam_rationale_sft as builder
from scripts.cmexam_prompts import LEGACY_EXPLAIN_SYSTEM_PROMPT, LEGACY_EXPLAIN_USER_SUFFIX, MODE_COT, build_prompt_messages


class RationaleBuilderPromptTests(unittest.TestCase):
    def test_builder_prompt_matches_shared_legacy_style(self):
        """Evaluation and RL render rationale prompts through cmexam_prompts; they must match the builder byte for byte."""
        self.assertEqual(builder.SYSTEM_PROMPT, LEGACY_EXPLAIN_SYSTEM_PROMPT)
        row = {"Question": "题干", "Options": "A. 甲\nB. 乙\nC. 丙", "Answer": "A", "Explanation": "因为甲。"}
        converted, explanation_length = builder.validate_and_convert(row)
        self.assertEqual(explanation_length, 4)
        messages = converted["messages"]
        rendered = build_prompt_messages("题干", [("A", "甲"), ("B", "乙"), ("C", "丙")], MODE_COT, style="legacy_explain")
        self.assertEqual(messages[0]["content"], rendered[0]["content"])
        self.assertEqual(messages[1]["content"], rendered[1]["content"])
        self.assertTrue(messages[1]["content"].endswith(LEGACY_EXPLAIN_USER_SUFFIX))
        self.assertEqual(messages[2]["content"], "解析：因为甲。\n答案：A")


if __name__ == "__main__":
    unittest.main()
