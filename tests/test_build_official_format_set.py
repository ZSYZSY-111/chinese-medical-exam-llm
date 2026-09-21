import unittest

from scripts.build_official_format_set import OFFICIAL_SYSTEM, convert_rows, index_cmb, official_user_prompt
from scripts.cmexam_prompts import MODE_DIRECT, build_prompt_messages


class OfficialFormatTests(unittest.TestCase):
    def test_prompt_matches_evaluator_and_metadata_join(self):
        options = [("A", "甲"), ("B", "乙"), ("C", "丙"), ("D", "丁"), ("E", "")]
        prompt = official_user_prompt("医师考试", "执业医师", "单项选择题", "题干", options)
        self.assertTrue(prompt.startswith("以下是中国医师考试中执业医师考试的一道单项选择题，不需要做任何分析和解释，直接输出答案选项。\n题干\nA. 甲\n"))
        self.assertIn("\nE. \n只能输出选项字母。单选题例如：A；多选题例如：ABC。", prompt)
        cmb_rows = [{"question": "题干", "option": {"A": "甲", "B": "乙", "C": "丙", "D": "丁", "E": ""}, "exam_type": "药师考试",
                     "exam_class": "执业中药师", "question_type": "单项选择题", "answer": "A"}]
        index = index_cmb(cmb_rows)
        rows = [{"messages": build_prompt_messages("题干", options, MODE_DIRECT) + [{"role": "assistant", "content": "A"}]},
                {"messages": build_prompt_messages("别的题", options[:4], MODE_DIRECT) + [{"role": "assistant", "content": "AB"}]}]
        out, counts = convert_rows(rows, index, "医师考试", "执业医师")
        self.assertEqual(counts, {"cmb_matched": 1, "unmatched_as_cmexam": 1})
        self.assertEqual(out[0]["messages"][0]["content"], OFFICIAL_SYSTEM)
        self.assertIn("药师考试中执业中药师考试的一道单项选择题", out[0]["messages"][1]["content"])
        self.assertIn("医师考试中执业医师考试的一道多项选择题", out[1]["messages"][1]["content"])
        self.assertEqual(out[1]["messages"][-1]["content"], "AB")


if __name__ == "__main__":
    unittest.main()
