import json
import unittest
from pathlib import Path

from scripts.cmexam_prompts import (
    ADAPTIVE_SYSTEM_PROMPT,
    DIRECT_SYSTEM_PROMPT,
    MODE_ADAPTIVE,
    MODE_COT,
    MODE_DIRECT,
    build_prompt_messages,
    detect_slices,
    extract_knowledge_point,
    is_shuffle_safe,
    make_permutation,
    parse_user_content,
    permute_options,
    remap_answer,
    render_user_content,
    stable_fraction,
    unmap_answer,
)


OPTIONS = [
    ("A", "患高血压病3年"),
    ("B", "心绞痛反复发作3年"),
    ("C", "3年前开始多饮，多食，多尿"),
    ("D", "吞咽困难，进行性加重已1月余"),
    ("E", "某医院确诊为肺癌，介绍病人来诊"),
]


class PromptRenderingTests(unittest.TestCase):
    def test_direct_render_reproduces_existing_dataset_user_content(self):
        path = Path("data/cmexam_sft_train.jsonl")
        if not path.exists():
            self.skipTest("dataset not downloaded (see data/README.md)")
        with path.open("r", encoding="utf-8") as input_file:
            record = json.loads(next(input_file))
        system, user = record["messages"][0], record["messages"][1]
        parsed = parse_user_content(user["content"])
        rendered = render_user_content(parsed["question"], parsed["options"], MODE_DIRECT)
        self.assertEqual(rendered, user["content"])
        self.assertEqual(system["content"], DIRECT_SYSTEM_PROMPT)

    def test_hint_round_trip_and_mode_prompts(self):
        content = render_user_content("题干", OPTIONS, MODE_ADAPTIVE, hint="主诉的书写要求")
        parsed = parse_user_content(content)
        self.assertEqual(parsed["hint"], "主诉的书写要求")
        self.assertEqual(parsed["question"], "题干")
        self.assertEqual(parsed["options"], OPTIONS)
        messages = build_prompt_messages("题干", OPTIONS, MODE_ADAPTIVE)
        self.assertEqual(messages[0]["content"], ADAPTIVE_SYSTEM_PROMPT)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("解析", build_prompt_messages("题干", OPTIONS, MODE_COT)[1]["content"])

    def test_parse_rejects_bad_structure(self):
        with self.assertRaises(ValueError):
            parse_user_content("没有题目结构")
        with self.assertRaises(ValueError):
            parse_user_content("题目：\nq\n\n选项：\nA. 甲\nC. 丙\n\n指令")


class ShuffleTests(unittest.TestCase):
    def test_unsafe_options_are_detected(self):
        self.assertTrue(is_shuffle_safe(OPTIONS))
        self.assertFalse(is_shuffle_safe(OPTIONS[:4] + [("E", "以上都是")]))
        self.assertFalse(is_shuffle_safe(OPTIONS[:4] + [("E", "A和B")]))
        self.assertFalse(is_shuffle_safe(OPTIONS[:2]))

    def test_permutation_is_deterministic_non_identity_and_respects_forbidden(self):
        first = make_permutation(5, 42, "sample", 0)
        again = make_permutation(5, 42, "sample", 0)
        self.assertEqual(first, again)
        self.assertNotEqual(first, [0, 1, 2, 3, 4])
        self.assertEqual(sorted(first), [0, 1, 2, 3, 4])
        second = make_permutation(5, 42, "sample", 1, forbidden=[first])
        self.assertNotEqual(second, first)

    def test_remap_round_trip_keeps_option_text_aligned(self):
        permutation = [3, 0, 4, 1, 2]
        shuffled = permute_options(OPTIONS, permutation)
        self.assertEqual([letter for letter, _ in shuffled], ["A", "B", "C", "D", "E"])
        self.assertEqual(shuffled[0][1], OPTIONS[3][1])
        new_answer = remap_answer("D", permutation)
        self.assertEqual(new_answer, "A")
        self.assertEqual(dict(shuffled)[new_answer], dict(OPTIONS)["D"])
        self.assertEqual(remap_answer("BDE", permutation), "ACD")
        self.assertEqual(unmap_answer("ACD", permutation), "BDE")
        self.assertIsNone(unmap_answer(None, permutation))
        with self.assertRaises(ValueError):
            permute_options(OPTIONS, [0, 1, 2])


class SliceAndKnowledgePointTests(unittest.TestCase):
    def test_slices(self):
        slices = detect_slices("下列除哪项外均符合问诊的要求", OPTIONS, "D")
        self.assertTrue(slices["negation"])
        self.assertFalse(slices["multi_choice"])
        case = detect_slices("患者，女，63岁，高血压病史10年，宜加用的药物是", OPTIONS, "AB")
        self.assertTrue(case["case"])
        self.assertTrue(case["multi_choice"])
        self.assertEqual(case["option_count"], 5)

    def test_knowledge_point_extraction(self):
        self.assertEqual(
            extract_knowledge_point("本题考查的是穿心莲的主治病证。既治湿热泻痢…（D对）"),
            "穿心莲的主治病证",
        )
        self.assertEqual(extract_knowledge_point("本题考查降压药物的联合应用。二氢…"), "降压药物的联合应用")
        self.assertIsNone(extract_knowledge_point("主诉一般包括就诊主要症状。"))
        self.assertIsNone(extract_knowledge_point("本题考查的是穿心莲（D对）。"))
        self.assertIsNone(extract_knowledge_point("本题考查" + "很长" * 60 + "。"))
        self.assertIsNone(extract_knowledge_point(None))

    def test_stable_fraction(self):
        value = stable_fraction(42, "easy", "abc")
        self.assertEqual(value, stable_fraction(42, "easy", "abc"))
        self.assertTrue(0 <= value < 1)
        self.assertNotEqual(value, stable_fraction(43, "easy", "abc"))

class EmptyOptionRoundTripTests(unittest.TestCase):
    def test_trailing_empty_option_parses_and_round_trips(self):
        from scripts.cmexam_prompts import MODE_DIRECT, build_prompt_messages, parse_user_content
        options = [("A", "50℃左右"), ("B", "60℃左右"), ("C", "70℃左右"), ("D", "80℃左右"), ("E", "")]
        messages = build_prompt_messages("印模膏加热到多少度变软", options, MODE_DIRECT)
        self.assertIn("\nE. \n", messages[1]["content"] + "\n")
        parsed = parse_user_content(messages[1]["content"])
        self.assertEqual(parsed["options"], options)
        rebuilt = build_prompt_messages(parsed["question"], parsed["options"], MODE_DIRECT)
        self.assertEqual(rebuilt[1]["content"], messages[1]["content"])

class TypeHintTests(unittest.TestCase):
    def test_type_hint_round_trip(self):
        from scripts.cmexam_prompts import MODE_DIRECT, build_prompt_messages, parse_user_content, type_hint_for
        self.assertEqual(type_hint_for("A"), "本题是单项选择题")
        self.assertEqual(type_hint_for("ACD"), "本题是多项选择题")
        self.assertEqual(type_hint_for("A", "多项选择题"), "本题是多项选择题")
        self.assertIsNone(type_hint_for(None))
        messages = build_prompt_messages("题干", [("A", "甲"), ("B", "乙")], MODE_DIRECT, hint=type_hint_for("AB"))
        parsed = parse_user_content(messages[1]["content"])
        self.assertEqual(parsed["hint"], "本题是多项选择题")
        self.assertEqual(parsed["question"], "题干")


if __name__ == "__main__":
    unittest.main()
