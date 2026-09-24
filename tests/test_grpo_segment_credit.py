import unittest

from scripts.grpo_segment_credit import (char_spans_to_token_ranges, first_token_covering, plan_completion, probe_text,
                                         segment_credits, select_boundaries, split_rationale, token_credit_rows)


class CharTokenizer:
    """每个字符一个 token；偶尔把一个字符拆成两个 token（模拟字节级 BPE 的半个汉字）。"""

    def __init__(self, split_chars=()):
        self.split_chars = set(split_chars)

    def encode(self, text, add_special_tokens=False):
        ids = []
        for char in text:
            ids.extend([("half", char), ("full", char)] if char in self.split_chars else [("full", char)])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for kind, char in ids:
            if kind == "half":
                out.append("�")
            else:
                if out and out[-1] == "�":
                    out.pop()
                out.append(char)
        return "".join(out)


class PureFunctionTests(unittest.TestCase):
    def test_split_rationale_handles_prefix_and_marker(self):
        self.assertEqual(split_rationale("解析：甲乙丙。\n答案：A"), (3, 7, "甲乙丙。"))
        self.assertEqual(split_rationale("甲乙丙。\n答案：A"), (0, 4, "甲乙丙。"))
        self.assertEqual(split_rationale("解析：被截断了没有答案"), (3, 11, "被截断了没有答案"))
        self.assertEqual(split_rationale("A"), (0, 0, ""))   # 不是解析格式

    def test_select_boundaries_merges_evenly_and_keeps_tiling(self):
        spans = [(i * 10, (i + 1) * 10) for i in range(11)]
        merged = select_boundaries(spans, 4)
        self.assertEqual(len(merged), 4)
        self.assertEqual(merged[0][0], 0)
        self.assertEqual(merged[-1][1], 110)
        for (_, end), (start, _) in zip(merged, merged[1:]):
            self.assertEqual(end, start)
        self.assertEqual(select_boundaries(spans[:3], 4), spans[:3])
        with self.assertRaises(ValueError):
            select_boundaries(spans, 0)

    def test_first_token_covering_with_split_characters(self):
        tokenizer = CharTokenizer(split_chars={"乙"})
        ids = tokenizer.encode("甲乙丙")          # 甲 | 半个乙 | 乙 | 丙 → 4 个 token
        self.assertEqual(len(ids), 4)
        cache = {}
        self.assertEqual(first_token_covering(tokenizer, ids, 0, cache), 0)
        self.assertEqual(first_token_covering(tokenizer, ids, 1, cache), 1)   # 半个乙就已经让长度超过 1
        self.assertEqual(first_token_covering(tokenizer, ids, 2, cache), 3)
        self.assertIsNone(first_token_covering(tokenizer, ids, 3, cache))
        self.assertIsNone(first_token_covering(tokenizer, [], 0, {}))

    def test_token_ranges_tile_the_rationale(self):
        tokenizer = CharTokenizer()
        text = "解析：第一段的内容很重要，第二段的内容比较一般。\n答案：B"
        ids = tokenizer.encode(text)
        plan = plan_completion(tokenizer, ids, text, max_probes=8)
        self.assertEqual(plan["segments"], 2)
        self.assertEqual(plan["prefixes"], ["", "第一段的内容很重要，", "第一段的内容很重要，第二段的内容比较一般。"])
        first, second = plan["token_ranges"]
        self.assertEqual(text[first[0]:first[1] + 1], "第一段的内容很重要，")
        self.assertEqual(text[second[0]:second[1] + 1], "第二段的内容比较一般。")
        self.assertEqual(first[1] + 1, second[0])

    def test_no_rationale_means_no_segments(self):
        tokenizer = CharTokenizer()
        plan = plan_completion(tokenizer, tokenizer.encode("B"), "B")
        self.assertEqual(plan["segments"], 0)
        self.assertEqual(plan["prefixes"], [""])

    def test_credits_telescope_and_rows(self):
        for got, want in zip(segment_credits([0.2, 0.5, 0.4, 0.9]), [0.3, -0.1, 0.5]):
            self.assertAlmostEqual(got, want)
        self.assertAlmostEqual(sum(segment_credits([0.2, 0.5, 0.4, 0.9])), 0.9 - 0.2)
        row = token_credit_rows(8, [(1, 2), None, (5, 9)], [0.3, -0.1, 0.5])
        self.assertEqual(row, [0.0, 0.3, 0.3, 0.0, 0.0, 0.5, 0.5, 0.5])

    def test_ranges_out_of_text_are_none(self):
        tokenizer = CharTokenizer()
        ids = tokenizer.encode("短")
        self.assertEqual(char_spans_to_token_ranges(tokenizer, ids, [(0, 1), (5, 9), (3, 3)]), [(0, 0), None, None])

    def test_probe_text_format(self):
        self.assertEqual(probe_text("P", "甲乙"), "P解析：甲乙\n答案：")


if __name__ == "__main__":
    unittest.main()
