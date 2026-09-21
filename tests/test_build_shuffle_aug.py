import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_shuffle_aug import build
from scripts.cmexam_prompts import MODE_DIRECT, build_prompt_messages, parse_user_content

OPTIONS = [("A", "甲"), ("B", "乙"), ("C", "丙"), ("D", "丁"), ("E", "戊")]


def record(question, answer, options=OPTIONS):
    return {"messages": build_prompt_messages(question, options, MODE_DIRECT) + [{"role": "assistant", "content": answer}]}


class ShuffleAugTests(unittest.TestCase):
    def test_copies_remap_answers_and_skip_unsafe_questions(self):
        unsafe = [("A", "甲"), ("B", "乙"), ("C", "丙"), ("D", "以上都是")]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.jsonl"
            rows = [record("单选题干", "B"), record("多选题干", "AD"), record("互指题干", "D", unsafe)]
            source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            first, stats = build(source, copies=1, seed=42)
            second, _ = build(source, copies=1, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(stats["original"], 3)
        self.assertEqual(stats["shuffled"], 2)
        self.assertEqual(stats["shuffle_unsafe_questions"], 1)
        gold_text = {"单选题干": {"乙"}, "多选题干": {"甲", "丁"}, "互指题干": {"以上都是"}}
        for row in first:
            parsed = parse_user_content(row["messages"][1]["content"])
            answer = row["messages"][-1]["content"]
            self.assertEqual(answer, "".join(sorted(answer)))
            self.assertEqual({text for letter, text in parsed["options"] if letter in answer}, gold_text[parsed["question"]])


if __name__ == "__main__":
    unittest.main()
