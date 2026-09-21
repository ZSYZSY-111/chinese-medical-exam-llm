import json
import tempfile
import unittest
from pathlib import Path

from scripts.cmexam_prompts import MODE_DIRECT, build_prompt_messages
from scripts.data_utils import extract_valid_letters, parse_messages_record, validate_messages_file

OPTIONS = [("A", "甲"), ("B", "乙"), ("C", "丙"), ("D", "丁"), ("E", "戊")]


def record(question, answer, options=OPTIONS):
    return {"messages": build_prompt_messages(question, options, MODE_DIRECT) + [{"role": "assistant", "content": answer}]}


class DataUtilsTests(unittest.TestCase):
    def test_parse_record_and_sample_id(self):
        parsed = parse_messages_record(record("题干", "CA"))
        self.assertEqual(parsed["answer"], "AC")
        self.assertTrue(parsed["is_multi_choice"])
        self.assertEqual(parsed["valid_letters"], "ABCDE")
        self.assertEqual(len(parsed["sample_id"]), 64)
        self.assertEqual(parsed["sample_id"], parse_messages_record(record("题干", "A"))["sample_id"])
        self.assertEqual([m["role"] for m in parsed["prompt"]], ["system", "user"])

    def test_six_option_questions_are_valid(self):
        six = OPTIONS + [("F", "己")]
        self.assertEqual(extract_valid_letters(record("题干", "F", six)["messages"][:-1]), "ABCDEF")

    def test_rejects_bad_records(self):
        with self.assertRaises(ValueError):
            parse_messages_record({"messages": record("题干", "A")["messages"], "extra": 1})
        with self.assertRaises(ValueError):
            parse_messages_record(record("题干", "G"))

    def test_validate_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("".join(json.dumps(record(f"题干{i}", "B"), ensure_ascii=False) + "\n" for i in range(3)), encoding="utf-8")
            report = validate_messages_file(path)
            self.assertTrue(report)


if __name__ == "__main__":
    unittest.main()
