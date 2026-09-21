import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_cmexam_direct_sft import main, normalize_question, parse_options


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Question", "Options", "Answer", "Explanation"])
        writer.writeheader()
        writer.writerows(rows)


OPTIONS = "A 甲\nB 乙\nC 丙\nD 丁\nE 戊"


class BuildCmexamDirectTests(unittest.TestCase):
    def test_option_parsing_and_question_key(self):
        self.assertEqual([letter for letter, _ in parse_options(OPTIONS)], list("ABCDE"))
        with self.assertRaises(ValueError):
            parse_options("A 甲\nC 丙")
        with self.assertRaises(ValueError):
            parse_options("A 甲\nB 甲\nC 丙")
        self.assertEqual(normalize_question("ＨＩＶ 感染 "), normalize_question("hiv感染"))

    def test_splits_stay_disjoint_and_duplicates_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "src", Path(tmp) / "out"
            src.mkdir()
            write_csv(src / "test_with_annotations.csv", [{"Question": "测试题", "Options": OPTIONS, "Answer": "A", "Explanation": ""}])
            write_csv(src / "val.csv", [{"Question": "验证题", "Options": OPTIONS, "Answer": "B", "Explanation": ""},
                                        {"Question": "测试题", "Options": OPTIONS, "Answer": "A", "Explanation": ""}])
            write_csv(src / "train.csv", [{"Question": "训练题一", "Options": OPTIONS, "Answer": "c a", "Explanation": ""},
                                          {"Question": "训练题一", "Options": OPTIONS, "Answer": "C", "Explanation": ""},
                                          {"Question": "验证题", "Options": OPTIONS, "Answer": "B", "Explanation": ""},
                                          {"Question": "测试题", "Options": OPTIONS, "Answer": "A", "Explanation": ""},
                                          {"Question": "坏题", "Options": OPTIONS, "Answer": "F", "Explanation": ""}])
            main(["--input-dir", str(src), "--output-dir", str(out)])
            train = [json.loads(l) for l in open(out / "cmexam_sft_train.jsonl", encoding="utf-8")]
            val = [json.loads(l) for l in open(out / "cmexam_sft_validation.jsonl", encoding="utf-8")]
            report = json.load(open(out / "cmexam_sft_report.json", encoding="utf-8"))
        self.assertEqual(len(train), 1)
        self.assertEqual(train[0]["messages"][-1]["content"], "AC")
        self.assertEqual(len(val), 1)
        counts = report["counts"]["train"]["rejection_counts"]
        self.assertEqual(counts["duplicate_question"], 1)
        self.assertEqual(counts["overlap_with_later_split"], 2)
        self.assertEqual(counts["answer_not_in_options"], 1)


if __name__ == "__main__":
    unittest.main()
