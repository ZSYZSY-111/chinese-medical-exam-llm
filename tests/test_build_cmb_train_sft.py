import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.build_cmb_train_sft import build, normalize_stem, parse_cmb_row
from scripts.data_utils import validate_messages_file


def cmb(question, answer, exam_type="医师考试", qtype="单项选择题", options=None):
    options = options or {"A": "甲", "B": "乙", "C": "丙", "D": "丁", "E": "戊"}
    return {"exam_type": exam_type, "exam_class": "x", "exam_subject": "y", "question": question,
            "answer": answer, "question_type": qtype, "option": options}


class BuildCmbTrainTests(unittest.TestCase):
    def test_parse_and_normalize(self):
        row = parse_cmb_row(cmb("HIV 患者最常感染的是？", "d"))
        self.assertEqual(row["answer"], "D")
        self.assertEqual(normalize_stem("HIV 患者，最常感染的是？"), normalize_stem("hiv患者最常感染的是"))
        with self.assertRaises(ValueError):
            parse_cmb_row(cmb("题", "F"))
        four = parse_cmb_row(cmb("四选项题", "D", options={"A": "甲", "B": "乙", "C": "丙", "D": "丁", "E": ""}))
        self.assertEqual(four["options"][-1], ("E", ""))
        self.assertTrue(four["has_empty_option"])
        with self.assertRaises(ValueError):
            parse_cmb_row(cmb("答案指向空选项", "E", options={"A": "甲", "B": "乙", "C": "丙", "D": "丁", "E": ""}))
        with self.assertRaises(ValueError):
            parse_cmb_row(cmb("中间为空", "A", options={"A": "甲", "B": "", "C": "丙", "D": "丁", "E": "戊"}))
        with self.assertRaises(ValueError):
            parse_cmb_row(cmb("题", "A", options={"A": "甲", "C": "丙"}))

    def test_decontamination_dedup_and_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            test_rows = [{"id": 1, "question": "急性胰腺炎最常见的病因是什么", "option": {"A": "胆道疾病", "B": "酗酒", "C": "高脂血症", "D": "外伤", "E": "药物"}}]
            val_rows = [{"id": 2, "question": "下列哪项是糖尿病酮症酸中毒的特征", "option": {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}}]
            train_rows = [
                cmb("急性胰腺炎最常见的病因是什么", "A"),                        # 精确重合 test
                cmb("急性胰腺炎最常见的病因是什么呢？", "A"),                    # 近重复 test
                cmb("下列哪项是糖尿病酮症酸中毒的特征", "B"),                    # 精确重合 val
                cmb("完全不同的题目一：肺炎链球菌肺炎的典型痰液是", "C"),
                cmb("完全不同的题目一：肺炎链球菌肺炎的典型痰液是", "C"),        # 训练集内重复
                cmb("完全不同的题目二：属于β受体阻滞剂的药物有哪些", "BD", qtype="多项选择题"),
                cmb("完全不同的题目三：C型题", "A", qtype="C型选择题"),
                cmb("完全不同的题目四：休克的早期表现是", "E"),
            ]
            for name, rows in (("train.json", train_rows), ("test.json", test_rows), ("val.json", val_rows)):
                (root / name).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            args = Namespace(cmb_train=str(root / "train.json"), cmb_test=str(root / "test.json"), cmb_val=str(root / "val.json"),
                             cmexam_csv=[], merge_direct_file=None, output_dir=str(root / "out"), internal_val_size=1,
                             near_threshold=0.7, borderline_threshold=0.5, min_ngram_chars=8, multi_shuffle_copies=1,
                             merge_multi_shuffle_copies=0, drop_question_types="C型选择题", max_chars=900, limit=0, seed=42,
                             overwrite=True, check_only=False)
            final, val_records, report, removed, borderline = build(args)
            stats = report["stats"]
            self.assertEqual(stats["removed_exact_stem"], 2)
            self.assertEqual(stats["removed_near_duplicate"], 1)
            self.assertEqual(stats["duplicate_within_train"], 1)
            self.assertEqual(stats["dropped_question_type"], 1)
            self.assertEqual(stats["internal_val"], 1)
            self.assertEqual(stats["cmb_train_kept"], 2)
            self.assertEqual(report["near_duplicate_method"], "bruteforce")
            reasons = {r["reason"] for r in removed}
            self.assertEqual(reasons, {"exact_stem", "near_duplicate"})
            sources = report["final_by_source"]
            self.assertEqual(sources["cmb_train/original"], 2)
            self.assertLessEqual(sources.get("cmb_train/shuffled", 0), 1)
            out = root / "check.jsonl"
            with out.open("w", encoding="utf-8") as handle:
                for record, _ in final:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.assertEqual(validate_messages_file(out)["samples"], len(final))
            val_stems = {normalize_stem(v["messages"][1]["content"]) for v in val_records}
            train_stems = {normalize_stem(r["messages"][1]["content"]) for r, _ in final}
            self.assertTrue(val_stems.isdisjoint(train_stems))


if __name__ == "__main__":
    unittest.main()
