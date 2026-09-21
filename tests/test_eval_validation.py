import json
import tempfile
import unittest
from pathlib import Path

from scripts.cmexam_prompts import MODE_ADAPTIVE, MODE_DIRECT, build_prompt_messages
from scripts.eval_validation import (
    bucket_of,
    finalize_shuffle_consistency,
    load_examples,
    majority_vote,
    summarize_records,
)


OPTIONS = [("A", "甲"), ("B", "乙"), ("C", "丙"), ("D", "丁"), ("E", "戊")]


class HelperTests(unittest.TestCase):
    def test_majority_vote(self):
        self.assertEqual(majority_vote(["A", "B", "A", None]), "A")
        self.assertEqual(majority_vote(["B", "A", "A", "B"]), "B")
        self.assertIsNone(majority_vote([None, None]))

    def test_finalize_shuffle_consistency_handles_repeated_indexes(self):
        records = [
            {"greedy_answer": "A", "permutations": [{"answer_original_letters": "A"}, {"answer_original_letters": "A"}]},
            {"greedy_answer": "B", "permutations": [{"answer_original_letters": "B"}, {"answer_original_letters": "C"}]},
            {"greedy_answer": "C"},
        ]
        finalize_shuffle_consistency(records, [0, 0, 1, 1])
        self.assertTrue(records[0]["shuffle_consistent"])
        self.assertFalse(records[1]["shuffle_consistent"])
        self.assertNotIn("shuffle_consistent", records[2])

    def test_bucket_of(self):
        self.assertEqual(bucket_of(8, 8), "easy")
        self.assertEqual(bucket_of(0, 8), "zero")
        self.assertEqual(bucket_of(6, 8), "medium")
        self.assertEqual(bucket_of(3, 8), "hard")

    def test_summary(self):
        records = [
            {
                "slices": {"negation": True},
                "greedy_correct": True, "greedy_strict": True, "greedy_used_cot": False,
                "correct_count": 8, "majority_correct": True, "bucket": "easy",
                "constrained_correct": True, "shuffle_consistent": True,
                "permutations": [{"correct": True}, {"correct": True}],
            },
            {
                "slices": {"negation": False},
                "greedy_correct": False, "greedy_strict": True, "greedy_used_cot": False,
                "correct_count": 3, "majority_correct": False, "bucket": "hard",
                "constrained_correct": True, "shuffle_consistent": False,
                "permutations": [{"correct": True}, {"correct": False}],
            },
        ]
        report = summarize_records(records, 8)
        overall = report["overall"]
        self.assertEqual(overall["pass1_greedy"], 0.5)
        self.assertEqual(overall["pass_at_k"], 1.0)
        self.assertEqual(overall["sc_at_k"], 0.5)
        self.assertEqual(overall["sharpening_headroom"], 0.5)
        self.assertEqual(overall["constrained_accuracy_single"], 1.0)
        self.assertEqual(overall["format_gap"], 0.5)
        self.assertEqual(overall["shuffle_consistency_rate"], 0.5)
        self.assertEqual(overall["shuffle_accuracy"], 0.75)
        self.assertEqual(report["slices"]["negation"]["questions"], 1)
        self.assertEqual(report["sampled_bucket_histogram"], {"easy": 1, "hard": 1})
        self.assertEqual(report["correct_count_histogram"], {"3": 1, "8": 1})


class LoadExamplesTests(unittest.TestCase):
    def test_load_and_rerender(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "val.jsonl"
            messages = build_prompt_messages("患者，男，40岁，下列不属于表现的是", OPTIONS, MODE_DIRECT)
            messages.append({"role": "assistant", "content": "AC"})
            path.write_text(json.dumps({"messages": messages}, ensure_ascii=False) + "\n", encoding="utf-8")
            examples, stats = load_examples(path, MODE_DIRECT, 0)
            self.assertEqual(stats["loaded"], 1)
            self.assertEqual(examples[0]["messages"], messages[:-1])
            self.assertEqual(examples[0]["answer"], "AC")
            self.assertTrue(examples[0]["slices"]["multi_choice"])
            self.assertTrue(examples[0]["slices"]["negation"])
            adaptive, _ = load_examples(path, MODE_ADAPTIVE, 0)
            self.assertNotEqual(adaptive[0]["messages"][0]["content"], messages[0]["content"])

    def test_shuffle_seed_reorders_but_keeps_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "val.jsonl"
            with path.open("w", encoding="utf-8") as output_file:
                for index in range(20):
                    messages = build_prompt_messages(f"题干{index}", OPTIONS, MODE_DIRECT)
                    messages.append({"role": "assistant", "content": "A"})
                    output_file.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            ordered, _ = load_examples(path, MODE_DIRECT, 5)
            shuffled, stats = load_examples(path, MODE_DIRECT, 5, shuffle_seed=42)
            self.assertEqual(stats["loaded"], 5)
            self.assertEqual([e["line_number"] for e in ordered], [1, 2, 3, 4, 5])
            self.assertNotEqual([e["line_number"] for e in shuffled], [1, 2, 3, 4, 5])
            again, _ = load_examples(path, MODE_DIRECT, 5, shuffle_seed=42)
            self.assertEqual([e["sample_id"] for e in shuffled], [e["sample_id"] for e in again])


if __name__ == "__main__":
    unittest.main()
