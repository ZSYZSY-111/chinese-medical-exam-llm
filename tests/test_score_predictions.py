import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.score_predictions import (bootstrap_accuracy, compare, load_predictions, mcnemar_chi2, mcnemar_exact,
                                       mde_paired, paired_bootstrap_delta, score_single, wilson)


class StatsTests(unittest.TestCase):
    def test_mcnemar_exact_known_values(self):
        self.assertAlmostEqual(mcnemar_exact(5, 0), 2 * 0.5 ** 5)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(3, 3), 1.0)
        self.assertLess(mcnemar_exact(469, 249), 1e-10)
        self.assertLess(mcnemar_chi2(469, 249)[1], 1e-10)

    def test_wilson_and_bootstrap_bracket_the_point(self):
        p, lo, hi = wilson(840, 1000)
        self.assertAlmostEqual(p, 0.84)
        self.assertTrue(lo < 0.84 < hi and hi - lo < 0.05)
        flags = [True] * 840 + [False] * 160
        blo, bhi = bootstrap_accuracy(flags, resamples=2000, seed=1)
        self.assertTrue(blo < 0.84 < bhi)
        self.assertLess(abs((blo + bhi) / 2 - 0.84), 0.01)

    def test_paired_bootstrap_delta(self):
        point, lo, hi = paired_bootstrap_delta((9000, 469, 249, 1482), resamples=2000, seed=3)
        self.assertAlmostEqual(point, (469 - 249) / 11200 * 100)
        self.assertTrue(lo < point < hi and lo > 0)

    def test_mde_decreases_with_n_and_increases_with_discordance(self):
        self.assertGreater(mde_paired(2784, 0.064), mde_paired(11200, 0.064))
        self.assertGreater(mde_paired(11200, 0.12), mde_paired(11200, 0.02))
        self.assertAlmostEqual(mde_paired(11200, 718 / 11200), 2.8016 * math.sqrt(718) / 11200 * 100, places=3)


class LoadAndCompareTests(unittest.TestCase):
    def write(self, path, rows):
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_auto_fields_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            b = Path(tmp) / "b.jsonl"
            rows_a = [{"id": i, "answer": "AC", "prediction": "AC" if i % 4 else "A", "exam_type": "医师考试" if i % 2 else "护理考试",
                       "question_type": "多项选择题"} for i in range(40)]
            rows_b = [{"id": i, "gold": "AC", "pred": "CA" if i % 5 else "B", "exam_type": "医师考试" if i % 2 else "护理考试",
                       "question_type": "多项选择题"} for i in range(40)]
            self.write(a, rows_a)
            self.write(b, rows_b)
            ra = load_predictions(str(a))
            rb = load_predictions(str(b))
            self.assertEqual(sum(c for _, c, _ in ra), 30)
            self.assertEqual(sum(c for _, c, _ in rb), 32)  # 字母顺序不同也算对
            single = score_single(ra, ["exam_type", "question_type"], resamples=500, seed=0)
            self.assertAlmostEqual(single["accuracy"], 75.0)
            self.assertIn("医师考试", single["slices"]["exam_type"])
            result = compare(ra, rb, ["exam_type"], resamples=500, seed=0)
            self.assertEqual(result["n"], 40)
            self.assertAlmostEqual(result["acc_a"], 75.0)
            self.assertAlmostEqual(result["acc_b"], 80.0)
            self.assertAlmostEqual(result["delta_a_minus_b"], -5.0)
            self.assertEqual(result["only_a"] - result["only_b"], -2)

    def test_correct_field_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            self.write(a, [{"question_id": "x", "correct": True}, {"question_id": "y", "correct": 0}])
            rows = load_predictions(str(a))
            self.assertEqual([c for _, c, _ in rows], [True, False])


if __name__ == "__main__":
    unittest.main()
