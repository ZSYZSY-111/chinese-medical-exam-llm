import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_headroom_records import cells_of, compare, load_records, main, summarize


def record(sid, correct, answer="A", multi=False):
    return {"sample_id": sid, "greedy_correct": correct, "greedy_answer": answer,
            "slices": {"multi_choice": multi, "negation": False, "case": False, "calc": False, "long_stem": False}}


class CompareTests(unittest.TestCase):
    def test_cells_and_delta_direction(self):
        pairs = [(True, True), (True, False), (True, False), (False, True), (False, False)]
        self.assertEqual(cells_of(pairs), (1, 2, 1, 1))
        s = summarize(pairs, resamples=200)
        self.assertEqual((s["acc_a"], s["acc_b"]), (60.0, 40.0))
        self.assertEqual(s["delta_b_minus_a"], -20.0)   # B 比 A 低 20 分
        self.assertLessEqual(s["delta_ci"][0], s["delta_b_minus_a"])
        self.assertGreaterEqual(s["delta_ci"][1], s["delta_b_minus_a"])
        self.assertEqual(s["cells"]["only_a"], 2)

    def test_compare_joins_on_sample_id_and_reports_format_failures(self):
        a = {f"s{i}": record(f"s{i}", i % 2 == 0) for i in range(10)}
        b = {f"s{i}": record(f"s{i}", i % 3 == 0, answer=None if i == 1 else "B", multi=(i < 4)) for i in range(2, 12)}
        report = compare(a, b, resamples=200)
        self.assertEqual((report["common"], report["only_in_a"], report["only_in_b"]), (8, 2, 2))
        self.assertEqual(report["format_failure_a"], 0.0)
        self.assertEqual(report["format_failure_b"], 0.0)   # s1 不在共同集合里
        self.assertEqual(report["slices"]["multi_choice"]["n"], 2)
        self.assertNotIn("negation", report["slices"])

    def test_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb, out = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl", Path(tmp) / "out.json"
            pa.write_text("\n".join(json.dumps(record(f"s{i}", True)) for i in range(5)) + "\n", encoding="utf-8")
            pb.write_text("\n".join(json.dumps(record(f"s{i}", i > 0)) for i in range(5)) + "\n", encoding="utf-8")
            self.assertEqual(main(["--a", str(pa), "--b", str(pb), "--json", str(out), "--resamples", "100"]), 0)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["overall"]["cells"]["only_a"], 1)
            self.assertEqual(len(load_records(pa)), 5)


if __name__ == "__main__":
    unittest.main()
