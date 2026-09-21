import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_ablation_sets import build_variant, load_audit, load_rows, main, select_units
from scripts.build_cmb_train_sft import normalize_stem, stem_option_hash
from scripts.cmexam_prompts import MODE_DIRECT, build_prompt_messages, make_permutation, permute_options, remap_answer
import hashlib
from collections import defaultdict

OPTS = [("A", "甲"), ("B", "乙"), ("C", "丙"), ("D", "丁"), ("E", "戊")]


def sid(messages):
    return hashlib.sha256("\n".join(m["content"] for m in messages if m["role"] == "user").encode("utf-8")).hexdigest()


def make_set(tmp, n_single=8, n_multi=4, n_cmexam=3, seed=42):
    train, meta = [], []
    def add(question, options, gold, source, variant, exam_type, qtype):
        messages = build_prompt_messages(question, options, MODE_DIRECT) + [{"role": "assistant", "content": gold}]
        train.append({"messages": messages})
        meta.append({"sample_id": sid(messages), "source": source, "variant": variant, "exam_type": exam_type, "question_type": qtype})
    for i in range(n_single):
        add(f"单选题干{i}", OPTS, "ABCDE"[i % 5], "cmb_train", "original", "医师考试" if i % 2 else "药师考试", "单项选择题")
    for i in range(n_multi):
        q = f"多选题干{i}"
        add(q, OPTS, "AC", "cmb_train", "original", "医学考研", "多项选择题")
        perm = make_permutation(5, seed, normalize_stem(q), 0)
        add(q, permute_options(OPTS, perm), remap_answer("AC", perm), "cmb_train", "shuffled", "医学考研", "多项选择题")
    for i in range(n_cmexam):
        add(f"CMExam题干{i}", OPTS, "B", "cmexam", "original", "", "")
    t = Path(tmp) / "train.jsonl"; m = Path(tmp) / "meta.jsonl"
    t.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train), encoding="utf-8")
    m.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in meta), encoding="utf-8")
    return t, m


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.train, self.meta = make_set(self.tmp.name)
        self.rows = load_rows(str(self.train), str(self.meta))
        self.units = defaultdict(list)
        for r in self.rows:
            if r["source"] == "cmb_train":
                self.units[r["unit"]].append(r)

    def tearDown(self):
        self.tmp.cleanup()

    def test_units_group_original_with_shuffled_copy(self):
        self.assertEqual(len(self.units), 12)
        self.assertEqual(sum(1 for v in self.units.values() if len(v) == 2), 4)

    def test_scale_subsets_are_nested_and_keep_cmexam(self):
        half = select_units(self.units, 0.5, 42)
        quarter = select_units(self.units, 0.25, 42)
        self.assertTrue(quarter <= half)
        zero, _, _ = build_variant(self.rows, self.units, 0.0, 1, None, 42)
        self.assertEqual([r["source"] for r in zero], ["cmexam"] * 3)
        full, chosen, _ = build_variant(self.rows, self.units, 1.0, 1, None, 42)
        self.assertEqual(len(full), len(self.rows))
        self.assertEqual(len(chosen), 12)

    def test_dose_variants(self):
        no_copies, _, _ = build_variant(self.rows, self.units, 1.0, 0, None, 42)
        self.assertEqual(sum(1 for r in no_copies if r["variant"] == "shuffled"), 0)
        three, _, _ = build_variant(self.rows, self.units, 1.0, 3, None, 42)
        extras = [r for r in three if r["variant"].startswith("shuffled_extra")]
        self.assertEqual(len(extras), 8)  # 4 道多选题各加 2 份
        for r in extras:
            self.assertEqual(sorted(r["gold"]), list(r["gold"]))
            gold_texts = {text for letter, text in r["options"] if letter in r["gold"]}
            self.assertEqual(gold_texts, {"甲", "丙"})
        self.assertEqual(len({r["sample_id"] for r in three}), len(three))

    def test_clean_variant_drops_flagged_units(self):
        multi_unit = next(u for u, v in self.units.items() if len(v) == 2)
        single_unit = next(u for u, v in self.units.items() if len(v) == 1)
        audit = Path(self.tmp.name) / "audit"
        audit.mkdir()
        (audit / "resolutions.jsonl").write_text(json.dumps({"hash": multi_unit, "resolution": "tie", "keep_content": None}) + "\n", encoding="utf-8")
        (audit / "structural_flags.jsonl").write_text(json.dumps({"hash": single_unit, "reason": "figure_reference", "index": 0}) + "\n", encoding="utf-8")
        cleaned, _, dropped = build_variant(self.rows, self.units, 1.0, 1, load_audit(str(audit)), 42)
        self.assertEqual(dropped["tie_conflict"], 2)
        self.assertEqual(dropped["figure_reference"], 1)
        self.assertEqual(len(cleaned), len(self.rows) - 3)

    def test_type_hint_variant_rerenders_prompts(self):
        from scripts.cmexam_prompts import parse_user_content
        tagged, _, _ = build_variant(self.rows, self.units, 1.0, 0, None, 42, type_hint=True)
        self.assertEqual(len(tagged), len([r for r in self.rows if r["variant"] != "shuffled"]))
        for r in tagged:
            parsed = parse_user_content(r["record"]["messages"][1]["content"])
            expected = "本题是多项选择题" if len(r["gold"]) > 1 else "本题是单项选择题"
            self.assertEqual(parsed["hint"], expected)
            self.assertEqual(r["record"]["messages"][-1]["content"], r["gold"])
        self.assertEqual(len({r["sample_id"] for r in tagged}), len(tagged))
        out = Path(self.tmp.name) / "extra"
        main(["--train-file", str(self.train), "--metadata", str(self.meta), "--output-dir", str(out), "--skip-defaults",
              "--extra-variant", "tagged_100:1.0:0:0:1"])
        plan = json.load(open(out / "plan_extra.json", encoding="utf-8"))
        self.assertEqual(plan[0]["name"], "tagged_100")
        self.assertTrue(plan[0]["type_hint"])

    def test_main_writes_plan(self):
        out = Path(self.tmp.name) / "out"
        main(["--train-file", str(self.train), "--metadata", str(self.meta), "--output-dir", str(out), "--scales", "0,0.5", "--dose-copies", "0,3"])
        plan = json.load(open(out / "plan.json", encoding="utf-8"))
        self.assertEqual([p["name"] for p in plan], ["scale_000", "scale_050", "dose_050_copies0", "dose_050_copies3"])
        self.assertTrue((out / "scale_050" / "train.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
