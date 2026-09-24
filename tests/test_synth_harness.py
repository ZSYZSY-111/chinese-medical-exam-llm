import json
import re
import tempfile
import unittest
from pathlib import Path

from scripts.cmexam_prompts import MODE_DIRECT, build_prompt_messages
from scripts.synth_harness.pipeline import (estimate_cost, load_items, load_reference_stems, run, select_items, write_outputs)
from scripts.synth_harness.providers import Budget, FakeProvider, ResponseCache, complete_many
from scripts.synth_harness.tasks import (ParaphraseTask, RationaleTask, StatementTask, letters_from_text, numbers_in)

OPTIONS = [("A", "青霉素"), ("B", "头孢曲松"), ("C", "红霉素"), ("D", "庆大霉素"), ("E", "万古霉素")]
QUESTIONS = [
    ("患者男性，45岁，社区获得性肺炎，首选的抗生素是", "A"),
    ("患者女性，30岁，对青霉素过敏的肺炎，不宜选用的药物是", "A"),
    ("新生儿，出生3天，B族链球菌脑膜炎，首选", "A"),
    ("错题：患者，60岁，铜绿假单胞菌感染，首选", "D"),
    ("MRSA 感染首选的抗生素是", "E"),
    ("以下哪些药物属于β内酰胺类（多选）", "AB"),
]
GOLD_BY_QUESTION = {q: a for q, a in QUESTIONS}
CONFIG = {
    "providers": {"teacher": {"base_url": "http://x", "model": "fake", "temperature": 0.7, "max_tokens": 100,
                              "prices_cny_per_million": {"input": 2.0, "output": 3.0}},
                  "verifier": {"base_url": "http://x", "model": "fake", "temperature": 0.0, "max_tokens": 8,
                               "prices_cny_per_million": {"input": 2.0, "output": 3.0}}},
    "tasks": {"rationale": {"provider": "teacher", "max_chars": 150},
              "paraphrase": {"provider": "teacher", "verifier": "verifier"},
              "statement": {"provider": "teacher", "max_chars": 120}},
    "budget": {"max_cost_cny": 100}, "decontam": {"threshold": 0.7, "min_chars": 10},
}


def question_of(messages):
    text = messages[-1]["content"]
    match = re.search(r"题[目干]：\s*(.+?)(?:\n|$)", text)
    return match.group(1).strip() if match else ""


def responder(messages):
    text = messages[-1]["content"]
    question = question_of(messages)
    gold = GOLD_BY_QUESTION.get(question)
    if "只输出正确选项字母" in text:  # 盲答校验：改写题的题干找不到原题，按关键词还原
        for q, a in QUESTIONS:
            if q.replace("患者", "病人").replace("首选", "应首先选用") == question or q == question:
                return a
        return "C"
    if "改写" in text:
        if question.startswith("错题"):
            return question.replace("60岁", "65岁")  # 改了数字，应被 numbers_preserved 拦下
        return question.replace("患者", "病人").replace("首选", "应首先选用")
    if "陈述句" in text:
        gold_texts = re.search(r"正确答案：\w+（(.+?)）", text).group(1)
        if question.startswith("错题"):
            return "选项 D. 庆大霉素是铜绿假单胞菌感染的首选药物。"  # 引用字母，应被 no_option_letters 拦下
        return f"{question.rstrip('是')}是{gold_texts}，这是临床常用的首选方案。"
    # rationale：错题答错，其余答对
    answer = "B" if question.startswith("错题") else gold
    return f"解析：根据病原学与指南推荐，{question[:12]}应选择相应药物，其余选项不符合。\n答案：{answer}"


def write_input(path):
    with open(path, "w", encoding="utf-8") as handle:
        for question, gold in QUESTIONS:
            messages = build_prompt_messages(question, OPTIONS, MODE_DIRECT) + [{"role": "assistant", "content": gold}]
            handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")


class HelperTests(unittest.TestCase):
    def test_letters_and_numbers(self):
        self.assertEqual(letters_from_text("解析：……\n答案：C、A"), "AC")
        self.assertEqual(letters_from_text("ACD"), "ACD")
        self.assertEqual(letters_from_text("我不知道"), "")
        self.assertEqual(numbers_in("45岁，体温38.5℃，3天"), numbers_in("3天后体温38.5℃的45岁患者"))

    def test_task_versions_change_with_templates(self):
        self.assertTrue(RationaleTask().version.startswith("rationale-"))
        self.assertNotEqual(RationaleTask().version, ParaphraseTask().version)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.input = Path(self.tmp.name) / "train.jsonl"
        write_input(self.input)
        self.items, self.load_counts = load_items(str(self.input))

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_and_select_with_refusal(self):
        self.assertEqual(self.load_counts["loaded"], 6)
        ref = Path(self.tmp.name) / "test.json"
        ref.write_text(json.dumps([{"question": "MRSA感染首选的抗生素是"}], ensure_ascii=False), encoding="utf-8")
        hashes, keys = load_reference_stems([str(ref)])
        selected, counts = select_items(self.items, exclude_hashes=frozenset(hashes), seed=1)
        self.assertEqual(counts["refused_reference_overlap"], 1)
        self.assertEqual(len(selected), 5)
        self.assertNotIn("MRSA 感染首选的抗生素是", [it["question"] for it in selected])

    def test_select_by_scores_and_limit(self):
        scores = {it["sample_id"]: {"p_gold": 0.99 if i % 2 else 0.3} for i, it in enumerate(self.items)}
        selected, counts = select_items(self.items, scores=scores, p_max=0.95, limit=2)
        self.assertEqual(len(selected), 2)
        self.assertGreaterEqual(counts["already_learned_skipped"], 1)
        self.assertTrue(all(it["p_gold"] <= 0.95 for it in selected))

    def test_select_hardest_first(self):
        scores = {it["sample_id"]: {"p_gold": 0.9 - 0.1 * i} for i, it in enumerate(self.items)}
        selected, counts = select_items(self.items, scores=scores, limit=3, order="p_gold")
        self.assertEqual(len(selected), 3)
        self.assertEqual([it["p_gold"] for it in selected], sorted(it["p_gold"] for it in selected))
        self.assertLess(selected[0]["p_gold"], 0.5)
        self.assertEqual(counts["candidates_before_cut"], 6)

    def test_full_run_filters_provenance_cache_and_outputs(self):
        provider = FakeProvider(responder)
        providers = {"teacher": provider, "verifier": provider}
        tasks = [RationaleTask(CONFIG["tasks"]["rationale"]), ParaphraseTask(CONFIG["tasks"]["paraphrase"]), StatementTask(CONFIG["tasks"]["statement"])]
        budget = Budget({"input": 2.0, "output": 3.0}, 100)
        cache_path = Path(self.tmp.name) / "cache.jsonl"
        cache = ResponseCache(str(cache_path))
        records, exports, funnels = run(CONFIG, self.items, tasks, providers, budget, cache, workers=2)
        # rationale：错题答错被 answer_verified 拦下，其余保留
        self.assertEqual(funnels["rationale"]["kept"], 5)
        self.assertEqual(funnels["rationale"]["drop_reasons"], {"answer_verified": 1})
        # paraphrase：错题改了数字被拦下；其余盲答正确且相似度在范围内
        self.assertEqual(funnels["paraphrase"]["drop_reasons"].get("numbers_preserved"), 1)
        self.assertGreaterEqual(funnels["paraphrase"]["kept"], 4)
        # statement：错题引用了字母被拦下
        self.assertEqual(funnels["statement"]["drop_reasons"].get("no_option_letters"), 1)
        self.assertEqual(funnels["statement"]["kept"], 5)
        # 溯源字段齐全
        required = {"gen_id", "sample_id", "task", "task_version", "provider", "model", "request_hash", "cached", "params",
                    "usage", "cost_cny", "parsed", "filters", "kept", "drop_reason", "created_at"}
        self.assertTrue(all(required <= set(r) for r in records))
        self.assertTrue(all(r["cached"] is False for r in records if r["request_hash"]))
        # 导出格式
        self.assertEqual(exports["rationale"][0]["messages"][-1]["content"].split("\n")[-1][:3], "答案：")
        self.assertEqual(exports["paraphrase"][0]["messages"][-1]["content"], GOLD_BY_QUESTION[exports["paraphrase"][0]["original_question"]])
        self.assertIn("text", exports["statement"][0])
        # 缓存：第二次运行零调用、零费用
        calls_before = provider.calls
        budget2 = Budget({"input": 2.0, "output": 3.0}, 100)
        cache2 = ResponseCache(str(cache_path))
        records2, _, funnels2 = run(CONFIG, self.items, tasks, providers, budget2, cache2, workers=2)
        self.assertEqual(provider.calls, calls_before)
        self.assertEqual(budget2.cost, 0.0)
        self.assertEqual(funnels2["rationale"]["cache_hits"], 6)
        self.assertEqual([r["kept"] for r in records2], [r["kept"] for r in records])
        # 输出文件与清单
        out = Path(self.tmp.name) / "out"
        manifest = write_outputs(out, CONFIG, json.dumps(CONFIG), [str(self.input)], tasks, records, exports, funnels, budget, {}, self.load_counts, str(cache_path), 0.0)
        for name in ("provenance.jsonl", "manifest.json", "funnel.md", "data_rationale.jsonl", "data_paraphrase.jsonl", "data_statement.jsonl"):
            self.assertTrue((out / name).exists(), name)
        self.assertEqual(manifest["exports"]["rationale"], 5)
        self.assertIn(str(self.input), manifest["inputs"])
        self.assertNotIn("DEEPSEEK", json.dumps(manifest))  # 清单里没有任何 key

    def test_budget_stops_new_requests(self):
        provider = FakeProvider(responder)
        budget = Budget({"input": 2.0, "output": 3.0}, max_cost_cny=1e-9)
        tasks = [RationaleTask(CONFIG["tasks"]["rationale"])]
        records, exports, funnels = run(CONFIG, self.items, tasks, {"teacher": provider}, budget, None, workers=1)
        self.assertEqual(funnels["rationale"]["requested"], 1)
        self.assertEqual(funnels["rationale"]["budget_stopped"], 5)
        self.assertTrue(budget.exceeded())

    def test_refusal_is_hard_inside_run(self):
        provider = FakeProvider(responder)
        budget = Budget({"input": 2.0, "output": 3.0}, 100)
        with self.assertRaises(RuntimeError):
            run(CONFIG, self.items, [RationaleTask()], {"teacher": provider}, budget, None, workers=1,
                exclude_hashes=frozenset({self.items[0]["stem_hash"]}))
        self.assertEqual(provider.calls, 0)

    def test_decontam_blocks_paraphrase_near_test_stem(self):
        provider = FakeProvider(responder)
        providers = {"teacher": provider, "verifier": provider}
        refs = ["病人男性45岁社区获得性肺炎应首先选用的抗生素是"]  # 与改写结果几乎一样
        _, _, funnels = run(CONFIG, self.items[:1], [ParaphraseTask(CONFIG["tasks"]["paraphrase"])], providers,
                            Budget({"input": 2.0, "output": 3.0}, 100), None, workers=1, decontam_refs=refs)
        self.assertEqual(funnels["paraphrase"]["drop_reasons"].get("decontaminated"), 1)

    def test_estimate_and_complete_many_cache(self):
        rows, total = estimate_cost(CONFIG, [RationaleTask(), ParaphraseTask(), StatementTask()], self.items)
        self.assertEqual(len(rows), 3)
        self.assertGreater(total, 0)
        provider = FakeProvider(lambda m: "答案：A")
        budget = Budget({"input": 2.0, "output": 3.0}, 100)
        reqs = [{"key": "k", "messages": [{"role": "user", "content": "x"}], "params": {"temperature": 0, "max_tokens": 4}}]
        first = complete_many(provider, reqs, budget, None, 1)["k"]
        self.assertEqual(first["cached"], False)
        cache = ResponseCache(None)
        complete_many(provider, reqs, budget, cache, 1)
        again = complete_many(provider, reqs, budget, cache, 1)["k"]
        self.assertTrue(again["cached"])
        self.assertEqual(provider.calls, 2)


if __name__ == "__main__":
    unittest.main()
