import json
import tempfile
import unittest
from pathlib import Path

from scripts.cmexam_prompts import MODE_COT, build_prompt_messages
from scripts.sentence_split import split_sentences, split_spans
from scripts.synth_harness.pipeline import load_items, run, select_items
from scripts.synth_harness.providers import Budget, FakeProvider, ResponseCache
from scripts.synth_harness.tasks import SentenceScoreTask

OPTIONS = [("A", "青霉素"), ("B", "头孢曲松"), ("C", "红霉素"), ("D", "庆大霉素"), ("E", "万古霉素")]
EXPLANATION = ("本题考查社区获得性肺炎的经验性治疗。肺炎链球菌是最常见的病原体，对青霉素敏感（A对）；"
               "红霉素主要用于支原体感染（C错）。故选A。")
CONFIG = {
    "providers": {"scorer": {"base_url": "http://x", "model": "fake", "temperature": 0.0, "max_tokens": 96,
                             "prices_cny_per_million": {"input": 2.0, "output": 3.0}}},
    "tasks": {"sentence_score": {"provider": "scorer"}},
    "budget": {"max_cost_cny": 100},
}


def explained_record(question, gold, explanation=EXPLANATION):
    messages = build_prompt_messages(question, OPTIONS, MODE_COT, style="legacy_explain")
    messages.append({"role": "assistant", "content": f"解析：{explanation}\n答案：{gold}"})
    return {"messages": messages}


class SplitTests(unittest.TestCase):
    def test_spans_cover_text_exactly(self):
        for text in (EXPLANATION, "只有一句没有句号", "第一句话是这样的。\n第二句话换行了！第三句话在这里？”然后继续说完这一句。", ""):
            for level in ("sentence", "clause"):
                self.assertEqual("".join(text[s:e] for s, e in split_spans(text, level=level)), text)
            spans = split_spans(text)
            for (_, end), (start, _) in zip(spans, spans[1:]):
                self.assertEqual(end, start)

    def test_does_not_split_inside_brackets_or_quotes_or_decimals(self):
        text = "医生应避免诱导性提问，如：“你胸痛放射至左手，对吗？”（D错；为本题答案）。血钾低于3.5mmol/L为低钾血症。"
        self.assertEqual(len(split_sentences(text, level="sentence")), 2)
        clauses = split_sentences(text)
        self.assertEqual(clauses, ["医生应避免诱导性提问，", "如：“你胸痛放射至左手，对吗？”（D错；为本题答案）。", "血钾低于3.5mmol/L为低钾血症。"])

    def test_short_fragments_are_merged(self):
        self.assertEqual(split_sentences("肺炎链球菌对青霉素敏感，首选青霉素治疗。故选A。", level="sentence"), ["肺炎链球菌对青霉素敏感，首选青霉素治疗。故选A。"])
        self.assertEqual(len(split_sentences("对。肺炎链球菌对青霉素敏感，首选青霉素治疗。", level="sentence")), 1)

    def test_clause_level_splits_at_commas(self):
        text = "牛乳蛋白质主要由酪蛋白（79.6%）、乳清蛋白和乳球蛋白组成，其主要成分是酪蛋白（B对ACDE错）。"
        self.assertEqual(split_sentences(text), ["牛乳蛋白质主要由酪蛋白（79.6%）、乳清蛋白和乳球蛋白组成，", "其主要成分是酪蛋白（B对ACDE错）。"])

    def test_clause_level_short_connective_merges_forward_and_tail_merges_back(self):
        self.assertEqual(split_sentences("因此，肺炎链球菌肺炎首选青霉素治疗，疗效确切。"), ["因此，肺炎链球菌肺炎首选青霉素治疗，疗效确切。"])
        self.assertEqual(split_sentences("因此，肺炎链球菌肺炎首选青霉素治疗，其余抗生素均不是首选药物。"),
                         ["因此，肺炎链球菌肺炎首选青霉素治疗，", "其余抗生素均不是首选药物。"])

    def test_clause_level_keeps_digit_grouping_commas(self):
        self.assertEqual(split_sentences("白细胞计数为12,000个每微升属于明显升高，提示体内存在明显的细菌感染。"),
                         ["白细胞计数为12,000个每微升属于明显升高，", "提示体内存在明显的细菌感染。"])

    def test_unknown_level_is_rejected(self):
        with self.assertRaises(ValueError):
            split_spans("任意文本。", level="word")

    def test_unbalanced_bracket_does_not_swallow_the_rest(self):
        text = "第一句里有个没配对的括号（注意这里。" + "这是很长的一句话用来超过作废阈值" * 6 + "。最后一句也要能切出来才行。"
        self.assertGreaterEqual(len(split_sentences(text)), 2)


class SentenceScoreTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "train.jsonl"
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(explained_record("社区获得性肺炎首选的抗生素是", "A"), ensure_ascii=False) + "\n")
            handle.write(json.dumps(explained_record("数量对不上的题目应当被丢弃，首选", "A"), ensure_ascii=False) + "\n")
            handle.write(json.dumps(explained_record("分数越界的题目应当被丢弃，首选", "A"), ensure_ascii=False) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_items_reads_explanation_records(self):
        items, counts = load_items(self.path)
        self.assertEqual(counts["loaded"], 3)
        self.assertEqual(counts["loaded_with_explanation"], 3)
        self.assertEqual(items[0]["gold"], "A")
        self.assertEqual(items[0]["explanation"], EXPLANATION)
        self.assertEqual(items[0]["messages"][-1]["content"], f"解析：{EXPLANATION}\n答案：A")

    def test_prompt_numbers_sentences_and_shows_gold(self):
        items, _ = load_items(self.path)
        task = SentenceScoreTask(CONFIG["tasks"]["sentence_score"])
        user = task.stage1_messages(items[0])[-1]["content"]
        count = len(split_spans(EXPLANATION))
        self.assertEqual(count, 4)  # 按逗号、句号切；末尾的短句“故选A。”并入前一段
        self.assertIn("[1] 本题考查", user)
        self.assertIn(f"[{count}] ", user)
        self.assertNotIn(f"[{count + 1}] ", user)
        self.assertIn("正确答案：A", user)
        self.assertIn(f"（{count} 个）", user)

    def test_parse_and_filters(self):
        task = SentenceScoreTask()
        self.assertEqual(task.parse_stage1("[2, 5, 4]"), {"scores": [2, 5, 4]})
        self.assertEqual(task.parse_stage1("打分如下：\n```json\n[2,5,1]\n```"), {"scores": [2, 5, 1]})
        self.assertIsNone(task.parse_stage1("无法判断"))
        items, _ = load_items(self.path)
        ok = task.filters(items[0], {"scores": [1, 3, 5, 2]}, None, {})
        self.assertTrue(all(passed for passed, _ in ok.values()))
        self.assertFalse(task.filters(items[0], {"scores": [2, 5]}, None, {})["count_matches"][0])
        self.assertFalse(task.filters(items[0], {"scores": [2, 5, 9, 1]}, None, {})["range_ok"][0])

    def test_full_run_exports_spans_with_scores(self):
        items, _ = load_items(self.path)
        selected, _ = select_items(items, limit=0)

        def responder(messages):
            text = messages[-1]["content"]
            if "数量对不上" in text:
                return "[3, 3]"
            if "分数越界" in text:
                return "[3, 7, 1, 1]"
            return "[1, 3, 5, 2]"

        providers = {"scorer": FakeProvider(responder)}
        budget = Budget({"input": 2.0, "output": 3.0}, 100)
        cache = ResponseCache(str(Path(self.tmp.name) / "cache.jsonl"))
        task = SentenceScoreTask(CONFIG["tasks"]["sentence_score"])
        records, exports, funnels = run(CONFIG, selected, [task], providers, budget, cache, workers=2)
        self.assertEqual(funnels["sentence_score"]["kept"], 1)
        self.assertEqual(funnels["sentence_score"]["drop_reasons"], {"count_matches": 1, "range_ok": 1})
        row = exports["sentence_score"][0]
        self.assertEqual([s["score"] for s in row["sentences"]], [1, 3, 5, 2])
        self.assertEqual("".join(row["explanation"][s["start"]:s["end"]] for s in row["sentences"]), row["explanation"])
        self.assertEqual(row["messages"][-1]["content"], f"解析：{EXPLANATION}\n答案：A")


if __name__ == "__main__":
    unittest.main()
