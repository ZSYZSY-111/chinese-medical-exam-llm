import json
import tempfile
import unittest
from pathlib import Path

from scripts import build_sentence_score_sft
from scripts.sentence_split import split_spans
from scripts.train_sft_score_head import (collate, encode_example, normalize_score, pearson, ranks, sentence_char_ends,
                                          summarize_score_predictions)

try:
    import torch
except ImportError:  # 本机没有 torch，损失相关测试在服务器上跑
    torch = None

EXPLANATION = "本题考查社区获得性肺炎的经验性治疗。肺炎链球菌对青霉素敏感（A对）；红霉素主要用于支原体感染（C错）。"
SCORES = [1, 5, 3]


def make_row(explanation=EXPLANATION, scores=SCORES, gold="A", sample_id="s1"):
    spans = split_spans(explanation)
    assert len(spans) == len(scores), (spans, scores)
    return {
        "sample_id": sample_id, "explanation": explanation, "gold": gold,
        "messages": [{"role": "system", "content": "系统"}, {"role": "user", "content": "题目"},
                     {"role": "assistant", "content": f"解析：{explanation}\n答案：{gold}"}],
        "sentences": [{"start": s, "end": e, "score": score} for (s, e), score in zip(spans, scores)],
    }


class CharTokenizer:
    """每个字符一个 token 的假分词器：够用来验证“句末字符 → token 位置”的对齐。"""
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</s>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class AlignmentTests(unittest.TestCase):
    def test_normalize_score(self):
        self.assertEqual([normalize_score(s) for s in (1, 3, 5)], [0.0, 0.5, 1.0])
        with self.assertRaises(ValueError):
            normalize_score(6)

    def test_sentence_ends_point_at_last_visible_char(self):
        row = make_row("第一句话说的是这个内容。  \n第二句话说的是另一个内容。", [2, 4])
        content = row["messages"][-1]["content"]
        ends = sentence_char_ends(content, row["explanation"], row["sentences"])
        self.assertEqual([content[end] for end, _, _ in ends], ["。", "。"])
        self.assertEqual([score for _, _, score in ends], [2, 4])

    def test_encode_example_aligns_positions_labels_and_weights(self):
        tokenizer, row = CharTokenizer(), make_row()
        example = encode_example(tokenizer, row, max_length=4096)
        text = tokenizer.decode(example["input_ids"])
        self.assertEqual(len(example["score_positions"]), 3)
        for position, sentence in zip(example["score_positions"], row["sentences"]):
            expected_last = row["explanation"][:sentence["end"]].rstrip()[-1]
            self.assertEqual(text[position], expected_last)
            self.assertNotEqual(example["labels"][position], -100)
        self.assertEqual(example["score_targets"], [0.0, 1.0, 0.5])
        prompt_length = len(tokenizer.apply_chat_template(row["messages"][:-1], add_generation_prompt=True))
        self.assertTrue(all(label == -100 for label in example["labels"][:prompt_length]))
        self.assertTrue(all(label != -100 for label in example["labels"][prompt_length:]))
        # 句内 token 的权重等于该句分数；句子以外（“解析：”前缀、答案行）保持 1
        second = row["sentences"][1]
        offset = prompt_length + len("解析：")
        self.assertEqual(set(example["sentence_weights"][offset + second["start"]:offset + second["end"]]), {5.0})
        self.assertEqual(example["sentence_weights"][prompt_length], 1.0)
        self.assertEqual(example["sentence_weights"][-1], 1.0)

    def test_overlength_returns_none(self):
        self.assertIsNone(encode_example(CharTokenizer(), make_row(), max_length=10))

    def test_explanation_must_be_in_target(self):
        row = make_row()
        row["explanation"] = "完全不同的解析文本，和目标对不上。"
        with self.assertRaises(ValueError):
            encode_example(CharTokenizer(), row, max_length=4096)


class MetricTests(unittest.TestCase):
    def test_pearson_and_ranks(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertIsNone(pearson([1, 1, 1], [1, 2, 3]))
        self.assertEqual(ranks([10, 30, 20, 30]), [1.0, 3.5, 2.0, 3.5])

    def test_summary(self):
        summary = summarize_score_predictions([0.1, 0.9, 0.5, 0.2, 0.3], [0.0, 1.0, 0.5, 1.0, 0.0], [0, 0, 0, 1, 1])
        self.assertEqual(summary["sentences"], 5)
        self.assertEqual(summary["examples"], 2)
        self.assertEqual(summary["top_sentence_hit_rate"], 0.5)
        self.assertGreater(summary["mae_points_constant_baseline"], summary["mae_points"])


class BuildSplitTests(unittest.TestCase):
    def test_split_is_stable_and_validates_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "export.jsonl"
            rows = [make_row(sample_id=f"id{i}") for i in range(30)]
            broken = make_row(sample_id="broken")
            broken["sentences"][0]["end"] -= 1
            duplicate = make_row(sample_id="id3")
            with open(source, "w", encoding="utf-8") as handle:
                for row in rows + [broken, duplicate]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            outputs = []
            for name in ("a", "b"):
                out_dir = Path(temp_dir) / name
                self.assertEqual(build_sentence_score_sft.main(["--input", str(source), "--output-dir", str(out_dir), "--dev-size", "5"]), 0)
                outputs.append(((out_dir / "train.jsonl").read_text(encoding="utf-8"), (out_dir / "dev.jsonl").read_text(encoding="utf-8")))
            self.assertEqual(outputs[0], outputs[1])
            report = json.loads((Path(temp_dir) / "a" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual((report["train"], report["dev"]), (25, 5))
            self.assertEqual(report["dropped"], {"spans_not_contiguous": 1, "duplicate_sample_id": 1})
            train_ids = {json.loads(line)["sample_id"] for line in outputs[0][0].splitlines()}
            dev_ids = {json.loads(line)["sample_id"] for line in outputs[0][1].splitlines()}
            self.assertFalse(train_ids & dev_ids)


@unittest.skipIf(torch is None, "需要 torch（在服务器上运行）")
class LossTests(unittest.TestCase):
    def setUp(self):
        from scripts.train_sft_score_head import predict_scores, score_loss, weighted_lm_loss
        self.predict_scores, self.score_loss, self.weighted_lm_loss = predict_scores, score_loss, weighted_lm_loss
        torch.manual_seed(0)
        self.hidden = torch.randn(2, 6, 8, requires_grad=True)
        self.lm_head = torch.nn.Linear(8, 11, bias=False)
        self.score_head = torch.nn.Linear(8, 1)
        self.labels = torch.tensor([[-100, -100, 3, 4, 5, 6], [-100, 2, 3, 4, 5, -100]])
        self.cot = torch.tensor([[0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0]], dtype=torch.bool)
        self.answer = torch.tensor([[0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0]], dtype=torch.bool)
        self.weights = torch.tensor([[1, 1, 5, 1, 1, 1], [1, 2, 2, 1, 1, 1]], dtype=torch.float32)

    def lm(self, use_sentence_weights=False, cot_weight=0.2):
        return self.weighted_lm_loss(self.hidden, self.lm_head, self.labels, self.cot, self.answer, self.weights, cot_weight, 0.8, 3,
                                     torch, use_sentence_weights)

    def test_lm_loss_matches_manual_per_sample_normalisation(self):
        logits = self.lm_head(self.hidden[:, :-1, :]).float()
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), self.labels[:, 1:].reshape(-1), reduction="none",
                                               ignore_index=-100).reshape(2, 5)
        cot, answer = self.cot[:, 1:].float(), self.answer[:, 1:].float()
        manual = (0.2 * (ce * cot).sum(1) / cot.sum(1) + 0.8 * (ce * answer).sum(1) / answer.sum(1)).mean()
        self.assertAlmostEqual(float(self.lm()), float(manual), places=5)

    def test_sentence_weighting_changes_only_the_cot_part(self):
        self.assertNotAlmostEqual(float(self.lm(True)), float(self.lm(False)), places=6)
        self.assertAlmostEqual(float(self.lm(True, cot_weight=0.0)), float(self.lm(False, cot_weight=0.0)), places=6)

    def test_score_loss_is_masked_mse_and_reaches_the_backbone(self):
        positions = torch.tensor([[2, 4], [3, -1]])
        targets = torch.tensor([[0.0, 1.0], [0.5, 0.0]])
        predicted, valid = self.predict_scores(self.hidden, self.score_head, positions, torch)
        self.assertEqual(valid.tolist(), [[True, True], [True, False]])
        expected = torch.sigmoid(self.score_head(self.hidden[0, 2].float())).item()
        self.assertAlmostEqual(predicted[0, 0].item(), expected, places=6)
        loss = self.score_loss(predicted, targets, valid, torch)
        manual = sum((predicted[r, c].item() - targets[r, c].item()) ** 2 for r, c in ((0, 0), (0, 1), (1, 0))) / 3
        self.assertAlmostEqual(float(loss), manual, places=6)
        loss.backward()
        self.assertGreater(float(self.hidden.grad[0, 2].abs().sum()), 0.0)   # 句末位置收到梯度
        self.assertEqual(float(self.hidden.grad[1, 5].abs().sum()), 0.0)     # 填充位置不受影响

    def test_collate_pads_every_field(self):
        batch = [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7], "score_positions": [2], "score_targets": [1.0], "sentence_weights": [1.0, 5.0, 5.0]},
                 {"input_ids": [5, 6], "labels": [-100, 6], "score_positions": [], "score_targets": [], "sentence_weights": [1.0, 3.0]}]
        out = collate(batch, 0, torch)
        self.assertEqual(out["input_ids"].tolist(), [[5, 6, 7], [5, 6, 0]])
        self.assertEqual(out["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(out["labels"].tolist(), [[-100, 6, 7], [-100, 6, -100]])
        self.assertEqual(out["score_positions"].tolist(), [[2], [-1]])


if __name__ == "__main__":
    unittest.main()
