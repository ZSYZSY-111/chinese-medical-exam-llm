"""Annotation tasks and their filters. The released task is `sentence_score`: rate each clause-level segment of an
official rationale (1–5) for how decisive it is for the answer. A task's version is a hash of its prompt templates.
"""
import hashlib
import re
import sys
from pathlib import Path

try:
    from ..cmexam_prompts import render_options
    from ..sentence_split import split_spans
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cmexam_prompts import render_options
    from sentence_split import split_spans


def version_of(name, *templates):
    digest = hashlib.sha256("\n".join(templates).encode("utf-8")).hexdigest()[:12]
    return f"{name}-{digest}"


class Task:
    name = "base"
    templates = ()
    expected_output_chars = 100

    def __init__(self, config=None):
        self.config = dict(config or {})

    @property
    def version(self):
        return version_of(self.name, *self.templates)

    def stage1_messages(self, item):
        raise NotImplementedError

    def parse_stage1(self, text):
        raise NotImplementedError

    def filters(self, item, parsed, context):
        raise NotImplementedError

    def export(self, item, parsed):
        raise NotImplementedError


SCORER_SYSTEM = "你是中国医学考试的资深命题专家，熟悉各类医学资格考试。"


SENTENCE_SCORE_USER = ("下面是一道中国医学考试选择题、它的正确答案，以及官方解析。解析已按逗号和句号切成小片段并编号，"
                       "片段可能不是完整的句子，请结合上下文理解它。"
                       "请给每个片段打分，表示这个片段里的医学知识对“判断出正确答案”有多关键：\n"
                       "5 = 决定性依据：没有它就无法确定答案\n"
                       "4 = 重要依据：直接支持正确选项，或排除某个错误选项\n"
                       "3 = 有帮助的相关知识，但不是判断的关键\n"
                       "2 = 背景介绍或一般性陈述\n"
                       "1 = 套话、单纯复述题干，或只宣布结论而不含医学依据（如“故选D”）\n\n"
                       "题目：\n{question}\n\n选项：\n{options}\n\n正确答案：{gold}\n\n解析片段：\n{sentences}\n\n"
                       "只输出一个 JSON 数组，长度必须等于片段数（{count} 个），元素是 1～5 的整数，例如 [2,5,4,1]。不要输出其他内容。")
SCORE_ARRAY = re.compile(r"\[[\d\s,，]*\]")


class SentenceScoreTask(Task):
    """Score each clause-level segment of an official rationale (1–5) for how decisive it is for the answer.

    The model only scores human-written rationales; it produces no answers and no new text. Segmentation is fixed by
    sentence_split and the model must return an integer array of exactly that length, otherwise the row is dropped.
    The scores serve as auxiliary supervision during training.
    """
    name = "sentence_score"
    templates = (SCORER_SYSTEM, SENTENCE_SCORE_USER)
    expected_output_chars = 30

    def spans(self, item):
        return split_spans(item.get("explanation") or "", self.config.get("min_chars", 8), self.config.get("level", "clause"))

    def stage1_messages(self, item):
        explanation = item.get("explanation") or ""
        numbered = "\n".join(f"[{index}] {explanation[start:end].strip()}" for index, (start, end) in enumerate(self.spans(item), start=1))
        return [{"role": "system", "content": SCORER_SYSTEM},
                {"role": "user", "content": SENTENCE_SCORE_USER.format(question=item["question"], options=render_options(item["options"]),
                                                                       gold=item["gold"], sentences=numbered, count=len(self.spans(item)))}]

    def parse_stage1(self, text):
        matches = SCORE_ARRAY.findall(text or "")
        if not matches:
            return None
        try:
            scores = [int(value) for value in re.findall(r"\d+", matches[-1])]
        except ValueError:
            return None
        return {"scores": scores} if scores else None

    def filters(self, item, parsed, context):
        scores = parsed["scores"]
        count = len(self.spans(item))
        return {
            "count_matches": (len(scores) == count, f"{len(scores)} vs {count}"),
            "range_ok": (all(1 <= value <= 5 for value in scores), None),
        }

    def export(self, item, parsed):
        sentences = [{"start": start, "end": end, "score": score} for (start, end), score in zip(self.spans(item), parsed["scores"])]
        return {"sample_id": item["sample_id"], "task": self.name, "messages": item["messages"], "explanation": item["explanation"],
                "gold": item["gold"], "sentences": sentences}


TASKS = {cls.name: cls for cls in (SentenceScoreTask,)}
