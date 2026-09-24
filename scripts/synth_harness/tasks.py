"""Generation and annotation tasks with their filters: rationale (blind answer must equal gold), stem paraphrase (blind-answer check + numbers/negations preserved + decontamination), knowledge statement (entity and number constraints), and segment scoring of official rationales.

Each task's version is a hash of its prompt templates, so any template edit changes the version.
"""
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from ..cmexam_prompts import MODE_COT, MODE_DIRECT, build_prompt_messages, render_options
    from ..build_cmb_train_sft import normalize_stem
    from ..sentence_split import split_spans
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cmexam_prompts import MODE_COT, MODE_DIRECT, build_prompt_messages, render_options
    from build_cmb_train_sft import normalize_stem
    from sentence_split import split_spans

ANSWER_LINE = re.compile(r"答案\s*[:：]\s*([A-Za-z][A-Za-z\s,，、]*)")
NEGATION_PATTERN = re.compile(r"不正确|不是|不包括|不包含|不含|不属于|不宜|不能|不会|不需要|不必|不应|不得|并非|错误|除外|无关|最不|不符合|不可能")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
OPTION_LETTER_PATTERN = re.compile(r"(?<![A-Za-z])[A-F]\s*[.．、]")
LETTERS_ONLY = re.compile(r"[A-Za-z][A-Za-z\s,，、]{0,11}")


def letters_from_text(text):
    text = (text or "").strip()
    matches = ANSWER_LINE.findall(text)
    if matches:
        raw = matches[-1]
    elif LETTERS_ONLY.fullmatch(text):
        raw = text
    else:
        return ""
    return "".join(sorted(set(re.findall(r"[A-Z]", raw.upper()))))


def numbers_in(text):
    return Counter(NUMBER_PATTERN.findall(text or ""))


def has_negation(text):
    return NEGATION_PATTERN.search(text or "") is not None


def char_ngrams(text, n=3):
    key = normalize_stem(text or "")
    if len(key) < n:
        return {key} if key else set()
    return {key[i:i + n] for i in range(len(key) - n + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def gold_texts(item):
    return [text for letter, text in item["options"] if letter in item["gold"]]


def version_of(name, *templates):
    digest = hashlib.sha256("\n".join(templates).encode("utf-8")).hexdigest()[:12]
    return f"{name}-{digest}"


class Task:
    name = "base"
    templates = ()
    stages = 1
    expected_output_chars = 100
    stage2_output_chars = 0

    def __init__(self, config=None):
        self.config = dict(config or {})

    @property
    def version(self):
        return version_of(self.name, *self.templates)

    def stage1_messages(self, item):
        raise NotImplementedError

    def parse_stage1(self, text):
        raise NotImplementedError

    def stage2_messages(self, item, parsed):
        return None

    def parse_stage2(self, text):
        return None

    def needs_decontam(self, parsed):
        return None

    def filters(self, item, parsed, stage2, context):
        raise NotImplementedError

    def export(self, item, parsed):
        raise NotImplementedError


RATIONALE_SYSTEM = "你是中国医学考试的资深命题专家，熟悉各类医学资格考试。"
RATIONALE_USER = ("题目：\n{question}\n\n选项：\n{options}\n\n"
                  "请先写不超过{max_chars}字的解析，说明判断依据；然后另起一行，只写“答案：”加正确选项字母"
                  "（多选题按字母顺序连写，例如ACD）。不要输出其他内容。")


class RationaleTask(Task):
    """The model answers blind; only rationales whose answer equals gold are kept, so model output never becomes a label directly."""
    name = "rationale"
    templates = (RATIONALE_SYSTEM, RATIONALE_USER)
    expected_output_chars = 130

    def stage1_messages(self, item):
        return [{"role": "system", "content": RATIONALE_SYSTEM},
                {"role": "user", "content": RATIONALE_USER.format(question=item["question"], options=render_options(item["options"]),
                                                                  max_chars=self.config.get("max_chars", 150))}]

    def parse_stage1(self, text):
        text = (text or "").strip()
        matches = list(ANSWER_LINE.finditer(text))
        if not matches:
            return None
        last = matches[-1]
        answer = "".join(sorted(set(re.findall(r"[A-Z]", last.group(1).upper()))))
        rationale = text[:last.start()].strip()
        rationale = re.sub(r"^解析\s*[:：]\s*", "", rationale).strip()
        if not answer:
            return None
        return {"rationale": rationale, "answer": answer}

    def filters(self, item, parsed, stage2, context):
        max_chars = self.config.get("max_chars", 150)
        rationale = parsed["rationale"]
        return {
            "answer_verified": (parsed["answer"] == item["gold"], f"{parsed['answer']} vs gold {item['gold']}"),
            "length_ok": (10 <= len(rationale) <= int(max_chars * 1.5), len(rationale)),
            "not_letters_only": (re.fullmatch(r"[A-Za-z\s，,、。.]*", rationale) is None, None),
        }

    def export(self, item, parsed):
        messages = build_prompt_messages(item["question"], item["options"], MODE_COT)
        messages.append({"role": "assistant", "content": f"解析：{parsed['rationale']}\n答案：{item['gold']}"})
        text = f"{item['question']}\n{render_options(item['options'])}\n解析：{parsed['rationale']}\n答案：{item['gold']}"
        return {"sample_id": item["sample_id"], "task": self.name, "messages": messages, "text": text}


PARAPHRASE_USER = ("请把下面这道医学选择题的题干改写一遍。要求：意思完全不变；保留所有数字、单位、否定词和专业术语；"
                   "不改动选项；不泄露答案，不增加任何提示；只输出改写后的题干，不要加引号或说明。\n\n"
                   "题干：{question}\n\n选项（仅供理解题意，不要改写）：\n{options}")
VERIFY_USER = "题目：\n{question}\n\n选项：\n{options}\n\n只输出正确选项字母；多选题按字母顺序连写，例如ACD。不要解释。"


class ParaphraseTask(Task):
    """The label stays the original gold; the model only rewrites the stem. A paraphrase must be answered correctly blind, keep numbers and negations, and not be a near-duplicate of any evaluation question."""
    name = "paraphrase"
    templates = (PARAPHRASE_USER, VERIFY_USER)
    stages = 2
    expected_output_chars = 80
    stage2_output_chars = 4

    def stage1_messages(self, item):
        return [{"role": "user", "content": PARAPHRASE_USER.format(question=item["question"], options=render_options(item["options"]))}]

    def parse_stage1(self, text):
        stem = (text or "").strip().strip("“”\"'「」『』").strip()
        stem = re.sub(r"^(改写后的?题干|改写|题干)\s*[:：]\s*", "", stem).strip()
        stem = " ".join(stem.split())
        return {"stem": stem} if stem else None

    def stage2_messages(self, item, parsed):
        return [{"role": "user", "content": VERIFY_USER.format(question=parsed["stem"], options=render_options(item["options"]))}]

    def parse_stage2(self, text):
        return {"answer": letters_from_text(text)}

    def needs_decontam(self, parsed):
        return parsed["stem"]

    def filters(self, item, parsed, stage2, context):
        original, new = item["question"], parsed["stem"]
        similarity = jaccard(char_ngrams(original), char_ngrams(new))
        low = self.config.get("min_similarity", 0.0)  # short stems can have zero 3-gram overlap after rewriting; the blind-answer check guards the meaning
        high = self.config.get("max_similarity", 0.85)
        ratio = len(new) / max(1, len(original))
        decontam_hit = (context.get("decontam") or {}).get(item["sample_id"])
        verified = bool(stage2) and stage2.get("answer") == item["gold"]
        return {
            "not_identical": (normalize_stem(new) != normalize_stem(original), None),
            "similarity_in_range": (low <= similarity <= high, round(similarity, 3)),
            "length_ratio_ok": (0.5 <= ratio <= 2.0, round(ratio, 2)),
            "numbers_preserved": (numbers_in(new) == numbers_in(original), None),
            "negation_preserved": (has_negation(new) == has_negation(original), None),
            "blind_answer_verified": (verified, (stage2 or {}).get("answer")),
            "decontaminated": (decontam_hit is None, decontam_hit),
        }

    def export(self, item, parsed):
        messages = build_prompt_messages(parsed["stem"], item["options"], MODE_DIRECT)
        messages.append({"role": "assistant", "content": item["gold"]})
        return {"sample_id": item["sample_id"], "task": self.name, "messages": messages, "original_question": item["question"]}


STATEMENT_USER = ("下面是一道医学选择题和它的正确答案。请把这道题考查的知识写成1～2句陈述句，不超过{max_chars}字："
                  "直接陈述事实；不要提到“题目”“选项”“正确答案”或选项字母；不要添加原题和选项之外的数字。只输出陈述句。\n\n"
                  "题目：{question}\n选项：\n{options}\n正确答案：{gold_letters}（{gold_texts}）")


class StatementTask(Task):
    """Turn a question into knowledge: the statement must contain the answer content, introduce no new numbers and cite no option letters."""
    name = "statement"
    templates = (STATEMENT_USER,)
    expected_output_chars = 80

    def stage1_messages(self, item):
        return [{"role": "user", "content": STATEMENT_USER.format(
            question=item["question"], options=render_options(item["options"]), gold_letters=item["gold"],
            gold_texts="；".join(gold_texts(item)), max_chars=self.config.get("max_chars", 120))}]

    def parse_stage1(self, text):
        statement = " ".join((text or "").strip().strip("“”\"'").split())
        return {"statement": statement} if statement else None

    def filters(self, item, parsed, stage2, context):
        statement = parsed["statement"]
        max_chars = self.config.get("max_chars", 120)
        source_numbers = numbers_in(item["question"] + " " + " ".join(text for _, text in item["options"]))
        new_numbers = numbers_in(statement)
        numbers_ok = set(new_numbers) <= set(source_numbers)  # no numbers beyond those in the question and options; repeating one is fine
        normalized = normalize_stem(statement)
        coverage = []
        for text in gold_texts(item):
            key = normalize_stem(text)
            if not key:
                continue
            if key in normalized:
                coverage.append(1.0)
            else:
                chars = set(key)
                coverage.append(sum(1 for ch in chars if ch in normalized) / len(chars))
        min_coverage = min(coverage) if coverage else 0.0
        return {
            "length_ok": (10 <= len(statement) <= int(max_chars * 1.5), len(statement)),
            "no_option_letters": (OPTION_LETTER_PATTERN.search(statement) is None and "选项" not in statement and "题目" not in statement, None),
            "numbers_subset_of_source": (numbers_ok, None),
            "contains_answer_content": (min_coverage >= self.config.get("min_answer_coverage", 0.6), round(min_coverage, 2)),
        }

    def export(self, item, parsed):
        return {"sample_id": item["sample_id"], "task": self.name, "text": parsed["statement"], "question": item["question"],
                "gold": item["gold"], "gold_texts": gold_texts(item)}


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
    templates = (RATIONALE_SYSTEM, SENTENCE_SCORE_USER)
    expected_output_chars = 30

    def spans(self, item):
        return split_spans(item.get("explanation") or "", self.config.get("min_chars", 8), self.config.get("level", "clause"))

    def stage1_messages(self, item):
        explanation = item.get("explanation") or ""
        numbered = "\n".join(f"[{index}] {explanation[start:end].strip()}" for index, (start, end) in enumerate(self.spans(item), start=1))
        return [{"role": "system", "content": RATIONALE_SYSTEM},
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

    def filters(self, item, parsed, stage2, context):
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


TASKS = {cls.name: cls for cls in (RationaleTask, ParaphraseTask, StatementTask, SentenceScoreTask)}
