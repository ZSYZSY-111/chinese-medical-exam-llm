"""Answer parsing shared by data builders and evaluators.

Strict letter-set handling for Chinese medical multiple-choice questions:
normalise an answer string to sorted unique option letters, extract the predicted
answer from a model completion, and parse reference targets from SFT records.
"""
import re

try:
    from .cmexam_prompts import MODE_ADAPTIVE, MODE_COT, MODE_DIRECT, MODES
except ImportError:
    from cmexam_prompts import MODE_ADAPTIVE, MODE_COT, MODE_DIRECT, MODES


ANSWER_ONLY = "letters_only"


ANSWER_WITH_ANALYSIS = "answer_with_analysis"


ANALYSIS_WITH_FINAL_ANSWER = "analysis_with_final_answer"


EXPLANATION_WITH_ANSWER = "explanation_with_answer"


_ANSWER_BODY = r"[A-Za-z](?:[\s,，、/]*[A-Za-z])*"


_LABELED_ANSWER_PATTERN = re.compile(
    rf"(?im)^[ \t]*(?:最终答案|答案)[ \t]*[:：][ \t]*"
    rf"(?P<answer>{_ANSWER_BODY})[ \t]*$"
)


_ANSWER_WITH_ANALYSIS_PATTERN = re.compile(
    r"^答案：(?P<answer>[A-Z]+)\n解析：(?P<analysis>\S[\s\S]*)$"
)


_ANALYSIS_WITH_FINAL_ANSWER_PATTERN = re.compile(
    r"^分析：(?P<analysis>\S[\s\S]*?)\n最终答案：(?P<answer>[A-Z]+)$"
)


_EXPLANATION_WITH_ANSWER_PATTERN = re.compile(
    r"^解析：(?P<analysis>\S[\s\S]*?)\n答案：(?P<answer>[A-Z]+)$"
)


def normalize_answer(raw_answer, valid_letters=None):
    """规范化答案集合；非法字母、重复字母和空答案返回 None。"""
    if not isinstance(raw_answer, str):
        return None

    compact = re.sub(r"[\s,，、/]", "", raw_answer).upper()
    if not compact or re.fullmatch(r"[A-Z]+", compact) is None:
        return None
    if len(set(compact)) != len(compact):
        return None

    normalized = "".join(sorted(compact))
    if valid_letters is not None:
        valid = set(valid_letters)
        if not valid or any(letter not in valid for letter in normalized):
            return None
    return normalized


def completion_to_text(completion):
    """同时兼容 TRL 的普通文本 completion 和对话式 completion。"""
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list) or len(completion) != 1:
        return ""
    message = completion[0]
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def extract_predicted_answer(completion, valid_letters=None):
    """提取唯一答案；额外文字允许存在，但多个答案标签会被拒绝。"""
    text = completion_to_text(completion).replace("\r\n", "\n").strip()
    if not text:
        return None

    pure_answer = normalize_answer(text, valid_letters)
    if pure_answer is not None:
        return pure_answer

    matches = list(_LABELED_ANSWER_PATTERN.finditer(text))
    if len(matches) != 1:
        return None
    return normalize_answer(matches[0].group("answer"), valid_letters)


def parse_reference_target(target, valid_letters=None):
    """Parse the reference answer and its format from an SFT assistant target."""
    if not isinstance(target, str) or not target.strip():
        raise ValueError("empty_assistant_target")
    text = target.replace("\r\n", "\n").replace("\r", "\n").strip()

    pure_answer = normalize_answer(text, valid_letters)
    if pure_answer is not None:
        return {
            "answer": pure_answer,
            "answer_format": ANSWER_ONLY,
            "reference_explanation": "",
        }

    match = _ANSWER_WITH_ANALYSIS_PATTERN.fullmatch(text)
    if match is not None:
        answer = normalize_answer(match.group("answer"), valid_letters)
        if answer is None:
            raise ValueError("invalid_assistant_answer")
        return {
            "answer": answer,
            "answer_format": ANSWER_WITH_ANALYSIS,
            "reference_explanation": match.group("analysis").strip(),
        }

    match = _ANALYSIS_WITH_FINAL_ANSWER_PATTERN.fullmatch(text)
    if match is not None:
        answer = normalize_answer(match.group("answer"), valid_letters)
        if answer is None:
            raise ValueError("invalid_assistant_answer")
        return {
            "answer": answer,
            "answer_format": ANALYSIS_WITH_FINAL_ANSWER,
            "reference_explanation": match.group("analysis").strip(),
        }

    match = _EXPLANATION_WITH_ANSWER_PATTERN.fullmatch(text)
    if match is not None:
        answer = normalize_answer(match.group("answer"), valid_letters)
        if answer is None:
            raise ValueError("invalid_assistant_answer")
        return {
            "answer": answer,
            "answer_format": EXPLANATION_WITH_ANSWER,
            "reference_explanation": match.group("analysis").strip(),
        }

    raise ValueError("unsupported_assistant_target")


_COT_PATTERN = re.compile(
    r"^解析：(?P<analysis>\S[\s\S]*?)\n答案：(?P<answer>[A-Z]+)$"
)


_ADAPTIVE_PATTERN = re.compile(
    r"^(?:解析：(?P<analysis>\S[\s\S]*?)\n)?答案：(?P<answer>[A-Z]+)$"
)


_LABELED_ANSWER_LINE = re.compile(
    r"(?im)^[ \t]*(?:最终答案|答案)[ \t]*[:：][ \t]*([A-Za-z](?:[\s,，、/]*[A-Za-z])*)[ \t]*$"
)


_ANY_ANSWER_LABEL = re.compile(r"(?m)^\s*(?:最终答案|答案)\s*[:：]")


def _extract_labeled_answer(text, valid_letters):
    matches = list(_LABELED_ANSWER_LINE.finditer(text))
    if len(matches) != 1:
        return None
    return normalize_answer(matches[0].group(1), valid_letters)


def parse_completion(completion, mode, valid_letters, lenient_answer=False):
    """解析一条输出；返回答案、是否严格合规、是否写了解析、解析字数。

    direct：答案接受纯字母（也容忍单个“答案：X”行），严格格式要求只有纯字母。
    cot / adaptive：答案只接受唯一一行“答案：X”；lenient_answer=True 时
    （格式热身阶段）纯字母输出也计入答案，但格式奖励仍为 0。
    """
    if mode not in MODES:
        raise ValueError(f"未知输出模式: {mode!r}")
    text = completion_to_text(completion).replace("\r\n", "\n").strip()
    result = {
        "text": text,
        "answer": None,
        "strict": False,
        "used_cot": False,
        "analysis_chars": 0,
    }
    if not text:
        return result

    if mode == MODE_DIRECT:
        result["answer"] = extract_predicted_answer(text, valid_letters)
        normalized = normalize_answer(text, valid_letters)
        result["strict"] = normalized is not None and text == normalized
        return result

    labeled = _extract_labeled_answer(text, valid_letters)
    if labeled is not None:
        result["answer"] = labeled
    elif lenient_answer:
        result["answer"] = normalize_answer(text, valid_letters)

    pattern = _COT_PATTERN if mode == MODE_COT else _ADAPTIVE_PATTERN
    match = pattern.fullmatch(text)
    if match is None:
        result["used_cot"] = text.startswith("解析：")
        return result

    analysis = match.group("analysis")
    raw_answer = match.group("answer")
    normalized = normalize_answer(raw_answer, valid_letters)
    strict = normalized is not None and raw_answer == normalized
    if analysis and _ANY_ANSWER_LABEL.search(analysis):
        strict = False
    result["strict"] = strict
    result["used_cot"] = analysis is not None
    result["analysis_chars"] = len(analysis.strip()) if analysis else 0
    return result
