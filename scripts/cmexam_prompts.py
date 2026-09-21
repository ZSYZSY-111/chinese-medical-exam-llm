"""Prompt utilities for Chinese medical multiple-choice questions: parsing, rendering, option shuffling, slice tagging.

Pure text processing with no torch / transformers dependency, so everything here is unit-tested on CPU.

Three output modes share one question rendering and differ only in the system prompt and the final user instruction:

    direct    answer letters only (the format of all released training data)
    cot       a short explanation (at most 150 characters) followed by a final "答案：X" line
    adaptive  the model decides whether to explain; the last line must be "答案：X"
"""

import hashlib
import re
import unicodedata


MODE_DIRECT = "direct"
MODE_COT = "cot"
MODE_ADAPTIVE = "adaptive"
MODES = (MODE_DIRECT, MODE_COT, MODE_ADAPTIVE)

# Byte-identical to the instructions in the released direct-answer data, so re-rendering a row in direct mode
# never changes its sample_id.
DIRECT_SYSTEM_PROMPT = (
    "你是一名医学考试答题助手。回答中国医学考试选择题。"
    "不要解释，直接输出正确选项字母；多选题按字母顺序连续输出，例如ACD。"
)
DIRECT_USER_SUFFIX = "请直接输出正确选项字母；多选题按字母顺序连续输出。"

COT_SYSTEM_PROMPT = (
    "你是一名医学考试答题助手。回答中国医学考试选择题。"
    "请先写不超过150字的简洁医学解析，再在最后一行给出正确选项字母；"
    "多选题按字母顺序连续输出。严格使用以下格式：\n解析：简洁分析\n答案：X"
)
COT_USER_SUFFIX = "请先写不超过150字的简洁解析，最后一行严格写成“答案：选项字母”。"

ADAPTIVE_SYSTEM_PROMPT = (
    "你是一名医学考试答题助手。回答中国医学考试选择题。"
    "如果这道题需要推理或逐项比较，先写“解析：”给出不超过150字的简洁分析，再作答；"
    "如果是可以直接判断的知识点，就不要写解析，直接作答。"
    "最后一行必须严格写成“答案：选项字母”；多选题按字母顺序连续输出，例如“答案：ACD”。"
)
ADAPTIVE_USER_SUFFIX = (
    "需要推理时先写不超过150字的“解析：”，否则直接作答；"
    "最后一行严格写成“答案：选项字母”。"
)

SYSTEM_PROMPTS = {
    MODE_DIRECT: DIRECT_SYSTEM_PROMPT,
    MODE_COT: COT_SYSTEM_PROMPT,
    MODE_ADAPTIVE: ADAPTIVE_SYSTEM_PROMPT,
}
USER_SUFFIXES = {
    MODE_DIRECT: DIRECT_USER_SUFFIX,
    MODE_COT: COT_USER_SUFFIX,
    MODE_ADAPTIVE: ADAPTIVE_USER_SUFFIX,
}

USER_CONTENT_PATTERN = re.compile(
    r"^题目：\n(?P<question>[\s\S]+?)\n\n选项：\n(?P<options>[\s\S]+?)\n\n"
    r"(?:提示：(?P<hint>[^\n]+)\n\n)?(?P<instruction>[^\n]+)$"
)
# Empty option text is allowed: CMB stores four-option questions with an empty "E", rendered by the evaluator as "E. " ("E." after rstrip).
OPTION_LINE_PATTERN = re.compile(r"^([A-Z])\.(?: ?(.*))?$")

# Options that refer to other options, or to their order, must not be shuffled.
SHUFFLE_UNSAFE_PATTERN = re.compile(
    r"(以上|上述|前述|均不|都不|均是|都是|均可|全部|所有|皆|两者|二者|三者|"
    r"其他选项|其它选项|无正确|没有正确|不确定|"
    r"[A-E]\s*[和与及、,，或+＋]\s*[A-E])"
)

# Slice heuristics: used for statistics, oversampling and analysis only; never part of the model input.
NEGATION_PATTERN = re.compile(
    r"(除外|除.{1,8}外|不正确|错误的是|不是|不包括|不属于|不宜|不应|禁忌|"
    r"不符合|无关|最不|不可能|不会出现|不需要|不必|不能)"
)
CASE_PATTERN = re.compile(r"(患者|患儿|病人|产妇|孕妇|男，|女，|男性|女性|\d+\s*岁)")
CALC_PATTERN = re.compile(r"(计算|mg|ml|mmol|kg|℃|%|/L|/min|mmHg|U/L|IU|μg|ug|mEq)")
KNOWLEDGE_POINT_PATTERN = re.compile(
    r"^本题考查(?:的是|了|的知识点是|的内容是)?[:：]?\s*(?P<point>.+?)[。．；;\n]"
)
LONG_STEM_CHARS = 80
MAX_KNOWLEDGE_POINT_CHARS = 60


def stable_hash(*parts):
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_fraction(*parts):
    """Map any key to a stable value in [0, 1) for reproducible sub-sampling."""
    digest = stable_hash(*parts)
    return int(digest[:12], 16) / float(16 ** 12)


def normalize_question_key(text):
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", "", normalized)


SINGLE_TYPE_HINT = "本题是单项选择题"
MULTI_TYPE_HINT = "本题是多项选择题"


def type_hint_for(answer=None, question_type=None):
    """Question-type hint aligned with the official CMB prompt: use the dataset's question_type when available, otherwise infer it from the number of answer letters."""
    if question_type:
        return MULTI_TYPE_HINT if "多项" in str(question_type) else SINGLE_TYPE_HINT
    if answer is None:
        return None
    return MULTI_TYPE_HINT if len(str(answer).strip()) > 1 else SINGLE_TYPE_HINT


def parse_options_block(block):
    options = []
    for raw_line in block.split("\n"):
        line = raw_line.rstrip()
        if not line:
            continue
        match = OPTION_LINE_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed option line: {line!r}")
        options.append((match.group(1), (match.group(2) or "").strip()))
    letters = [letter for letter, _ in options]
    expected = [chr(ord("A") + index) for index in range(len(options))]
    if not options or letters != expected:
        raise ValueError(f"invalid option letter sequence: {''.join(letters)!r}")
    return options


def parse_user_content(content):
    """Split the user content of an SFT row into question stem, options, optional hint and instruction."""
    if not isinstance(content, str):
        raise ValueError("user content must be a string")
    match = USER_CONTENT_PATTERN.fullmatch(content.replace("\r\n", "\n").strip())
    if match is None:
        raise ValueError("user content does not follow the question / options / instruction layout")
    question = match.group("question").strip()
    options = parse_options_block(match.group("options"))
    return {
        "question": question,
        "options": options,
        "hint": match.group("hint"),
        "instruction": match.group("instruction").strip(),
    }


def render_options(options):
    return "\n".join(f"{letter}. {text}" for letter, text in options)


def render_user_content(question, options, mode, hint=None):
    if mode not in MODES:
        raise ValueError(f"unknown output mode: {mode!r}")
    parts = [
        f"题目：\n{question}",
        f"选项：\n{render_options(options)}",
    ]
    if hint:
        parts.append(f"提示：{hint}")
    parts.append(USER_SUFFIXES[mode])
    return "\n\n".join(parts)


def build_prompt_messages(question, options, mode, hint=None):
    """Returns system + user messages only; the reference answer never enters the prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[mode]},
        {
            "role": "user",
            "content": render_user_content(question, options, mode, hint),
        },
    ]


def is_shuffle_safe(options):
    if len(options) < 3:
        return False
    return not any(SHUFFLE_UNSAFE_PATTERN.search(text) for _, text in options)


def make_permutation(count, *seed_parts, forbidden=()):
    """Non-identity permutation from a stable hash; returned only if it differs from every permutation in `forbidden`."""
    if count < 2:
        raise ValueError("at least two options are needed to shuffle")
    forbidden = {tuple(item) for item in forbidden}
    identity = tuple(range(count))
    for attempt in range(64):
        digest = stable_hash(*seed_parts, attempt)
        order = list(range(count))
        # Fisher-Yates, with randomness taken from consecutive bytes of the hash
        for index in range(count - 1, 0, -1):
            chunk = digest[(index * 4) % 56:(index * 4) % 56 + 4]
            swap_with = int(chunk, 16) % (index + 1)
            order[index], order[swap_with] = order[swap_with], order[index]
        candidate = tuple(order)
        if candidate != identity and candidate not in forbidden:
            return list(candidate)
    raise ValueError("could not generate a new non-identity permutation")


def permute_options(options, permutation):
    """New position i holds the original option permutation[i]; letters are re-assigned A, B, C, ..."""
    if sorted(permutation) != list(range(len(options))):
        raise ValueError("permutation must be a permutation of 0..n-1")
    return [
        (chr(ord("A") + new_index), options[old_index][1])
        for new_index, old_index in enumerate(permutation)
    ]


def remap_answer(answer, permutation):
    """Map answer letters from the original order to the shuffled order; the result is sorted."""
    remapped = []
    for letter in answer:
        old_index = ord(letter) - ord("A")
        if old_index < 0 or old_index >= len(permutation):
            raise ValueError(f"answer letter {letter!r} is outside the option range")
        new_index = permutation.index(old_index)
        remapped.append(chr(ord("A") + new_index))
    return "".join(sorted(remapped))


def unmap_answer(answer, permutation):
    """Map predicted letters from the shuffled order back to the original order (used by the consistency diagnostic)."""
    if answer is None:
        return None
    original = []
    for letter in answer:
        new_index = ord(letter) - ord("A")
        if new_index < 0 or new_index >= len(permutation):
            return None
        original.append(chr(ord("A") + permutation[new_index]))
    return "".join(sorted(original))


def stem_length_bucket(question):
    length = len(question)
    if length < 30:
        return "short"
    if length < LONG_STEM_CHARS:
        return "medium"
    return "long"


def detect_slices(question, options, answer):
    option_text = " ".join(text for _, text in options)
    return {
        "multi_choice": len(answer) > 1,
        "negation": bool(NEGATION_PATTERN.search(question)),
        "case": bool(CASE_PATTERN.search(question)),
        "calc": bool(CALC_PATTERN.search(question + " " + option_text)),
        "long_stem": len(question) >= LONG_STEM_CHARS,
        "stem_bucket": stem_length_bucket(question),
        "option_count": len(options),
        "shuffle_safe": is_shuffle_safe(options),
    }


def extract_knowledge_point(explanation):
    """Extract the knowledge-point phrase from an official explanation that starts with "本题考查…"; None if absent."""
    if not isinstance(explanation, str):
        return None
    text = explanation.replace("\r\n", "\n").strip()
    match = KNOWLEDGE_POINT_PATTERN.match(text)
    if match is None:
        return None
    point = match.group("point").strip().strip("“”\"'")
    if not point or len(point) > MAX_KNOWLEDGE_POINT_CHARS:
        return None
    # A knowledge point that states which option is right or wrong would leak the label; discard it.
    if re.search(r"[（(][A-E](对|错)", point):
        return None
    return point
