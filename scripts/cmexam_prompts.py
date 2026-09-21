"""CMExam 选择题 prompt 工具：解析、渲染、选项乱序、切片标注。

本模块只做纯文本处理，不依赖 torch / transformers，可在本地直接单测。

三种输出模式共用同一套题目渲染，只有 system prompt 和 user 末尾指令不同：

    direct    只输出答案字母（与 cmexam_data/no_explanation 完全一致）
    cot       先写 ≤150 字解析，再写“答案：X”
    adaptive  模型自行决定要不要写解析，最后一行必须是“答案：X”
"""

import hashlib
import re
import unicodedata


MODE_DIRECT = "direct"
MODE_COT = "cot"
MODE_ADAPTIVE = "adaptive"
MODES = (MODE_DIRECT, MODE_COT, MODE_ADAPTIVE)

# 与 cmexam_data/no_explanation 的 system / user 指令逐字一致，保证 direct 模式
# 重新渲染后 sample_id 不变。
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
# 允许空选项文本（CMB 四选项题把 E 存成空串，评测器渲染为 "E. "），rstrip 后是 "E."
OPTION_LINE_PATTERN = re.compile(r"^([A-Z])\.(?: ?(.*))?$")

# 选项内容引用了其他选项或依赖顺序时，不能乱序。
SHUFFLE_UNSAFE_PATTERN = re.compile(
    r"(以上|上述|前述|均不|都不|均是|都是|均可|全部|所有|皆|两者|二者|三者|"
    r"其他选项|其它选项|无正确|没有正确|不确定|"
    r"[A-E]\s*[和与及、,，或+＋]\s*[A-E])"
)

# 切片启发式：只用于统计、过采样和分析，不进入模型输入。
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
    """把任意 key 稳定映射到 [0, 1)，用于可复现的子采样。"""
    digest = stable_hash(*parts)
    return int(digest[:12], 16) / float(16 ** 12)


def normalize_question_key(text):
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", "", normalized)


SINGLE_TYPE_HINT = "本题是单项选择题"
MULTI_TYPE_HINT = "本题是多项选择题"


def type_hint_for(answer=None, question_type=None):
    """与 CMB 官方评测 prompt 对齐的题型提示：优先用数据集的 question_type，否则按答案字母数推断。"""
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
            raise ValueError(f"选项行格式不合法: {line!r}")
        options.append((match.group(1), (match.group(2) or "").strip()))
    letters = [letter for letter, _ in options]
    expected = [chr(ord("A") + index) for index in range(len(options))]
    if not options or letters != expected:
        raise ValueError(f"选项字母序列不合法: {''.join(letters)!r}")
    return options


def parse_user_content(content):
    """把已有 SFT 数据的 user content 拆成题干、选项、提示和指令。"""
    if not isinstance(content, str):
        raise ValueError("user content 必须是字符串")
    match = USER_CONTENT_PATTERN.fullmatch(content.replace("\r\n", "\n").strip())
    if match is None:
        raise ValueError("user content 不符合“题目/选项/指令”三段结构")
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
        raise ValueError(f"未知输出模式: {mode!r}")
    parts = [
        f"题目：\n{question}",
        f"选项：\n{render_options(options)}",
    ]
    if hint:
        parts.append(f"提示：{hint}")
    parts.append(USER_SUFFIXES[mode])
    return "\n\n".join(parts)


def build_prompt_messages(question, options, mode, hint=None):
    """只返回 system + user，不含 assistant；金答案永远不进入 prompt。"""
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
    """由稳定哈希生成非恒等排列；与 forbidden 中的排列都不同时才返回。"""
    if count < 2:
        raise ValueError("至少需要两个选项才能乱序")
    forbidden = {tuple(item) for item in forbidden}
    identity = tuple(range(count))
    for attempt in range(64):
        digest = stable_hash(*seed_parts, attempt)
        order = list(range(count))
        # Fisher–Yates，随机源取自哈希的连续字节
        for index in range(count - 1, 0, -1):
            chunk = digest[(index * 4) % 56:(index * 4) % 56 + 4]
            swap_with = int(chunk, 16) % (index + 1)
            order[index], order[swap_with] = order[swap_with], order[index]
        candidate = tuple(order)
        if candidate != identity and candidate not in forbidden:
            return list(candidate)
    raise ValueError("无法生成新的非恒等排列")


def permute_options(options, permutation):
    """新位置 i 放原来的第 permutation[i] 个选项，并重新贴 A、B、C…。"""
    if sorted(permutation) != list(range(len(options))):
        raise ValueError("permutation 必须是 0..n-1 的排列")
    return [
        (chr(ord("A") + new_index), options[old_index][1])
        for new_index, old_index in enumerate(permutation)
    ]


def remap_answer(answer, permutation):
    """把原顺序下的答案字母映射到乱序后的字母，输出按字母排序。"""
    remapped = []
    for letter in answer:
        old_index = ord(letter) - ord("A")
        if old_index < 0 or old_index >= len(permutation):
            raise ValueError(f"答案字母 {letter!r} 超出选项范围")
        new_index = permutation.index(old_index)
        remapped.append(chr(ord("A") + new_index))
    return "".join(sorted(remapped))


def unmap_answer(answer, permutation):
    """把乱序后的预测字母映射回原顺序，供一致性诊断使用。"""
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
    """从“本题考查…”开头的官方解析中抽取知识点短语；抽不到返回 None。"""
    if not isinstance(explanation, str):
        return None
    text = explanation.replace("\r\n", "\n").strip()
    match = KNOWLEDGE_POINT_PATTERN.match(text)
    if match is None:
        return None
    point = match.group("point").strip().strip("“”\"'")
    if not point or len(point) > MAX_KNOWLEDGE_POINT_CHARS:
        return None
    # 知识点里若直接写出了答案字母判定，会造成标签泄漏，直接放弃。
    if re.search(r"[（(][A-E](对|错)", point):
        return None
    return point
