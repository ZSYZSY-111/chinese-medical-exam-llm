"""Split a rationale into short segments and return their character spans.

The scoring task (numbered segments sent to the annotator) and the training-data builders (segment-end
positions) must agree on the boundaries, so the splitter lives in this dependency-free module. It stays
Python 3.9 compatible because the annotation harness runs under the system interpreter.

Two granularities:
- level="sentence": cut only at sentence-final punctuation (。！？； !?; and newlines). Official rationales
  often contain one or two full stops per paragraph, so the decisive fact and the padding share one long
  sentence and a score cannot separate them; the scoring task therefore does not use this level by default.
- level="clause" (default): also cut at commas, giving clause-level segments.

Shared rules:
- decimal points and English abbreviations are not cut; the enumeration comma 、 is not cut either
  ("酪蛋白、乳清蛋白和乳球蛋白" is one unit);
- a closing quote or bracket directly after the punctuation belongs to the same segment;
- no cut inside brackets or quotes, so "（A错；B对）" and quoted questions stay whole;
- fragments shorter than min_chars are merged: comma-ended connectives ("因此，", "其中，") join the
  following segment, sentence-ended fragments join the preceding one;
- spans are contiguous and cover the whole text, so joining them reproduces the input byte for byte.
"""

SENTENCE_TERMINATORS = "。！？；!?;\n"
CLAUSE_TERMINATORS = SENTENCE_TERMINATORS + "，,"
CLOSERS = "”’」』）)】》〉"
OPEN_TO_CLOSE = {"（": "）", "(": ")", "【": "】", "《": "》", "“": "”", "‘": "’", "「": "」", "『": "』"}
LEVEL_SENTENCE, LEVEL_CLAUSE = "sentence", "clause"
DEFAULT_LEVEL = LEVEL_CLAUSE
DEFAULT_MIN_CHARS = 8
MAX_OPEN_CHARS = 80
# short fragments ending with one of these join the next segment (all other short fragments join the previous one)
FORWARD_MERGE_ENDINGS = "，,"


def _is_decimal_comma(text, index):
    """Do not cut at the thousands separator in numbers such as 1,000."""
    return text[index] == "," and 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit()


def split_spans(text, min_chars=DEFAULT_MIN_CHARS, level=DEFAULT_LEVEL):
    """Return [(start, end), ...] with exclusive ends; an empty text gives []."""
    if level not in (LEVEL_SENTENCE, LEVEL_CLAUSE):
        raise ValueError(f"unknown split level: {level!r}")
    if not text:
        return []
    terminators = CLAUSE_TERMINATORS if level == LEVEL_CLAUSE else SENTENCE_TERMINATORS
    cuts = []
    stack = []
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        # an unmatched opening bracket or quote must not suppress cuts forever: drop it at a newline or after MAX_OPEN_CHARS characters
        while stack and (char == "\n" or index - stack[-1][1] > MAX_OPEN_CHARS):
            stack.pop()
        if char in OPEN_TO_CLOSE:
            stack.append((OPEN_TO_CLOSE[char], index))
        elif stack and char == stack[-1][0]:
            stack.pop()
        elif char in terminators and not stack and not _is_decimal_comma(text, index):
            end = index + 1
            while end < length and (text[end] in terminators or text[end] in CLOSERS or text[end] in " \t　"):
                end += 1
            cuts.append(end)
            index = end
            continue
        index += 1
    if not cuts or cuts[-1] != length:
        cuts.append(length)

    spans, start = [], 0
    for end in cuts:
        spans.append([start, end])
        start = end

    def visible(span):
        return len(text[span[0]:span[1]].strip())

    def ends_with_forward_mark(span):
        body = text[span[0]:span[1]].rstrip()
        return bool(body) and body[-1] in FORWARD_MERGE_ENDINGS

    merged, carry_start = [], None
    for span in spans:
        if carry_start is not None:
            span = [carry_start, span[1]]
            carry_start = None
        if visible(span) < min_chars:
            if ends_with_forward_mark(span):
                carry_start = span[0]          # connective: hand it to the next segment
                continue
            if merged:
                merged[-1][1] = span[1]        # short sentence-final fragment: join the previous segment
                continue
        merged.append(span)
    if carry_start is not None:                # a short fragment left over at the very end
        if merged:
            merged[-1][1] = length
        else:
            merged.append([carry_start, length])
    if len(merged) > 1 and visible(merged[0]) < min_chars:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return [(start, end) for start, end in merged]


def split_sentences(text, min_chars=DEFAULT_MIN_CHARS, level=DEFAULT_LEVEL):
    return [text[start:end] for start, end in split_spans(text, min_chars, level)]
