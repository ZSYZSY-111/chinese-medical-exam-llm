#!/usr/bin/env python3
"""Segment credit for GRPO: split the sparse "one correctness signal per rationale" advantage into per-segment
contributions.

For every sampled rationale, insert "\\n答案：" after each clause-level segment (sentence_split) and read the
current policy's own probability of the gold answer, Φ_k, normalised over the valid option letters. Segment k
earns c_k = Φ_k − Φ_{k−1}; every token in that segment gets advantage = the sequence-level GRPO advantage + λ·c_k.
The answer part and the "解析：" prefix keep the sequence-level advantage only. The credits telescope to
Φ_K − Φ_0, the net confidence the rationale added, so padding earns nothing and misleading segments are penalised.

The pure functions (splitting, character-to-token alignment, credits, advantage assembly) have no torch
dependency and are unit-tested; the probe forward pass lives in the GRPOTrainer subclass built by
make_segment_credit_trainer.
"""
import sys
import time
from pathlib import Path

try:
    from .sentence_split import split_spans
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sentence_split import split_spans

RATIONALE_PREFIX = "解析："
ANSWER_MARKER = "\n答案："
DEFAULT_MAX_PROBES = 8


def split_rationale(text):
    """Return (start, end, rationale) as character offsets into the completion text.
    The rationale runs from after the "解析：" prefix to the last "\\n答案：" marker; with only the prefix it runs to
    the end, with only the marker it starts at 0. With neither (a bare letter, say) the completion is not in
    rationale format: return an empty rationale so the completion keeps the sequence-level advantage only."""
    has_prefix = text.startswith(RATIONALE_PREFIX)
    marker_at = text.rfind(ANSWER_MARKER)
    if not has_prefix and marker_at < 0:
        return 0, 0, ""
    start = len(RATIONALE_PREFIX) if has_prefix else 0
    end = marker_at if marker_at >= start else len(text)
    return start, end, text[start:end]


def select_boundaries(spans, max_probes=DEFAULT_MAX_PROBES):
    """When there are more than max_probes segments, merge neighbours evenly into max_probes segments, keeping them contiguous and the final end fixed."""
    if max_probes < 1:
        raise ValueError("max_probes must be at least 1")
    if len(spans) <= max_probes:
        return [tuple(span) for span in spans]
    n = len(spans)
    picks = sorted({round((i + 1) * n / max_probes) - 1 for i in range(max_probes)})
    if picks[-1] != n - 1:
        picks[-1] = n - 1
    merged, start = [], spans[0][0]
    for index in picks:
        merged.append((start, spans[index][1]))
        start = spans[index][1]
    return merged


def prefix_length(tokenizer, ids, count, cache):
    """Character length of decode(ids[:count]), memoised."""
    if count not in cache:
        cache[count] = len(tokenizer.decode(ids[:count], skip_special_tokens=True))
    return cache[count]


def first_token_covering(tokenizer, ids, char_index, cache):
    """Smallest token index t such that decode(ids[:t+1]) is longer than char_index; None if beyond the text.
    Prefix decode lengths are non-decreasing in t, which makes binary search valid."""
    if not ids or prefix_length(tokenizer, ids, len(ids), cache) <= char_index:
        return None
    lo, hi = 0, len(ids) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if prefix_length(tokenizer, ids, mid + 1, cache) > char_index:
            hi = mid
        else:
            lo = mid + 1
    return lo


def char_spans_to_token_ranges(tokenizer, ids, char_spans):
    """[(start_char, end_char)] → [(first_token, last_token)] (inclusive); segments that cannot be aligned give None."""
    cache = {}
    ranges = []
    for start, end in char_spans:
        if end <= start:
            ranges.append(None)
            continue
        first = first_token_covering(tokenizer, ids, start, cache)
        last = first_token_covering(tokenizer, ids, end - 1, cache)
        ranges.append((first, last) if first is not None and last is not None and last >= first else None)
    return ranges


def segment_credits(phis):
    """phis = [Φ_0, Φ_1, …, Φ_K] → [Φ_1−Φ_0, …, Φ_K−Φ_{K−1}]。"""
    return [phis[k] - phis[k - 1] for k in range(1, len(phis))]


def plan_completion(tokenizer, ids, text, max_probes=DEFAULT_MAX_PROBES):
    """Credit plan for one completion: the rationale prefixes to probe (including the empty prefix) and each segment's token range."""
    start, end, rationale = split_rationale(text)
    spans = select_boundaries(split_spans(rationale), max_probes) if rationale.strip() else []
    prefixes = [""] + [rationale[:span_end] for _, span_end in spans]
    ranges = char_spans_to_token_ranges(tokenizer, ids, [(start + a, start + b) for a, b in spans])
    return {"prefixes": prefixes, "token_ranges": ranges, "rationale_chars": len(rationale), "segments": len(spans)}


def token_credit_rows(length, token_ranges, credits):
    """Spread segment credits over tokens: a list of length `length`, zero outside the segments."""
    row = [0.0] * length
    for token_range, credit in zip(token_ranges, credits):
        if token_range is None:
            continue
        first, last = token_range
        for index in range(first, min(last, length - 1) + 1):
            row[index] = credit
    return row


def probe_text(prompt_text, prefix):
    return f"{prompt_text}{RATIONALE_PREFIX}{prefix}{ANSWER_MARKER}"


def make_segment_credit_trainer(base_cls, torch):
    """Return a GRPOTrainer subclass that adds λ × segment credit to the sequence-level advantages the parent computed, producing (B, T) token-level advantages."""

    class SegmentCreditGRPOTrainer(base_cls):
        def __init__(self, *args, credit_lambda=2.0, max_probes=DEFAULT_MAX_PROBES, probe_batch_size=64, **kwargs):
            super().__init__(*args, **kwargs)
            self.credit_lambda = float(credit_lambda)
            self.max_probes = int(max_probes)
            self.probe_batch_size = int(probe_batch_size)
            self._letter_ids = {}

        def _letter_id(self, letter):
            if letter not in self._letter_ids:
                encoded = self.processing_class.encode(letter, add_special_tokens=False)
                if len(encoded) != 1:
                    raise ValueError(f"letter {letter!r} is not a single token: {encoded}")
                self._letter_ids[letter] = encoded[0]
            return self._letter_ids[letter]

        @torch.no_grad()
        def _probe_phi(self, texts, gold_letters, valid_letters):
            """Next-token distribution at the end of each probe text, restricted to the valid letters → confidence in the gold answer's first letter. Returns (phis, letter_mass)."""
            tokenizer = self.processing_class
            model = self.model
            was_training = model.training
            model.eval()
            phis, masses = [], []
            try:
                for start in range(0, len(texts), self.probe_batch_size):
                    chunk = texts[start:start + self.probe_batch_size]
                    encoded = tokenizer(chunk, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
                    input_ids = encoded["input_ids"].to(model.device)
                    attention_mask = encoded["attention_mask"].to(model.device)
                    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                   logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
                    probs = torch.softmax(logits, dim=-1)
                    for row, index in enumerate(range(start, start + len(chunk))):
                        letters = valid_letters[index]
                        ids = [self._letter_id(letter) for letter in letters]
                        letter_probs = probs[row, ids]
                        mass = float(letter_probs.sum())
                        gold = gold_letters[index]
                        gold_prob = float(probs[row, self._letter_id(gold)]) if gold in letters else 0.0
                        phis.append(gold_prob / mass if mass > 0 else 0.0)
                        masses.append(mass)
            finally:
                model.train(was_training)
            return phis, masses

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            if self.credit_lambda <= 0 or not self.model.training:
                return output
            started = time.time()
            tokenizer = self.processing_class
            completion_ids, completion_mask = output["completion_ids"], output["completion_mask"]
            batch_size, length = completion_ids.shape
            seq_adv = output["advantages"]
            if seq_adv.dim() != 1:
                raise ValueError("the parent trainer should provide (B,) sequence-level advantages")

            plans, probe_texts, probe_gold, probe_valid, owners = [], [], [], [], []
            for i in range(batch_size):
                count = int(completion_mask[i].sum())
                ids = completion_ids[i, :count].tolist()
                text = tokenizer.decode(ids, skip_special_tokens=True)
                plan = plan_completion(tokenizer, ids, text, self.max_probes)
                plans.append(plan)
                prompt_text = tokenizer.apply_chat_template(inputs[i]["prompt"], tokenize=False, add_generation_prompt=True)
                gold = str(inputs[i]["answer"])[:1]
                valid = str(inputs[i]["valid_letters"])
                for prefix in plan["prefixes"]:
                    probe_texts.append(probe_text(prompt_text, prefix))
                    probe_gold.append(gold)
                    probe_valid.append(valid)
                    owners.append(i)

            phis, masses = self._probe_phi(probe_texts, probe_gold, probe_valid)
            credit = torch.zeros((batch_size, length), dtype=torch.float32)
            phi_first, phi_last, all_credits, cursor = [], [], [], 0
            for i, plan in enumerate(plans):
                count = len(plan["prefixes"])
                own = phis[cursor:cursor + count]
                cursor += count
                credits = segment_credits(own)
                phi_first.append(own[0])
                phi_last.append(own[-1])
                all_credits.extend(credits)
                credit[i] = torch.tensor(token_credit_rows(length, plan["token_ranges"], credits), dtype=torch.float32)
            credit = credit.to(seq_adv.device) * completion_mask.float()
            token_adv = seq_adv.unsqueeze(1).float() + self.credit_lambda * credit
            token_adv = token_adv * completion_mask.float()
            output["advantages"] = token_adv

            mode = "train"
            active = completion_mask.float()
            nonzero = ((token_adv.abs() > 1e-8).float() * active).sum() / active.sum().clamp(min=1.0)
            seq_nonzero = ((seq_adv.abs() > 1e-8).float().unsqueeze(1) * active).sum() / active.sum().clamp(min=1.0)
            self._metrics[mode]["credit/probe_seconds"].append(time.time() - started)
            self._metrics[mode]["credit/probes_per_completion"].append(len(probe_texts) / max(batch_size, 1))
            self._metrics[mode]["credit/phi_first_mean"].append(sum(phi_first) / max(len(phi_first), 1))
            self._metrics[mode]["credit/phi_last_mean"].append(sum(phi_last) / max(len(phi_last), 1))
            self._metrics[mode]["credit/abs_credit_mean"].append(sum(abs(c) for c in all_credits) / max(len(all_credits), 1))
            self._metrics[mode]["credit/letter_mass_mean"].append(sum(masses) / max(len(masses), 1))
            self._metrics[mode]["credit/token_frac_nonzero_adv"].append(float(nonzero))
            self._metrics[mode]["credit/token_frac_nonzero_seq_adv"].append(float(seq_nonzero))
            self._metrics[mode]["credit/frac_no_segments"].append(sum(1 for p in plans if p["segments"] == 0) / max(batch_size, 1))
            return output

    return SegmentCreditGRPOTrainer
