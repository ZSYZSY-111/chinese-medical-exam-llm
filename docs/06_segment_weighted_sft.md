# Segment-level supervision for rationale SFT: the same signal, two ways in

*Code: `scripts/sentence_split.py`, `scripts/synth_harness/` (task `sentence_score`), `scripts/build_sentence_score_sft.py`, `scripts/train_sft_score_head.py`. Numbers: `results/segment_weighted_sft.json`. Released labels: `data/segment_scores/`.*

## 1. Why segment scores

A rationale-then-answer training example carries one verifiable bit, the answer, plus a few hundred characters of human-written explanation in which the decisive fact sits next to boilerplate ("本题考查…"), background and restatements. Plain SFT spends its loss evenly over all of it. The question here was whether telling the model *which* segments matter changes what it learns, and, if so, how that information has to enter the loss.

## 2. Getting the scores

Each official CMExam rationale is split into clause-level segments by `sentence_split.py` (commas, sentence-final punctuation and newlines; no cut inside brackets, quotes or numbers; fragments under 8 characters are merged). The median rationale has 9 segments. Splitting only at sentence-final punctuation was tried first and rejected: 22% of rationales then had one or two segments, so the decisive fact and the padding shared a score.

DeepSeek V4 Pro rated every segment from 1 to 5 for how decisive it is for reaching the correct answer, seeing the question, the options, the gold answer and the numbered segments. The harness (`scripts/synth_harness/`) enforces that the reply is an integer array of exactly the right length, refuses to send any evaluation-set stem, caches responses by request hash and caps spending. 10,000 questions were scored (9,985 kept; 111,294 segments; 50.5 CNY at list prices). Re-scoring the same 281 segments twice with sampling agreed within one point 93% of the time, and the top score was stable 85% of the time. Boilerplate segments average 1.1, segments that name the correct option average 4.3.

The scores are released without any question text in `data/segment_scores/cmexam_rationale_segment_scores.jsonl.gz`: one row per question with its `sample_id` (the SHA-256 of the prompt, reproducible from the data builders), the character spans and the score of each segment.

## 3. Two ways to use them

All arms train the same LoRA (r = 16) on the same 9,484 questions for two epochs with the region-weighted rationale loss (0.2 on rationale tokens, 0.8 on answer tokens), same seed. Segment ends are aligned to tokens exactly through the tokenizer's offset mapping.

- **control**: the rationale LM loss only.
- **predict**: the LM loss plus an auxiliary linear head on the hidden state of each segment's last token, trained to regress the teacher score (MSE, weight 1). The head's gradient flows into the backbone, so this arm learns to *judge* segments.
- **weighted**: no head. The score of a segment becomes the loss weight of its tokens, normalised within the rationale; decisive segments are learned five times as hard as boilerplate, and the rationale's total weight is unchanged.

## 4. Results

Rationale-then-answer accuracy on the CMExam validation split (6,657 questions), paired against the control arm:

| Arm | Accuracy | vs control | Head on held-out segments (Spearman / top-segment hit rate) |
|---|---:|---|---|
| control | 76.27 | | −0.02 / 21% (untrained head, chance level) |
| predict | 73.74 | **−2.52**, CI −3.49 to −1.55, *p* = 4 × 10⁻⁷ | **0.57 / 64%** |
| weighted | **77.50** | **+1.23**, CI +0.53 to +1.94, *p* = 8 × 10⁻⁴ | −0.02 / 21% |

The auxiliary head learns what it is asked to learn: on 500 held-out questions its Spearman correlation with the teacher is 0.57 and it picks the most decisive segment 64% of the time against a 21% chance rate. Yet the model that learned to judge segments answers worse, on every slice, and its LM loss on held-out rationales is higher than the control's. Judging which segment is decisive and generating one do not share a representation, and at weight 1 the auxiliary objective competes with generation.

The same scores placed in the generation loss help: +1.23 overall, +2.6 on negation questions (*p* = 0.046). The gain is specific to rationale-mode answering. Under the direct-answer prompt the weighted arm is +0.18 (free generation) and +0.20 (constrained letter scoring) over the control, neither significant, so segment weighting changed how the model reasons in writing rather than what it knows; the predict arm is worse in direct mode too (−0.85 on constrained scoring, *p* = 8 × 10⁻⁵), so its damage reaches the knowledge representation.

## 5. Reading

- Segment-level information is useful only when it acts on the tokens being generated. Turning it into a prediction target makes the model a critic of rationales, not a better writer of them.
- The weighted arm is the initial policy for the GRPO experiments in `docs/05_segment_credit_grpo.md`, where the per-segment signal comes from the policy itself rather than from a teacher.
- One seed per arm; the +1.23 sits above the 1.0-point minimum detectable difference at the observed discordance but should be replicated before being relied on.
