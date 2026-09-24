# Segment-credit GRPO: credit assignment inside a rationale from the policy's own confidence

*Code: `scripts/grpo_segment_credit.py`, `scripts/train_grpo.py --segment-credit-lambda`. Numbers: `results/segment_credit_grpo.json`.*

## 1. The problem

When a model writes a rationale and then an answer, and only the answer can be verified, GRPO gives every token of a completion the same advantage: the group-normalised correctness of the final letters. Two things go wrong at once.

- **No credit assignment.** A rationale of 150 characters that reaches the right answer is rewarded as a whole, including the sentence that nearly led it astray; a rationale that reaches the wrong answer is penalised as a whole, including the sentence that was right.
- **Unanimous groups carry nothing.** With 8 samples per question, a group that is all-right or all-wrong has zero advantage for every token. On this task, standard GRPO reaches 60–80% unanimous groups within 50 steps, so most sampled tokens stop contributing a gradient.

The starting model makes the first problem visible before any RL: on questions the direct-answer model gets right, writing a rationale first *lowers* accuracy. On 1,500 held-out questions, 173 are answered correctly directly but wrongly after a rationale, against 45 the other way round. Rationales lead the model astray more often than they help, and an outcome reward cannot say which sentence did it.

## 2. The idea

Multiple-choice answers are enumerable, so the policy's belief in the gold answer can be read exactly at any point of its own rationale. After segment *k* of a sampled rationale, append the answer marker `\n答案：` and take the next-token distribution over the valid option letters. Its probability on the gold letter, normalised over the letters, is

Φ<sub>k</sub> = *p*<sub>θ</sub>(gold letter | prompt, rationale segments 1..k, "答案："),

with Φ<sub>0</sub> the belief before any rationale. Segment *k* earns the change it caused:

c<sub>k</sub> = Φ<sub>k</sub> − Φ<sub>k−1</sub>.

Every token inside segment *k* receives the advantage

A<sub>token</sub> = A<sub>sequence</sub> + λ · c<sub>k</sub>,

where A<sub>sequence</sub> is the ordinary group-normalised outcome advantage and λ = 2 by default (a +0.5 change in confidence is worth +1, the scale of the outcome advantage under `scale_rewards="none"`). Tokens of the answer part and of the "解析：" prefix keep A<sub>sequence</sub> only.

Three properties follow directly from the definition.

1. **The credits telescope.** Σ<sub>k</sub> c<sub>k</sub> = Φ<sub>K</sub> − Φ<sub>0</sub>, the net confidence the whole rationale added. Padding earns nothing, restating the question earns nothing, and a segment that lowers the belief in the right answer is charged for it. This is the potential-based shaping structure of Ng, Harada and Russell (1999), applied inside a sequence.
2. **Unanimous groups still learn.** A<sub>sequence</sub> is zero for an all-right group, but c<sub>k</sub> is not: the segments that raised the belief are reinforced and those that lowered it are suppressed, within a completion that the outcome reward could not distinguish from its siblings. In training, the share of tokens with a non-zero advantage stays at 86–100% throughout, against 20–80% for standard GRPO.
3. **Nothing external is needed.** The probe is the policy being trained, at its current parameters. There is no teacher, no reward model, no process labels, and no extra generation; the cost is one forward pass per segment boundary with `logits_to_keep=1`, about 14 s per step on the 7B model.

## 3. What has to be exact

- **Segment boundaries in token space.** Completions are decoded, split at clause level by `sentence_split.py` (commas and sentence-final punctuation; never inside brackets or quotes), and each character span is mapped back to token indices with a binary search over prefix decode lengths, which are non-decreasing. A completion is capped at 8 probes; longer rationales have neighbouring segments merged evenly.
- **The probe reads the same model.** Probes run under `no_grad` in eval mode with left padding, explicit position ids and `logits_to_keep=1`, then the model is returned to training mode. The gold letter is the first letter of the answer set; probabilities are normalised over the question's valid letters, and the letter mass (0.97–1.00 in practice) is logged as a sanity check.
- **Token-level advantages go through TRL unchanged.** TRL 1.8's `_compute_loss` accepts advantages of shape (B, T), so the subclass only replaces `advantages` in the output of `_generate_and_score_completions`; clipping, loss aggregation and masking are the library's own.

## 4. The comparison

Both arms start from the same adapter (the segment-weighted SFT model, `docs/06_segment_weighted_sft.md`), see the same 4,000-question prompt pool in the same order, draw 8 samples per prompt for 16 prompts per step, and train for 150 steps with identical hyper-parameters and seed. The pool contains no question used in the segment-scored SFT and no evaluation question. The only difference is λ: 0 for standard GRPO, 2 for segment credit.

| Arm | Accuracy | 95% CI | Paired difference | Output length (mean / median chars) | Answer letter within first 20 chars |
|---|---:|---|---|---:|---:|
| RL start (segment-weighted SFT) | 77.50 | 76.48 – 78.48 | | 151 / 123 | 34.8% |
| Standard GRPO | 82.71 | 81.78 – 83.60 | +5.21 vs start | 59 / 37 | 16.5% |
| **Segment-credit GRPO** | **83.73** | 82.83 – 84.60 | **+1.02 vs standard**, CI +0.59 to +1.47, McNemar *p* = 8 × 10⁻⁶ | 50 / 36 | 12.9% |

The 2×2 table between the two GRPO arms is 5,426 both right / 80 only standard / 148 only segment credit / 1,003 both wrong. The discordance is 3.4%, which puts the minimum detectable difference at 0.64 points; the observed +1.02 clears it. Per slice, negation questions gain +2.6 (*p* = 0.003) and case questions +1.3 (*p* = 0.01); calculation, long-stem and multi-answer questions move in the same direction without reaching significance. Against the direct-answer SFT model on the 1,500 shared questions, the rationale-mode gap shrinks from −8.5 (start) to −4.3 (standard GRPO) to −3.3 (segment credit).

## 5. What the training dynamics say

| Last 20 steps, mean | Standard GRPO | Segment credit |
|---|---:|---:|
| Training reward | 0.837 | 0.862 |
| Rationale length (tokens) | 54 | 46 |
| Unanimous groups | 72% | 79% |
| Entropy | 0.79 | 0.40 |
| Tokens with non-zero advantage | 20–60% | ~90% |

Both arms learn to write less: rationales shrink to a third of their initial length, which is by itself the main reason accuracy rises, because a shorter rationale has fewer chances to mislead. Segment credit does not prevent this, and it sharpens faster (lower entropy). What it changes is what survives inside the short rationale: the arm with segment credit fixes 589 questions the start got wrong and breaks 174, against 527 and 180 for standard GRPO. The share of rationales that state an answer letter in their first 20 characters is lower with segment credit (12.9% vs 16.5%), so the gain does not come from announcing the answer early to inflate Φ.

Early in training the probe shows Φ<sub>K</sub> below Φ<sub>0</sub> on average: writing the rationale lowers the belief in the right answer. By the end, the two are equal, which is what the credit pushes towards.

## 6. Limits and open questions

- One seed per arm. The paired test bounds evaluation noise, not training noise.
- λ was set once (2) and not tuned; the cap of 8 probes per completion and the clause-level split were fixed a priori.
- The comparison isolates segment credit from standard GRPO but not from a sequence-level shaping baseline (reward + λ(Φ<sub>K</sub> − Φ<sub>0</sub>) with no per-segment split). That ablation is the natural next experiment, together with a shuffled-credit control.
- The initial adapter was trained with teacher-scored segment weights (`docs/06`); re-running both arms from the control SFT adapter would remove that dependency. The RL procedure itself uses no external model.
- Both arms are still 3–4 points below the direct-answer model on this benchmark. Segment credit is a claim about how to use a sparse reward, not a claim that rationales beat direct answers here.

## 7. Engineering notes

- TRL 1.8 passes all completions of a step (prompts × samples) to a single `generate` call; `GRPOConfig.generation_batch_size` does not bound that call. `train_grpo.py` generates in chunks of `--generation-chunk` completions and, if a chunk still runs out of memory, releases the exception before freeing the cache and retries in halves. This took the 7B peak from 22–29 GB to 19–20 GB and removed the sporadic out-of-memory failures that had stopped three earlier attempts.
- Memory is logged per step from a `log` override, because a `TrainerCallback.on_log` fires after the entry has already been written to `log_history`.
- Checkpoints every 10 steps and an automatic resume from the latest checkpoint make a 3-hour run survive an interruption at the cost of at most 10 steps.
