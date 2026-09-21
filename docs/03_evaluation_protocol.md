# 3 · Evaluation protocol

## Rules

1. **The test set is used once per final model.** All selection — checkpoints, averaging weights, data recipes — happens on held-out validation data (3,000 CMB-train questions that never enter training, plus 6,657 CMExam validation questions).
2. **One fixed measurement.** `eval_cmb.py` uses the official CMB zero-shot prompt, greedy decoding, at most 12 new tokens, and counts an answer as correct only if the predicted letter set equals the reference set.
3. **Every reported difference is paired.** Two models are compared on the same questions with an exact McNemar test and a bootstrap interval of the difference (`score_cmb.py`, `score_predictions.py`).

## Uncertainty

`score_predictions.py` reports a Wilson 95% interval for each accuracy and a paired bootstrap interval (10,000 resamples of the four-cell agreement table) for each difference. On 11,200 questions a single accuracy near 84% has a half-width of 0.68 points.

The smallest difference a paired comparison can detect with 80% power at α = 0.05 depends on how often the two models disagree. With *n* questions and a disagreement rate *d*,

MDE ≈ (1.96 + 0.84) · √(n·d) / n.

| *n* | disagreement | MDE (points) |
|---:|---:|---:|
| 11,200 | 2% (an averaged model vs its component) | 0.37 |
| 11,200 | 6.4% (two independent training runs) | 0.67 |
| 3,000 | 5% | 1.13 |

Differences below these values are not reported as improvements anywhere in this repository.

## How noisy is the evaluation itself

| Measurement | Result |
|---|---|
| same model, same settings, run twice | 0 of 3,000 answers change |
| batch size 32 vs 8 (bf16, padding changes the arithmetic) | 20 of 3,000 answers change (0.67%); accuracy differs by 0.07, CI −0.17 to +0.33 |
| training prompt vs official prompt, same model | 216 of 3,000 answers change; accuracy +1.43, CI +0.63 to +2.23 |

The third row is why validation results are reported under the official prompt: `make_internal_val_cmb_json.py` restores the exam metadata of the validation questions so that `eval_cmb.py` can score them exactly as it scores the test set.
