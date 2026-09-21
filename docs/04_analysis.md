# 4 · Analysis

## Where the gain comes from

| Test accuracy | zero-shot | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|---:|
| single-answer (9,999) | 82.66 | 87.01 | 87.01 | 87.03 |
| multi-answer (1,190) | 46.39 | 43.11 | 47.90 | 61.51 |
| overall (11,200) | 78.79 | 82.34 | 82.85 | 84.30 |

Stage 1 buys the single-answer improvement and slightly hurts multi-answer questions: only 3.2% of CMExam answers have more than one letter, so the model learns that an answer is one letter, and under exact set matching a one-letter reply to a multi-answer question is always wrong. Stage 2 recovers part of that through shuffling and averaging. Stage 3 adds 23,614 in-domain multi-answer questions and moves multi-answer accuracy by 18 points, which accounts for almost exactly the overall gain (10.6% of the test set × 18.4 = 1.95). By exam family, the gains follow the multi-answer share: postgraduate entrance +9.3, specialty knowledge +2.4, the rest within noise.

## Data-scale ablation

Nested, stratified subsets of CMB-train; every point is a full retrain with the Stage 3 hyper-parameters, scored on the 3,000-question validation set.

| Share of CMB-train | overall, official prompt | single-answer | multi-answer |
|---:|---:|---:|---:|
| 0% | 81.03 | 85.36 | 30.79 |
| 25% | 82.40 | 84.14 | 48.68 |
| 50% | 83.03 | 85.10 | 49.01 |
| 100% | 83.40 | 85.17 | 53.31 |

(The last two columns use the training prompt.) The multi-answer decision rule is mostly learned from the first quarter of the data; single-answer accuracy does not respond to data volume at all. The model also scores only 85% on questions it was trained on, so the remaining single-answer errors are not a data-coverage problem at this adapter size and training length.

## The prompt-format effect

The training prompt does not say whether a question has one or several answers; the official evaluation prompt does. Counting outputs that cannot possibly be right:

| Condition | single-answer question, several letters | multi-answer question, one letter |
|---|---:|---:|
| CMExam only, training prompt | 0.8% | 44.0% |
| + CMB-train, training prompt | 2.3% | 11.9% |
| + CMB-train, official prompt | 0% | 5.0% |
| question type stated during training | 0% | 0% |

Stating the type during training (`cmexam_prompts.type_hint_for`, `build_ablation_sets.py --extra-variant`) removes the problem and is significantly better on validation under that prompt (+0.97, *p* = 0.02). Evaluated once on the test set with the official prompt, which already carries the type, it scores 84.33 against 84.30 (+0.03, CI −0.35 to +0.41, *p* = 0.93). A variant trained directly on the official template scored 82.93 on validation against 83.40, also within noise. Format alignment is therefore reported as an evaluation finding rather than a training stage: it explains a 1.4-point gap between two ways of measuring the same model, and it confirms that the recipe has plateaued at 84.3.
