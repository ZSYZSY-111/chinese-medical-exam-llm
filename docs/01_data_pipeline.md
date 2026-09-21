# 1 · Data pipeline

## Record format

Every training row is a three-message chat record whose target is only the answer letters:

```json
{"messages": [
  {"role": "system", "content": "你是一名医学考试答题助手。回答中国医学考试选择题。不要解释，直接输出正确选项字母；多选题按字母顺序连续输出，例如ACD。"},
  {"role": "user", "content": "题目：\n…\n\n选项：\nA. …\nB. …\n\n请直接输出正确选项字母；多选题按字母顺序连续输出。"},
  {"role": "assistant", "content": "AC"}]}
```

`sample_id` is the SHA-256 of the user content. Every tool in the repository (validation, ablation subsets, audits) joins rows on it, so a row keeps its identity across files. `scripts/cmexam_prompts.py` renders and parses this template and round-trips exactly, including an optional one-line hint slot.

## Stage 1 · CMExam

`build_cmexam_direct_sft.py` converts the official CMExam CSV splits. A row is rejected when its options are not a contiguous `A..` sequence, when two options have the same text, or when the answer is not a set of existing option letters. Questions are keyed by NFKC-normalised, whitespace-free text; a key is written once, and train questions that also occur in val or test are dropped so the splits stay disjoint. `decontaminate_cmexam_against_cmb.py` then removes every row whose key occurs in CMB-test.

| | train | validation |
|---|---:|---:|
| rows read | 54,497 | 6,811 |
| duplicate question | −1,041 | −16 |
| also in a later split | −589 | −58 |
| also in CMB-test | −432 | −70 |
| malformed options or answer | −66 | −10 |
| **written** | **52,369** | **6,657** |

The builder is deterministic: the output files have SHA-256 `e15ff4a5…050a830` (train) and `95da75fc…2950aaf` (validation).

## Stage 2 · Option-shuffle augmentation

`build_shuffle_aug.py` adds one copy of each question with the options permuted and the answer letters remapped. The permutation is a Fisher–Yates shuffle driven by a stable hash of `(seed, sample_id, copy_index)`, never the identity, so the output is reproducible byte for byte. Questions whose options refer to each other or to their order (以上都是, A和B, 均不是 …) are *shuffle-unsafe* and kept once: 2,754 of 52,369. Result: 101,984 rows.

## Stage 3 · CMB-train with three-layer decontamination

`build_cmb_train_sft.py` checks every candidate row against 24,699 reference stems from CMB-test, CMB-val and the CMExam validation and test splits:

1. **Exact stem** — NFKC, lower-case, all whitespace and punctuation removed.
2. **Stem + option set** — SHA-256 of the normalised stem and the sorted normalised option texts; also used to deduplicate inside CMB-train.
3. **Near duplicate** — MinHash-LSH over character 5-grams (128 permutations), followed by the exact Jaccard similarity of the candidate pairs. Rows at or above 0.7 are removed; rows between 0.5 and 0.7 are kept and written to a borderline audit file.

Two decisions deserve a note. CMB stores four-option questions with an empty option `E`, and the evaluator renders it as `E. `; trailing empty options are therefore kept and rendered identically instead of rejected, which preserves 16,478 training questions. And 3,000 questions are held out by stable hash as the validation set that every later decision uses.

Outputs: the training file, the validation split, per-row metadata (`source`, `variant`, `exam_type`, `question_type`), a JSON report with the full funnel, `removed_audit.jsonl` and `borderline_audit.jsonl`. The final training file has 293,973 rows: 226,923 CMB questions, 15,971 shuffled copies of multi-answer CMB questions, and the 51,079-row CMExam component.

## Leakage check on the result

Removing from the test set the 250 questions that fall in the 0.5–0.7 similarity band to any training question changes the Stage 3 gain over Stage 1 from +1.96 to +1.88.
