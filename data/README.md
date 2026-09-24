# Data

No dataset is redistributed here. Both sources are public; download them and keep their licences.

| Dataset | Source | Used for |
|---|---|---|
| CMExam | https://github.com/williamliujl/CMExam (`data/train.csv`, `val.csv`, `test_with_annotations.csv`) | Stage 1 and 2 training; its val and test splits are decontamination references |
| CMB-Exam | https://github.com/FreedomIntelligence/CMB (`data/CMB.zip`, `data/CMB-test-choice-answer.json`) | Stage 3 training split, the held-out validation split, and the final test set |

Expected layout after download:

```
data/
  CMExam/data/{train,val,test_with_annotations}.csv
  CMB/CMB-Exam/CMB-train/CMB-train-merge.json
  CMB/CMB-Exam/CMB-val/CMB-val-merge.json
  CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json
  CMB/CMB-test-choice-answer.json
```

Build the training files:

```bash
# Stage 1: CMExam -> chat-format direct-answer SFT, then drop stems that also occur in CMB-test
python scripts/build_cmexam_direct_sft.py --input-dir data/CMExam/data --output-dir data
python scripts/decontaminate_cmexam_against_cmb.py \
  --cmb-test-file data/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json \
  --train-file data/cmexam_sft_train.jsonl --validation-file data/cmexam_sft_validation.jsonl

# expected: 52,369 train rows (sha256 e15ff4a5c7d6...050a830) and 6,657 validation rows (sha256 95da75fcbe70...2950aaf)

# Stage 2: one shuffled copy per shuffle-safe question (52,369 -> 101,984 rows)
python scripts/build_shuffle_aug.py --source-file data/cmexam_sft_train.jsonl \
  --output-file data/cmexam_sft_train_shuffle_aug.jsonl --copies 1 --seed 42

# Stage 3: CMB-train with three-layer decontamination, merged with the Stage 1 file (-> 293,973 rows)
python scripts/build_cmb_train_sft.py \
  --cmb-train data/CMB/CMB-Exam/CMB-train/CMB-train-merge.json \
  --cmb-test data/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json \
  --cmb-val data/CMB/CMB-Exam/CMB-val/CMB-val-merge.json \
  --cmexam-csv data/CMExam/data/test_with_annotations.csv --cmexam-csv data/CMExam/data/val.csv \
  --merge-direct-file data/cmexam_sft_train.jsonl --output-dir data/cmb_sft \
  --internal-val-size 3000 --near-threshold 0.7 --borderline-threshold 0.5 \
  --multi-shuffle-copies 1 --merge-multi-shuffle-copies 2 --seed 42
```

`data/cmb_sft/` then contains the training file, the 3,000-question validation split, per-row metadata,
`cmb_sft_report.json` with the full funnel, and two audit files: every removed row with its reason, and every
borderline row (Jaccard 0.5–0.7) that was kept.

The test set is touched by exactly one script, `scripts/eval_cmb.py`, and only for final evaluation.

## Rationale-style data (Stage 4 and 5)

The segment-supervision and GRPO experiments use CMExam questions *with* their official rationales, rendered as
`解析：<rationale>\n答案：<letters>` under the prompt in `cmexam_prompts.LEGACY_EXPLAIN_SYSTEM_PROMPT`:

```bash
python scripts/build_cmexam_rationale_sft.py --input-dir data/CMExam/data --output-dir data/cmexam_rationale
# rewrites both files in place, dropping stems that also occur in CMB-test (45,030 -> 44,669 training rows)
python scripts/decontaminate_cmexam_against_cmb.py \
  --cmb-test-file data/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json \
  --train-file data/cmexam_rationale/cmexam_rationale_sft_train.jsonl \
  --validation-file data/cmexam_rationale/cmexam_rationale_sft_validation.jsonl \
  --report-file data/cmexam_rationale/cmexam_rationale_decontamination_report.json
```

### Segment scores

`data/segment_scores/cmexam_rationale_segment_scores.jsonl.gz` holds the released annotation: for 9,984 training
questions, the character spans of each clause-level segment of the official rationale and its 1–5 decisiveness
score (DeepSeek V4 Pro, `configs/synth_harness_deepseek.json`, task `sentence_score`). No question text is
included; rows carry the `sample_id` = SHA-256 of the prompt's user content, so they join back onto the rows you
build above. `split` marks the 9,484 training / 500 held-out questions used in the paper-style comparison.

To re-create the scored SFT rows from the released scores:

```bash
python - <<'EOF_'
import gzip, hashlib, json
scores = {json.loads(l)["sample_id"]: json.loads(l) for l in gzip.open("data/segment_scores/cmexam_rationale_segment_scores.jsonl.gz", "rt", encoding="utf-8")}
out = {"train": open("data/segment_scores/sft/train.jsonl", "w", encoding="utf-8"), "dev": open("data/segment_scores/sft/dev.jsonl", "w", encoding="utf-8")}
for line in open("data/cmexam_rationale/cmexam_rationale_sft_train.jsonl", encoding="utf-8"):
    row = json.loads(line)
    sid = hashlib.sha256("\n".join(m["content"] for m in row["messages"] if m["role"] == "user").encode("utf-8")).hexdigest()
    if sid not in scores:
        continue
    content = row["messages"][-1]["content"]
    explanation = content[len("解析："):content.rindex("\n答案：")]
    assert len(explanation) == scores[sid]["rationale_chars"], sid
    out[scores[sid]["split"]].write(json.dumps({"sample_id": sid, "task": "sentence_score", "messages": row["messages"], "explanation": explanation,
        "gold": content.rsplit("答案：", 1)[-1].strip(), "sentences": scores[sid]["segments"]}, ensure_ascii=False) + "\n")
EOF_
```

(`mkdir -p data/segment_scores/sft` first.) To score new rationales instead, run the harness with your own API key
in the environment variable named in the config; it never sends an evaluation-set stem:

```bash
python scripts/synth_harness/run_harness.py --config configs/synth_harness_deepseek.json \
  --input data/cmexam_rationale/cmexam_rationale_sft_train.jsonl \
  --exclude-stems data/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json data/CMB/CMB-Exam/CMB-val/CMB-val-merge.json \
                  data/CMExam/data/test_with_annotations.csv data/CMExam/data/val.csv \
  --tasks sentence_score --min-sentences 3 --max-sentences 40 --limit 10000 --output-dir data/segment_scores/run --dry-run
python scripts/build_sentence_score_sft.py --input data/segment_scores/run/data_sentence_score.jsonl --output-dir data/segment_scores/sft --dev-size 500
```

### RL prompt pool

4,000 questions from the decontaminated rationale file that were **not** used in the scored SFT (sample-id
disjoint), with user content of at most 450 characters; 256 validation prompts for TRL's periodic evaluation:

```bash
python scripts/build_cot_rl_pool.py --input data/cmexam_rationale/cmexam_rationale_sft_train.jsonl \
  --exclude data/segment_scores/sft/train.jsonl data/segment_scores/sft/dev.jsonl --limit 4000 --max-user-chars 450 \
  --output data/rl_cot_pool/train.jsonl
python scripts/build_cot_rl_pool.py --input data/cmexam_rationale/cmexam_rationale_sft_validation.jsonl \
  --limit 256 --seed 7 --output data/rl_cot_pool/validation.jsonl
```
