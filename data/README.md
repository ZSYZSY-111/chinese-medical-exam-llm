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
