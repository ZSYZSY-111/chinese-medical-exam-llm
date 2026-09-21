#!/usr/bin/env bash
# Stage 2: train on the option-shuffled set (101,984 rows, LoRA r=64), then average it with the Stage 1 adapter in weight space.
set -euo pipefail
source "$(dirname "$0")/common.sh"
$PY scripts/build_shuffle_aug.py --source-file "$DATA/cmexam_sft_train.jsonl" --output-file "$DATA/cmexam_sft_train_shuffle_aug.jsonl" --copies 1 --seed 42
train_sft stage2_shuffle_aug_r64 "$DATA/cmexam_sft_train_shuffle_aug.jsonl" "$DATA/cmexam_sft_validation.jsonl" 16 2 64 128
# The 0.5 / 0.5 weights were chosen on the validation set; the merge runs on CPU and checks itself numerically.
$PY scripts/merge_lora_soup.py --base "$BASE" \
  --component "$OUT/stage1_sft_cmexam/final_adapter:0.5" --component "$OUT/stage2_shuffle_aug_r64/final_adapter:0.5" \
  --out "$OUT/stage2_lora_soup"
