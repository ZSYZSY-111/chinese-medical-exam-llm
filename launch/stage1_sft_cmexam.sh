#!/usr/bin/env bash
# Stage 1: direct-answer SFT on decontaminated CMExam (52,369 rows). LoRA r=16, effective batch 32, about 1 hour on one RTX 5090.
set -euo pipefail
source "$(dirname "$0")/common.sh"
train_sft stage1_sft_cmexam "$DATA/cmexam_sft_train.jsonl" "$DATA/cmexam_sft_validation.jsonl" 32 1 16 32
