#!/usr/bin/env bash
# Stage 3: CMExam + decontaminated CMB-train (293,973 rows). LoRA r=64, effective batch 32, about 6.5 hours on one RTX 5090.
# Build data/cmb_sft first (see data/README.md).
set -euo pipefail
source "$(dirname "$0")/common.sh"
train_sft stage3_cmb_train "$DATA/cmb_sft/cmb_sft_train.jsonl" "$DATA/cmb_sft/cmb_internal_val.jsonl" 16 2 64 128
