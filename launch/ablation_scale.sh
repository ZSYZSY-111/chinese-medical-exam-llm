#!/usr/bin/env bash
# Data-scale ablation: retrain from the base model on nested, stratified subsets of CMB-train (0 / 25 / 50 %), identical hyper-parameters,
# then score each on the held-out validation set. The 100 % point is the Stage 3 model.
set -euo pipefail
source "$(dirname "$0")/common.sh"
$PY scripts/build_ablation_sets.py --train-file "$DATA/cmb_sft/cmb_sft_train.jsonl" --metadata "$DATA/cmb_sft/cmb_sft_metadata.jsonl" \
  --output-dir "$DATA/ablation" --scales 0,0.25,0.5 --dose-copies "" --seed 42
for name in scale_000 scale_025 scale_050; do
  train_sft "ablation_$name" "$DATA/ablation/$name/train.jsonl" "$DATA/cmb_sft/cmb_internal_val.jsonl" 16 2 64 128
  $PY -u scripts/eval_validation.py --model "$BASE" --adapter "$OUT/ablation_$name/final_adapter" --input-file "$DATA/cmb_sft/cmb_internal_val.jsonl" \
    --output-dir "$OUT/eval_$name" --mode direct --limit 0 --batch-size 32 --skip-sampling --skip-shuffle --skip-constrained --seed 42 --overwrite
done
