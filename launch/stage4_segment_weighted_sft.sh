#!/usr/bin/env bash
# Stage 4: segment-level supervision for rationale SFT (docs/06_segment_weighted_sft.md).
# Three arms on the same 9,484 scored questions, LoRA r=16, 2 epochs, seed 42; only the loss differs:
#   control   rationale LM loss only
#   predict   + auxiliary head regressing the teacher's segment score (weight 1)
#   weighted  segment scores as per-token loss weights (the RL starting point)
# Each arm trains in about 1 hour on one RTX 5090 and is then evaluated in rationale mode on the CMExam validation split.
# Inputs: data/segment_scores/sft/{train,dev}.jsonl (see data/README.md, "Segment scores").
set -euo pipefail
source "$(dirname "$0")/common.sh"
SFT=$DATA/segment_scores/sft
ARMS=${ARMS:-"control:0: predict:1.0: weighted:0:score"}
for arm in $ARMS; do
  IFS=: read -r name weight weighting <<< "$arm"
  extra=()
  [ -n "${weighting:-}" ] && extra=(--sentence-weighting "$weighting")
  $PY -u scripts/train_sft_score_head.py --model "$BASE" --train-file "$SFT/train.jsonl" --dev-file "$SFT/dev.jsonl" \
    --output-dir "$OUT/stage4_$name" --score-loss-weight "$weight" "${extra[@]}" \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --learning-rate 1e-5 --head-learning-rate 1e-3 --warmup-ratio 0.05 \
    --num-train-epochs 2 --per-device-train-batch-size 8 --gradient-accumulation-steps 4 --seed 42 2>&1 | tee "logs/stage4_$name.log"
  # rationale-mode accuracy on the validation split, rendered with the prompt the rationale data was written in
  $PY -u scripts/eval_validation.py --model "$BASE" --adapter "$OUT/stage4_$name/final_adapter" \
    --input-file "$DATA/cmexam_sft_validation.jsonl" --output-dir "$OUT/stage4_$name/eval_cmexam_val" \
    --mode cot --prompt-style legacy_explain --limit 0 --batch-size 64 --max-new-tokens 768 \
    --skip-sampling --skip-shuffle --seed 42 --overwrite 2>&1 | tee "logs/stage4_${name}_eval.log"
done
# paired comparison, e.g. weighted vs control:
#   python scripts/compare_headroom_records.py --a $OUT/stage4_control/eval_cmexam_val/headroom_records.jsonl \
#     --b $OUT/stage4_weighted/eval_cmexam_val/headroom_records.jsonl --label-a control --label-b weighted
