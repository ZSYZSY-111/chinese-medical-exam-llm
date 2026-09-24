#!/usr/bin/env bash
# Stage 5: GRPO with segment credit versus standard GRPO (docs/05_segment_credit_grpo.md).
# Both arms start from the Stage 4 "weighted" adapter, use the same 4,000-question pool, 16 prompts x 8 samples per
# step, 150 steps, seed 42; the only difference is --segment-credit-lambda (0 = standard GRPO, 2 = segment credit).
# On one RTX 5090: standard GRPO ~2 h, segment credit ~3 h (probes add ~14 s per step), evaluation ~20 min each.
# Checkpoints every 10 steps; a failed arm resumes from its latest checkpoint (up to 6 attempts).
# Inputs: data/rl_cot_pool/{train,validation}.jsonl (see data/README.md, "RL prompt pool").
set -euo pipefail
source "$(dirname "$0")/common.sh"
START=${START:-$OUT/stage4_weighted/final_adapter}
POOL=$DATA/rl_cot_pool
STEPS=${STEPS:-150}
common=(--model "$BASE" --adapter "$START" --train-file "$POOL/train.jsonl" --validation-file "$POOL/validation.jsonl"
        --reward-mode cot --prompts-per-step 16 --num-generations 8 --micro-batch 4 --generation-chunk 32
        --max-completion-length 384 --temperature 1.0 --learning-rate 1e-5 --scale-rewards none --loss-type dapo
        --epsilon 0.2 --epsilon-high 0.28 --eval-steps 50 --save-steps 10 --save-total-limit 3 --max-eval-samples 64
        --seed 42 --log-completions)

train_arm() {  # train_arm <name> <lambda>
  local name=$1 lam=$2 attempt latest resume
  for attempt in 1 2 3 4 5 6; do
    resume=()
    latest=$(ls -d "$OUT/stage5_$name"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    [ -n "$latest" ] && resume=(--resume-from-checkpoint "$latest")
    if $PY -u scripts/train_grpo.py "${common[@]}" --output-dir "$OUT/stage5_$name" --max-steps "$STEPS" \
        --segment-credit-lambda "$lam" "${resume[@]}" > "logs/stage5_${name}_try$attempt.log" 2>&1; then
      return 0
    fi
    echo "attempt $attempt of $name failed; resuming from ${latest:-scratch}" >&2
  done
  return 1
}

evaluate_arm() {  # evaluate_arm <name>
  $PY -u scripts/eval_validation.py --model "$BASE" --adapter "$OUT/stage5_$1/final_adapter" \
    --input-file "$DATA/cmexam_sft_validation.jsonl" --output-dir "$OUT/stage5_$1/eval_cmexam_val" \
    --mode cot --prompt-style legacy_explain --limit 0 --batch-size 64 --max-new-tokens 768 \
    --skip-sampling --skip-shuffle --seed 42 --overwrite 2>&1 | tee "logs/stage5_$1_eval.log"
}

train_arm standard 0 && evaluate_arm standard
train_arm segment_credit 2.0 && evaluate_arm segment_credit
$PY scripts/compare_headroom_records.py --a "$OUT/stage5_standard/eval_cmexam_val/headroom_records.jsonl" \
  --b "$OUT/stage5_segment_credit/eval_cmexam_val/headroom_records.jsonl" --label-a "standard GRPO" --label-b "segment credit" \
  --json "$OUT/stage5_standard_vs_segment_credit.json"
