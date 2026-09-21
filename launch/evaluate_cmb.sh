#!/usr/bin/env bash
# Evaluate one adapter on the full CMB-Exam test set with the official zero-shot prompt (greedy, exact letter-set match).
#   bash launch/evaluate_cmb.sh <adapter-dir|none> <name> [reference-predictions.json]
# With a reference file the script also writes a paired comparison with intervals and an exact McNemar test.
# Run this once per final model: selection belongs on the validation set (scripts/eval_validation.py).
set -euo pipefail
source "$(dirname "$0")/common.sh"
ADAPTER=$1; NAME=$2; REF=${3:-}
Q=$DATA/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json
A=$DATA/CMB/CMB-test-choice-answer.json
mkdir -p benchmark_outputs
ARGS=(--base-model "$BASE" --questions "$Q" --output "benchmark_outputs/${NAME}.json" --batch-size 32 --max-length 2048 --max-new-tokens 12)
[ "$ADAPTER" != "none" ] && ARGS+=(--adapter "$ADAPTER")
$PY -u scripts/eval_cmb.py "${ARGS[@]}"
if [ -n "$REF" ]; then
  $PY scripts/score_cmb.py --questions "$Q" --answers "$A" --base "$REF" --sft "benchmark_outputs/${NAME}.json" --out-dir "benchmark_outputs/compare_${NAME}"
  $PY scripts/score_predictions.py "benchmark_outputs/compare_${NAME}/question_level_results.jsonl" --correct-field sft_correct \
    --compare-correct-field base_correct --slices exam_type,question_type --output "benchmark_outputs/compare_${NAME}/score_with_ci.json"
fi
