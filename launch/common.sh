# Shared settings. Override any of these from the environment.
export PY=${PY:-python}
export BASE=${BASE:-Qwen/Qwen2.5-7B-Instruct}
export DATA=${DATA:-data}
export OUT=${OUT:-training_outputs}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Tokenised datasets are cached per run; keep the cache off a small system disk.
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PWD/hf_cache}
mkdir -p "$OUT" logs "$HF_DATASETS_CACHE"

train_sft() {  # train_sft <run-name> <train-file> <validation-file> <batch> <grad-accum> <lora-r> <lora-alpha>
  $PY -u scripts/train_sft.py --model "$BASE" --train-file "$2" --validation-file "$3" --output-dir "$OUT/$1" \
    --max-eval-samples 3000 --max-length 768 --num-train-epochs 1 --learning-rate 2e-5 --warmup-ratio 0.03 --weight-decay 0.01 \
    --per-device-train-batch-size "$4" --per-device-eval-batch-size 32 --gradient-accumulation-steps "$5" \
    --logging-steps 20 --eval-steps 1000 --save-steps 1000 --save-total-limit 1 \
    --lora-r "$6" --lora-alpha "$7" --lora-dropout 0.05 --gradient-checkpointing \
    --dataset-num-proc 8 --dataloader-num-workers 4 --seed 42 --report-to none --loss-mode completion 2>&1 | tee "logs/$1.log"
}
