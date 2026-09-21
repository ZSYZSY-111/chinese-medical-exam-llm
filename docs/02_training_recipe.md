# 2 · Training recipe

All runs fine-tune `Qwen/Qwen2.5-7B-Instruct` in bf16 with LoRA on all linear projections, one epoch, completion-only loss (only the answer tokens contribute), AdamW, cosine schedule with 3% warm-up, weight decay 0.01, learning rate 2 × 10⁻⁵, maximum length 768, gradient checkpointing, seed 42, on one RTX 5090 (32 GB). `scripts/train_sft.py` writes the full argument set, resolved sample counts and library versions to `run_config.json`; the recorded files are in [`configs/`](../configs).

| Run | Rows | LoRA r / α | Micro-batch × accumulation | Time |
|---|---:|---|---|---|
| Stage 1 · CMExam | 52,369 | 16 / 32 | 32 × 1 | ≈ 1 h |
| Stage 2 · shuffle-augmented | 101,984 | 64 / 128 | 16 × 2 | ≈ 2 h |
| Stage 3 · + CMB-train | 293,973 | 64 / 128 | 16 × 2 | ≈ 6.5 h |

## LoRA weight averaging

Two adapters trained from the same base on overlapping data sit in the same loss basin, so their updates can be averaged: ΔW = Σᵢ wᵢ ΔWᵢ. For LoRA this cannot be done by averaging the A and B matrices separately, and the two adapters here do not even share a rank. `merge_lora_soup.py` uses PEFT's `add_weighted_adapter(combination_type="cat")`, which concatenates the factors (rank 16 + 64 = 80) and folds each adapter's α/r scaling and weight into its A matrix, giving the exact weighted sum. Before saving, the script recomputes ΔW on one attention projection and asserts that it matches Σ wᵢ ΔWᵢ; the recorded error was 1 × 10⁻⁶.

Candidate weights were compared on the validation set only. The 0.5 / 0.5 average was the best candidate and was the only one evaluated on the test set.

## Why direct answers

The target is the answer letters and nothing else. Two explanation-supervised variants were tried during development and both scored below the direct-answer model on the test protocol, so the released recipe keeps the entire loss on the decision.
