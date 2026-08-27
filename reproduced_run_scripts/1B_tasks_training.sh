#!/usr/bin/env bash
set -euo pipefail

# Set CONNECTOR_DIR to the pretrain output's final/ directory or checkpoint-N.
CONNECTOR_DIR="${CONNECTOR_DIR:-ckpts/speedup/1b_connector_test/checkpoint-155000}"

accelerate launch --gpu_ids all main.py \
  --task sft \
  --model_dir base_model/Llama-3.2-1B-Instruct_resized_embedding \
  --connector_path "${CONNECTOR_DIR}" \
  --dataset_dir ../../LLM4Graph_tracked/datasets_local/all_downstream_dataset_32 \
  --dataset_names WN18RR \
  --connector_dim 768 \
  --model_save_path ckpts/speedup/1b_sft_30epoch_lr3e-5 \
  --log_dir logs/speedup/1b_sft_30epoch_lr3e-5 \
  --run_name 1b_sft_speedup_30epoch_lr3e-5 \
  --wandb_project NodeConceptLLM \
  --use_LoRA \
  --precision bf16 \
  --attn_implementation sdpa \
  --train_size 8 \
  --gradient_accumulation_steps 1 \
  --eval_size 16 \
  --lr 3e-5 \
  --epoch 30 \
  --logging_steps 500 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 10 \
  --eval_samples 2000 \
  --eval_strategy steps \
  --save_strategy steps \
  --metric_for_best_model label_in_output_accuracy \
  --greater_is_better \
  --load_best_model_at_end \
  --label_in_output \
  --save_resume_state \
  "$@"
