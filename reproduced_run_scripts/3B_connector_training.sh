#!/usr/bin/env bash
set -euo pipefail

PRETRAIN_DATASET_DIR="${PRETRAIN_DATASET_DIR:-../../LLM4Graph_tracked/datasets_local/all_connector_pretrain_with_names}"

accelerate launch --gpu_ids all main.py \
  --task pretrain \
  --model_dir base_model/Llama-3.2-3B-Instruct_resized_embedding \
  --dataset_dir "${PRETRAIN_DATASET_DIR}" \
  --connector_dim 768 \
  --model_save_path ckpts/speedup/3b_connector \
  --log_dir logs/speedup/3b_connector \
  --run_name 3b_connector_speedup \
  --wandb_project NodeConceptLLM \
  --precision bf16 \
  --attn_implementation sdpa \
  --train_size 2 \
  --gradient_accumulation_steps 4 \
  --eval_size 8 \
  --lr 3e-1 \
  --epoch 10 \
  --logging_steps 100 \
  --eval_steps 5000 \
  --save_steps 5000 \
  --save_total_limit 3 \
  --save_resume_state \
  "$@"
