#!/usr/bin/env bash
set -euo pipefail

CONNECTOR_DIR="${CONNECTOR_DIR:-ckpts/speedup/3b_connector/final}"

accelerate launch --gpu_ids all main.py \
  --task sft \
  --model_dir base_model/Llama-3.2-3B-Instruct_resized_embedding \
  --connector_path "${CONNECTOR_DIR}" \
  --dataset_dir ../../LLM4Graph_tracked/datasets_local/all_downstream_dataset_32 \
  --dataset_names WN18RR \
  --connector_dim 768 \
  --model_save_path ckpts/speedup/3b_sft_wn18rr \
  --log_dir logs/speedup/3b_sft_wn18rr \
  --run_name 3b_sft_wn18rr_speedup \
  --wandb_project NodeConceptLLM \
  --use_LoRA \
  --precision bf16 \
  --attn_implementation sdpa \
  --train_size 2 \
  --gradient_accumulation_steps 2 \
  --eval_size 8 \
  --lr 3e-3 \
  --epoch 10 \
  --logging_steps 20 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 3 \
  "$@"
