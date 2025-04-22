# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Full training
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns

# LoRA:
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from datasets import load_from_disk
import os, sys
sys.path.append(os.path.abspath(os.path.curdir))

from data_utils.data_collators import DPOCollator
from models.LLaMa import LLama4Graph
from data_utils.prompt_str import *
from trainers.flex_dpo_trainer import FlexDPOTrainer
from peft import LoraConfig, TaskType, AutoPeftModel

def main(script_args, training_args, model_args):
    ################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    
    
     # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    # )
    tokenizer = AutoTokenizer.from_pretrained('base_models/Llama-3.2-3B-Instruct_node_graph_combine_pretrain')
    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token =  "<|finetune_right_pad_id|>"
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    # )
    
    
    model = LLama4Graph.from_pretrained(
            'base_models/llama3_2_3B_Instruct_arxiv_Graph_QA_LoRA_node_combine_pretrain_1_1_checkpoint-78000_merged')
    # train_layer_count = 0
    # for param in model.parameters():
    #     if param.requires_grad:
    #         train_layer_count += 1
    model.embedding_mask_id = tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0]
    for param in model.node_embedding_connect.parameters():
        param.requires_grad = False
    for param in model.model.parameters():
        param.requires_grad = False
    
    
        
    # peft_config = get_peft_config(model_args)
    peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    
    
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = load_from_disk('datasets_local/arxiv_dpo_graph_QA')
    use_think = True
    def map_function(dp):
        dp['prompt'] = dp.pop('question')
        if use_think:
            dp['chosen'] = dp.pop('thinking_answer')
        else:
            dp['chosen'] = dp.pop('qwq_answer')
        dp['rejected'] = dp['text_label']
        return dp
    dataset = dataset.map(map_function)
    data_collator = DPOCollator(pad_token_id=tokenizer.pad_token_type_id,embedding_id=tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0])

    ##########
    # Training
    ################
    # trainer = DPOTrainer(
    #     model,
    #     ref_model,
    #     args=training_args,
    #     train_dataset=dataset,#[script_args.dataset_train_split],
    #     eval_dataset=dataset,#[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
    #     processing_class=tokenizer,
    #     peft_config=peft_config,
    #     data_collator=data_collator,
    # )
    trainer = FlexDPOTrainer(
        registered_model_input_keys=['embedding_positions','unaligned_inputs_embeds'],
        model = model,
        ref_model = ref_model,
        args=training_args,
        train_dataset=dataset,#[script_args.dataset_train_split],
        # eval_dataset=dataset,#[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=data_collator,
    )
    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(training_args.optim)
    main(script_args, training_args, model_args)
