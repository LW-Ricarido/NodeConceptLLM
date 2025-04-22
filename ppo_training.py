
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator, PartialState
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from models.reward_models import ValueRewardModel
from models.LLaMa import LLama4Graph
from datasets import load_from_disk
import numpy as np
from mine_pure_eval import LeftPaddingPredictionCollator

@dataclass
class ScriptArguments:
    model_dir: str
    dataset_dir: str
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})
    # local_tokenizer: str
    epochs: int = field(default=10)
    

if __name__ == '__main__':
    parser = HfArgumentParser((ScriptArguments, PPOConfig))
    args, ppo_config = parser.parse_args_into_dataclasses()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = '<|finetune_right_pad_id|>'
    
    reward_model = ValueRewardModel(tokenizer=tokenizer)
    dataset = load_from_disk(args.dataset_dir)
    split_set = np.array(dataset['split_set'])
    train_list = np.where(split_set == 'train')[0].tolist()
    test_list = np.where(split_set == 'valid')[0].tolist()
    train_dataset = dataset.select(train_list)
    test_dataset = dataset.select(test_list)
    
    base_model = LLama4Graph.from_pretrained(args.model_dir)
    if args.use_peft:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = AutoModelForCausalLMWithValueHead(base_model,peft_config = lora_config)
        model = model.half()
    collator = LeftPaddingPredictionCollator(tokenizer,mlm=False)
    ppo_trainer = PPOTrainer(
        model = model,
        config = ppo_config,
        tokenizer = tokenizer,
        dataset = train_dataset,
        training_data_collator = collator,
        data_collator = collator
    )
    # import ipdb; ipdb.set_trace()
    for epoch in tqdm(range(args.epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            unaligned_inputs_embeds= [batch['unaligned_inputs_embeds'][i].half().to(batch['input_ids'].device) for i in range(len(batch['unaligned_inputs_embeds']))]
            embedding_positions = [batch['embedding_positions'][i].to(batch['input_ids'].device) for i in range(len(batch['embedding_positions']))]
            batch['unaligned_inputs_embeds'] = unaligned_inputs_embeds
            batch['embedding_positions'] = embedding_positions
            for i in range(len(batch['embedding_positions'])):
                print(i, ": ", batch['embedding_positions'][i].max())
            # import ipdb; ipdb.set_trace()
            ppo_trainer.model.pretrained_model.task_type = "prediction"
            tmp = ppo_trainer.model.generate(**batch)
            attn_mask = torch.ones_like(tmp,dtype=torch.long, device = tmp.device)
            labels = [1,0]
            rewards = reward_model(tmp, attn_mask, labels)
            query_tensor = [batch['input_ids'][i].cpu() for i in range(batch['input_ids'].shape[0])]
            response_tensor = [tmp[i].cpu() for i in range(tmp.shape[0])]
            rewards = [rewards[i].cpu() for i in range(rewards.shape[0])]
            stats = ppo_trainer.step(query_tensor, response_tensor, rewards)
            print('======get in =========')
            
            import ipdb; ipdb.set_trace()
            pass
    pass
