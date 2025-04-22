from trl import PPOTrainer,AutoModelForCausalLMWithValueHead
import torch
import torch.nn as nn

class ValueRewardModel(nn.Module):
    def __init__(self, tokenizer,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.base_model_perfix = self
    
    def forward(self, input_ids, attention_mask,labels,**kwargs):
        real_input_ids = input_ids#[attention_mask]
        tokenized_text = self.tokenizer.batch_decode(real_input_ids,skip_special_tokens=True)
        scores = torch.zeros(len(tokenized_text))
        for i in range(len(tokenized_text)):
            scores[i] = self.get_reward(tokenized_text[i], labels[i])
        return scores
    
    def get_reward(self, text, label):
        try:
            value = float(text)
            if value < 0 or value > 1:
                return 1
            if label == 1:
                return 100 * value
            else:
                return 10 * (1 - value)
        except:
            return -1000
    
    def score(self, reward_input):
        '''
            This function is only used for hack trl PPOTrainer get_reward function
        '''
        return reward_input