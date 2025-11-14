from transformers import Qwen2ForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union

class Qwen24Graph(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.node_embedding_connect = nn.Sequential(
            nn.Linear(768, config.hidden_size)
        )
        self.task_type = None
        self.pad_token_id = None
        self.embedding_mask_id = None
        self.post_init()
    def forward(self, input_ids = None, attention_mask = None, position_ids = None, past_key_values = None, inputs_embeds = None, labels = None, use_cache = None, output_attentions = None, output_hidden_states = None, cache_position = None, logits_to_keep = 0, **kwargs):
        return super().forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, cache_position, logits_to_keep, **kwargs)