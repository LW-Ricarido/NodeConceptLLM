import ipdb.stdout
from transformers import LlamaModel,LlamaPreTrainedModel, GenerationMixin, LlamaForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import(
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
import numpy as np
from trl import AutoModelForCausalLMWithValueHead

logger = logging.get_logger(__name__)

# class LLama4Graph(LlamaPreTrainedModel, GenerationMixin):
#     _tied_weights_keys = ["lm_head.weight"]
#     _tp_plan = {"lm_head": "colwise_rep"}
class LLama4Graph(LlamaForCausalLM):  
    def __init__(self, config):
        super().__init__(config)
        # self.model = LlamaModel(config)
        # self.vocab_size = config.vocab_size
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.node_embedding_connect = nn.Sequential(
            nn.Linear(768, config.hidden_size) # 768 for clip
        )
        self.task_type = None
        self.pad_token_id = None
        self.embedding_mask_id = None
        self.post_init()
        
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        unaligned_inputs_embeds: Optional[List[torch.FloatTensor]] = None,
        embedding_positions: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if input_ids is not None and (input_ids == self.embedding_mask_id).sum() > 0 and unaligned_inputs_embeds is None:
            import traceback
            print(traceback.print_stack())
        
            import ipdb; ipdb.set_trace()
        if unaligned_inputs_embeds is not None and embedding_positions is None:
            embedding_positions = []
            for i in range(input_ids.shape[0]):
                embedding_positions.append(torch.nonzero(input_ids[i] == self.embedding_mask_id).flatten())
                if embedding_positions[-1].shape[0] != unaligned_inputs_embeds[i].shape[0]:
                    import ipdb; ipdb.set_trace()
        if inputs_embeds is not None and unaligned_inputs_embeds is not None and embedding_positions is None:
            import traceback
            print(traceback.print_stack())
        
            import ipdb; ipdb.set_trace()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None:
            assert torch.sum(input_ids >= self.model.get_input_embeddings().num_embeddings) == 0, 'have some tokens not include in embedding layer, max input id: {}, vocab size: {}'.format(torch.max(input_ids).item(),self.model.get_input_embeddings().num_embeddings)      
            # input_ids[input_ids >= self.model.get_input_embeddings().num_embeddings] = 0   
            
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if unaligned_inputs_embeds is not None:
            if unaligned_inputs_embeds is torch.tensor:
                aligned_input_embeds = self.node_embedding_connect(unaligned_inputs_embeds)
            else:
                aligned_input_embeds = [self.node_embedding_connect(unaligned_inputs_embed) for unaligned_inputs_embed in unaligned_inputs_embeds]
           
            for i in range(len(aligned_input_embeds)):
                # for auto casting
                inputs_embeds[i][embedding_positions[i]] = aligned_input_embeds[i].to(inputs_embeds.dtype)

        # if not self.training and self.task_type == 'prediction' and unaligned_inputs_embeds is not None:
            
        #     input_lengths = []
        #     for i in range(labels.shape[0]):
        #         input_lengths.append(torch.nonzero(labels[i] != -100).flatten()[0].item())
        #     max_input_length = np.max(input_lengths)

        #     new_inputs_embeds = torch.zeros(inputs_embeds.shape[0],max_input_length,inputs_embeds.shape[-1],dtype=inputs_embeds.dtype,device=inputs_embeds.device)
        #     new_attention_mask = torch.ones(inputs_embeds.shape[0], max_input_length, dtype=input_ids.dtype, device=inputs_embeds.device)
        #     pad_embed = self.model.get_input_embeddings()(self.pad_token_id * torch.ones(1,dtype=input_ids.dtype,device=input_ids.device))
        #     max_new_tokens =  torch.sum(labels != -100,dim=1).max().item()
        #     for i in range(inputs_embeds.shape[0]):
        #         current_input_length = input_lengths[i]
        #         new_inputs_embeds[i,max_input_length - current_input_length:] = inputs_embeds[i,:current_input_length]
        #         ### Attention!!!: pad for generation for on the left, not right
        #         new_inputs_embeds[i, :max_input_length - current_input_length] = pad_embed.repeat(max_input_length - current_input_length,1)
        #         new_attention_mask[i, : max_input_length - current_input_length] = 0
        #     inputs_embeds = new_inputs_embeds
        #     outputs = self.generate(
        #         inputs_embeds=inputs_embeds, 
        #         max_new_tokens=max_new_tokens,
        #         do_sample=False,
        #         output_logits=True, 
        #         return_dict_in_generate=True,
        #         pad_token_id=self.pad_token_id,
        #         attention_mask = new_attention_mask,
        #         output_hidden_states=True,
        #     )
        #     logits = torch.stack(outputs['logits'],dim=1)
        #     new_labels = -100 * torch.ones((labels.shape[0], max_new_tokens),dtype=labels.dtype,device=labels.device)
        #     for i in range(labels.shape[0]):
        #         new_labels[i][:torch.sum(labels[i] != -100)] = labels[i][torch.sum(labels[i] == -100):]
        #     if logits.shape[1] != new_labels.shape[1]:
        #         pad_logits = torch.zeros((logits.shape[0],new_labels.shape[1] - logits.shape[1],logits.shape[-1]),dtype=logits.dtype,device=logits.device)
        #         pad_logits[:,:,self.pad_token_id] = 1
        #         logits = torch.cat((logits, pad_logits),dim=1)
        #     assert logits.shape[1] == new_labels.shape[1], "Sentence length dose not match, logits: {}, new_labels: {} max_new_tokens:{}".format(logits.shape[1],new_labels.shape[1],max_new_tokens)
        #     import ipdb;ipdb.set_trace()
        #     loss = self.loss_function(
        #         logits=logits,
        #         labels=new_labels,
        #         vocab_size=logits.shape[-1],
        #         ** kwargs
        #     )
        #     return CausalLMOutputWithPast(
        #         loss=loss,
        #         logits=logits,
        #         hidden_states = outputs.hidden_states[0],
        #     )


        
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # this one use aligned_input_embeds
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        loss = None
        if labels is not None:
            #TODO: a better way to set vocab_size
            loss = self.loss_function(logits=logits,labels=labels, vocab_size=logits.shape[-1], **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def generate(self, *args, **kwargs):
        if 'input_ids' in kwargs.keys():
            input_ids = kwargs.pop('input_ids')
            inputs_embeds = self.get_input_embeddings()(input_ids)
            
            unaligned_inputs_embeds = kwargs.pop('unaligned_inputs_embeds')
            embedding_positions = kwargs.pop("embedding_positions")
            aligned_embeds = list(map(self.node_embedding_connect, unaligned_inputs_embeds))
            for i in range(len(aligned_embeds)):
                inputs_embeds[i][embedding_positions[i]] = aligned_embeds[i]
            outputs = super().generate(
                inputs_embeds = inputs_embeds,
                **kwargs
            )
            return outputs
        else:
            return super().generate(**kwargs)


class LLama4GraphWithValueHead(AutoModelForCausalLMWithValueHead):
    base_model_prefix = 'pretrained_model'
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, is_for_sft=True,positive_id=9642, *model_args, **kwargs):
        model =  super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.is_for_sft = is_for_sft
        ## TODO: a better way to set positive_id
        model.positive_id = positive_id
        return model
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=None,
        **kwargs,
    ):
        original_output = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            return_past_key_values=return_past_key_values,
            **kwargs
        )
        if return_past_key_values:
            lm_logits, loss, value, past_key_values = original_output
        else:
            lm_logits, loss, value = original_output
        if 'labels' in kwargs:
            labels = kwargs['labels']
            first_output_token_idx = (labels != -100).int().argmax(dim=1) - 1
            classification_value = value[torch.arange(value.shape[0],device=value.device),first_output_token_idx].unsqueeze(1)
            target = (labels == self.positive_id).sum(dim=1).unsqueeze(1)
            target = target.to(classification_value.dtype)
            loss += F.binary_cross_entropy_with_logits(classification_value,target)
        else:
            raise ValueError('labels should be provided')
        if 'return_dict' in kwargs.keys():
            return CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
            )
        if self.is_for_sft:
            if not self.training:
                output =  CausalLMOutputWithPast(
                    loss=loss,
                    logits=lm_logits,
                )
                output['predicted_values']= torch.cat([target,classification_value],dim=1)
                return output
            return CausalLMOutputWithPast(
                loss=loss,
                past_key_values=past_key_values,
            )
        if return_past_key_values:
            return (lm_logits, loss, value, past_key_values)
        return (lm_logits, loss, value)
    
    def generate(self, require_value=False, *args, **kwargs):
        if not self.training or require_value:
            outputs = super().generate(
                output_logits=True, 
                return_dict_in_generate=True,
                output_hidden_states=True,
                *args, 
                **kwargs
            )
            classification_value = self.v_head(outputs.hidden_states[0][-1][:,-1])
            generated_tokens = torch.stack(outputs['logits'],dim=1).argmax(dim=-1)
            return generated_tokens, classification_value
        return super().generate(*args, **kwargs)

    def score(self,hidden_state):
        return self.v_head(hidden_state)