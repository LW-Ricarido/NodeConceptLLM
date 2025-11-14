from transformers import DataCollatorForLanguageModeling
import torch
from transformers.data.data_collator import *
import numpy as np
# from trl.trainer.dpo_trainer import DataCollatorForPreference
from data_utils.prompt_str import *

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        if not isinstance(examples, torch.Tensor):
            return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer.pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

class Embeds2TextCollator(DataCollatorForLanguageModeling):
    
    # fp16: bool = True 
    def torch_call(self, examples):
        batch =  super().torch_call(examples)
        if 'input_embedding' in batch:
            unaligned_inputs_embeds = batch.pop('input_embedding')
            batch['unaligned_inputs_embeds'] = unaligned_inputs_embeds
        unaligned_inputs_embeds = batch.pop('input_embedding')
        batch['unaligned_inputs_embeds'] = unaligned_inputs_embeds#.half()
        return batch

class InstructEmbedsPretrainCollator(DataCollatorForLanguageModeling):
    

    def __init__(self,use_fp16: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fp16 = use_fp16
    def torch_call(self, examples):
        '''
            examples.keys:
                - input_ids: list[int]
                - labels: list[int]
                - embedding_positions: list[int]
                - unaligned_input_embeds: list[tensor]
                
        '''
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, {'input_ids':[example['input_ids'] for example in examples]}, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
        )
        labels = batch["input_ids"].clone()
        
        for i in range(labels.shape[0]):
            labels[i,:len(examples[i]['labels'])] = torch.tensor(examples[i]['labels'],dtype=labels.dtype,device=labels.device)
        batch['labels'] = labels
        if self.tokenizer.pad_token_id is not None:
            batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100
        embedding_length = set()
        if 'unaligned_input_embeds' in examples[0].keys():
            for example in examples:
                embedding_length.add(len(example['unaligned_input_embeds']))
            # if embedding_length == {1}:
            #     if self.use_fp16:
            #         batch['unaligned_inputs_embeds'] = torch.stack([torch.tensor(example['unaligned_input_embeds'][0],device=labels.device).half() for example in examples])
            #     else:
            #         batch['unaligned_inputs_embeds'] = torch.stack([torch.tensor(example['unaligned_input_embeds'][0],device=labels.device) for example in examples])
            # else:
            if self.use_fp16:
                batch['unaligned_inputs_embeds'] = [torch.tensor(example['unaligned_input_embeds'],device=labels.device).half() for example in examples]
            else:
                batch['unaligned_inputs_embeds'] = [torch.tensor(example['unaligned_input_embeds'],device=labels.device) for example in examples]
            embedding_mask_id =  self.tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0]
            batch['embedding_positions'] = []
            for i in range(len(examples)):
                batch['embedding_positions'].append(torch.nonzero(batch['input_ids'][i] == embedding_mask_id).flatten())
        # batch['embedding_positions'] = [torch.tensor(example['embedding_positions'],dtype=torch.long,device=labels.device) for example in examples]
        return batch
        # return super().torch_call(examples)

class LeftPaddingInstructEmbedsPretrainWithLabelCollator(DataCollatorForLanguageModeling):
    def __init__(self,use_fp16: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fp16 = use_fp16
    def torch_call(self, examples):
        '''
            examples.keys:
                - input_ids: list[int]
                - labels: list[int]
                - embedding_positions: list[int]
                - unaligned_input_embeds: list[tensor]
                
        '''
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, {'input_ids':[example['input_ids'] for example in examples]}, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of,padding_side="left"
        )
        labels = self.tokenizer([example['answer'] for example in examples],
                                       padding=True,return_tensors='pt')['input_ids']
        
        # labels = pad_without_fast_tokenizer_warning(
        #     self.tokenizer, {'inputs':[example['answer'] for example in examples]}, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
        # )
        # for i in range(labels.shape[0]):
        #     labels[i,:len(examples[i]['labels'])] = torch.tensor(examples[i]['labels'],dtype=labels.dtype,device=labels.device)
        batch['labels'] = labels
        embedding_length = set()
        if 'unaligned_input_embeds' in examples[0].keys():
            for example in examples:
                embedding_length.add(len(example['unaligned_input_embeds']))
            if embedding_length == {1}:
                if self.use_fp16:
                    batch['unaligned_inputs_embeds'] = torch.stack([torch.tensor(example['unaligned_input_embeds'][0],device=labels.device).half() for example in examples])
                else:
                    batch['unaligned_inputs_embeds'] = torch.stack([torch.tensor(example['unaligned_input_embeds'][0],device=labels.device) for example in examples])
            else:
                if self.use_fp16:
                    batch['unaligned_inputs_embeds'] = [torch.tensor(example['unaligned_input_embeds'],device=labels.device).half() for example in examples]
                else:
                    batch['unaligned_inputs_embeds'] = [torch.tensor(example['unaligned_input_embeds'],device=labels.device) for example in examples]
            embedding_mask_id =  self.tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0]
            batch['embedding_positions'] = []
            for i in range(len(examples)):
                batch['embedding_positions'].append(torch.nonzero(batch['input_ids'][i] == embedding_mask_id).flatten())
        # batch['embedding_positions'] = [torch.tensor(example['embedding_positions'],dtype=torch.long,device=labels.device) for example in examples]
        return batch

class Embeds2PredictionCollator(DataCollatorForLanguageModeling):
    
    def torch_call(self, examples):
        '''
            examples.keys:
                - input_ids: List[int]
                - label_begin_pos: int
                - label_end_pos: int
                - text_label: str
                - unaligned_inputs_embeds: tensor
        '''
        if isinstance(examples[0], Mapping):
            max_length = np.max([len(example['unaligned_input_embeds']) for example in examples])
            for i in range(len(examples)):
                
                # if len(examples[i]['unaligned_inputs_embeds']) < max_length:
                if type(examples[i]['input_ids']) == list:
                    examples[i]['input_ids'] = [self.tokenizer.pad_token_id] * len(examples[i]['unaligned_input_embeds']) + examples[i]['input_ids']
                else:
                    examples[i]['input_ids'] = torch.cat((
                        self.tokenizer.pad_token_id * torch.ones(len(examples[i]['unaligned_input_embeds']),dtype=examples[i]['input_ids'].dtype),
                        examples[i]['input_ids']
                        ))
            
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, {'input_ids':[example['input_ids'] for example in examples]}, return_tensors="pt",pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch({'input_ids':[example['input_ids'] for example in examples]}, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        for i in range(labels.shape[0]):
            labels[i,:len(examples[i]['labels'])] = torch.tensor(examples[i]['labels'],dtype=labels.dtype,device=labels.device)
        batch['labels'] = labels
        # unaligned_inputs_embeds = batch.pop('unaligned_inputs_embeds')
        batch['unaligned_inputs_embeds'] = [torch.tensor(example['unaligned_input_embeds'],dtype=torch.float,device=labels.device).squeeze(1) for example in examples]
        # batch['unaligned_inputs_embeds'] = unaligned_inputs_embeds#.half()
        batch['text_label'] = [example['text_label'] for example in examples]
        batch['embedding_positions'] = [torch.tensor(example['embedding_positions'],dtype=torch.long,device=labels.device) for example in examples]
        return batch

# @dataclass   
# class DPOCollator(DataCollatorForPreference):
    
#     embedding_id: int = -100
    
#     def torch_call(self, examples):
#         output = super().torch_call(examples)
#         output['unaligned_inputs_embeds'] = [torch.tensor(example['unaligned_inputs_embeds'],dtype=torch.float,device=output["prompt_input_ids"].device).squeeze(1) for example in examples]
#         output['text_label'] = [example['text_label'] for example in examples]
#         output['embedding_positions'] = [torch.nonzero(input_ids == self.embedding_id).flatten() for input_ids in output['prompt_input_ids']]
#         return output