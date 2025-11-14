import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, EvalPrediction, DataCollatorForLanguageModeling,set_seed
from models.LLaMa import LLama4Graph, LLama4GraphWithValueHead
from data_utils.data_loader import getCoraEmbeds2TextDataset
from sentence_transformers import SentenceTransformer
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.graphproppred import Evaluator
from datasets import load_from_disk,Dataset
import os
from transformers.data.data_collator import pad_without_fast_tokenizer_warning 
# from data_utils.data_collators import pad_without_fast_tokenizer_warning, InstructEmbedsPretrainCollator
from data_utils.prompt_str import embedding_mask_str, begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str
from tqdm.auto import tqdm
from typing import List
import numpy as np
from peft import PeftModel, get_peft_model
from torch.utils.data import DataLoader
import json
from tensorboardX import SummaryWriter

global_ds_name = 'cora'
max_new_tokens = 200
batch_size = 1
use_half = True
if global_ds_name == 'pubmed':
    # all_labels = ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
    all_labels = ['Diabetes Experiments', 'Diabetes Type 1', 'Diabetes Type 2']
elif global_ds_name == 'cora':
    all_labels = ['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning',"Theory"]
elif global_ds_name == 'arxiv':
    all_labels = ['Artificial Intelligence', 'Hardware Architecture', 'Computational Complexity', 'Computational Engineering, Finance, and Science', 'Computational Geometry', 'Computation and Language', 'Cryptography and Security', 'Computer Vision and Pattern Recognition', 'Computers and Society', 'Databases', 'Distributed, Parallel, and Cluster Computing', 'Digital Libraries', 'Discrete Mathematics', 'Data Structures and Algorithms', 'Emerging Technologies', 'Formal Languages and Automata Theory', 'General Literature', 'Graphics', 'Computer Science and Game Theory', 'Human-Computer Interaction', 'Information Retrieval', 'Information Theory', 'Machine Learning', 'Logic in Computer Science', 'Multiagent Systems', 'Multimedia', 'Mathematical Software', 'Numerical Analysis', 'Neural and Evolutionary Computing', 'Networking and Internet Architecture', 'Other Computer Science', 'Operating Systems', 'Performance', 'Programming Languages', 'Robotics', 'Symbolic Computation', 'Sound', 'Software Engineering', 'Social and Information Networks', 'Systems and Control']
else:
    all_labels = ['Yes', 'Nope']


def is_all_label_contained(predicted_sentences):
    cnt = 0
    for label in all_labels:
        if label.lower() in predicted_sentences.lower():
            cnt += 1
    if cnt == len(all_labels):
        return True
    else:
        return False



class LeftPaddingPredictionCollator(DataCollatorForLanguageModeling):
    
    def torch_call(self, examples):
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, {'input_ids':[np.array(example['input_ids'])[np.array(example['labels']) == -100] for example in examples]}, return_tensors = 'pt',
            padding_side = 'left', pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch['attention_mask'] = torch.ones_like(batch['input_ids'])
        batch['attention_mask'][batch['input_ids'] == self.tokenizer.pad_token_id] = 0
        batch['unaligned_inputs_embeds'] = [torch.tensor(example['unaligned_input_embeds'],device=batch['input_ids'].device) for example in examples]
        embedding_mask_id =  self.tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0]
        batch['embedding_positions'] = []
        for i in range(len(examples)):
            batch['embedding_positions'].append(torch.nonzero(batch['input_ids'][i] == embedding_mask_id).flatten())
        return batch

def answer_with_label(answer):
    for label in all_labels:
        if label.lower() in answer:
            return True
    return False

if __name__ == "__main__":
    set_seed(42)
    device = torch.device('cuda:0')
    '''
        This is for Graph Embedding QA prediction Task
    '''
    ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/{}_graph_embedding_QA'.format(global_ds_name))
    # ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/{}_graph_embedding_QA_pure_nodes_with_element_type_count_cot'.format(global_ds_name))
    if global_ds_name == 'arxiv' or global_ds_name == 'mutag' or global_ds_name == 'molhiv':
        test_set_ids = np.where((np.array(ds['split_set']) == 'test') & (np.array(ds['task_type']) == 'classification'))[0]
    else:
        test_set_ids = np.where((np.array(ds['task_type']) == 'classification'))[0]
    ds = ds.select(test_set_ids)
    print("=============length of :",len(test_set_ids))
    
    base_model_path = 'base_models/llama3_2_3B_Instruct_pretrain_lr03_ckpt_197000'
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = '<|finetune_right_pad_id|>'
   
    use_value_head = False
    parent_checkpoint_path = 'ckpts/llama3_2_3B_Instruct/r_16_with_link_eval_test_set2025-05-08 15:35:35.357742'
    
    ckpts_dirs = ['105000']
    count_dict = {}
    for curr_ckpt in ckpts_dirs:
        peft_model = None
        base_model = None
        torch.cuda.empty_cache()
        base_model = LLama4Graph.from_pretrained(base_model_path)
        base_model.embedding_mask_id = tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0]
        if  curr_ckpt != 'zero_shot':
            if not use_value_head:
                base_model.load_adapter('{}/checkpoint-{}/lora_adapter'.format(parent_checkpoint_path,curr_ckpt))
                peft_model = base_model
            else:
                peft_model = LLama4GraphWithValueHead.from_pretrained(base_model)
                peft_model.load_state_dict(torch.load('{}/checkpoint-{}/pytorch_model.bin'.format(parent_checkpoint_path,curr_ckpt),map_location='cpu'),strict=False)
                peft_model.pretrained_model.load_adapter('{}/checkpoint-{}'.format(parent_checkpoint_path,curr_ckpt))
            
            peft_model.generation_config = base_model.generation_config
            # peft_model.generation_config.temperature = 1
            # peft_model.generation_config.top_p = 1
            # peft_model.generation_config.do_sample = False

        else:
            peft_model = base_model
        if use_half:
            peft_model = peft_model.half()
        peft_model = peft_model.to(device)
        collator = LeftPaddingPredictionCollator(tokenizer,mlm=False)
        
        print('curr ckpt {}'.format(curr_ckpt))
        res_list = [] 
        raw_input_list =[]
        res_values = []
        peft_model.generation_config.pad_token_id = tokenizer.eos_token_id
        # peft_model.generation_config.do_sample = False
        print(peft_model.generation_config)
        print(tokenizer.eos_token_id)
        
        test_dataloader = DataLoader(
            ds,
            batch_size = batch_size,
            # collate_fn=collator,
            num_workers=4,
            shuffle=False
        )
        reasoning_save_list = []
        embedding_mask_id = tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0]
        eot_id = tokenizer.encode('<|eot_id|>',add_special_tokens=False)[0]
        correct_count = 0
        not_answered_count = 0
        self_corrected_count = 0
        correct_attempt = 0
        tqdm_bar = tqdm(ds, desc='Testing Process')
        with torch.no_grad():
            for index, batch_data in enumerate(tqdm_bar):
                answer_label = batch_data['answer']
                answer_label = answer_label.replace('Diabetes Mellitus, Experimental', 'Diabetes Experiments')
                answer_label = answer_label.replace('Diabetes Mellitus Type 1', 'Diabetes Type 1')
                answer_label = answer_label.replace('Diabetes Mellitus Type 2', 'Diabetes Type 2')
                QA_json =[
                    {
                        "role":'user',
                        'content': batch_data['question'].replace(":Diabetes Mellitus, Experimental; Diabetes Mellitus Type 1; Diabetes Mellitus Type 2",
                                            ": Diabetes Type 1; Diabetes Type 2; Diabetes Experiments")
                    }
                        
                ]
                get_answer = False
                ask_round = 0
                if use_half:
                    unaligned_inputs_embeds = torch.tensor(batch_data['unaligned_input_embeds']).half().to(device)
                else:
                    unaligned_inputs_embeds = torch.tensor(batch_data['unaligned_input_embeds']).to(device)
                aligned_embeds = peft_model.node_embedding_connect(unaligned_inputs_embeds)
                while not get_answer and ask_round < 3:
                    input_ids = tokenizer.apply_chat_template(QA_json, return_tensors='pt', add_generation_prompt=True)
                    embedding_positions = torch.nonzero(input_ids[0] == embedding_mask_id).flatten()
                    batch_input_embeds = peft_model.get_input_embeddings()(input_ids.to(device))
                    batch_input_embeds[0][embedding_positions] = aligned_embeds
                    res = peft_model.generate(
                        inputs_embeds = batch_input_embeds,
                        attention_mask = torch.ones_like(input_ids).to(device),
                        do_sample = True,
                        max_new_tokens = max_new_tokens,
                    )
                    generated_answer = tokenizer.decode(res[0][res[0] != eot_id])
                    import ipdb; ipdb.set_trace()
                    if True:#not answer_with_label(generated_answer.lower()):
                        QA_json.extend(
                            [
                                {
                                    "role": "assistant",
                                    "content": generated_answer
                                },
                                {
                                    'role': "user",
                                    "content": "Please answer why it should be classify into that category." ''#'"{}" is not one of the required categories. Please answer with correct category.'.format(generated_answer)
                                }
                            ]
                            
                        )
                        ask_round += 1
                    else:
                        get_answer = True
                if get_answer:
                    if answer_label.lower() in generated_answer.lower() and not is_all_label_contained(generated_answer.lower()):
                        correct_count += 1
                        if ask_round > 1:
                            print('corrected answer:', generated_answer)
                            self_corrected_count += 1
                else:
                    not_answered_count += 1
                if ask_round > 1:
                    correct_attempt += 1 
                tqdm_bar.set_postfix({"Correct": correct_count,  "Not Answered": not_answered_count, "correct_attempt": correct_attempt,"self_corrected_count": self_corrected_count})
        print('acc:{}'.format(correct_count / len(ds)))
        print('not answered count: ', not_answered_count)