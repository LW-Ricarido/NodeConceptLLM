import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, EvalPrediction, DataCollatorForLanguageModeling,set_seed
from datasets import load_from_disk,Dataset
import os
from transformers.data.data_collator import pad_without_fast_tokenizer_warning 
# from data_utils.data_collators import pad_without_fast_tokenizer_warning, InstructEmbedsPretrainCollator
from data_utils.prompt_str import embedding_mask_str, begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str
from tqdm.auto import tqdm
from typing import List
import numpy as np
from torch.utils.data import DataLoader
import json
from peft import PeftModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
# from tensorboardX import SummaryWriter

global_ds_name = 'arxiv'
max_new_tokens = 800
batch_size = 32
use_half = True
use_thinking_mode=True
use_vllm = True
if global_ds_name == 'pubmed':
    # all_labels = ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
    all_labels = ['Diabetes Experiments', 'Diabetes Type 1', 'Diabetes Type 2']
elif global_ds_name == 'cora':
    all_labels = ['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning', 'Theory']
elif global_ds_name == 'arxiv':
    all_labels = ['Artificial Intelligence', 'Hardware Architecture', 'Computational Complexity', 'Computational Engineering, Finance, and Science', 'Computational Geometry', 'Computation and Language', 'Cryptography and Security', 'Computer Vision and Pattern Recognition', 'Computers and Society', 'Databases', 'Distributed, Parallel, and Cluster Computing', 'Digital Libraries', 'Discrete Mathematics', 'Data Structures and Algorithms', 'Emerging Technologies', 'Formal Languages and Automata Theory', 'General Literature', 'Graphics', 'Computer Science and Game Theory', 'Human-Computer Interaction', 'Information Retrieval', 'Information Theory', 'Machine Learning', 'Logic in Computer Science', 'Multiagent Systems', 'Multimedia', 'Mathematical Software', 'Numerical Analysis', 'Neural and Evolutionary Computing', 'Networking and Internet Architecture', 'Other Computer Science', 'Operating Systems', 'Performance', 'Programming Languages', 'Robotics', 'Symbolic Computation', 'Sound', 'Software Engineering', 'Social and Information Networks', 'Systems and Control']
else:
    all_labels = ['Yes', 'Nope']


def label_contained(predicted_sentences):
    cnt = 0
    for label in all_labels:
        if label.lower() in predicted_sentences.lower():
            cnt += 1
    return cnt

class LeftPaddingPredictionCollator(DataCollatorForLanguageModeling):
    
    def torch_call(self, examples):
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, {'input_ids':[example['input_ids'] for example in examples]}, return_tensors = 'pt',
            padding_side = 'left', pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch['attention_mask'] = torch.ones_like(batch['input_ids'])
        batch['attention_mask'][batch['input_ids'] == self.tokenizer.pad_token_id] = 0
        return batch

if __name__ == "__main__":
    device = torch.device("cuda:0")
    model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    adapter_dir = 'ckpts/DeepSeek-R1-Distill-Qwen-1.5B_even_ds_format_reward_4_card/checkpoint-8500'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_from_disk('datasets_local/with_node_index/arxiv_paper_summary')
    splits = np.array(ds['split_set'])
    test_list = np.where(splits == 'test')[0]
    ds = ds.select(test_list)
    print(model_name, adapter_dir)
    if not use_vllm:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # tokenizer.pad_token = '<|finetune_right_pad_id|>'
        model.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.do_sample = False
        model.generation_config.top_p = 1
        model.generation_config.temperature = 1
        model.generation_config.top_k = None
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(model,adapter_dir)
        if use_half:
            model = model.half()
        model = model.to(device)
        
        # ds = load_from_disk('datasets_local/with_node_index/arxiv_paper_summary_even_distribution')
        def chat_map(dp):
            QA_json = []
            # question_str = "This is the information of a paper:\n " + dp.pop('paper summary')
            # question_str += "Please classify it into one the categories:"
            # for label in all_labels:
            #     question_str += " " + label+";"
            # question_str = question_str[:-1] + '. Please answer with only one category.'
            question_str = dp.pop('question')
            QA_json.append(
                {
                    'role':'user',
                    'content': question_str
                }
            )
            dp['input_ids'] = tokenizer.apply_chat_template(QA_json,return_tensors='np', add_generation_prompt=True)[0]
            return dp
        ds = ds.map(chat_map,num_proc=16)
        
        collator = LeftPaddingPredictionCollator(tokenizer,mlm=False)
        test_dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=8,
            shuffle=False,
            collate_fn=collator
        )
        res_list = []
        with torch.no_grad():
            for index, batch_data in enumerate(tqdm(test_dataloader, desc='Testing Process')):
                input_ids = batch_data['input_ids'].to(device)
                batch_res = model.generate(
                    input_ids,
                    attention_mask = batch_data['attention_mask'].to(device),
                    do_sample = False,
                    max_new_tokens = max_new_tokens,
                    generation_config=model.generation_config
                )
                batch_res = batch_res[:,input_ids.shape[1]:]
                # import ipdb; ipdb.set_trace()
                res_list.append(batch_res)
        preds = []
        for i in range(len(res_list)):
            for j in range(res_list[i].shape[0]):
                preds.append(res_list[i][j])
    else:
        llm = LLM(model=model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.65,
                enable_lora=True,
                dtype='half',
                  )
        lora_request = LoRARequest(lora_name='default',lora_int_id=1,lora_path=adapter_dir)
        def chat_map(dp):
            QA_json = []
            # question_str = "This is the information of a paper:\n " + dp.pop('paper summary')
            # question_str += "Please classify it into one the categories:"
            # for label in all_labels:
            #     question_str += " " + label+";"
            # question_str = question_str[:-1] + '. Please answer with only one category.'
            question_str = dp.pop('question')
            QA_json.append(
                {
                    'role':'user',
                    'content': question_str
                }
            )
            dp['input_str'] = tokenizer.apply_chat_template(QA_json,tokenize=False, add_generation_prompt=True)
            return dp
        ds = ds.map(chat_map,num_proc=16)
        sampleParam = SamplingParams(
            max_tokens = max_new_tokens,
            top_p=1,
        )
        # test_dataloader = DataLoader(
        #     ds,
        #     batch_size=batch_size,
        #     num_workers=8,
        #     shuffle=False,
        # )
        prompts = []
        for i in range(len(ds)):
            prompts.append(ds[i]['input_str'])
        # prompts.append(ds[0]['input_str'])
        # prompts.append(ds[1]['input_str'])
        with torch.no_grad():
            # for index, batch_data in enumerate(tqdm(test_dataloader, desc='Testing Process')):
            #     prompt = batch_data['input_str'].to(device)
                batch_res = llm.generate(
                    prompts,
                    lora_request=lora_request,
                    sampling_params=sampleParam
                )
        preds = []
        for i in range(len(batch_res)):
            preds.append(batch_res[i].outputs[0].token_ids)
    correct = 0
    not_finished = 0
    not_finished_list = []
    wrong_list = []
    prediction_wrong_map = {}
    with tqdm(range(len(preds))) as pbar:
        for i in pbar:
            if use_thinking_mode:
                full_response = tokenizer.decode(preds[i])
                if preds[i][-1] != tokenizer.eos_token_id:
                    not_finished += 1
                    not_finished_list.append(i)
                if len(full_response.split('</think>')) == 1:
                    after_think_response = None
                else:
                    after_think_response = full_response.split('</think>')[-1]
                if after_think_response is not None and ds[i]['answer'].lower() in after_think_response.lower() and label_contained(after_think_response.lower()) < 2:
                    correct += 1
                else:
                    wrong_list.append(i)
                    correct_ans = ds[i]['answer'].lower()
                    wrong_predict = str(after_think_response).replace('\n','')
                    wrong_predict = str(wrong_predict).replace('<｜end▁of▁sentence｜>','')
                    curr_key = "{}_to_{}".format(correct_ans,wrong_predict)
                    if curr_key not in prediction_wrong_map.keys():
                        prediction_wrong_map[curr_key] = 0
                    prediction_wrong_map[curr_key] += 1
            else:
                if ds[i]['answer'].lower() in tokenizer.decode(preds[i]).lower() and label_contained(tokenizer.decode(preds[i]).lower()) < 2:
                    correct += 1
    print('{} {} correct count:{} acc: {:.4f} not finished count:{}'.format(model_name, adapter_dir,correct, correct / len(ds),not_finished))
    print(not_finished_list)
    print(wrong_list[0:100])
    print(dict(sorted(prediction_wrong_map.items(), key=lambda item:item[1],reverse=True)))
    # print(prediction_wrong_map)