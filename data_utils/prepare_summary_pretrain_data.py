from ollama import chat, ChatResponse, Client
import os,sys
sys.path.append(os.curdir)
# os.
from data_utils.load_raw_data import load_arxiv_raw_data
import multiprocessing as mp
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

raw_ds = load_arxiv_raw_data(True, True, True, True, True)

prompt_pools =[
    "This is a paper with {}. Please summarize this paper in no more than 30 words.",
    # "This is a paper with {}. Please based on it's original title and abstract, provide a new title. Please only answer with the new title."
]
model_dicts = {}
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
# for i in range(4):
#     model_dicts[i] = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct').to(torch.device('cuda:{}'.format(i)))
    

def get_answer_from_hf(start_index, end_index,gpu_idx):
    dp_lists = []
    # model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct').to(torch.device('cuda:{}'.format(gpu_idx)))
    print('==========get model================')
    for i in tqdm(
        range(start_index, end_index),
        position=gpu_idx,
        # leave=True,
        desc=f"Worker {gpu_idx}"
        # dynamic_ncols=True
        ):
        # print(raw_ds['raw_text'][i])
        messages=[
            {
                "role":"user",
                "content": prompt_pools[0].format(raw_ds['raw_text'][i])
            }
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,return_tensors='pt',add_generation_prompt=True
        )
        original_length = input_ids.shape[1]
        inputs = {}
        inputs['input_ids'] = input_ids.to(model_dicts[gpu_idx].device)
        inputs['attention_mask'] = torch.ones_like(input_ids,device = model_dicts[gpu_idx].device)
        response = tokenizer.decode(model_dicts[gpu_idx].generate(max_new_tokens=300,**inputs)[0][original_length:],skip_special_tokens=True)
        dp = {'original_node_idx':i, 'type':'summary','answer':response}
        dp_lists.append(dp)
        messages=[
            {
                "role":"user",
                "content": prompt_pools[1].format(raw_ds['raw_text'][i])
            }
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,return_tensors='pt',add_generation_prompt=True
        )
        original_length = input_ids.shape[1]
        inputs = {}
        inputs['input_ids'] = input_ids.to(model_dicts[gpu_idx].device)
        inputs['attention_mask'] = torch.ones_like(input_ids,device = model_dicts[gpu_idx].device)
        response = tokenizer.decode(model_dicts[gpu_idx].generate(max_new_tokens=300,**inputs)[0][original_length:],skip_special_tokens=True)
        dp = {'original_node_idx':i, 'type':'alter title','answer':response}
        dp_lists.append(dp)
    return dp_lists


def get_answer_from_ollama(start_index,end_index,gpu_idx, global_rank):
    dp_lists = []
    # print(start_index, end_index, global_rank)
    client = Client(
        host='http://localhost:9233'
    )
    for i in tqdm(
        range(start_index, end_index),
        position=global_rank,
        # leave=True,
        desc=f"Worker {global_rank}"
        # dynamic_ncols=True
        ):
        # print(raw_ds['raw_text'][i])
        response: ChatResponse = client.chat(
            model="qwen2.5:1.5b",
            messages=[
                {
                    "role":"user",
                    "content": prompt_pools[0].format(raw_ds['raw_text'][i])
                }
            ],
            options={
                "main_gpu": gpu_idx,
                # "num_batch": 2,
            },
            keep_alive=-1
        )
        dp = {'original_node_idx':i, 'type':'summary','answer':response.message.content}
        # print(dp)
        dp_lists.append(dp)
        response: ChatResponse = chat(
            model="qwen2.5:1.5b",
            messages=[
                {
                    "role":"user",
                    "content": prompt_pools[1].format(raw_ds['raw_text'][i])
                }
            ],
            options={
                "main_gpu": gpu_idx,
                # "num_batch": 0
            },
            keep_alive=-1
        )
        dp = {'original_node_idx':i, 'type':'alter title','answer':response.message.content}
        dp_lists.append(dp)
    return dp_lists


if __name__ == "__main__":
    processes_number = 4
    my_pool = []
    seg = len(raw_ds['raw_text'])// processes_number
    ds_for_save = []
    # test = get_answer_from_ollama(0,seg,0)
    # test = get_answer_from_hf(0,seg, 0)
    # mp.set_start_method('spawn')
    # with mp.Pool(processes=processes_number,) as pool:
    #     for i in range(processes_number):
    #         current_res = pool.apply_async(get_answer_from_hf,(seg * i, min(seg * (i+1), len(raw_ds['raw_text'])), i%4))
    #         my_pool.append(current_res)
    #     print('=======actived=========')
    #     for i in range(processes_number):
    #         current_res = my_pool[i].get()
    #         ds_for_save.extend(current_res)
    rank = 3
    model_dicts[rank] = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct').to(torch.device('cuda:{}'.format(rank)))
    # import ipdb; ipdb.set_trace()
    ds_for_save = get_answer_from_hf(rank * seg, min(seg* (rank+1), len(raw_ds['raw_text'])), rank)
    ds_for_save = Dataset.from_list(ds_for_save)
    ds_for_save.save_to_disk('datasets_local/new_reasoning_datasets/arxiv_alter_pretrain_{}'.format(rank))