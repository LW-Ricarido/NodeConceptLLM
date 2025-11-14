from vllm import LLM, SamplingParams
import pandas as pd
import os,sys
import requests
import json
from datasets import Dataset,load_from_disk
sys.path.append(os.path.abspath(os.path.curdir))
# from data_utils.load_raw_data import load_arxiv_raw_data
from tqdm.auto import tqdm
from data_utils.prompt_str import *
# from data_utils.prepare_instruct_dataset import remove_reverse_edge
from transformers import AutoTokenizer
import time,torch
import numpy as np
candidate_str = "Artificial Intelligence; Hardware Architecture; Computational Complexity; Computational Engineering, Finance, and Science; Computational Geometry; Computation and Language; Cryptography and Security; Computer Vision and Pattern Recognition; Computers and Society; Databases; Distributed, Parallel, and Cluster Computing; Digital Libraries; Discrete Mathematics; Data Structures and Algorithms; Emerging Technologies; Formal Languages and Automata Theory; General Literature; Graphics; Computer Science and Game Theory; Human-Computer Interaction; Information Retrieval; Information Theory; Machine Learning; Logic in Computer Science; Multiagent Systems; Multimedia; Mathematical Software; Numerical Analysis; Neural and Evolutionary Computing; Networking and Internet Architecture; Other Computer Science; Operating Systems; Performance; Programming Languages; Robotics; Symbolic Computation; Sound; Software Engineering; Social and Information Networks; Systems and Control."
if __name__ == "__main__":
    max_nodes = 10
    begin_of_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    # data_dict = load_arxiv_raw_data(embeddings=True,graph=True,text_label=True,split_ids=True, raw_text=True)
    # node_index_ds = load_from_disk("datasets_local/with_node_index/arxiv")
    dp_list = []
    # graph = data_dict['graph']
    raw_text = torch.load('/data/sharefile/wei/dataset/ogbn_arxiv/raw_text.bin',map_location='cpu')
    ds = load_from_disk('datasets_local/with_node_index/arxiv_paper_summary_by_openai_gpt-oss-20b')
    train_list = np.where(np.array(ds['split_set']) =='train')[0].tolist()
    ds = ds.select(train_list[15000:22000])
    # raw_text = torch.load('datasets_local/pubmed_embedding_to_text.pt',map_location='cpu')
    # text_label = data_dict['text_label']
    # train_list = data_dict['split_ids']['train'].tolist()
    sampling_params = SamplingParams(temperature=0.5, top_p=1.0,max_tokens=10240)
    prompts = []
    gl_list = []
    total_length = len(ds)
    segment = 200 #total_length // 4
    rank = 0
    with tqdm(range(total_length)) as pbar:
        for i in pbar:
            
            # current_center_node_id = train_list[i]
            dp = {}
#             question = '''You are given the title and abstract of a scientific paper.

# Task: Write a concise summary (no more than 25 words) that captures the paper’s main topic, methods, and contributions.

# The summary should focus on what the paper is about — key ideas, technical approach, and findings — but should not include category labels or explicit classification terms.
# Input:
# {}

# Output:
# A 1-2 sentence summary (≤25 words) accurately representing the core content of the paper, written in a clear, factual academic tone.
#             '''.format(raw_text[i])
            question = '''{}Give the following candidate label list: {} Explain why a specific node should be classified as {} in clear, human language.
Reasoning order:

1. Analyze only the target node's own text first.

2. If that's not enough, use its 1-hop neighbors only (edges are undirected unless stated otherwise).

3. Write a short essay with separate paragraphs, then provide a neighbor evidence list with one line per neighbor you used.

4. Return only the block below. Do not add anything outside it.

The output block format is below:
<BEGIN_OUTPUT>
<FEATURE_PARAGRAPH>
{{Human-style paragraph discussing only the target node's own text. Highlight words/phrases/themes that connect to the target label.}}
</FEATURE_PARAGRAPH>

<NEIGHBOR_PARAGRAPH>
{{Only include if needed. A natural paragraph explaining how the 1-hop neighborhood shapes or confirms the label; mention neighbor ids in prose.}}
</NEIGHBOR_PARAGRAPH>

<NEIGHBOR_EVIDENCE>
{{One line per neighbor actually used (1-hop only). Follow this exact pattern:}}
ID {{neighbor_id}}: "{{short quote or keywords}}" — relevance: {{why it supports or refines the label}}; link strength: {{strong|moderate|weak}}
</NEIGHBOR_EVIDENCE>

<CONCLUSION_PARAGRAPH>
{{Short verdict tying it together in human language. End by explicitly naming the most fitting category.}}
</CONCLUSION_PARAGRAPH>

FINAL_CATEGORY: {{one category from the provided list}}
</END_OUTPUT>'''.format(ds[i]['question'].split('Based on its own and other')[0], candidate_str,ds[i]['answer'])
            json_for_template = [
                {
                    "role":"user",
                    "content": question
                 },
            ]
            prompts.append(tokenizer.apply_chat_template(json_for_template,tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False))    
            # import ipdb; ipdb.set_trace()
            # dp["original_ids"] = original_ids
            # dp['question'] = question
            # # dp['answer'] = answer
            # dp['text_label'] = text_label[current_center_node_id]
            # dp['truncated'] = truncated
            # dp['prompts'] = prompts[-1]
            # dp_list.append(dp)
            # if dp['text_label'] == 'General Literature':
            #     gl_list.append(i)
    llm = LLM(model='openai/gpt-oss-20b',  
            tensor_parallel_size=4,
            max_model_len=10240,
            gpu_memory_utilization=0.65,
            max_num_seqs = 1,
            )
    batch_size = 4
    outputs = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        dp_list.append({})
        dp_list[i]['answer'] = output.outputs[0].text
        split_list_str = output.outputs[0].text.split('assistantfinal')
        if len(split_list_str) > 1:
            dp_list[i]['pure_summary'] = split_list_str[1]
        else:
            dp_list[i]['pure_summary'] = "NO Result"
        dp_list[i]['finish_reason'] = output.outputs[0].finish_reason
        dp_list[i]['stop_reason'] = output.outputs[0].stop_reason
        dp_list[i]['orginal_ds_index'] = train_list[i]
        # if '</think>' not in output.outputs[0].text:
    dataset = Dataset.from_list(dp_list)
    dataset.save_to_disk('datasets_local/summarize_data/arxiv_sft_warmup_openai_gpt-oss-20b_15000_22000')
    