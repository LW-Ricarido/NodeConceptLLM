import torch
# from ogb.nodeproppred import DglNodePropPredDataset
import pandas as pd
import os, sys
import requests
import json
# from ogb.nodeproppred import DglNodePropPredDataset
from datasets import Dataset, load_from_disk
sys.path.append(os.path.abspath(os.path.curdir))
from data_utils.load_raw_data import load_arxiv_raw_data, load_cora_raw_data,load_pubmed_raw_data
from tqdm.auto import tqdm
from data_utils.prompt_str import *
from data_utils.prepare_instruct_dataset import remove_reverse_edge
import time, re
import numpy as np
if __name__ == "__main__":
    begin_of_time = time.time()
    model_name = "llama3.2"
    llama_url = "http://localhost:11434/api/chat"
    max_nodes = 10
    data_dict = load_arxiv_raw_data(embeddings=True,graph=True,text_label=True,split_ids=True, raw_text=True)
    # data_dict = load_pubmed_raw_data(embeddings=True, graph=True, text_label=True, split_ids=True)
    # data_dict = load_cora_raw_data(embeddings=True, graph=True, text_label=True, split_ids=True)
    node_index_ds = load_from_disk('datasets_local/with_node_index/arxiv')
    dp_list = []
    # graph = data_dict['graph']
    raw_text = data_dict['raw_text']
    # raw_text = torch.load('datasets_local/pubmed_embedding_to_text.pt',weights_only=False)
    text_label = data_dict['text_label']
    # train_list = data_dict['split_ids']['train'].tolist()
    train_list = np.where(np.array(node_index_ds['split_set']) == 'train')[0].tolist()
    try:
        with tqdm(range(len(train_list))) as pbar:
            for i in pbar:
                current_center_node_id = train_list[i]
                dp = node_index_ds[current_center_node_id]
                question = "This is a citation graph: "
                
                question += begin_of_nodes_str
                for j, node_id in enumerate(node_index_ds[current_center_node_id]['original_node_idx']):
                    # if (raw_text[node_id]['input_embedding'] == torch.tensor(dp['unaligned_input_embeds'][j])).sum() ==raw_text[node_id]['input_embedding'].shape[0]: 
                    #     # question += "{}<begin_of_node_feature>{}<end_of_node_feature>".format(j,raw_text[node_id])
                    #     question += "{}<begin_of_node_feature>{}<end_of_node_feature>".format(j,raw_text[node_id]['label'])
                    # else:
                    #     print('=======================embedding error===================')
                    #     import ipdb; ipdb.set_trace()
                    question += "{}<begin_of_node_feature>{}<end_of_node_feature>".format(j,raw_text[node_id])
                question += end_of_nodes_str + " "
                
                edge_str = re.search(r"<\|begin_of_edges\|>(.*?)<\|end_of_edges\|>", node_index_ds[current_center_node_id]['question']).group(0)
                question += edge_str +". "
                
                assert text_label[current_center_node_id] == dp['answer'], "node index does not much"

                # question += "Given the following list of academic categories in computer science: Artificial Intelligence; Hardware Architecture; Computational Complexity; Computational Engineering, Finance, and Science; Computational Geometry; Computation and Language; Cryptography and Security; Computer Vision and Pattern Recognition; Computers and Society; Databases; Distributed, Parallel, and Cluster Computing; Digital Libraries; Discrete Mathematics; Data Structures and Algorithms; Emerging Technologies; Formal Languages and Automata Theory; General Literature; Graphics; Computer Science and Game Theory; Human-Computer Interaction; Information Retrieval; Information Theory; Machine Learning; Logic in Computer Science; Multiagent Systems; Multimedia; Mathematical Software; Numerical Analysis; Neural and Evolutionary Computing; Networking and Internet Architecture; Other Computer Science; Operating Systems; Performance; Programming Languages; Robotics; Symbolic Computation; Sound; Software Engineering; Social and Information Networks; Systems and Control. Based on its own and other nodes' node features and the graph structure, explain why node 0 should be classified under {}. Ensure the reasoning is logical and specific.".format(text_label[current_center_node_id])
     
                # question += "Given the following list of academic categories in computer science: Case Based; Genetic Algorithms; Neural Networks; Probabilistic Methods; Reinforcement Learning; Rule Learning; Theory. Based on its own and other nodes' node features and the graph structure, explain why node 0 should be classified under {}. Ensure the reasoning is logical and specific.".format(text_label[current_center_node_id])
                
                question += "Given the following list of categories: Diabetes Mellitus, Experimental; Diabetes Mellitus Type 1; Diabetes Mellitus Type 2. Based on its own and other nodes' node features and the graph structure, explain why node 0 should be classified under {}. Ensure the reasoning is logical and specific.".format(text_label[current_center_node_id])
                
                data_for_ollama = {
                    "model": model_name,
                    "messages":[
                        {
                            "role":"user",
                            "content":question
                        }
                    ],
                    "stream": False,
                    "options":{
                        "seed": 3047,
                    }
                }
                headers = {
                    'Content-Type': 'application/json'
                }
                
                response = requests.post(llama_url, headers=headers, json=data_for_ollama)
                answer = response.json()['message']['content']
                
                dp['answer'] = answer
                dp['text_label'] = text_label[current_center_node_id]
                # import ipdb; ipdb.set_trace()
                dp_list.append(dp)
    except KeyboardInterrupt:
        print('=============get keyboard interrupt')
        pass 
    end_of_time = time.time()           
    dataset = Dataset.from_list(dp_list)
    dataset.save_to_disk("datasets_local/with_node_index/pubmed_reasoning_based_on_{}".format(model_name))
    print('ollama 11 dp test time:{:.4f}'.format(end_of_time - begin_of_time))