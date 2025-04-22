import torch
from ogb.nodeproppred import DglNodePropPredDataset
import pandas as pd
import os,dgl,sys
import requests
import json
from ogb.nodeproppred import DglNodePropPredDataset
from datasets import Dataset
sys.path.append(os.path.abspath(os.path.curdir))
from data_utils.load_raw_data import load_arxiv_raw_data
from tqdm.auto import tqdm
from data_utils.prompt_str import *
from data_utils.prepare_instruct_dataset import remove_reverse_edge
import time
if __name__ == "__main__":
    begin_of_time = time.time()
    model_name = "llama3.2"
    llama_url = "http://localhost:11434/api/chat"
    max_nodes = 10
    data_dict = load_arxiv_raw_data(embeddings=True,graph=True,text_label=True,split_ids=True, raw_text=True)
    dp_list = []
    graph = data_dict['graph']
    raw_text = data_dict['raw_text']
    text_label = data_dict['text_label']
    train_list = data_dict['split_ids']['train'].tolist()
    try:
        with tqdm(range(len(train_list))) as pbar:
            for i in pbar:
                current_center_node_id = train_list[i]
                dp = {}
                question = "This is a citation graph: "
                subgraph_node_list = graph.successors(current_center_node_id).tolist()
                if len(subgraph_node_list) > max_nodes:
                    subgraph_node_list = subgraph_node_list[:max_nodes]
                    truncated = True
                subgraph_node_list = [current_center_node_id] + subgraph_node_list
                ego_subgraph = dgl.node_subgraph(graph, subgraph_node_list)
                original_ids = []
                
                question += begin_of_nodes_str
                for j, node_id in enumerate(ego_subgraph.ndata[dgl.NID].tolist()):
                    question += "{}<begin_of_node_feature>{}<end_of_node_feature>".format(j,raw_text[node_id])
                    original_ids.append(node_id)
                question += end_of_nodes_str + " "
                
                question += begin_of_edges_str
                
                ego_subgraph = remove_reverse_edge(ego_subgraph)
                u, v = ego_subgraph.edges()[0].tolist(), ego_subgraph.edges()[1].tolist()
                for j in range(len(u)):
                    question += one_edge_str + str(u[j]) + ', ' + str(v[j])
                question += end_of_edges_str + ""
                question += "Given the following list of academic categories in computer science: Artificial Intelligence; Hardware Architecture; Computational Complexity; Computational Engineering, Finance, and Science; Computational Geometry; Computation and Language; Cryptography and Security; Computer Vision and Pattern Recognition; Computers and Society; Databases; Distributed, Parallel, and Cluster Computing; Digital Libraries; Discrete Mathematics; Data Structures and Algorithms; Emerging Technologies; Formal Languages and Automata Theory; General Literature; Graphics; Computer Science and Game Theory; Human-Computer Interaction; Information Retrieval; Information Theory; Machine Learning; Logic in Computer Science; Multiagent Systems; Multimedia; Mathematical Software; Numerical Analysis; Neural and Evolutionary Computing; Networking and Internet Architecture; Other Computer Science; Operating Systems; Performance; Programming Languages; Robotics; Symbolic Computation; Sound; Software Engineering; Social and Information Networks; Systems and Control. Based on its own and other nodes' node features and the graph structure, explain why node 0 should be classified under {}. Ensure the reasoning is logical and specific.".format(text_label[current_center_node_id])
                
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
                
                dp["original_ids"] = original_ids
                dp['question'] = question
                dp['answer'] = answer
                dp['text_label'] = text_label[current_center_node_id]
                dp['truncated'] = truncated
                import ipdb; ipdb.set_trace()
                dp_list.append(dp)
    except KeyboardInterrupt:
        print('=============get keyboard interrupt')
        pass 
    end_of_time = time.time()           
    dataset = Dataset.from_list(dp_list)
    dataset.save_to_disk("datasets_local/reasoning_data/arxiv_reasoning_based_on_{}".format(model_name))
    print('ollama 11 dp test time:{:.4f}'.format(end_of_time - begin_of_time))