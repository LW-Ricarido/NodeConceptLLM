import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from ogb.graphproppred import DglGraphPropPredDataset
import periodictable
from tqdm import tqdm
import dgl
from dgl.data import TUDataset
from datasets import Dataset
import sys,os
sys.path.append(os.path.abspath(os.path.curdir)) 
from data_utils.prompt_str import *
atom_map = {
        0 : "Carbon",
        1 : "Nitrogen",
        2 : "Oxygen",
        3 : "Fluorine",
        4 : "Iodine",
        5 : "Chlorine",
        6 : "Bromine",
    }
correct_orders = [0, 1, 2, 3, 5, 6, 4]
def load_MUTAG():
    device = torch.device("cuda:0")
    PLM  = SentenceTransformer('all-mpnet-base-v2').to(device)
    MUTAG_dataset = TUDataset("MUTAG")
    map_index = {}
    over_all_idx = 0
    dp_list = []
   
    for graph, label in MUTAG_dataset:
        for i in range(graph.number_of_nodes()):
            current_node_feature = [0,0]
            current_node_feature[0] = graph.ndata['node_labels'][i][0].item()
            edge_idx = graph.edge_ids(graph.predecessors(i).tolist(),[i] * graph.predecessors(i).shape[0])
            if (graph.edata['edge_labels'][edge_idx] == 0).sum() > 0:
                current_node_feature[1] = 1
            edge_idx =  graph.edge_ids([i] * graph.successors(i).shape[0],graph.successors(i).tolist())
            if (graph.edata['edge_labels'][edge_idx] == 0).sum() > 0:
                current_node_feature[1] = 1
            feat_key = tuple(current_node_feature)
            if feat_key in map_index.keys():
                continue
            
            map_index[feat_key] = over_all_idx
            over_all_idx += 1
            
            node_description = "This atom is {}.".format(atom_map[current_node_feature[0]].lower())
            if current_node_feature[1] == 1:
                node_description += " This atom is part of an aromatic ring."
            dp = {
                "embedding": PLM.encode(node_description, convert_to_tensor=True).cpu(),
                "label": node_description,
                "element_name": atom_map[current_node_feature[0]],
                "aromatic_ring": current_node_feature[1]
            }
            dp_list.append(dp)
    print("overall number: ", over_all_idx)
    torch.save(dp_list,'datasets_local/mutag_text_to_embedding.bin')
    torch.save(map_index,'datasets_local/mutag_map_index_vector_to_text.bin')
    dataset_dp_list = []
    for i in range(len(map_index.keys())):
        curr_key = list(map_index.keys())[i]
        dp = {}
        dp['type'] = "unknown"
        dp['question'] = "This <|embedding_mask|> is embedding of an atom in a molecular. What the element type of this atom?"
        dp['answer'] =  "This atom is {}".format(dp_list[map_index[curr_key]]['element_name'].lower())
        dp['unaligned_input_embeds'] = [dp_list[map_index[curr_key]]['embedding']]
        dataset_dp_list.append(dp)
        
        dp = {}
        dp['type'] = "unknown"
        dp['question'] = 'This <|embedding_mask|> is embedding of an atom in a molecular. Is this atom part of an aromatic ring?'
        if dp_list[map_index[curr_key]]['aromatic_ring'] == 1:
            dp['answer'] = 'Indeed, this atom is incorporated into an aromatic system.'
        else:
            dp['answer'] = 'Nope, this atom is not involved in the aromatic ring system.'
        dp['unaligned_input_embeds'] = [dp_list[map_index[curr_key]]['embedding']]
        dataset_dp_list.append(dp)
    ds = Dataset.from_list(dataset_dp_list)
    ds.save_to_disk('datasets_local/json_texts_datasets/mutag_pretrain')

def load_MUTAG_graph_prediction_dataset():
    map_index = torch.load('datasets_local/mutag_map_index_vector_to_text.bin',map_location='cpu')
    text_to_embeddings = torch.load('datasets_local/mutag_text_to_embedding.bin',map_location='cpu')
    graph_datasets = TUDataset("MUTAG")
    dp_list = []
    with tqdm(range(len(graph_datasets))) as pbar:
        for i in pbar:
            dp = {}
            graph, current_label = graph_datasets[i]
            graph_description_str = begin_of_nodes_str
            graph_embeddings = []
            element_dict = {}
            for j in range(graph.num_nodes()):
                current_node_feature = [0,0]
                current_node_feature[0] = graph.ndata['node_labels'][j][0].item()
                edge_idx = graph.edge_ids(graph.predecessors(j).tolist(),[j] * graph.predecessors(j).shape[0])
                if (graph.edata['edge_labels'][edge_idx] == 0).sum() > 0:
                    current_node_feature[1] = 1
                edge_idx =  graph.edge_ids([j] * graph.successors(j).shape[0],graph.successors(j).tolist())
                if (graph.edata['edge_labels'][edge_idx] == 0).sum() > 0:
                    current_node_feature[1] = 1
                feat_key = tuple(current_node_feature)
                graph_description_str += embedding_mask_str + str(j)
                current_node_embedding = text_to_embeddings[map_index[feat_key]]['embedding']
                current_element = graph.ndata['node_labels'][j][0].item()
                if current_element not in element_dict.keys():
                    element_dict[current_element] = 0
                element_dict[current_element] += 1
                graph_embeddings.append(current_node_embedding)
            graph_description_str += end_of_nodes_str
            
            classification_question = " Is this molecule likely to exhibit mutagenic effects on Salmonella typhimurium? Additionally, identify the different types of atoms in the molecule, count the number of each type, and provide the results in ascending order of their atomic numbers."
            question = graph_description_str + classification_question
            if current_label.item() == 1:
                answer = "Yes."
            else:
                answer = "Nope."
            answer += " There are: "
            for order in correct_orders:
                if order in element_dict.keys():
                    atom_count = element_dict[order]
                    atom_name = atom_map[order]
                    if atom_count > 1:
                        answer += " {} {} atoms,".format(atom_count, atom_name)
                    else:
                        answer += " {} {} atom,".format(atom_count, atom_name)
            anwer = answer[:-1] + "."
            if i < int(0.8 * len(graph_datasets)):
                split_set = 'train'
            elif i < int(0.9 * len(graph_datasets)):
                split_set = 'valid'
            else:
                split_set = 'test'
            dp['question'] = question
            dp['answer'] = answer
            dp['unaligned_input_embeds'] = graph_embeddings
            dp['task_type'] = 'classification'
            dp['split_set'] = split_set
            dp['task_level'] = 'graph'
            dp['truncated'] =  False
            dp_list.append(dp)
    ds = Dataset.from_list(dp_list)
    ds.save_to_disk('datasets_local/json_texts_datasets/prediction_datasets/mutag_graph_embedding_QA_pure_nodes_with_element_type_count_cot')
            
if __name__ == "__main__":
    load_MUTAG_graph_prediction_dataset()
    