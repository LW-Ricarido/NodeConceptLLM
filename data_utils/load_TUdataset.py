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

def load_MUTAG():
    device = torch.device("cuda:0")
    PLM  = SentenceTransformer('all-mpnet-base-v2').to(device)
    MUTAG_dataset = TUDataset("MUTAG")
    map_index = {}
    over_all_idx = 0
    dp_list = []
    atom_map = {
        0 : "Carbon",
        1 : "Nitrogen",
        2 : "Oxygen",
        3 : "Fluorine",
        4 : "Iodine",
        5 : "Chlorine",
        6 : "Bromine",
    }
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
if __name__ == "__main__":
    load_MUTAG()
    