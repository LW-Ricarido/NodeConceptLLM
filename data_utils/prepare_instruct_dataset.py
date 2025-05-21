import dgl
import torch
import json, os
from sentence_transformers import SentenceTransformer
import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
# from data_utils.prepare_dataset import cora_text_label_list, arxiv_category_mapping_dict
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.graphproppred  import DglGraphPropPredDataset
import sys
sys.path.append(os.path.abspath(os.path.curdir))
from data_utils.prompt_str import embedding_mask_str, begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str
from data_utils.load_raw_data import load_arxiv_raw_data, arxiv_category_mapping_dict, load_cora_raw_data, cora_text_label_list, load_pubmed_raw_data, pubmed_text_label_list
import periodictable
import re

def remove_reverse_edge(reverse_graph:dgl.DGLGraph):
    for_remove_edge = []
    us,vs = reverse_graph.edges()
    us,vs = us.tolist(), vs.tolist()
    remove_us, remove_vs = [], []
    for i in range(len(us)):
        if us[i] < vs[i]:
            remove_us.append(vs[i])
            remove_vs.append(us[i])
    try:
        for_remove_edge = reverse_graph.edge_ids(remove_us,remove_vs).tolist()    
        removed_graph = dgl.remove_edges(reverse_graph,for_remove_edge)
    except Exception as e:
        print(e)
        print(remove_us)
        print(remove_vs)
        print(for_remove_edge)
        print(us)
        print(vs)
        print(reverse_graph.edges())
        import ipdb; ipdb.set_trace()

    return removed_graph

def mask_out_question_label(labels, start_header_id, assistant_id, end_header_id, eot_id):
    prev_id = 0
    my_cnter = 0
    
    while my_cnter < len(labels):
        if labels[my_cnter] == start_header_id and labels[my_cnter+1] == assistant_id and labels[my_cnter+2] == end_header_id:
            labels[prev_id:my_cnter + 3] = [-100]*len(labels[prev_id:my_cnter + 3])
            prev_id = labels.index(eot_id,my_cnter)+1
            my_cnter = prev_id
        else:
            my_cnter += 1
    return labels

def pretrain_prepare_dp(json_for_template, embedding,tokenizer, embedding_mask_id, start_header_id, assistant_id, end_header_id, eot_id):
        input_text = tokenizer.apply_chat_template(json_for_template,tokenize=False)
        dp = {}
        dp['input_ids'] = tokenizer.encode(input_text, add_special_tokens=False)
        dp['labels'] = dp['input_ids'].copy()
        dp['embedding_positions'] = [ i for i,x in enumerate(dp['input_ids']) if x == embedding_mask_id]
        dp['input_ids'].index(embedding_mask_id)
        dp['labels'] = mask_out_question_label(dp['labels'], start_header_id, assistant_id, end_header_id, eot_id)
        dp['unaligned_input_embeds'] = [embedding]
        return dp

def prepare_graph_QA_dp(embeddings, input_ids, tokenizer, task_type, is_truncated,split_set, embedding_mask_id, start_header_id, assistant_id, end_header_id, eot_id,task_level='node'):
        dp = {}
        dp['input_ids'] = input_ids
        dp['labels'] = dp['input_ids'].copy()
        dp['embedding_positions'] = [ i for i,x in enumerate(dp['input_ids']) if x == embedding_mask_id]
        dp['input_ids'].index(tokenizer.convert_tokens_to_ids(embedding_mask_str))
        dp['labels'] = mask_out_question_label(dp['labels'], start_header_id, assistant_id, end_header_id, eot_id)
        dp['unaligned_input_embeds'] = embeddings
        dp['task_type'] = task_type
        dp['truncated'] = is_truncated
        dp['split_set'] = split_set
        dp['task_level'] = task_level
        return dp

def prepare_pretrain_dataset(tokenizer,tokenizer_name,dataset_name):
    # Load the dataset
    if 'arxiv' in dataset_name:
        raw_texts = torch.load('your_local_root_path/ogbn_arxiv/raw_text.bin',map_location='cpu')
        text_embeddings = torch.load('your_local_root_path/ogbn_arxiv/raw_tensor.pt',map_location='cpu')
        data = DglNodePropPredDataset('ogbn-arxiv', root='your_local_root_path')
        graph, label = data[0]
        test_list = data.get_idx_split()['test']
    elif 'cora' in dataset_name:
        raw_data = torch.load('datasets_local/cora_embedding_to_text.pt',map_location='cpu')
        raw_texts = [data['label'] for data in raw_data]
        text_embeddings = torch.stack([torch.tensor(data['input_embedding']) for data in raw_data])
        test_list = [i for i in range(len(raw_data))]
    elif 'pubmed' in dataset_name:
        raw_data = torch.load('datasets_local/pubmed_embedding_to_text.pt',map_location='cpu')
        raw_texts = [data['label'] for data in raw_data]
        text_embeddings = torch.stack([torch.tensor(data['input_embedding']) for data in raw_data])
        test_list = [i for i in range(len(raw_data))]
    else:
        raise NotImplementedError
    dp_list = []
    start_header_id, assistant_id, end_header_id,eot_id, embedding_mask_id = tokenizer.convert_tokens_to_ids(['<|start_header_id|>','assistant','<|end_header_id|>','<|eot_id|>',embedding_mask_str])
    label2category = pd.read_csv(os.path.join('your_local_root_path/','ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz'), compression='gzip')

    
    with tqdm(range(len(raw_texts))) as pbar:
        for idx in pbar:
            title, abstract = raw_texts[idx].split('\n')
            embedding = text_embeddings[idx]
            ### title question
            json_for_template = [
                {
                    "role":"user",
                    "content":"This {} is embedding of a paper. What's the title of this paper?".format(embedding_mask_str)
                 },
                {
                    "role":"assistant",
                    "content":"The title of this paper is: {}.".format(title.replace('Title: ','',1))
                },
                # {
                #     "role":"user",
                #     "content": "This is a test question."
                # },
                # {
                #     "role":"assistant",
                #     'content': "This is a test answer."
                # }
                
            ]
            
            dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
            dp_list.append(dp)
            ### abstract question
            json_for_template = [
                {
                    "role":"user",
                    "content":"This {} is embedding of a paper. What's the abstract of this paper?".format(embedding_mask_str)
                 },
                {
                    "role":"assistant",
                    "content":"The abstract of this paper is: {}.".format(abstract.replace('Abstract: ','',1))
                }
            ]
            dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
            dp_list.append(dp)
            if idx not in test_list:
                ### category question
                json_for_template = [
                    {
                        "role":"user",
                        "content":"This {} is embedding of a paper. What's the category of this paper?".format(embedding_mask_str)
                     },
                    {
                        "role":"assistant",
                        "content":"The category of this paper is: {}.".format(arxiv_category_mapping_dict[label2category['arxiv category'][label[idx].item()]])
                    }
                ]
                dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
        
    dataset = Dataset.from_list(dp_list)
    if not os.path.exists(os.path.join('datasets_local',tokenizer_name)):
        os.makedirs(os.path.join('datasets_local',tokenizer_name))
    dataset.save_to_disk(os.path.join('datasets_local',tokenizer_name,dataset_name))
    
def prepare_molhiv_pretrain_dataset(tokenizer, tokenizer_name, dataset_name):
    text_to_embedding = torch.load('your_local_root_path/ogbg_molhiv/text_to_embedding.bin', map_location='cpu')
    map_idx = torch.load('your_local_root_path/ogbg_molhiv/map_index_vector_to_text.bin', map_location='cpu')
    dp_list = []
    start_header_id, assistant_id, end_header_id,eot_id, embedding_mask_id = tokenizer.convert_tokens_to_ids(['<|start_header_id|>','assistant','<|end_header_id|>','<|eot_id|>',embedding_mask_str])
    label2category = pd.read_csv(os.path.join('your_local_root_path/','ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz'), compression='gzip')
    with tqdm(range(len(map_idx.keys()))) as pbar:
        key_list = list(map_idx.keys())
        for i in pbar:
            curr_key = key_list[i]
            embedding = text_to_embedding[i]['embedding']
            if len(curr_key) == 9:
                ### element task
                json_for_template = [
                    {
                        "role": "user",
                        "content": "This {} is embedding of an atom in a molecular. What the element type of this atom?".format(embedding_mask_str)
                    },
                    {
                        "role": "assistant",
                        "content": "This atom is {}.".format(text_to_embedding[i]['element_name'])
                    }
                ]
                dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
                
                ### chirality task 
                if np.random.rand() < 0.1:
                    json_for_template = [
                        {
                            "role": "user",
                            "content": "This {} is embedding of an atom in a molecular. What the chirality type of this atom?".format(embedding_mask_str)
                        },
                        {
                            "role": "assistant",
                            "content": "The chirality type of this atom is {}.".format(text_to_embedding[i]['chirality'])
                        }
                    ]
                    dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                    dp_list.append(dp)
                
                ### formal_charge task
                
                json_for_template = [
                    {
                        "role": "user",
                        "content": "This {} is embedding of an atom in a molecular. What the formal charge of this atom?".format(embedding_mask_str)
                    },
                    {
                        "role": "assistant",
                        "content": "The formal charge of this atom is {}.".format(text_to_embedding[i]['formal_charge'])
                    }
                ]
                dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
                
                ### number of h task
                
                json_for_template = [
                    {
                        "role": "user",
                        "content": "This {} is embedding of an atom in a molecular. How many hydrogen atoms does this atom bond with?".format(embedding_mask_str)
                    },
                    {
                        "role": "assistant",
                        "content": "The atom bonds with {} hydrogen atoms.".format(text_to_embedding[i]['number_of_h'])
                    }
                ]
                dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
                
                ### number_of_radical_e task
                
                json_for_template = [
                    {
                        "role": "user",
                        "content": "This {} is embedding of an atom in a molecular. How many radical electrons does this atom have?".format(embedding_mask_str)
                    },
                    {
                        "role": "assistant",
                        "content": "The atom bonds with {} hydrogen atoms.".format(text_to_embedding[i]['number_of_radical_e'])
                    }
                ]
                dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
                
                ### hybridization task
                json_for_template = [
                    {
                        "role": "user",
                        "content": "This {} is embedding of an atom in a molecular. What is the hybridization type of this atom?".format(embedding_mask_str)
                    },
                    {
                        "role": "assistant",
                        "content": "The atom bonds with {} hydrogen atoms.".format(text_to_embedding[i]['hybridization'])
                    }
                ]
                ### aromatic_ring task
                if text_to_embedding[i]['aromatic_ring'] == 1:
                    aromatic_answer = 'Indeed, this atom is incorporated into an aromatic system.'
                else:
                    aromatic_answer = "Nope, this atom is not involved in the aromatic ring system."
                json_for_template = [
                    {
                        "role": "user",
                        "content": "This {} is embedding of an atom in a molecular. Is this atom part of an aromatic ring?".format(embedding_mask_str)
                    },
                    {
                        "role": "assistant",
                        "content": aromatic_answer
                    }
                ]
                dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
                
                ### ring task
                if text_to_embedding[i]['ring'] == 1:
                    ring_answer = "Yes, it plays a structural role in maintaining the ring's integrity."
                else:
                    ring_answer = "No, this atom is positioned outside of any ring structure."
                json_for_template = [
                    {
                        "role": "user",
                        "content": "This {} is embedding of an atom in a molecular. Does this atom belong to a ring structure?".format(embedding_mask_str)
                    },
                    {
                        "role": "assistant",
                        "content": ring_answer
                    }
                ]
                dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
                
                ### degree task
                if np.random.rand() < 0.3:
                    json_for_template = [
                        {
                            "role": "user",
                            "content": "This {} is embedding of an atom in a molecular. What is the valency of this atom?".format(embedding_mask_str)
                        },
                        {
                            "role": "assistant",
                            "content": "It is {}.".format(text_to_embedding[i]['degree'])
                        }
                    ]
                    dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                    dp_list.append(dp)
                
            elif len(curr_key) == 3:
                ### bond type task
                json_for_template = [
                    {
                        "role": "user",
                        "content": "This {} is embedding of a chemical bond in a molecular. What is type of this bond?".format(embedding_mask_str)
                    },
                    {
                        "role": "assistant",
                        "content": "The type of this bond is {}.".format(text_to_embedding[i]['bond_type'])
                    }
                ]
                dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
                ### bond_stereochemistry task
                json_for_template = [
                    {
                        "role": "user",
                        "content": "This {} is embedding of a chemical bond in a molecular. What type of bond stereochemistry does this bond have?".format(embedding_mask_str)
                    },
                    {
                        "role": "assistant",
                        "content": "The bond exhibits {} stereochemistry.".format(text_to_embedding[i]['bond_stereochemistry'])
                    }
                ]
                dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
                ###  conjugated task
                if text_to_embedding[i]['conjugated'] == 1:
                    conjugated_answer = "This edge is conjugated."
                else:
                    conjugated_answer = "This bond is isolated."
                json_for_template = [
                    {
                        "role": "user",
                        "content": "This {} is embedding of a chemical bond in a molecular. What is the conjugation status of this bond?".format(embedding_mask_str)
                    },
                    {
                        "role": "assistant",
                        "content": conjugated_answer
                    }
                ]
                dp = pretrain_prepare_dp(json_for_template,embedding, tokenizer, embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
        dataset = Dataset.from_list(dp_list)
    if not os.path.exists(os.path.join('datasets_local',tokenizer_name)):
        os.makedirs(os.path.join('datasets_local',tokenizer_name))
    dataset.save_to_disk(os.path.join('datasets_local',tokenizer_name,dataset_name))
    
def prepare_embedding_prediction_dataset(tokenizer, tokenizer_name, dataset_name):
    raw_data = load_arxiv_raw_data(embeddings=True, graph=True, text_label=True, split_ids=True)
    unaligned_embeddings, graph, text_label, split_ids = raw_data['embeddings'], raw_data['graph'], raw_data['text_label'], raw_data['split_ids']
    dp_list = []
    start_header_id, assistant_id, end_header_id,eot_id, embedding_mask_id = tokenizer.convert_tokens_to_ids(['<|start_header_id|>','assistant','<|end_header_id|>','<|eot_id|>',embedding_mask_str])
    label2category = pd.read_csv(os.path.join('your_local_root_path/','ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz'), compression='gzip')
    categories_string = "Please classify the paper into one of the following categories:"
    for key in arxiv_category_mapping_dict.keys():
        categories_string += arxiv_category_mapping_dict[key] + '; '
    categories_string = categories_string[:-2] + '.'
    def prepare_dp(embeddings, input_ids, tokenizer,text_label,split_set):
        dp = {}
        dp['unaligned_input_embeds'] = embeddings
        dp['input_ids'] = input_ids
        dp['labels'] = dp['input_ids'].copy()
        dp['labels'] = mask_out_question_label(dp['labels'], start_header_id, assistant_id, end_header_id, eot_id)
        
        dp['embedding_positions'] = [ i for i,x in enumerate(dp['input_ids']) if x == embedding_mask_id]
        dp['input_ids'].index(tokenizer.convert_tokens_to_ids(embedding_mask_str))
        dp['text_label'] = text_label
        dp['split_set'] = split_set
        return dp
    with tqdm(range(graph.num_nodes())) as pbar:
        for idx in pbar:
            dp = {}
            embedding = unaligned_embeddings[idx]
            json_for_template = [
                {
                    "role":"user",
                    "content":"This {} is embedding of a paper. ".format(embedding_mask_str) + categories_string
                 },
                {
                    "role":"assistant",
                    'content': text_label[idx]
                }
            ]
           
            dp['unaligned_inputs_embeds'] = [embedding]
            input_text = tokenizer.apply_chat_template(json_for_template,tokenize=False) 
            input_ids = tokenizer.encode(input_text,add_special_tokens=False)
            dp = prepare_dp([embedding],input_ids,tokenizer,text_label[idx],split_set='train' if idx in split_ids['train'] else 'valid' if idx in split_ids['valid'] else 'test')
            dp_list.append(dp)
    dataset = Dataset.from_list(dp_list)
    if not os.path.exists(os.path.join('datasets_local',tokenizer_name)):
        os.makedirs(os.path.join('datasets_local',tokenizer_name))
    dataset.save_to_disk(os.path.join('datasets_local',tokenizer_name,dataset_name))

def prepare_graph_embedding_QA_dataset(tokenizer, tokenizer_name, dataset_name, max_nodes=10, edge_sort_type='random'):
    if 'arxiv' in dataset_name:
        raw_data = load_arxiv_raw_data(embeddings=True, graph=True, text_label=True, split_ids=True)
    elif 'cora' in dataset_name:
        raw_data = load_cora_raw_data(embeddings=True, graph=True, text_label=True,split_ids=True)
    elif 'pubmed' in dataset_name:
        raw_data = load_pubmed_raw_data(embeddings=True, graph=True, text_label=True,split_ids=True)
    else:
        raise NotImplementedError
    unaligned_embeddings, graph, text_label, split_ids = raw_data['embeddings'], raw_data['graph'], raw_data['text_label'], raw_data['split_ids']
    dp_list = []
    start_header_id, assistant_id, end_header_id,eot_id, embedding_mask_id = tokenizer.convert_tokens_to_ids(['<|start_header_id|>','assistant','<|end_header_id|>','<|eot_id|>',embedding_mask_str])
    
    with tqdm(range(graph.num_nodes())) as pbar:
        for i in pbar:
            subgraph_embeddings = []
            truncated = False
            subgraph_node_list = graph.successors(i).tolist()
            if len(subgraph_node_list) > max_nodes:
                subgraph_node_list = subgraph_node_list[:max_nodes]
                truncated = True
            subgraph_node_list = [i] + subgraph_node_list
            ego_subgraph = dgl.node_subgraph(graph, subgraph_node_list)
            graph_description_str = begin_of_nodes_str
            for j, node_id in enumerate(ego_subgraph.ndata[dgl.NID].tolist()):
                graph_description_str += embedding_mask_str + str(j)
                subgraph_embeddings.append(unaligned_embeddings[node_id])
            graph_description_str += end_of_nodes_str
            graph_description_str += begin_of_edges_str

            if edge_sort_type == 'random':
                ego_subgraph = remove_reverse_edge(ego_subgraph)
                u,v = ego_subgraph.edges()[0].tolist(), ego_subgraph.edges()[1].tolist()
                random_list = torch.randperm(len(u)).tolist()
                for j in random_list:
                    graph_description_str += one_edge_str + str(u[j]) +', ' + str(v[j])
            else:
                raise NotImplementedError('edge_sort_type {} is not implemented'.format(edge_sort_type))
            graph_description_str += end_of_edges_str
            
            classification_question = "Please classify the node 0 into one of the following categories:"
            if 'arxiv' in dataset_name :
                for key in arxiv_category_mapping_dict.keys():
                    classification_question += arxiv_category_mapping_dict[key] + '; '
            elif 'cora' in dataset_name :
                for key in cora_text_label_list:
                    classification_question += key + '; '
            elif 'pubmed' in dataset_name:
                for key in pubmed_text_label_list:
                    classification_question += key + '; '
            classification_question = classification_question[:-2] + '.'
            # import ipdb; ipdb.set_trace()
            
            split_set = 'train' if i in split_ids['train'] else 'valid' if i in split_ids['valid'] else 'test'
            # dp = {}
            # # dp['unaligned_inputs_embeds'] = subgraph_embeddings
            # # dp['truncate'] = truncated
            prediction_json_for_template = [
                {
                    "role":"user",
                    "content": "This is a graph: " + graph_description_str + " " + classification_question
                },
                {
                    'role':'assistant',
                    'content': text_label[i]
                }
            ]
            input_text = tokenizer.apply_chat_template(prediction_json_for_template,tokenize=False)
            input_ids = tokenizer.encode(input_text,add_special_tokens=False)
            dp = prepare_graph_QA_dp(subgraph_embeddings, input_ids, tokenizer,'classification',truncated,split_set,embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
            dp_list.append(dp)
                
            # 0.6 prob ask a node count question
            if np.random.rand() < 0.6:
                node_count_json_for_template = [
                    {
                        "role":"user",
                        "content": "This is a graph: " + graph_description_str + " How many nodes are there in this graph?"
                    },
                    {
                        'role':'assistant',
                        'content': str(len(subgraph_node_list))
                    }
                ]
                input_text = tokenizer.apply_chat_template(node_count_json_for_template,tokenize=False)
                input_ids = tokenizer.encode(input_text,add_special_tokens=False)
                dp = prepare_graph_QA_dp(subgraph_embeddings, input_ids, tokenizer,'node_count',truncated,split_set,embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
                
            # 0.2 prob to ask a edge count question
            if np.random.rand() < 0.2:
                edge_count_json_for_template = [
                    {
                        "role":"user",
                        "content": "This is a graph: " + graph_description_str + " How many edges are there in this graph?"
                    },
                    {
                        'role':'assistant',
                        'content': str(ego_subgraph.num_edges())
                    }
                ]
                input_text = tokenizer.apply_chat_template(edge_count_json_for_template,tokenize=False)
                input_ids = tokenizer.encode(input_text,add_special_tokens=False)
                dp = prepare_graph_QA_dp(subgraph_embeddings, input_ids, tokenizer,'edge_count',truncated,split_set,embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)    
            # 0.7 prob to ask a edge existence question
            if np.random.rand() < 0.7:
                edge_check_ids = np.random.choice(len(subgraph_node_list),2)
                if edge_check_ids[0] < edge_check_ids[1]:
                    u, v = edge_check_ids[0], edge_check_ids[1]
                else:
                    u, v = edge_check_ids[1], edge_check_ids[0]
                edge_existence_json_for_template = [
                    {
                        "role":"user",
                        "content": "This is a graph: " + graph_description_str + " Does the edge between node {} and node {} exist?".format(u,v)
                    },
                    {
                        'role':'assistant',
                        'content': "Yes" if ego_subgraph.has_edges_between([u],[v]) else "No"
                    }
                ]
                input_text = tokenizer.apply_chat_template(edge_existence_json_for_template,tokenize=False)
                input_ids = tokenizer.encode(input_text,add_special_tokens=False)
                dp = prepare_graph_QA_dp(subgraph_embeddings, input_ids, tokenizer,'edge_existence',truncated,split_set,embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id)
                dp_list.append(dp)
    print("length of dp_list: ",len(dp_list))
    # split_datasets = []
    # single_size = 5000
    # for i in range(0,len(dp_list),single_size):
    #     print('current i: ',i)
    #     current_dataset = Dataset.from_list(dp_list[i:i+single_size])
    #     current_dataset.save_to_disk(os.path.join('datasets_local',tokenizer_name,dataset_name+'/split_{}'.format(i)))
    #     split_datasets.append(current_dataset)
        # split_datasets.append(Dataset.from_list(dp_list[i:i+single_size]))
    # dataset = concatenate_datasets(split_datasets)
    dataset = Dataset.from_list(dp_list)
    dataset.save_to_disk(os.path.join('datasets_local',tokenizer_name,dataset_name))
            
        
def prepare_molhiv_graph_embedding_QA_dataset(tokenizer, tokenizer_name, dataset_name, edge_sort_type='random'):
    map_index = torch.load('your_local_root_path/ogbg_molhiv/map_index_vector_to_text.bin',map_location='cpu')
    text_to_embeddings = torch.load('your_local_root_path/ogbg_molhiv/text_to_embedding.bin', map_location='cpu')
    graph_datasets = DglGraphPropPredDataset(name='ogbg-molhiv',root='your_local_root_path')
    start_header_id, assistant_id, end_header_id,eot_id, embedding_mask_id = tokenizer.convert_tokens_to_ids(['<|start_header_id|>','assistant','<|end_header_id|>','<|eot_id|>',embedding_mask_str])

    split_ids = graph_datasets.get_idx_split()
    dp_list = []
    with tqdm(range(len(graph_datasets))) as pbar:
        for i in pbar:
            graph, current_label = graph_datasets[i]
            graph_description_str = begin_of_nodes_str
            graph_embeddings = []
            element_dict = {}
            for j in range(graph.num_nodes()):
                graph_description_str += embedding_mask_str + str(j)
                current_node_embedding = text_to_embeddings[map_index[tuple(graph.ndata['feat'][j].tolist())]]['embedding']
                current_element = graph.ndata['feat'][j][0].item()
                if current_element not in element_dict.keys():
                    element_dict[current_element] = 0
                element_dict[current_element] += 1
                graph_embeddings.append(current_node_embedding)
            graph_description_str += end_of_nodes_str
            
            # graph_description_str += begin_of_edges_str
            # if edge_sort_type == 'random':
            #     graph = remove_reverse_edge(graph)
            #     u,v = graph.edges()[0].tolist(), graph.edges()[1].tolist()
            #     random_list = torch.randperm(len(u)).tolist()
            #     for j in random_list:
            #         graph_description_str += one_edge_str + embedding_mask_str + str(u[j]) +', ' + str(v[j])
            #         edge_index = graph.edge_ids([u[j]],[v[j]])
            #         current_edge_embedding = text_to_embeddings[map_index[tuple(graph.edata['feat'][edge_index][0].tolist())]]['embedding']
            #         graph_embeddings.append(current_edge_embedding)
            # else:
            #     raise NotImplementedError('edge_sort_type {} is not implemented'.format(edge_sort_type))
            # graph_description_str += end_of_edges_str
            
            classification_question = 'Does the molecule have the ability to inhibit HIV virus replication? Additionally, identify the different types of atoms in the molecule, count the number of each type, and provide the results in ascending order of their atomic numbers.'
            if current_label.item() == 1:
                answer = 'Yes.'#, this compound exhibits antiviral properties against HIV.'
            else:
                answer = 'Nope.'# it fails to exhibit any antiviral effects on HIV.'
            answer += " There are:"
            element_ids = list(element_dict.keys())
            element_ids.sort()
            for element_id in element_ids:
                atom_count = element_dict[element_id]
                atom_name = periodictable.elements[element_id].name
                if atom_count > 1:
                    answer += " {} {} atoms,".format(atom_count,atom_name)
                else:
                    answer += " {} {} atom,".format(atom_count, atom_name)
            answer = answer[:-1] + "."
            split_set = 'train' if i in split_ids['train'] else 'valid' if i in split_ids['valid'] else 'test'
            prediction_json_for_template = [
                {
                    "role":'user',
                    "content": "This is a molecular graph: " + graph_description_str + " " + classification_question
                },
                {
                    'role': "assistant",
                    "content": answer
                }
            ]
            input_text = tokenizer.apply_chat_template(prediction_json_for_template, tokenize=False)
            input_ids = tokenizer.encode(input_text,add_special_tokens=False)
            dp = prepare_graph_QA_dp(graph_embeddings, input_ids, tokenizer,'classification',False,split_set,embedding_mask_id,start_header_id,assistant_id, end_header_id, eot_id,'graph')
            dp_list.append(dp)
    print("length of dp_list: ",len(dp_list))
    dataset = Dataset.from_list(dp_list)
    dataset.save_to_disk(os.path.join('datasets_local',tokenizer_name,dataset_name))

def prepare_arxiv_dpo_graph_QA_datasets(dataset_name):
    data_dict = load_arxiv_raw_data(embeddings=True, graph=True, text_label=True, split_ids=True)
    reasoning_ds = load_from_disk('datasets_local/reasoning_data/arxiv_reasoning_based_on_Qwen_QwQ-32B-AWQ_combined_2')
    dp_list = []
    with tqdm(range(len(reasoning_ds))) as pbar:
        for i in pbar:
            if '</think>' not in reasoning_ds[i]['answer']:
                continue
            dp = {}
            question = reasoning_ds[i]['question']
            reason_str, answer_str = reasoning_ds[i]['answer'].split('</think>')
            question = re.sub(r"<begin_of_node_feature>.*?<end_of_node_feature>",embedding_mask_str,question,flags=re.DOTALL)
            question = re.sub(r"Based on its.*?logical and specific.","Please classify the node 0 into one of these categories.",question,flags=re.DOTALL)
            thinking_answer = "Reasoning:\n" + reason_str + "\n\nAnswer: " + reasoning_ds[i]['text_label'] + '.'
            qwq_answer = "Reasoning:\n" + answer_str + "\n\nAnswer: " + reasoning_ds[i]['text_label'] + '.'

            dp['question'] = question
            dp['thinking_answer'] = thinking_answer
            dp['qwq_answer'] = qwq_answer
            dp['unaligned_inputs_embeds'] = [data_dict['embeddings'][j] for j in reasoning_ds[i]['original_ids']]
            dp['truncated'] = reasoning_ds[i]['truncated']
            dp['split_set'] = 'train'
            dp['text_label'] = reasoning_ds[i]['text_label']
            dp_list.append(dp)
    dataset = Dataset.from_list(dp_list)
    dataset.save_to_disk(os.path.join('datasets_local',dataset_name))
            

def prepare_link_prediction_dataset(dataset_name):
    print(dataset_name)
    np.random.seed(42)
    if dataset_name == 'pubmed':
        raw_data = load_pubmed_raw_data(embeddings=True, graph=True, text_label=True,split_ids=True)      
    elif dataset_name == 'cora':
        raw_data = load_cora_raw_data(embeddings=True, graph=True, text_label=True,split_ids=True)
    # 85: 5 : 10
    unaligned_embeddings, graph, text_label, split_ids = raw_data['embeddings'], raw_data['graph'], raw_data['text_label'], raw_data['split_ids']
    # graph = remove_reverse_edge(graph)
    edge_ids = graph.edges()
    max_nodes = 5
    dp_lists = []
    def prepare_graph_by_edge(og,graph_u_center, graph_v_center):
        truncated = False
        subgraph_embeddings  = []
        u_subgraph_node_list = graph.successors(graph_u_center).tolist()   
        if graph_v_center in u_subgraph_node_list:
            u_subgraph_node_list.remove(graph_v_center)
        v_subgraph_node_list = graph.successors(graph_v_center).tolist()
        if graph_u_center in v_subgraph_node_list:
            v_subgraph_node_list.remove(graph_u_center)
        if len(u_subgraph_node_list) > max_nodes:
            u_subgraph_node_list = u_subgraph_node_list[:max_nodes]
            truncated = True
        if len(v_subgraph_node_list) > max_nodes:
            v_subgraph_node_list = v_subgraph_node_list[:max_nodes]
            truncated = True
        u_subgraph_node_list = [graph_u_center] + u_subgraph_node_list
        v_subgraph_node_list = [graph_v_center] + v_subgraph_node_list
        u_size = len(u_subgraph_node_list)
        u_ego_subgraph = dgl.node_subgraph(og, u_subgraph_node_list)
        v_ego_subgraph = dgl.node_subgraph(og, v_subgraph_node_list)
        graph_description_str = begin_of_nodes_str
        
        for j, node_id in enumerate(u_ego_subgraph.ndata[dgl.NID].tolist()):
            graph_description_str += embedding_mask_str + str(j)
            subgraph_embeddings.append(unaligned_embeddings[node_id])
        for j, node_id in enumerate(v_ego_subgraph.ndata[dgl.NID].tolist()):
            graph_description_str += embedding_mask_str + str(j + u_size)
            subgraph_embeddings.append(unaligned_embeddings[node_id])
        graph_description_str += end_of_nodes_str
        u_ego_subgraph = remove_reverse_edge(u_ego_subgraph)
        v_ego_subgraph = remove_reverse_edge(v_ego_subgraph)
        u_graph_us, u_graph_vs = u_ego_subgraph.edges()[0].tolist(), u_ego_subgraph.edges()[1].tolist()

        for j in range(len(u_graph_us)):
            graph_description_str += one_edge_str + str(u_graph_us[j]) + ', ' + str(u_graph_vs[j])
        v_graph_us, v_graph_vs = v_ego_subgraph.edges()[0].tolist(), v_ego_subgraph.edges()[1].tolist()
        for j in range(len(v_graph_us)):
            graph_description_str += one_edge_str + str(v_graph_us[j] + u_size) + ', ' + str(v_graph_vs[j] + u_size)
        graph_description_str += end_of_edges_str
        return graph_description_str, subgraph_embeddings,u_size, truncated
    def prepare_dp(positive):
        dp  = {}
        dp['question'] = "This is a graph: " + graph_description +". " + "Should node 0 connect node {}?".format(check_node_id)
        if positive:
            dp['answer'] = "Yes, these two nodes should be connected."
        else:
            dp['answer'] = "Nope, these two nodes have no relation."
        dp['unaligned_input_embeds'] = subgraph_embeddings
        dp['task_type'] = 'classification'
        dp['split_set'] = split_set
        dp['truncated'] = truncated
        return dp
    neg_list = {'train':[], 'valid':[],"test":[]}
    positive_list = {'train':[], 'valid':[],"test":[]}
    cnt = 0
    for i in range(0, edge_ids[0].shape[0]):
        u_center = edge_ids[0][i].item()
        v_center = edge_ids[1][i].item()
        if u_center > v_center:
            continue
        
        graph_description,subgraph_embeddings, check_node_id, truncated =  prepare_graph_by_edge(graph,u_center, v_center)
        if cnt < int(edge_ids[0].shape[0]//2 * 0.85):
            split_set = 'train'
        elif cnt < int(edge_ids[0].shape[0]//2 * 0.90):
            split_set = 'valid'
        else:
            split_set = 'test'
        cnt += 1
        positive_list[split_set].append((u_center,v_center))
        dp = prepare_dp(True)
        dp_lists.append(dp)
        while graph.has_edges_between([u_center], [v_center])[0]:
            v_center = np.random.randint(u_center + 1, graph.num_nodes())
        neg_list[split_set].append((u_center,v_center))
        graph_description,subgraph_embeddings, check_node_id, truncated =  prepare_graph_by_edge(graph,u_center, v_center)
        dp  = prepare_dp(False)
        dp_lists.append(dp)
    dgl.save_graphs('your_local_root_path/for_NC_check/{}'.format(dataset_name),[graph])
    torch.save(neg_list,'your_local_root_path/for_NC_check/{}_neg'.format(dataset_name))
    torch.save(positive_list,'your_local_root_path/for_NC_check/{}_pos'.format(dataset_name))
    dataset = Dataset.from_list(dp_lists)
    dataset.save_to_disk('datasets_local/json_texts_datasets/prediction_datasets/{}_link_prediction'.format(dataset_name))
        
        
        
        
if __name__ == '__main__':
    prepare_link_prediction_dataset('cora')
    # tokenizer_name = 'Llama-3.2-3B-Instruct'
    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/{}'.format(tokenizer_name))
    # tokenizer.add_tokens([embedding_mask_str, begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str])
    # # prepare_graph_embedding_QA_dataset(tokenizer,tokenizer_name,'arxiv_graph_embedding_QA_new')
    # prepare_graph_embedding_QA_dataset(tokenizer, tokenizer_name,'pubmed_graph_embedding_QA')
    # prepare_molhiv_graph_embedding_QA_dataset(tokenizer, tokenizer_name,'molhiv_graph_embedding_QA_pure_nodes_with_element_type_count_cot')
    # prepare_embedding_prediction_dataset(tokenizer,'Llama-3.2-3B-Instruct','arxiv_pureEmbeds_prediction_new')
    # prepare_molhiv_pretrain_dataset(tokenizer,tokenizer_name,'molhiv_pretrain')
    # prepare_pretrain_dataset(token
    # prepare_arxiv_dpo_graph_QA_datasets('arxiv_dpo_graph_QA')
    