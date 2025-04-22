import torch
from ogb.nodeproppred import DglNodePropPredDataset
import pandas as pd
import os,dgl
arxiv_category_mapping_dict = {
        "arxiv cs ai" : "Artificial Intelligence", 
        "arxiv cs ar" : "Hardware Architecture",
        "arxiv cs cc" : "Computational Complexity",
        "arxiv cs ce" : "Computational Engineering, Finance, and Science",
        "arxiv cs cg" : "Computational Geometry",
        "arxiv cs cl" : "Computation and Language",
        "arxiv cs cr" : "Cryptography and Security",
        "arxiv cs cv" : "Computer Vision and Pattern Recognition",
        "arxiv cs cy" : "Computers and Society",
        "arxiv cs db" : "Databases",
        "arxiv cs dc" : "Distributed, Parallel, and Cluster Computing",
        "arxiv cs dl" : "Digital Libraries",
        "arxiv cs dm" : "Discrete Mathematics",
        "arxiv cs ds" : "Data Structures and Algorithms",
        "arxiv cs et" : "Emerging Technologies",
        "arxiv cs fl" : "Formal Languages and Automata Theory",
        "arxiv cs gl" : "General Literature",
        "arxiv cs gr" : "Graphics",
        "arxiv cs gt" : "Computer Science and Game Theory",
        "arxiv cs hc" : "Human-Computer Interaction",
        "arxiv cs ir" : "Information Retrieval",
        "arxiv cs it" : "Information Theory",
        "arxiv cs lg" : "Machine Learning",
        "arxiv cs lo" : "Logic in Computer Science",
        "arxiv cs ma" : "Multiagent Systems",
        "arxiv cs mm" : "Multimedia",
        "arxiv cs ms" : "Mathematical Software",
        "arxiv cs na" : "Numerical Analysis",
        "arxiv cs ne" : "Neural and Evolutionary Computing",
        "arxiv cs ni" : "Networking and Internet Architecture",
        "arxiv cs oh" : "Other Computer Science",
        "arxiv cs os" : "Operating Systems",
        "arxiv cs pf" : "Performance",
        "arxiv cs pl" : "Programming Languages",
        "arxiv cs ro" : "Robotics",
        "arxiv cs sc" : "Symbolic Computation",
        "arxiv cs sd" : "Sound",
        "arxiv cs se" : "Software Engineering",
        "arxiv cs si" : "Social and Information Networks",
        "arxiv cs sy" : "Systems and Control"
    }

cora_text_label_list = ['Case Based', 'Genetic Algorithms', 'Neural Networks','Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning', 'Theory']

pubmed_text_label_list = ["Diabetes Mellitus, Experimental", "Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2"]

def load_arxiv_raw_data(embeddings, graph=False, text_label=False,
                        split_ids=False, raw_text=False):
    data_dict = {}
    if embeddings:
        data_dict['embeddings'] = torch.load('/data/sharefile/wei/dataset/ogbn_arxiv/raw_tensor.pt',map_location='cpu')
    if raw_text:
        data_dict['raw_text']  = torch.load('/data/sharefile/wei/dataset/ogbn_arxiv/raw_text.bin',map_location='cpu')
    if graph:
        data = DglNodePropPredDataset('ogbn-arxiv', root='/data/sharefile/wei/dataset')
        data_dict['graph'], data_dict['label'] = data[0]
        data_dict['graph'] = dgl.add_reverse_edges(data_dict['graph'])
        data_dict['graph'] = dgl.to_simple_graph(data_dict['graph'])
        split_list = data.get_idx_split()
        if split_ids:
            data_dict['split_ids'] = split_list
    if text_label:
        label2category = pd.read_csv(os.path.join('/data/sharefile/wei/dataset/','ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz'), compression='gzip')
        text_label_list = []
        for label in data_dict['label']:
            text_label_list.append(arxiv_category_mapping_dict[label2category['arxiv category'][label.item()]])
        data_dict['text_label'] = text_label_list
    return data_dict

def load_cora_raw_data(embeddings, graph=False, text_label=False, split_ids=False):
    data_dict = {}
    raw_graph = dgl.load_graphs('datasets_local/cora_graph_with_PLMfeature.bin')[0][0]
    if embeddings:
        data_dict['embeddings'] = raw_graph.ndata['raw']
    if graph:
        raw_graph = dgl.add_reverse_edges(raw_graph)
        raw_graph = dgl.to_simple_graph(raw_graph)
        data_dict['graph'] = raw_graph
        if split_ids:
            idxs = torch.randperm(raw_graph.num_nodes())
            split_list = {}
            split_list['train'] = idxs[0:140].tolist()
            split_list['valid'] = idxs[140: -1000].tolist()
            split_list['test'] = idxs[-1000:].tolist()
            data_dict['split_ids'] = split_list
    if text_label:
        text_label_list = []
        for i in range(raw_graph.num_nodes()):
            text_label_list.append(cora_text_label_list[raw_graph.ndata['label'][i]])
        data_dict['text_label'] = text_label_list
    return data_dict
    

def load_citeseer_raw_data(embeddings,graph=False, text_label=False, split_ids = False):
    #TODO:
    pass

def load_pubmed_raw_data(embeddings,graph=False, text_label=False, split_ids = False):
    data_dict = {}
    raw_graph = dgl.load_graphs('datasets_local/pubmed_graph_with_PLMfeature.bin')[0][0]
    if embeddings:
        data_dict['embeddings'] = raw_graph.ndata['raw']
    if graph:
        raw_graph = dgl.add_reverse_edges(raw_graph)
        raw_graph = dgl.to_simple(raw_graph)
        data_dict['graph'] = raw_graph
        if split_ids:
            split_list = {}
            split_list['train'] = torch.nonzero(raw_graph.ndata['train_mask']).flatten().tolist()
            split_list['valid'] = torch.nonzero(raw_graph.ndata['val_mask']).flatten().tolist()
            split_list['test'] = torch.nonzero(raw_graph.ndata['test_mask']).flatten().tolist()
            data_dict['split_ids'] = split_list
    if text_label:
        text_label_list = []
        for i in range(raw_graph.num_nodes()):
            text_label_list.append(pubmed_text_label_list[raw_graph.ndata['y'][i]])
        data_dict['text_label'] = text_label_list
    return data_dict