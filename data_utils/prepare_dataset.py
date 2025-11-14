import dgl
import torch
import json, os
from sentence_transformers import SentenceTransformer
import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

cora_text_label_list = ['Case Based', 'Genetic Algorithms', 'Neural Networks','Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning', 'Theory']
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
def load_cora_raw_data(data_dir):
    graph = dgl.load_graphs(os.path.join(data_dir,'cora_8b_graph.pt'))[0][0]
    graph = dgl.add_reverse_edges(graph)
    graph = dgl.to_simple(graph)
    
    with open(os.path.join(data_dir,'cora_8b_all_processed.json'),'r') as fp:
        text_dict = json.load(fp)
        texts = [text_dict[str(i)]['raw'] for i in range(graph.number_of_nodes())]
    return graph, texts

def prepare_text_embedding_feature_dataset(texts,dataset_name):
    PLM = SentenceTransformer("all-mpnet-base-v2")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        PLM = PLM.to(device)
    data_list = []
    for i in range(len(texts)):
        data_instance = {}
        data_instance['input_embedding'] = PLM.encode(texts[i])
        data_instance['label'] = texts[i]
        data_list.append(data_instance)
    # with open('../datasets/'+dataset_name+".pt",'w') as fp:
    torch.save(data_list,'../datasets/'+dataset_name+".pt")

def clean_graphML_string(graphML_str):
    tree = ET.ElementTree(ET.fromstring(graphML_str))
    root = tree.getroot()
    ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}
    
    for edge in root.findall(".//graphml:edge",ns):
        del edge.attrib['id']
        for data in edge.findall("graphml:data",ns):
            edge.remove(data)
        if not list(edge):
            edge.text = None
    for key_element in root.findall(".//graphml:key",ns):
        root.remove(key_element)
    root.attrib.clear()
    modified_xml = ET.tostring(root,encoding='unicode')    
    modified_xml = modified_xml.replace('ns0:','')
    modified_xml = modified_xml.replace('xmlns:ns0="http://graphml.graphdrawing.org/xmlns"','')
    return modified_xml

def prepare_graphML_dataset(graph:dgl.DGLGraph,dataset_name, sub_graph_type='1_hop'):
    assert sub_graph_type in ['1_hop', '2_hop', 'random_walk']
    data_list = []
    
    
    
    for i in range(graph.number_of_nodes()):
        if sub_graph_type == '1_hop':
            sub_graph_nodes_list = [i] + graph.predecessors(i).tolist()
            node_size = len(sub_graph_nodes_list)
            sub_graph = dgl.node_subgraph(graph, sub_graph_nodes_list)
            nx_subgraph = sub_graph.to_networkx().to_undirected()
            graphML = ''
            for line in nx.generate_graphml(nx_subgraph):
                graphML += line
            graphML = clean_graphML_string(graphML)
            data_instance = {}
            degree_node = np.random.randint(1,node_size)
            edge_u_node = np.random.randint(0,node_size)
            edge_v_node =edge_u_node
            while edge_v_node == edge_u_node:
                edge_v_node = np.random.randint(0, node_size)
            data_instance['input'] = "According to this GraphML:'" + graphML + "', answer these two questions: 1. What's the degree of node {}? 2. Is there a edge between node {} and node {}?".format(degree_node, edge_u_node, edge_v_node)
            data_instance['label'] = "{}, {}".format(nx_subgraph.degree[degree_node], 'Yes' if nx_subgraph.has_edge(edge_u_node,edge_v_node) else "No")
            data_list.append(data_instance)
        else:
            raise NotImplemented
        # with open('../datasets/{}_{}.pt'.format(dataset_name,sub_graph_type),'w') as fp:
    torch.save(data_list,'../datasets/{}_{}.pt'.format(dataset_name,sub_graph_type))

def prepare_cora_dataset(data_dir):
    graph, texts = load_cora_raw_data(data_dir)
    prepare_text_embedding_feature_dataset(texts,'cora_embedding_to_text')
    prepare_graphML_dataset(graph,'cora_graphML_1degree_1edge')

def prepare_graphML_dataset(graph:dgl.DGLGraph,max_neighbor_number=10,graph_name='cora'):
    dp_list = []
    for i in range(graph.number_of_nodes()):
        dp = {}
        truncated = False
        subgraph_node_list = graph.predecessors(i).tolist()
        if len(subgraph_node_list) > max_neighbor_number:
            np.random.shuffle(subgraph_node_list)
            subgraph_node_list = subgraph_node_list[0:max_neighbor_number]
            truncated=True
        subgraph_node_list = [i] + subgraph_node_list
        sub_graph = dgl.node_subgraph(graph, subgraph_node_list)
        Origin_IDs = sub_graph.ndata['_ID'].tolist()
        nx_subgraph = sub_graph.to_networkx().to_undirected()
        graphML = ''
        for line in nx.generate_graphml(nx_subgraph):
            graphML += line
        graphML = clean_graphML_string(graphML)
        dp['origin_ids'] = Origin_IDs
        dp['truncated'] = truncated
        dp['graphML'] = graphML
        dp_list.append(dp)
    dataset = Dataset.from_list(dp_list)
    dataset.save_to_disk('datasets_local/{}_pure_graphML_max_{}_nodes_dataset'.format(graph_name,max_neighbor_number))

def prepare_cora_pure_embed_prediction_dataset(token_size,graph:dgl.DGLGraph,tokenizer):
    token_embeds_ds = load_from_disk('datasets_local/CoraEmbeds2Text_{}TokenDataset'.format(token_size))
    dp_list = []
    for i in range(graph.num_nodes()):
        dp = {}
        dp['unaligned_inputs_embeds'] = token_embeds_ds[i]['input_embedding']
        dp['input_ids'] = tokenizer.encode("This is the embedding of a research paper. Please classify the paper into one of the following categories based on its content: Case Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory. Your Classification: ")
        dp['label_begin_pos'] = len(dp['input_ids'])
        dp['input_ids'].extend(
            tokenizer.encode(cora_text_label_list[graph.ndata['label'][i]]) + [tokenizer.eos_token_id]
            )
        dp['label_end_pos'] = len(dp['input_ids'])
        dp['text_label'] = cora_text_label_list[graph.ndata['label'][i]]
        dp_list.append(dp)
    ds = Dataset.from_list(dp_list)
    ds.save_to_disk('datasets_local/CoraPureEmbeds2Prediction_{}TokenDataset'.format(token_size))  

def prepare_arxiv_pure_embed_prediction_dataset(token_size,graph:dgl.DGLGraph, tokenizer):
    token_embeds_ds = load_from_disk('datasets_local/ArxivEmbeds2Text_{}TokensDataset'.format(token_size))
    dp_list = []
    label2category = pd.read_csv(os.path.join('/data/sharefile/wei/dataset/','ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz'), compression='gzip')
    prompt_text = "This is the embedding of a research paper. Please classify the paper into one of the following categories based on its content:"
    for key in arxiv_category_mapping_dict.keys():
        prompt_text += " {};".format(arxiv_category_mapping_dict[key])
    prompt_text = prompt_text[:-1]
    prompt_text += ". Your Classification: "
    for i in range(graph.num_nodes()):
        dp = {}
        dp['unaligned_inputs_embeds'] = token_embeds_ds[i]['input_embedding']
        dp['input_ids'] = tokenizer.encode(prompt_text)
        dp['label_begin_pos'] = len(dp['input_ids'])
        dp['input_ids'].extend(
            tokenizer.encode(arxiv_category_mapping_dict[label2category['arxiv category'][graph.ndata['label'][i].item()]]) + [tokenizer.eos_token_id]
        )
        dp['label_end_pos'] = len(dp['input_ids'])
        dp['text_label'] = arxiv_category_mapping_dict[label2category['arxiv category'][graph.ndata['label'][i].item()]]
        dp_list.append(dp)
    ds =  Dataset.from_list(dp_list)
    ds.save_to_disk('datasets_local/ArxivPureEmbeds2Prediction_{}TokenDataset'.format(token_size))  
    pass

def prepare_arxiv_embed_graph_prediction_dataset(token_size, graph: dgl.DGLGraph, tokenizer, graphML_dataset, tokenizer_name):
    token_embeds_ds = load_from_disk('datasets_local/ArxivEmbeds2Text_{}TokensDataset'.format(token_size))
    dp_list = []
    label2category = pd.read_csv(os.path.join('/data/sharefile/wei/dataset/','ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz'), compression='gzip')
    prompt_text = "This is the embedding of a research paper."
    classify_text = "Please classify the first node into one of the following categories based on its content:"
    for key in arxiv_category_mapping_dict.keys():
        classify_text += " {};".format(arxiv_category_mapping_dict[key])
    classify_text += ". Your classification: "
    with tqdm(range(graph.num_nodes())) as pbar:
        for i in pbar:
            dp = {}
            origin_ids = graphML_dataset[i]['origin_ids']
            dp['truncated'] = graphML_dataset['truncated']
            prompt_text = "These are embeddings of {} research paper. This is their graph topology in GraphML format: {}".format(len(origin_ids), graphML_dataset[i]['graphML']) + '.'
            import ipdb; ipdb.set_trace()
            dp['unaligned_inputs_embeds'] = token_embeds_ds[origin_ids]['input_embedding']
            dp['input_ids'] = tokenizer.encode(prompt_text + classify_text)
            dp['label_begin_pos'] = len(dp['input_ids'])
            dp['input_ids'].extend(
                tokenizer.encode(
                    arxiv_category_mapping_dict[label2category['arxiv category'][graph.ndata['label'][i].item()]]
                ) +
                [tokenizer.eos_token_id]
            )
            dp['label_end_pos'] = len(dp["input_ids"])
            dp['text_label'] = arxiv_category_mapping_dict[label2category['arxiv category'][graph.ndata['label'][i].item()]]
            dp_list.append(dp)
    ds = Dataset.from_list(dp_list)
    ds.save_to_disk("datasets_local/ArxivEmbedsGraphML2Prediction_{}TokenDataset_Tokenizer{}_Repeat".format(token_size, tokenizer_name))
    
    

def filter_out_bos_token_id(for_filter_list, bos_token_id):
    return [i for i in for_filter_list if i != bos_token_id]
      
if __name__ == "__main__":
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset('ogbn-arxiv',root='/data/sharefile/wei/dataset')
    graph, labels = data[0]
    labels = labels.squeeze()
    graph.ndata['label'] = labels
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    # prepare_graphML_dataset(graph,max_neighbor_number=10,graph_name='arxiv')
    graphML_dataset = load_from_disk('datasets_local/arxiv_pure_graphML_max_{}_nodes_dataset'.format(10))
    prepare_arxiv_embed_graph_prediction_dataset(1, graph, tokenizer, graphML_dataset,'Llama-3.2-1B')    
    # prepare_arxiv_pure_embed_prediction_dataset(1, graph,tokenizer)
    # prepare_cora_pure_embed_prediction_dataset(1, graph,tokenizer)
    # prepare_cora_graphML_dataset(graph,max_neighbor_number=10)
    # prepare_cora_dataset('/data/sharefile/wei/workspace/LLM_Graph/processed_data')