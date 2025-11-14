import torch
from ogb.nodeproppred import DglNodePropPredDataset
import pandas as pd
import os,dgl
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
from sentence_transformers import SentenceTransformer

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

products_category_mapping_dict =  {0 :            "Home & Kitchen",
 1 :    "Health & Personal Care",
 2 :                    "Beauty",
 3 :         "Sports & Outdoors",
 4 :                     "Books",
 5 :      "Patio, Lawn & Garden",
 6 :              "Toys & Games",
 7 :               "CDs & Vinyl",
 8 : "Cell Phones & Accessories",
 9 :    "Grocery & Gourmet Food",
10 :     "Arts, Crafts & Sewing",
11 : "Clothing, Shoes & Jewelry",
12 :               "Electronics",
13 :               "Movies & TV",
14 :                  "Software",
15 :               "Video Games",
16 :                "Automotive",
17 :              "Pet Supplies",
18 :           "Office Products",
19 :   "Industrial & Scientific",
20 :       "Musical Instruments",
21 :  "Tools & Home Improvement",
22 :    "Magazine Subscriptions",
23 :             "Baby Products",
24 :                       "NaN",
25 :                "Appliances",
26 :          "Kitchen & Dining",
27 :   "Collectibles & Fine Art",
28 :                "All Beauty",
29 :             "Luxury Beauty",
30 :            "Amazon Fashion",
31 :                 "Computers",
32 :           "All Electronics",
33 :          "Purchase Circles",
34 : "MP3 Players & Accessories",
35 :                "Gift Cards",
36 :  "Office & School Supplies",
37 :          "Home Improvement",
38 :            "Camera & Photo",
39 :          "GPS & Navigation",
40 :             "Digital Music",
41 :           "Car Electronics",
42 :                      "Baby",
43 :              "Kindle Store",
44 :              "Buy a Kindle",
45 :    "Furniture & Decor",
46 :                   "unknown label"}

msrc_21_node_label_dict = {
0:"void",
1:"building",
2:"grass",
3:"tree",
4:"cow",
5:"horse",
6:"sheep",
7:"sky",
8:"mountain",
9:"aeroplane",
10:"water",
11:"face",
12:"car",
13:"bicycle",
14:"flower",
15:"sign",
16:"bird",
17:"book",
18:"chair",
19:"road",
20:"cat",
21:"dog",
22:"body",
23:"boat"
}

msrc_21_graph_label_dict = {
    
}
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

def load_products_raw_data(embeddings, graph=False, text_label=False, split_ids=False, raw_text=False):
    data_dict = {}
    if embeddings:
        embeddings_dict = torch.load('/data/sharefile/wei/dataset/ogbn_products/raw_tensor.bin',map_location='cpu',weights_only=False)
        data_dict['embeddings'] = [embeddings_dict[key] for key in range(len(embeddings_dict))]
        # data_dict['embeddings'] = torch.load('/data/sharefile/wei/dataset/ogbn_products/raw_tesnor.bin')
    
    if raw_text:
        text_dict = torch.load('/data/sharefile/wei/dataset/ogbn_products/raw_text.bin')
        data_dict['raw_text'] = [text_dict[key] for key in range(len(text_dict))]
    if graph:
        data = DglNodePropPredDataset('ogbn-products', root='/data/sharefile/wei/dataset')
        data_dict['graph'], data_dict['label'] = data[0]
        data_dict['graph'] = dgl.remove_self_loop(data_dict['graph'])
        data_dict['graph'] = dgl.add_reverse_edges(data_dict['graph'])
        data_dict['graph'] = dgl.to_simple_graph(data_dict['graph'])
        split_list = data.get_idx_split()
        if split_ids:
            data_dict['split_ids'] = split_list
    if text_label:
        text_label_list = []
        for label in data_dict['label']:
            text_label_list.append(products_category_mapping_dict[label.item()])
        data_dict['text_label'] = text_label_list
    return data_dict

def prepare_paper100_raw_texts():
    paperidx2nodeidx = pd.read_csv('/data/sharefile/wei/dataset/ogbn_papers100M/mapping/nodeidx2paperid.csv.gz', compression='gzip').set_index('paper id')['node idx'].to_dict()
    raw_texts = []
    node_size = len(paperidx2nodeidx.keys())
    paperidx2title = pd.read_csv('/data/sharefile/wei/dataset/ogbn_papers100M/raw/paperinfo/idx_title.tsv',sep="\t",header=None,names=['paper id','title']).set_index('paper id')['title'].to_dict()
    print('===============pass=======================')
    chunk_size = 100000
    embeddings = [0] * node_size
    raw_texts = [''] * node_size
    batch_size = 256
    device = torch.device('cuda:0')
    PLM = SentenceTransformer('all-mpnet-base-v2').to(device)
    chunk_count = 0
    pbar = tqdm(total=node_size)
    for chunk in pd.read_csv('/data/sharefile/wei/dataset/ogbn_papers100M/raw/paperinfo/idx_abs.tsv',sep="\t", header=None, names=['paper id', 'abs'], chunksize=chunk_size):
        current_paperidx2abs = chunk.set_index('paper id')['abs'].to_dict()
        paper_idxs = list(current_paperidx2abs.keys())
        current_raw_texts = []
        current_node_idxs = []
        for i in range(len(chunk)):
            current_paper_id = paper_idxs[i]
            if current_paper_id in paperidx2nodeidx.keys():
                current_node_id = paperidx2nodeidx[current_paper_id]
                try:
                    current_text = "Title: {}\n Abstract: {}".format(paperidx2title[current_paper_id], current_paperidx2abs[current_paper_id])
                except Exception as e:
                    print('============check')
                    import ipdb;ipdb.set_trace()
                raw_texts[current_node_id] = current_text
                current_raw_texts.append(current_text)
                current_node_idxs.append(current_node_id)
                pbar.update(1)
        for i in range(len(current_raw_texts)//batch_size):
            start_idx = i * batch_size
            end_idx = min((i+1) * batch_size, len(chunk))
            currents = PLM.encode(current_raw_texts[start_idx:end_idx]).tolist()
            for j in range(len(currents)):
                current_node_id = paperidx2nodeidx[current_node_idxs[start_idx+1]]
                embeddings[current_node_id] = currents[j]
    save_dict = {"embedding": embeddings, "text":raw_texts}
    torch.save(save_dict,"datasets_local/paper100M_text_to_embedding.bin")
    
    
    # paperidx2title = pd.read_csv('/data/sharefile/wei/dataset/ogbn_papers100M/raw/paperinfo/idx_title.tsv',sep="\t", header=None, names=['paper id', 'title']).set_index('paper id')['title'].to_dict()
    # print('========================get')
    # for i in tqdm(range(node_size)):
    #     current_paper_id = nodeidx2paperidx[i]
    #     current_text = "Title: {}\n Abstract: {}".format(paperidx2title[current_paper_id], paperidx2abs[current_paper_id])
    #     raw_texts.append(current_text)
    # batch_size = 256
    
    # for i in range(tqdm(node_size // batch_size)):
    #     start_idx = i * batch_size
    #     end_idx = min((i + 1) * batch_size, node_size)
    #     currents = raw_texts[start_idx:end_idx]
    #     embeddings.extend(PLM.encode(currents).tolist())
    # save_dict = {"embedding": embeddings, "text":raw_texts}
    # torch.save(save_dict,"datasets_local/paper100M_text_to_embedding.bin")
    
def prepare_products_raw_texts():
    raw_text_dict = dict()
    PLM = SentenceTransformer("all-mpnet-base-v2")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        PLM = PLM.to(device)
    with open('/data/sharefile/wei/dataset/ogbn_products/Amazon-3M.raw/trn.json','r') as fp:
        for line in fp.readlines():
            curr_info = json.loads(line)
            raw_text_dict[curr_info['uid']] = curr_info
    # for dp in trn:
    #     raw_text_dict[dp['uid']] = dp
    print('=========finish trn')
    with open('/data/sharefile/wei/dataset/ogbn_products/Amazon-3M.raw/tst.json','r') as fp:
        for line in fp.readlines():
            curr_info = json.loads(line)
            raw_text_dict[curr_info['uid']] = curr_info
    # for dp in tst:
    #     raw_text_dict[dp['uid']] = dp
    nodeid2asin = pd.read_csv('/data/sharefile/wei/dataset/ogbn_products/mapping/nodeidx2asin.csv.gz',compression='gzip')
    raw_text_list = {}
    raw_tensor_list = {}
    for index in tqdm(range(len(nodeid2asin))):
        row = nodeid2asin.iloc[index]
        current_dict = dict()
        current_info = raw_text_dict[row['asin']]
        current_dict['text'] = "Product name: {} Product description:{}".format(current_info['title'],current_info['content'])
        raw_text_list[row['node idx'].item()] = current_dict
        raw_tensor_list[row['node idx'].item()] = PLM.encode(current_dict['text'])
    torch.save(raw_text_list, '/data/sharefile/wei/dataset/ogbn_products/raw_text.bin')
    torch.save(raw_tensor_list, '/data/sharefile/wei/dataset/ogbn_products/raw_tensor.bin')
    
if __name__ == "__main__":
    prepare_products_raw_texts()