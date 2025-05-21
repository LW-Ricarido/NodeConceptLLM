import torch
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import numpy as np
import os, sys
sys.path.append(os.curdir)
from data_utils.load_raw_data import load_pubmed_raw_data
ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/molhiv_graph_embedding_QA_pure_nodes_with_element_type_count_cot')
task_type = np.where(np.array(ds['task_type'])=='classification')[0]
ds = ds.select(task_type)
raw_data = load_pubmed_raw_data(embeddings=True, graph=True, text_label=True,split_ids=True)
raw_embeddings = torch.load('your_local_root_path/ogbg_molhiv/text_to_embedding.bin',map_location='cpu')
raw_text = torch.load('your_local_root_path/ogbg_molhiv/text_to_embedding.bin',map_location='cpu')

# graph = raw_data['graph']
# graph = dgl.add_reverse_edges(graph)
# graph = dgl.to_simple_graph(graph)
map_embedding_to_id = {}
for i in range(len(raw_embeddings)):
    idx = tuple(raw_embeddings[i]['embedding'].tolist())
    map_embedding_to_id[idx] = i

def map_to_raw_text(center_idx,dp):
    raw_question_list = dp.pop('question').split('<|embedding_mask|>')
    new_question = raw_question_list[0]
    potential_check_id = list(range(len(raw_text)))# [center_idx] + graph.predecessors(center_idx).tolist()
    for j in range(len(dp['unaligned_input_embeds'])):
        current_id = None
        current_embedding = torch.tensor(dp['unaligned_input_embeds'][j])
        current_id = map_embedding_to_id[tuple(current_embedding.tolist())]
        # for check_id in potential_check_id:
        #     if (current_embedding == raw_embeddings[check_id]['input_embedding']).sum() == 768:
        #         current_id = check_id
        #         break
        # if current_id is None:
        #     print("FUCK==================")
        # potential_check_id.remove(current_id)
        new_question += raw_text[current_id]['label'] + ';' + raw_question_list[j + 1]
    dp['question'] = new_question
    return dp

new_dp_list = []
for i in tqdm(range(len(ds))):
    dp = ds[i]
    check_dp = map_to_raw_text(i,dp)
    # import ipdb;ipdb.set_trace()
    new_dp_list.append(check_dp)

new_ds = Dataset.from_list(new_dp_list)
new_ds.save_to_disk('datasets_local/json_texts_datasets/molhiv_pure_text_dataset')
import ipdb;ipdb.set_trace()