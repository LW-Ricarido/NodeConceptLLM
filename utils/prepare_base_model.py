import sys,os
sys.path.append(os.path.abspath(os.path.curdir))
from models.LLaMa import LLama4Graph, LLama4GraphWithValueHead
from transformers import Trainer, AutoTokenizer,AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments,LlamaConfig
import torch
import json
from data_utils.prompt_str import embedding_mask_str, begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str


if __name__ == '__main__':
    # model = LLama4Graph.from_pretrained('base_models/Llama-3.2-3B-Instruct_node_graph_combine_pretrain')
    node_embedding_path = "ckpts/llama3_2_3B_Instruct/node_graph_combine_pretrain/checkpoint-365000/node_embedding_connect.pt"
    save_path = "base_models/Llama-3.2-3B-Instruct_node_graph_combine_pretrain"
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
    tokenizer.add_tokens([embedding_mask_str, begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str])
    original_model = LLama4Graph.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    original_model.resize_token_embeddings(len(tokenizer))
    # import ipdb;ipdb.set_trace()
    # original_model.lm_head.load_state_dict(original_model.model.embed_tokens.state_dict())
    original_model.node_embedding_connect.load_state_dict(torch.load(node_embedding_path,map_location='cpu').state_dict())
   
    original_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(os.path.join(save_path,'connector_path.json'),'w') as f:
        json.dump({'node_embedding_connect':node_embedding_path},f)
    
    
