from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils.prompt_str import *


original_model_name = "meta-llama/Llama-3.2-1B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(original_model_name)
base_tokenizer = AutoTokenizer.from_pretrained(original_model_name)

base_tokenizer.add_tokens([embedding_mask_str ,begin_of_nodes_str,end_of_nodes_str,begin_of_edges_str,end_of_edges_str,one_edge_str])

config = base_model.config
if config.tie_word_embeddings:
    base_model.lm_head.load_state_dict(base_model.model.embed_tokens.state_dict())
    config.tie_word_embeddings = False
    base_model.config = config
base_model.resize_token_embeddings(len(base_tokenizer))
base_model.save_pretrained('base_model/{}_resized_embedding_untied'.format(original_model_name.split('/')[1]))
base_tokenizer.save_pretrained('base_model/{}_resized_embedding_untied'.format(original_model_name.split('/')[1]))