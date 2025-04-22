import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, EvalPrediction, DataCollatorForLanguageModeling
from models.LLaMa import LLama4Graph, LLama4GraphWithValueHead
from data_utils.data_loader import getCoraEmbeds2TextDataset
from sentence_transformers import SentenceTransformer
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.graphproppred import Evaluator
from datasets import load_from_disk
import os
from transformers.data.data_collator import pad_without_fast_tokenizer_warning 
# from data_utils.data_collators import pad_without_fast_tokenizer_warning, InstructEmbedsPretrainCollator
from data_utils.prompt_str import embedding_mask_str, begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str
from tqdm.auto import tqdm
from typing import List
import numpy as np
from peft import PeftModel, get_peft_model
from torch.utils.data import DataLoader
import json


# all_labels = ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
all_labels = ['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning']
# all_labels = ['Artificial Intelligence', 'Hardware Architecture', 'Computational Complexity', 'Computational Engineering, Finance, and Science', 'Computational Geometry', 'Computation and Language', 'Cryptography and Security', 'Computer Vision and Pattern Recognition', 'Computers and Society', 'Databases', 'Distributed, Parallel, and Cluster Computing', 'Digital Libraries', 'Discrete Mathematics', 'Data Structures and Algorithms', 'Emerging Technologies', 'Formal Languages and Automata Theory', 'General Literature', 'Graphics', 'Computer Science and Game Theory', 'Human-Computer Interaction', 'Information Retrieval', 'Information Theory', 'Machine Learning', 'Logic in Computer Science', 'Multiagent Systems', 'Multimedia', 'Mathematical Software', 'Numerical Analysis', 'Neural and Evolutionary Computing', 'Networking and Internet Architecture', 'Other Computer Science', 'Operating Systems', 'Performance', 'Programming Languages', 'Robotics', 'Symbolic Computation', 'Sound', 'Software Engineering', 'Social and Information Networks', 'Systems and Control']
def is_all_label_contained(predicted_sentences):
    cnt = 0
    for label in all_labels:
        if label.lower() in predicted_sentences.lower():
            cnt += 1
    if cnt == len(all_labels):
        return True
    else:
        return False
class LeftPaddingPredictionCollator(DataCollatorForLanguageModeling):
    
    def torch_call(self, examples):
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, {'input_ids':[np.array(example['input_ids'])[np.array(example['labels']) == -100] for example in examples]}, return_tensors = 'pt',
            padding_side = 'left', pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch['attention_mask'] = torch.ones_like(batch['input_ids'])
        batch['attention_mask'][batch['input_ids'] == self.tokenizer.pad_token_id] = 0
        batch['unaligned_inputs_embeds'] = [torch.tensor(example['unaligned_input_embeds'],device=batch['input_ids'].device) for example in examples]
        
        batch['embedding_positions'] = [torch.tensor(example['embedding_positions'],dtype=torch.long,device=batch['input_ids'].device) for example in examples]
        return batch

if __name__ == "__main__":
    device = torch.device('cuda:0')
    # '''
    #     This is for pure embedding prediction task
    # '''
    # ds = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/arxiv_pureEmbeds_prediction_test_set_only')
    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
    # base_model = LLama4Graph.from_pretrained("meta-llama/Llama-3.2-3B-Instruct").to(device)
    # base_model.lm_head.load_state_dict(base_model.model.embed_tokens.state_dict())
    # base_model.node_embedding_connect.load_state_dict(torch.load('ckpts/llama3_2_3B_Instruct/arxiv_pretrain_lr01/checkpoint-451000/node_embedding_connect.pt',map_location='cpu').state_dict())
    # peft_model = PeftModel.from_pretrained(base_model,model_id='ckpts/llama3_2_3B_Instruct/arxiv_pureEmbeds_LoRA/checkpoint-85000/lora_adapter.pt')
    
    '''
        This is for Graph Embedding QA prediction Task
    '''
    ds = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/cora_graph_embedding_QA/')
    # ds = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/molhiv_graph_embedding_QA_pure_nodes_with_element_type_count_cot')
    # test_set_ids = np.where((np.array(ds['split_set']) == 'test') & (np.array(ds['task_type']) == 'classification'))[0]
    test_set_ids = np.where((np.array(ds['task_type']) == 'classification'))[0]
    print("=============length of :",len(test_set_ids))
    ds = ds.select(test_set_ids)
    tokenizer = AutoTokenizer.from_pretrained('base_models/Llama-3.2-3B-Instruct_node_graph_combine_pretrain')
    # tokenizer.add_tokens([embedding_mask_str, begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str])
    tokenizer.pad_token = '<|finetune_right_pad_id|>'
    # base_model = LLama4Graph.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")# .to(device)
    # base_model.lm_head.load_state_dict(base_model.model.embed_tokens.state_dict())
    
    # base_model.node_embedding_connect.load_state_dict(torch.load('ckpts/llama3_2_3B_Instruct/node_graph_combine_pretrain/checkpoint-365000/node_embedding_connect.pt',map_location='cpu').state_dict())
    # base_model.resize_token_embeddings(len(tokenizer))
    use_value_head = False
    parent_checkpoint_path = 'ckpts/llama3_2_3B_Instruct/arxiv_Graph_QA_LoRA_node_combine_pretrain_1_1'
    
    ckpts_dirs = ['zero_shot']#,'80000']#,'24000','25000']#['34000','51000','60000']#,'67000','80000']#,'24000','25000']
    count_dict = {}
    max_new_tokens = 20
    batch_size = 32
    for curr_ckpt in ckpts_dirs:
        peft_model = None
        base_model = None
        torch.cuda.empty_cache()
        base_model = LLama4Graph.from_pretrained('base_models/Llama-3.2-3B-Instruct_node_graph_combine_pretrain')
        # base_model = LLama4Graph.from_pretrained("base_models/llama3_2_3B_Instruct_arxiv_Graph_QA_LoRA_node_combine_pretrain_1_1_checkpoint-78000_merged").to(device)
        base_model.embedding_mask_id = tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0]
        
        # if not use_value_head:
        #     base_model.load_adapter('{}/checkpoint-{}/lora_adapter'.format(parent_checkpoint_path,curr_ckpt))
        #     peft_model = base_model
        # else:
        #     peft_model = LLama4GraphWithValueHead.from_pretrained(base_model)
        #     peft_model.load_state_dict(torch.load('{}/checkpoint-{}/pytorch_model.bin'.format(parent_checkpoint_path,curr_ckpt),map_location='cpu'),strict=False)
        #     peft_model.pretrained_model.load_adapter('{}/checkpoint-{}'.format(parent_checkpoint_path,curr_ckpt))
        
        # peft_model.generation_config = base_model.generation_config
        # peft_model = peft_model#.half()
        # peft_model = peft_model.to(device)
        
        peft_model = base_model
        peft_model = peft_model.to(device)
        
        # tokenizer = AutoTokenizer.from_pretrained('ckpts/llama3_2_3B_Instruct/value_head_graph_test/checkpoint-{}'.format(curr_ckpt))
        # tokenizer.pad_token = '<|finetune_right_pad_id|>'
        collator = LeftPaddingPredictionCollator(tokenizer,mlm=False)
        
        print('curr ckpt {}'.format(curr_ckpt))
        res_list = [] 
        raw_input_list =[]
        res_values = []
        peft_model.generation_config.pad_token_id = tokenizer.eos_token_id
        print(tokenizer.eos_token_id)
        
        test_dataloader = DataLoader(
            ds,
            batch_size = batch_size,
            collate_fn=collator,
            num_workers=4,
            shuffle=False
        )
        correct_count = {'truncated':0,'correct_truncated': 0}
        reasoning_save_list = []
        with torch.no_grad():
            for index, batch_data in enumerate(tqdm(test_dataloader, desc='Testing Process')):
                
                if not use_value_head:
                    batch_input_embeds = peft_model.get_input_embeddings()(batch_data['input_ids'].to(device))
                else:
                    batch_input_embeds = peft_model.pretrained_model.get_input_embeddings()(batch_data['input_ids'].to(device))
                
                for i in range(len(batch_data['unaligned_inputs_embeds'])):
                    
                    if not use_value_head:
                        aligned_embeds = peft_model.node_embedding_connect(batch_data['unaligned_inputs_embeds'][i].to(device))#.half())
                    else:
                        aligned_embeds = peft_model.pretrained_model.node_embedding_connect(batch_data['unaligned_inputs_embeds'][i].to(device))
                    
                    new_position = torch.nonzero(batch_data['attention_mask'][i].bool())[batch_data['embedding_positions'][i]].flatten()
                    batch_input_embeds[i][new_position] = aligned_embeds
                if use_value_head:
                    batch_res = peft_model.generate(
                        inputs_embeds = batch_input_embeds,
                        attention_mask = batch_data['attention_mask'].to(device),
                        do_sample = False,
                        max_new_tokens = max_new_tokens,
                        require_value = True
                    )
                else:
                    batch_res = peft_model.generate(
                        inputs_embeds = batch_input_embeds,
                        attention_mask = batch_data['attention_mask'].to(device),
                        do_sample = False,
                        max_new_tokens = max_new_tokens,
                    )
                if not use_value_head:
                    res_list.append(batch_res)
                else:
                    res_list.append(batch_res[0])
                    res_values.append(batch_res[1])
                # if batch_res.shape[1] > 50:
                #     for k in range(batch_res.shape[0]):
                #         if (batch_res[k] != tokenizer.eos_token_id).sum() > 50:
                #             current_idx = i * batch_size + k
                #             current_label = torch.tensor(ds[current_idx]['labels'])
                #             # print("dp index:{}  label: {}".format(current_idx,tokenizer.decode(current_label[current_label != -100])))
                #             # print(tokenizer.decode(batch_res[k][batch_res[k] != tokenizer.eos_token_id]))
                #             reasoning_save_dict = {}
                #             reasoning_save_dict['index'] = current_idx
                #             reasoning_save_dict['label'] = tokenizer.decode(current_label[current_label != -100][1:-1])
                #             reasoning_save_dict['generated'] = tokenizer.decode(batch_res[k][batch_res[k] != tokenizer.eos_token_id])
                #             reasoning_save_list.append(reasoning_save_dict)
                #             # print(batch_data['unaligned_inputs_embeds'][k][0][0:10])
                #             # import ipdb;ipdb.set_trace()
                #             # print('================================================')
                   
        # save_name = "test_output/fp16_batch_{}_arxiv_check_llama3_2_3B_node_graph_combine_classification_checkpoint-{}".format(batch_size,curr_ckpt)
        # torch.save(res_list,"{}_result.pt".format(save_name))
        # torch.save(raw_input_list,"{}_inputs.pt".format(save_name))
        # with open('test_output/reasoning_check/arxiv_{}.json'.format(curr_ckpt),'w') as fp:
        #     json.dump(reasoning_save_list,fp)
        label_list = []
        preds = []
        values = []
        for i in range(len(res_list)):
            for j in range(res_list[i].shape[0]):
                preds.append(res_list[i][j])
                if use_value_head:
                    values.append(res_values[i][j].item())
                else:
                    values.append(0)
                
        correct = 0
        idxs = []
        y_pred = []
        y_true = []
        len_count = []
        for i in range(len(preds)):
            labels = torch.tensor(ds[i]['labels'])
            # if np.random.rand() < 0.005:
            #     print("dp: {}   label:   {}\n\n".format(i,tokenizer.decode(labels[labels != -100][1:-1])))
            #     print("Generated:\n{}".format(tokenizer.decode(preds[i][preds[i] != tokenizer.eos_token_id])))
            #     print('=========================\n')
            len_count.append(len(ds[i]['input_ids']))
            if ds[i]['truncated']:
                correct_count['truncated'] += 1
            if "Yes" in tokenizer.decode(labels[labels != -100][1:-1]):
                # print(tokenizer.decode(preds[i]))
                # import ipdb; ipdb.set_trace()
                y_true.append(1)
            else:
                y_true.append(0)
            if "Yes" in tokenizer.decode(preds[i]):
                y_pred.append(1)
            else:
                y_pred.append(0)
            if tokenizer.decode(labels[labels != -100][1:-1]).lower() in tokenizer.decode(preds[i]).lower():
                label_list.append(tokenizer.decode(labels[labels != -100][1:-1]))
                correct += 1
                idxs.append(i)
                if ds[i]['truncated']:
                    correct_count['correct_truncated'] += 1
                if is_all_label_contained(tokenizer.decode(preds[i])):
                    correct -= 1
            # else:
            #     if "Yes" not in tokenizer.decode(preds[i]) and "Nope" not in tokenizer.decode(preds[i]):
            #         print(tokenizer.decode(preds[i]))
            #         print(tokenizer.decode(labels[labels != -100][1:-1]))
            #         print('==================================================')

        # print(idxs)
        # import ipdb; ipdb.set_trace()
        evaluator = Evaluator(name='ogbg-molhiv')
        roc_auc = 0 # evaluator.eval({'y_true':np.expand_dims(np.array(y_true),axis=1),'y_pred':np.expand_dims(np.array(values),axis=1)})['rocauc']
        len_count = torch.tensor(len_count)
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
        true_positive = torch.logical_and(y_pred == 1, y_true == 1).sum()
        false_positive = torch.logical_and(y_pred == 1, y_true == 0).sum()
        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (y_true == 1).sum()
        F1 = 2 * precision * recall / (precision + recall)
        print("ACC {} ROC_AUC:{} F1:{:.4f} tp:{} fp :{} Precision:{:.4f} Recall:{:.4f}".format(correct/ len(test_set_ids),roc_auc, F1.item(), true_positive.item(),false_positive.item(), precision.item(), recall.item()))
        print('ckpt-{} correct number: {}'.format(curr_ckpt,correct))
        correct_count['overall_correct_number'] = correct
        correct_count['below_500_recall'] = torch.logical_and(torch.logical_and(len_count < 500, y_pred == 1), y_true == 1).sum()
        correct_count['ROC_AUC'] = roc_auc
        correct_count['F1'] = F1.item()
        correct_count['tp'] = true_positive.item()
        correct_count['fp'] = false_positive.item()
        correct_count['acc'] = correct/ len(test_set_ids) 
        
        # torch.save(label_list, "{}_labels.pt".format(save_name))
        print(correct_count)
        # import ipdb; ipdb.set_trace()
        count_dict[curr_ckpt] = correct_count
    print(count_dict)
    print(parent_checkpoint_path)

