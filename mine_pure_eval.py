import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, EvalPrediction, DataCollatorForLanguageModeling,set_seed
from models.LLaMa import LLama4Graph, LLama4GraphWithValueHead
from data_utils.data_loader import getCoraEmbeds2TextDataset
from sentence_transformers import SentenceTransformer
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.graphproppred import Evaluator
from datasets import load_from_disk,Dataset
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
from tensorboardX import SummaryWriter

global_ds_name = 'cora'
max_new_tokens = 20
batch_size = 1
use_half = True
if global_ds_name == 'pubmed':
    # all_labels = ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
    all_labels = ['Diabetes Experiments', 'Diabetes Type 1', 'Diabetes Type 2']
elif global_ds_name == 'cora':
    all_labels = ['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning','Theory']
elif global_ds_name == 'arxiv':
    all_labels = ['Artificial Intelligence', 'Hardware Architecture', 'Computational Complexity', 'Computational Engineering, Finance, and Science', 'Computational Geometry', 'Computation and Language', 'Cryptography and Security', 'Computer Vision and Pattern Recognition', 'Computers and Society', 'Databases', 'Distributed, Parallel, and Cluster Computing', 'Digital Libraries', 'Discrete Mathematics', 'Data Structures and Algorithms', 'Emerging Technologies', 'Formal Languages and Automata Theory', 'General Literature', 'Graphics', 'Computer Science and Game Theory', 'Human-Computer Interaction', 'Information Retrieval', 'Information Theory', 'Machine Learning', 'Logic in Computer Science', 'Multiagent Systems', 'Multimedia', 'Mathematical Software', 'Numerical Analysis', 'Neural and Evolutionary Computing', 'Networking and Internet Architecture', 'Other Computer Science', 'Operating Systems', 'Performance', 'Programming Languages', 'Robotics', 'Symbolic Computation', 'Sound', 'Software Engineering', 'Social and Information Networks', 'Systems and Control']
else:
    all_labels = ['Yes', 'Nope']


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
        embedding_mask_id =  self.tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0]
        batch['embedding_positions'] = []
        for i in range(len(examples)):
            batch['embedding_positions'].append(torch.nonzero(batch['input_ids'][i] == embedding_mask_id).flatten())
        return batch

if __name__ == "__main__":
    set_seed(42)
    device = torch.device('cuda:0')
    # my_writer = SummaryWriter('zero_shot_cora_3B')
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
    ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/{}_graph_embedding_QA'.format(global_ds_name))
    # ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/{}_graph_embedding_QA_pure_nodes_with_element_type_count_cot'.format(global_ds_name))
    if global_ds_name == 'arxiv' or global_ds_name == 'mutag' or global_ds_name == 'molhiv':
        test_set_ids = np.where((np.array(ds['split_set']) == 'test') & (np.array(ds['task_type']) == 'classification'))[0]
    else:
        test_set_ids = np.where((np.array(ds['task_type']) == 'classification'))[0]
    print("=============length of :",len(test_set_ids))
    print("==================batch size:", batch_size)
    
    # base_model_path = 'base_models/llama3_2_3B_Instruct_pretrain_lr03_ckpt_197000'
    base_model_path = 'base_models/llama3_2_1B_Instruct_pretrain_lr03_ckpt_149000'
    print(base_model_path)
    print()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # tokenizer.add_tokens([embedding_mask_str, begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str])
    tokenizer.pad_token = '<|finetune_right_pad_id|>'
    def chat_map(dp):
        QA_json = {}
        question_str = dp.pop('question')
        answer_str = dp.pop('answer')
        question_str = question_str.replace(":Diabetes Mellitus, Experimental; Diabetes Mellitus Type 1; Diabetes Mellitus Type 2",
                                            ": Diabetes Type 1; Diabetes Type 2; Diabetes Experiments")
        question_str = question_str.replace('Diabetes Mellitus, Experimental', 'Diabetes Experiments')
        question_str = question_str.replace('Diabetes Mellitus Type 1', 'Diabetes Type 1')
        question_str = question_str.replace('Diabetes Mellitus Type 2', 'Diabetes Type 2')
        answer_str = answer_str.replace('Diabetes Mellitus, Experimental', 'Diabetes Experiments')
        answer_str = answer_str.replace('Diabetes Mellitus Type 1', 'Diabetes Type 1')
        answer_str = answer_str.replace('Diabetes Mellitus Type 2', 'Diabetes Type 2')
        QA_json['full'] = [
            {
                'role': 'user',
                'content': question_str
            },
            {
                'role': 'assistant',
                'content': answer_str
            }
        ]
        QA_json['question_only'] = [
            {
                'role': 'user',
                'content': question_str
            }
        ]
        full_chat = tokenizer.apply_chat_template(QA_json['full'],return_tensors='pt')[0]
        question_only = tokenizer.apply_chat_template(QA_json['question_only'],return_tensors='pt',add_generation_prompt=True)[0]
        dp['input_ids'] = full_chat.tolist() #tokenizer.encode(full_chat,add_special_tokens=False,return_tensors='pt')
        labels = full_chat.clone()
        prompt_length = question_only.shape[0]
        labels[:prompt_length] = -100
        dp['labels'] = labels.tolist()
        dp['length'] = len(dp['input_ids'])
        # dp.pop('type')
        return dp
    
    ds = ds.select(test_set_ids)
    ds = ds.map(chat_map,num_proc=16)
    # import ipdb;ipdb.set_trace()
   
    use_value_head = False
    # parent_checkpoint_path = 'ckpts/llama3_2_3B_Instruct/r_16_with_link_eval_test_set2025-05-08 15:35:35.357742'
    parent_checkpoint_path = 'ckpts/llama3_2_1B_Instruct/r_16_with_link_eval_test_set2025-05-09 01:24:22.768621'
    
    ckpts_dirs = ['160000','105000']#,'150000','160000','167000']#,'67000','80000']#,'24000','25000']
    count_dict = {}
    # for ckpt_index in range(10000, 100000,1000):
    for curr_ckpt in ckpts_dirs:
        # curr_ckpt = str(ckpt_index)
        peft_model = None
        base_model = None
        torch.cuda.empty_cache()
        base_model = LLama4Graph.from_pretrained(base_model_path)
        # base_model = LLama4Graph.from_pretrained("base_models/llama3_2_3B_Instruct_arxiv_Graph_QA_LoRA_node_combine_pretrain_1_1_checkpoint-78000_merged").to(device)
        base_model.embedding_mask_id = tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0]
        if  curr_ckpt != 'zero_shot':
            if not use_value_head:
                base_model.load_adapter('{}/checkpoint-{}/lora_adapter'.format(parent_checkpoint_path,curr_ckpt))
                peft_model = base_model
            else:
                peft_model = LLama4GraphWithValueHead.from_pretrained(base_model)
                peft_model.load_state_dict(torch.load('{}/checkpoint-{}/pytorch_model.bin'.format(parent_checkpoint_path,curr_ckpt),map_location='cpu'),strict=False)
                peft_model.pretrained_model.load_adapter('{}/checkpoint-{}'.format(parent_checkpoint_path,curr_ckpt))
            
            peft_model.generation_config = base_model.generation_config
        else:
            peft_model = base_model
        if use_half:
            peft_model = peft_model.half()
        peft_model = peft_model.to(device)
        peft_model.generation_config.do_sample = False
        peft_model.generation_config.top_p = 1
        peft_model.generation_config.temperature = 1
        
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
                    if use_half:
                        unaligned_inputs_embeds = batch_data['unaligned_inputs_embeds'][i].to(device).half()
                    else:
                        unaligned_inputs_embeds = batch_data['unaligned_inputs_embeds'][i].to(device)
                    if not use_value_head:
                        aligned_embeds = peft_model.node_embedding_connect(unaligned_inputs_embeds)
                        
                        # if use_half:
                        #     aligned_embeds = peft_model.node_embedding_connect(batch_data['unaligned_inputs_embeds'][i].to(device).half())
                        # else:
                        #     aligned_embeds = peft_model.node_embedding_connect(batch_data['unaligned_inputs_embeds'][i].to(device))#.half())
                    else:
                        aligned_embeds = peft_model.pretrained_model.node_embedding_connect(unaligned_inputs_embeds)
                        # aligned_embeds = peft_model.pretrained_model.node_embedding_connect(batch_data['unaligned_inputs_embeds'][i].to(device))
                    
                    new_position = batch_data['embedding_positions'][i]
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
                # import ipdb;ipdb.set_trace()
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
        correct_output_list = []
        error_output_list = []
        T_F_correct = 0
        eot_id = tokenizer.encode('<|eot_id|>',add_special_tokens=False)[0]
        with tqdm(range(len(preds))) as pbar:
        
            for i in pbar:
                labels = torch.tensor(ds[i]['labels'])
                len_count.append(len(ds[i]['input_ids']))
                if ds[i]['truncated']:
                    correct_count['truncated'] += 1
                if "Yes" in tokenizer.decode(labels[labels != -100][:-1]):
                    # print(tokenizer.decode(preds[i]))
                    # import ipdb; ipdb.set_trace()
                    y_true.append(1)
                else:
                    y_true.append(0)
                if "Yes" in tokenizer.decode(preds[i]):
                    y_pred.append(1)
                elif "Nope" in tokenizer.decode(preds[i]):
                    y_pred.append(0)
                else:
                    y_pred.append(-1)
                if y_pred[-1] == y_true[-1]:
                    T_F_correct += 1
                if tokenizer.decode(labels[labels != -100][:-1]).lower() in tokenizer.decode(preds[i]).lower():
                    correct_dp = {}
                    correct_dp['label'] = tokenizer.decode(labels[labels != -100]).lower()
                    correct_dp['preds'] = tokenizer.decode(preds[i][preds[i] != eot_id]).lower()
                    correct_output_list.append(correct_dp)
                    label_list.append(tokenizer.decode(labels[labels != -100][:-1]))
                    correct += 1
                    idxs.append(i)
                    if ds[i]['truncated']:
                        correct_count['correct_truncated'] += 1
                    if is_all_label_contained(tokenizer.decode(preds[i])):
                        correct -= 1
                else:
                    error_dp = {}
                    error_dp['label'] = tokenizer.decode(labels[labels != -100]).lower()
                    error_dp['preds'] = tokenizer.decode(preds[i][preds[i] != eot_id]).lower()
                    error_output_list.append(error_dp)
                # else:
                #     if "Yes" not in tokenizer.decode(preds[i]) and "Nope" not in tokenizer.decode(preds[i]):
                #         print(tokenizer.decode(preds[i]))
                #         print(tokenizer.decode(labels[labels != -100][1:-1]))
        #         #         print('==================================================')
        # if len(correct_output_list) != 0:
        #     correct_ds = Dataset.from_list(correct_output_list)        
        #     correct_ds.save_to_disk('pubmed_check/{}_{}_{}_correct_bs_1'.format(base_model_path.split('/')[-1],global_ds_name, curr_ckpt))
        # if len(error_output_list) != 0:
        #     error_ds = Dataset.from_list(error_output_list)
        #     error_ds.save_to_disk('pubmed_check/{}_{}_{}_error_bs_1'.format(base_model_path.split('/')[-1],global_ds_name, curr_ckpt))
        # print(idxs)
        # import ipdb; ipdb.set_trace()
        evaluator = Evaluator(name='ogbg-molhiv')
        if use_value_head:
            roc_auc = evaluator.eval({'y_true':np.expand_dims(np.array(y_true),axis=1),'y_pred':np.expand_dims(np.array(values),axis=1)})['rocauc']
        else:
            roc_auc = 0
        len_count = torch.tensor(len_count)
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
        true_positive = torch.logical_and(y_pred == 1, y_true == 1).sum()
        false_positive = torch.logical_and(y_pred == 1, y_true == 0).sum()
        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (y_true == 1).sum()
        F1 = 2 * precision * recall / (precision + recall)
        print("ACC {:.4f} ROC_AUC:{} F1:{:.4f} tp:{} fp :{} Precision:{:.4f} Recall:{:.4f} T_C_ACC: {:.4f}".format(correct/ len(test_set_ids),roc_auc, F1.item(), true_positive.item(),false_positive.item(), precision.item(), recall.item(),T_F_correct / len(test_set_ids)))
        print('ckpt-{} correct number: {}'.format(curr_ckpt,correct))
        correct_count['overall_correct_number'] = correct
        correct_count['below_500_recall'] = torch.logical_and(torch.logical_and(len_count < 500, y_pred == 1), y_true == 1).sum()
        correct_count['ROC_AUC'] = roc_auc
        correct_count['F1'] = F1.item()
        correct_count['tp'] = true_positive.item()
        correct_count['fp'] = false_positive.item()
        correct_count['acc'] = correct/ len(test_set_ids) 
        correct_count['T_C_ACC'] = T_F_correct / len(test_set_ids)
        
        # torch.save(label_list, "{}_labels.pt".format(save_name))
        # my_writer.add_scalar('accuracy',correct/ len(test_set_ids), ckpt_index)
        print(correct_count)
        # import ipdb; ipdb.set_trace()
        count_dict[curr_ckpt] = correct_count
    print(count_dict)
    print(parent_checkpoint_path)
    print(base_model_path)
    print(global_ds_name)
    print('use Half: ', use_half)

