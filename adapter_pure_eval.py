import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, EvalPrediction, DataCollatorForLanguageModeling,set_seed
from models.LLaMa import LLama4Graph, LLama4GraphWithValueHead
from data_utils.data_loader import getCoraEmbeds2TextDataset
from sentence_transformers import SentenceTransformer
# from ogb.nodeproppred import DglNodePropPredDataset
# from ogb.graphproppred import Evaluator
from datasets import load_from_disk,Dataset
import os
from transformers.data.data_collator import pad_without_fast_tokenizer_warning 
# from data_utils.data_collators import pad_without_fast_tokenizer_warning, InstructEmbedsPretrainCollator
from data_utils.prompt_str import embedding_mask_str, begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str
from tqdm.auto import tqdm
from typing import List
import numpy as np
from peft import PeftModel, get_peft_model, LoraConfig,TaskType
from torch.utils.data import DataLoader
import json,re,time
# from tensorboardX import SummaryWriter

import multiprocess as mp
from models.graphAdapter import GraphAdapter4CausalLM

global_ds_name = 'arxiv'
max_new_tokens = 800
batch_size = 64
use_half = True
change_order = False
zero_modify = False
process_number = 32

if global_ds_name == 'pubmed':
    # all_labels = ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
    all_labels = ['Diabetes Experiments', 'Diabetes Type 1', 'Diabetes Type 2']
elif global_ds_name == 'cora':
    all_labels = ['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning','Theory']
elif global_ds_name == 'arxiv':
    all_labels = ['Artificial Intelligence', 'Hardware Architecture', 'Computational Complexity', 'Computational Engineering, Finance, and Science', 'Computational Geometry', 'Computation and Language', 'Cryptography and Security', 'Computer Vision and Pattern Recognition', 'Computers and Society', 'Databases', 'Distributed, Parallel, and Cluster Computing', 'Digital Libraries', 'Discrete Mathematics', 'Data Structures and Algorithms', 'Emerging Technologies', 'Formal Languages and Automata Theory', 'General Literature', 'Graphics', 'Computer Science and Game Theory', 'Human-Computer Interaction', 'Information Retrieval', 'Information Theory', 'Machine Learning', 'Logic in Computer Science', 'Multiagent Systems', 'Multimedia', 'Mathematical Software', 'Numerical Analysis', 'Neural and Evolutionary Computing', 'Networking and Internet Architecture', 'Other Computer Science', 'Operating Systems', 'Performance', 'Programming Languages', 'Robotics', 'Symbolic Computation', 'Sound', 'Software Engineering', 'Social and Information Networks', 'Systems and Control']
else:
    all_labels = ['Yes', 'No']


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
        if 'unaligned_input_embeds' in examples[0].keys():
            batch['unaligned_inputs_embeds'] = [torch.tensor(example['unaligned_input_embeds'],device=batch['input_ids'].device) for example in examples]
            embedding_mask_id =  self.tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0]
            batch['embedding_positions'] = []
            for i in range(len(examples)):
                batch['embedding_positions'].append(torch.nonzero(batch['input_ids'][i] == embedding_mask_id).flatten())
        return batch

class ChangeOrderLeftPaddingPredictionCollator(LeftPaddingPredictionCollator):
    change_zero: bool = False
    
    def torch_call(self, examples):
        for i in range(len(examples)):
            example = examples[i]
            unaligned_input_embeds = np.array(example['unaligned_input_embeds'])
            new_unaligned_input_embeds = np.zeros_like(unaligned_input_embeds)
            order_map = {}
            if not self.change_zero:
                new_unaligned_input_embeds[0] = unaligned_input_embeds[0]
                new_order = (torch.randperm(unaligned_input_embeds.shape[0] - 1) + 1).tolist()
                for j in range(len(new_order)):
                    order_map[str(j + 1)] = str(new_order[j])
                    new_unaligned_input_embeds[j+1] = unaligned_input_embeds[new_order[j]]
            else:
                new_order = (torch.randperm(unaligned_input_embeds.shape[0])).tolist()
                for j in range(len(new_order)):
                    order_map[str(j)] = str(new_order[j])
                    new_unaligned_input_embeds[j] = unaligned_input_embeds[new_order[j]]
            
            before_edge_str, after_edge_str = self.tokenizer.decode(example['input_ids']).split(begin_of_edges_str)
            digits = re.split(r'\D+', after_edge_str)
            delimiters = re.findall(r'\D+', after_edge_str)
            remapped_digits = [order_map.get(token, token) for token in digits]
            result = before_edge_str + begin_of_edges_str + ''.join(t + d for t, d in zip(remapped_digits, delimiters)) + (remapped_digits[-1] if len(remapped_digits) > len(delimiters) else '')
            
            example['input_ids'] = self.tokenizer.encode(result,add_special_tokens=False)
            example['unaligned_inputs_embeds'] = new_unaligned_input_embeds
            examples[i] = example
        return super().torch_call(examples)
        
if __name__ == "__main__":
    set_seed(42)
    device = torch.device('cuda:0')
    
    '''
        This is for Graph Embedding QA prediction Task
    '''
    load_ds_name = 'arxiv_graph_embedding_QA'
    # ds = load_from_disk('datasets_local/with_node_index/arxiv')
    ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/arxiv_graph_embedding_QA')
    # ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/{}_graph_embedding_QA_pure_nodes_with_element_type_count_cot'.format(global_ds_name))
    # ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/cora_link_prediction')
    connector_path_dir = "ckpts/Qwen2_5_1_5B/combine_pretrain_no_add_tokens2025-09-17 14:35:49.274360/checkpoint-1570000/node_embedding_connect.pt"
    lora_dir = "ckpts/Qwen2_5_1_5B/combine_pretrain_no_add_tokens2025-09-17 14:35:49.274360/checkpoint-1570000/lora_adapter"
    correct_node_idxs = []
    if global_ds_name == 'arxiv' or global_ds_name == 'mutag' or global_ds_name == 'molhiv' or 'link' in global_ds_name or 'cora' in global_ds_name:
        test_set_ids = np.where((np.array(ds['split_set']) == 'test') * (np.array(ds['task_type'])== 'classification'))[0]
    else:
        # test_set_ids = np.arange(len(ds))
        test_set_ids = np.where((np.array(ds['task_type']) == 'classification'))[0]
    
    print("=============length of :",len(test_set_ids))
    print("==================batch size:", batch_size)
    
    base_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print(base_model_path)
    print(load_ds_name)
    print('change order:{} zero modify:{}'.format(change_order, zero_modify))
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens([embedding_mask_str])#,begin_of_nodes_str,end_of_nodes_str,begin_of_edges_str,end_of_edges_str,one_edge_str])
    embedding_mask_id = tokenizer.encode(embedding_mask_str, add_special_tokens=False)[0]
    def chat_map(dp):
        QA_json = {}
        question_str = dp.pop('question')
        answer_str = dp.pop('answer')
        # dp.pop('unaligned_input_embeds')
        question_str = question_str.replace(":Diabetes Mellitus, Experimental; Diabetes Mellitus Type 1; Diabetes Mellitus Type 2",
                                            ": Diabetes Type 1; Diabetes Type 2; Diabetes Experiments")
        question_str = question_str.replace('Diabetes Mellitus, Experimental', 'Diabetes Experiments')
        question_str = question_str.replace('Diabetes Mellitus Type 1', 'Diabetes Type 1')
        question_str = question_str.replace('Diabetes Mellitus Type 2', 'Diabetes Type 2')
        answer_str = answer_str.replace('Diabetes Mellitus, Experimental', 'Diabetes Experiments')
        answer_str = answer_str.replace('Diabetes Mellitus Type 1', 'Diabetes Type 1')
        answer_str = answer_str.replace('Diabetes Mellitus Type 2', 'Diabetes Type 2')
        question_str = question_str.split('Please classify the node 0 into')[0] + " Based on its own and other nodes' node features and the graph structure, classify node 0 into one of the following list: Artificial Intelligence; Hardware Architecture; Computational Complexity; Computational Engineering, Finance, and Science; Computational Geometry; Computation and Language; Cryptography and Security; Computer Vision and Pattern Recognition; Computers and Society; Databases; Distributed, Parallel, and Cluster Computing; Digital Libraries; Discrete Mathematics; Data Structures and Algorithms; Emerging Technologies; Formal Languages and Automata Theory; General Literature; Graphics; Computer Science and Game Theory; Human-Computer Interaction; Information Retrieval; Information Theory; Machine Learning; Logic in Computer Science; Multiagent Systems; Multimedia; Mathematical Software; Numerical Analysis; Neural and Evolutionary Computing; Networking and Internet Architecture; Other Computer Science; Operating Systems; Performance; Programming Languages; Robotics; Symbolic Computation; Sound; Software Engineering; Social and Information Networks; Systems and Control."
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
        full_chat = tokenizer.apply_chat_template(QA_json['full'],return_tensors='pt',enable_thinking=False)[0]
        question_only = tokenizer.apply_chat_template(QA_json['question_only'],return_tensors='pt',add_generation_prompt=True, enable_thinking=False)[0]
        dp['input_ids'] = full_chat.tolist() #tokenizer.encode(full_chat,add_special_tokens=False,return_tensors='pt')
        labels = full_chat.clone()
        prompt_length = question_only.shape[0]
        labels[:prompt_length] = -100
        dp['labels'] = labels.tolist()
        dp['length'] = len(dp['input_ids'])
        return dp
    ds = ds.map(chat_map,num_proc=16)
    ds = ds.select(test_set_ids)
    
    
    use_value_head = False

    ckpts_dirs = ['40000']
    # for i in range(450000,600000,5000):
    #     ckpts_dirs.append(str(i))
    # # ckpts_dirs = ['105000']#,'6001','7001','9601']#,'16000']#,'150000','160000','167000']#,'67000','80000']#,'24000','25000']
    count_dict = {}
    parent_checkpoint_path = 'ckpts/Qwen2_5_1_5B/no_add_tokens_rl_warmup_no_random2025-09-29 23:46:59.366590'
    for curr_ckpt in ckpts_dirs:
        # curr_ckpt = str(ckpt_index)
        torch.cuda.empty_cache()
        model = GraphAdapter4CausalLM.from_pretrained(base_model_path, embedding_mask_id,connect_dir=connector_path_dir,LoRA_dir=lora_dir)
        # model.node_embedding_connect.load_state_dict(torch.load(connector_path_dir,weights_only=False).state_dict())
        if  curr_ckpt != 'zero_shot':
            if not use_value_head:
                peft_model = PeftModel.from_pretrained(model,"{}/checkpoint-{}/lora_adapter".format(parent_checkpoint_path,curr_ckpt))
            else:
                peft_model = LLama4GraphWithValueHead.from_pretrained(base_model)
                peft_model.load_state_dict(torch.load('{}/checkpoint-{}/pytorch_model.bin'.format(parent_checkpoint_path,curr_ckpt),map_location='cpu'),strict=False)
                peft_model.pretrained_model.load_adapter('{}/checkpoint-{}'.format(parent_checkpoint_path,curr_ckpt))
            
            # peft_model.generation_config = base_model.generation_config
        else:
            peft_model = model
        if use_half:
            peft_model = peft_model.half()
        peft_model = peft_model.to(device)
        peft_model.base_model.generation_config.do_sample = False
        peft_model.base_model.generation_config.top_p = 1
        peft_model.base_model.generation_config.temperature = 1
        peft_model.base_model.generation_config.top_k = None
        
        if change_order:
            collator = ChangeOrderLeftPaddingPredictionCollator(tokenizer, mlm=False)
            collator.change_zero = zero_modify
        else:
            collator = LeftPaddingPredictionCollator(tokenizer,mlm=False)
        
        print('curr ckpt {}'.format(curr_ckpt))
        res_list = [] 
        raw_input_list =[]
        res_values = []
        peft_model.base_model.generation_config.pad_token_id = tokenizer.eos_token_id
        print(tokenizer.eos_token_id)
        test_dataloader = DataLoader(
            ds,
            batch_size = batch_size,
            collate_fn=collator,
            num_workers=4,
            shuffle=False,
        )
        correct_count = {'truncated':0,'correct_truncated': 0}
        reasoning_save_list = []
        with torch.no_grad():
            for index, batch_data in enumerate(tqdm(test_dataloader, desc='Testing Process')):
                for i in range(len(batch_data['unaligned_inputs_embeds'])):
                    batch_data['unaligned_inputs_embeds'][i] = batch_data['unaligned_inputs_embeds'][i].to(peft_model.node_embedding_connect[0].weight)
                batch_data['input_ids'] = batch_data['input_ids'].to(device)
                batch_data['attention_mask'] = batch_data['attention_mask'].to(device)
                batch_res = peft_model.generate(max_new_tokens=max_new_tokens,generation_config = peft_model.base_model.generation_config, **batch_data)
                
                if not use_value_head:
                    res_list.append(batch_res)
                else:
                    res_list.append(batch_res[0])
                    res_values.append(batch_res[1])  
        label_list = []
        preds = []
        values = []
        for i in range(len(res_list)):
            for j in range(res_list[i].shape[0]):
                preds.append(res_list[i][j].tolist())
                if use_value_head:
                    values.append(res_values[i][j].item())
                else:
                    values.append(0)
                
        all_correct = 0
        all_pure_predict_format_right = 0
        all_pure_predict_correct = 0
        idxs = []
        all_y_pred = []
        all_y_true = []
        all_len_count = []
        correct_output_list = []
        error_output_list = []
        T_F_correct = 0
        all_degree_count = {}
        all_degree_correct = {}
        origin_1hop_count = {}
        origin_1hop_correct = {}
        # truncated_num = 0
        # truncated_correct = 0 
        eot_id = tokenizer.encode('<|eot_id|>',add_special_tokens=False)[0]
        
        def get_check_data(start_index, end_index):
            T_F_correct = 0
            return_dict = {}
            return_dict['truncated'] = 0
            return_dict['correct_truncated'] = 0
            return_dict['all_degree_count'] = {}
            return_dict['all_degree_correct'] = {}
            return_dict['origin_1_hop_count'] = {}
            return_dict['origin_1_hop_correct'] = {}
            pure_predict_correct = 0
            pure_predict_format_right = 0
            correct = 0
            len_count = []
            y_true =  []
            y_pred = []
            for idx in range(start_index, end_index):
                labels = torch.tensor(ds[idx]['labels'])
                len_count.append(len(ds[idx]['input_ids']))
                current_degree = len(ds[idx]['unaligned_input_embeds'])
                # origin_1_hop_size = ds[idx]['origin_1hop_size'] + 1
                # origin_1_hop_size = min(origin_1_hop_size,11)
                if  current_degree not in return_dict['all_degree_count'].keys():
                    return_dict['all_degree_count'][current_degree] = 0
                    return_dict['all_degree_correct'][current_degree] = 0
                return_dict['all_degree_count'][current_degree] += 1
                
                # if origin_1_hop_size not in return_dict['origin_1_hop_count'].keys():
                #     return_dict['origin_1_hop_correct'][origin_1_hop_size] = 0
                #     return_dict['origin_1_hop_count'][origin_1_hop_size] = 0
                # return_dict['origin_1_hop_count'][origin_1_hop_size] += 1
                current_predict = tokenizer.decode(preds[idx]).split('assistant\n')[-1]
                if len(current_predict.split('</think>')) > 1:
                    pure_predict = current_predict.split('</think>')[-1]
                else:
                    pure_predict = None
                if ds[idx]['truncated']:
                    return_dict['truncated'] += 1
                if "Yes" in tokenizer.decode(labels[labels != -100][:-1]):
                    # print(tokenizer.decode(preds[idx]))
                    # import ipdb; ipdb.set_trace()
                    y_true.append(1)
                else:
                    y_true.append(0)
                if "Yes" in current_predict:
                    y_pred.append(1)
                elif "Nope" in current_predict or "No" in current_predict:
                    y_pred.append(0)
                else:
                    y_pred.append(-1)
                if y_pred[-1] == y_true[-1]:
                    T_F_correct += 1
                if tokenizer.decode(labels[labels != -100][:-1]).lower() in current_predict.lower():
                    correct_dp = {}
                    correct_dp['label'] = tokenizer.decode(labels[labels != -100]).lower()
                    correct_dp['preds'] = current_predict.lower()
                    correct += 1
                    return_dict['all_degree_correct'][current_degree] += 1
                    # return_dict['origin_1_hop_correct'][origin_1_hop_size] += 1
                    if is_all_label_contained(current_predict):
                        correct -= 1
                    if ds[idx]['truncated']:
                        return_dict['correct_truncated'] += 1
                if pure_predict is not None:
                    pure_predict_format_right += 1
                    if tokenizer.decode(labels[labels != -100][:-1]).lower() in pure_predict.lower() and not is_all_label_contained(pure_predict):
                        pure_predict_correct += 1
                    
            return_dict['len_count'] = len_count
            return_dict['correct'] = correct
            return_dict['y_pred'] = y_pred
            return_dict['y_true'] = y_true
            return_dict['pure_predict_format_right'] = pure_predict_format_right
            return_dict['pure_predict_correct'] = pure_predict_correct
            
            return return_dict
        
        my_pool = []
        seg= len(preds) // process_number
        bot = time.time()
        with mp.Pool(processes=process_number) as pool:
            for i in range(process_number):
                current_res = pool.apply_async(get_check_data,(seg * i, min(seg * (i+1), len(preds))))
                my_pool.append(current_res)
            for i in range(process_number):
                current_res = my_pool[i].get()
                all_correct += current_res['correct']
                all_pure_predict_format_right += current_res['pure_predict_format_right']
                all_pure_predict_correct += current_res['pure_predict_correct']
                all_y_pred.extend(current_res['y_pred'])
                all_y_true.extend(current_res['y_true'])
                all_len_count.extend(current_res['len_count'])
                correct_count['correct_truncated'] += current_res['correct_truncated']
                correct_count['truncated'] += current_res['truncated']
                
                for key in current_res['all_degree_count'].keys():
                    if key not in all_degree_count.keys():
                        all_degree_count[key] = 0
                        all_degree_correct[key] = 0
                    all_degree_correct[key] += current_res['all_degree_correct'][key]
                    all_degree_count[key] += current_res['all_degree_count'][key]
                
                # for key in current_res['origin_1_hop_count'].keys():
                #     if key not in origin_1hop_count.keys():
                #         origin_1hop_count[key] = 0
                #         origin_1hop_correct[key] = 0
                #     origin_1hop_correct[key] += current_res['origin_1_hop_correct'][key]
                #     origin_1hop_count[key] += current_res['origin_1_hop_count'][key]
        correct = all_correct
        y_pred = all_y_pred
        y_true = all_y_true
        len_count = all_len_count
        eot = time.time()
        print('eval_time:{}'.format(eot-bot))
        
        if use_value_head:
            roc_auc = 0 # evaluator.eval({'y_true':np.expand_dims(np.array(y_true),axis=1),'y_pred':np.expand_dims(np.array(values),axis=1)})['rocauc']
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
        print("ACC {:.4f} ROC_AUC:{} F1:{:.4f} tp:{} fp :{} Precision:{:.4f} Recall:{:.4f} T_C_ACC: {:.4f} Format_Right:{} Predict_Right: {}".format(correct/ len(test_set_ids),roc_auc, F1.item(), true_positive.item(),false_positive.item(), precision.item(), recall.item(),T_F_correct / len(test_set_ids), all_pure_predict_format_right, all_pure_predict_correct))
        # for i in range(1,12):
        #     if i in all_degree_count.keys():
        #         print("Degree acc {}:  {:.4} {}".format(i, all_degree_correct[i]/all_degree_count[i], all_degree_count[i]))
        # for i in range(1,12):
        #     if i in origin_1hop_count.keys():
        #         print("Original 1 hop acc {}: {:.4}".format(i, origin_1hop_correct[i] / origin_1hop_count[i]))
        print('ckpt-{} correct number: {}'.format(curr_ckpt,correct))
        correct_count['overall_correct_number'] = correct
        correct_count['below_500_recall'] = torch.logical_and(torch.logical_and(len_count < 500, y_pred == 1), y_true == 1).sum()
        correct_count['ROC_AUC'] = roc_auc
        correct_count['F1'] = F1.item()
        correct_count['tp'] = true_positive.item()
        correct_count['fp'] = false_positive.item()
        correct_count['acc'] = correct/ len(test_set_ids) 
        correct_count['T_C_ACC'] = T_F_correct / len(test_set_ids)
        correct_count['truncated_acc'] = correct_count['correct_truncated']/ correct_count['truncated']
        correct_count['non_truncate_acc'] = ( correct - correct_count['correct_truncated']) / (len(test_set_ids) - correct_count['truncated'])
        correct_count['pure_format_right'] = all_pure_predict_format_right
        correct_count['pure_predict_correct'] = all_pure_predict_correct
        
        # torch.save(label_list, "{}_labels.pt".format(save_name))
        # my_writer.add_scalar('accuracy',correct/ len(test_set_ids), ckpt_index)
        print(correct_count)
        # import ipdb; ipdb.set_trace()
        count_dict[curr_ckpt] = correct_count
        # print(correct_node_idxs)
    print(count_dict)
    print(base_model_path)
    print(global_ds_name)
    print(load_ds_name)
    print('use Half: ', use_half)
    print('change order:{} zero_modify:{}'.format(change_order, zero_modify))

