from transformers import DefaultDataCollator,AutoTokenizer,AutoModelForCausalLM
from datasets import Dataset,load_dataset,load_from_disk, concatenate_datasets
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
# from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import os,re,sys
sys.path.append(os.path.abspath(os.path.curdir))
from data_utils.prompt_str import embedding_mask_str, begin_of_edges_str


def getCoraLLMPretrainDataset(text_path,graph_path, tokenizer:AutoTokenizer):
    graphML_list = torch.load(graph_path)
    text_embeds = torch.load(text_path)
    def chat_json_map(row):
        row_json = [
            {'role':"user","content":row['input']},
            {'role':"assistant","content":row['label']}
        ]
        return tokenizer.apply_chat_template(row_json,tokenize=False)
    graph_chat_list = list(map(chat_json_map, graphML_list))
    for i in range(len(text_embeds)):
        label_text = text_embeds[i].pop('label')
        text_embeds[i]['labels'] = tokenizer.encode(label_text)
    combined_list = []
    first_text_embed = text_embeds[0]
    first_graph_ML = tokenizer(graph_chat_list[0])
    for key in first_graph_ML.keys():
        if key not in first_text_embed.keys():
            first_text_embed[key] = None
    combined_list.append(first_text_embed)
    combined_list.append(first_graph_ML)
    for i in range(1,len(text_embeds)):
        combined_list.append(text_embeds)
        combined_list.append(graph_chat_list)
    
def getCoraEmbeds2TextDataset(text_path,tokenizer:AutoTokenizer):
    text_embeds = torch.load(text_path)
    labels = []
    for i in range(len(text_embeds)):
        label_text = text_embeds[i].pop('label')
        labels.append(label_text)
        text_embeds[i]['input_ids'] = tokenizer.encode(label_text)
        text_embeds[i]['input_ids'].append(tokenizer.eos_token_id)
        text_embeds[i]['input_embedding'] = torch.tensor(text_embeds[i]['input_embedding']).unsqueeze(0)
        # data_list.append(text_embeds[i])
    
    # CoraEmbeds2TextDataset = Dataset.from_list(data_list)
    CoraEmbeds2TextDataset = Dataset.from_list(text_embeds)
    return CoraEmbeds2TextDataset
def removeTooLongDatapoint(dataset, max_length=500):
    dp_index = []
    for i in range(dataset.num_rows):
        if len(dataset[i]['input_ids']) < max_length:
            dp_index.append(dataset[i])
    return dataset.select(dp_index)
def getArxivEmbeds2TextDataset(data_dir,max_length=500):
    raw_dataset = load_from_disk(data_dir)
    dp_list = []
    for i in range(raw_dataset.num_rows):
        if len(raw_dataset[i]['input_ids']) < max_length:
            dp_list.append(raw_dataset[i])
    return Dataset.from_list(dp_list)

def getImageTestDataset(tokenizer:AutoTokenizer,data_dir=None):
    if data_dir is None:
        dataset = load_dataset("cifar10")
        imageEncoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        train_data_list = []
        test_data_list = []
        label_to_text = {
        0:"Airplane",
        1:"Automobile",
        2:"Bird",
        3:"Cat",
        4:"Deer",
        5:"Dog",
        6:"Frog",
        7:"Horse",
        8:"Ship",
        9:"Truck",
        }
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            imageEncoder = imageEncoder.to(device)
            # processor = processor.to(device)
        def get_patchs(img, processor, model):
            with torch.no_grad():
                inputs = processor(images=img,return_tensors='pt')
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.device("cuda:0"))
                patch_embeddings = model.vision_model(**inputs,output_hidden_states=True).last_hidden_state[:,1:,].cpu()
            return patch_embeddings[0]
        for dp in dataset['train']:
            input_dp = {}
            patches = get_patchs(dp['img'],processor, imageEncoder)
            input_dp['input_embedding'] = patches
            input_text = "This is an image of {}.".format(label_to_text[dp['label']])
            input_ids = tokenizer.encode(input_text)
            input_ids.append(tokenizer.eos_token_id)
            input_dp['input_ids'] = input_ids
            # labels = input_ids.copy()
            # labels[0:-3] = [-100] * len(labels[0:-3])
            train_data_list.append(input_dp)
        print('train process ready')
        # train_dataset = Dataset.from_list(train_data_list)
        for dp in dataset['test']:
            input_dp = {}
            patches = get_patchs(dp['img'],processor, imageEncoder)
            input_dp['input_embedding'] = patches
            input_text = "This is an image of {}.".format(label_to_text[dp['label']])
            input_ids = tokenizer.encode(input_text)
            input_ids.append(tokenizer.eos_token_id)
            input_dp['input_ids'] = input_ids
            test_data_list.append(input_dp)
        # test_dataset = Dataset.from_list(test_data_list)
        train_dataset = Dataset.from_list(train_data_list)
        test_dataset = Dataset.from_list(test_data_list)
        train_dataset.save_to_disk('clip_cifar10_train')
        test_dataset.save_to_disk('clip_cifar10_test')
    else:
        train_dataset = load_from_disk(data_dir+'/clip_cifar10_train')
        test_dataset = load_from_disk(data_dir+'/clip_cifar10_test')
    print('pass dataset load')
    return train_dataset, test_dataset
    
def prepareCoraMultiTokenDataset(tokenizer:AutoTokenizer, data_dir,token_number,PLM:SentenceTransformer):
    text_embeds = torch.load(data_dir)
    dp_list = []
    all_decode_ids = []
    for i in range(len(text_embeds)):
        label_text = text_embeds[i].pop('label')
        text_embeds[i]['input_ids'] = tokenizer.encode(label_text)
        text_embeds[i]['input_ids'].append(tokenizer.eos_token_id)
        split_number = len(text_embeds[i]['input_ids']) // token_number
        for j in range(token_number - 1):
            all_decode_ids.append(text_embeds[i]['input_ids'][split_number*j:split_number*(j+1)])
        # all_decode_ids.append(text_embeds[i]['input_ids'])
        all_decode_ids.append(text_embeds[i]['input_ids'][split_number*token_number: ])
    sentences = tokenizer.batch_decode(all_decode_ids)
    print('decode finished')
    all_input_embeddings = PLM.encode(sentences,batch_size=256)
    for i in range(len(text_embeds)):
        dp = {}
        dp['input_ids'] = text_embeds[i]['input_ids']
        dp['input_embedding'] = all_input_embeddings[i*token_number:(i+1)*token_number]
        dp_list.append(dp)
    dataset = Dataset.from_list(dp_list)
    dataset.save_to_disk('datasets_local/CoraEmbeds2Text_{}TokenDataset'.format(token_number))
    

def prepareArxivMultiTokenDataset(tokenizer:AutoTokenizer, data_dir,token_number,PLM:SentenceTransformer):
    raw_dataset = load_from_disk(data_dir)
    dp_list = []
    all_decode_ids = []
    with tqdm(range(raw_dataset.num_rows)) as pbar:
        for i in pbar:

            input_ids = raw_dataset[i]['input_ids']
            split_number = len(input_ids) // token_number
            for j in range(token_number - 1):
                all_decode_ids.append(input_ids[split_number*j:split_number*(j+1)])
            all_decode_ids.append(input_ids[split_number*token_number: ])
    
    sentences = tokenizer.batch_decode(all_decode_ids)
    print('decode finished')
    all_input_embeddings = PLM.encode(sentences,batch_size=256)
    print(all_input_embeddings.shape)
    for i in range(raw_dataset.num_rows):
        dp = {}
        dp['input_ids'] = raw_dataset[i]['input_ids']
        dp['input_embedding'] = all_input_embeddings[i*token_number:(i+1)*token_number]
        dp_list.append(dp)
    dataset = Dataset.from_list(dp_list)
    dataset.save_to_disk('datasets_local/ArxivEmbeds2Text_{}TokensDataset'.format(token_number))

def change_node_order(ds, change_zero=False):
    def order_change_map(dp):
        unaligned_input_embeds = np.array(dp['unaligned_input_embeds'])
        new_unaligned_input_embeds = np.zeros_like(unaligned_input_embeds)
        order_map = {}
        if not change_zero:
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
        before_edge_str, after_edge_str = dp['question'].split(begin_of_edges_str)
        digits = re.split(r'\D+', after_edge_str)
        delimiters = re.findall(r'\D+', after_edge_str)
        remapped_digits = [order_map.get(token, token) for token in digits]
        result = before_edge_str + begin_of_edges_str + ''.join(t + d for t, d in zip(remapped_digits, delimiters)) + (remapped_digits[-1] if len(remapped_digits) > len(delimiters) else '')
        
        dp['question'] = result
        dp['unaligned_input_embeds'] = new_unaligned_input_embeds        
        return dp
    return ds.map(order_change_map,num_proc=32)
 
def load_dataset(dataset_dir,tokenizer:AutoTokenizer):
    if 'connector_pretrain' in dataset_dir:
        # pretrained_datasets = []
        # for dir_name  in os.listdir('datasets_local/connector_pretrain'):
        #     pretrained_datasets.append(load_from_disk(os.path.join('datasets_local/connector_pretrain',dir_name)))
        # train_dataset = concatenate_datasets(pretrained_datasets)
        train_dataset = load_from_disk('datasets_local/all_connector_pretrain')
        test_dataset = train_dataset.select(range(25355,25455))
    else:
        raise NotImplementedError
    # train_dataset.shuffle(seed=41)
    # test_dataset.shuffle(seed=41)
    return train_dataset, test_dataset
def old_load_dataset(dataset_dir,tokenizer:AutoTokenizer):
    if 'new_pretrain' in dataset_dir:
        pretrained_datasets = []
        for dir_name in os.listdir('datasets_local/new_reasoning_datasets'):
            if "pretrain" in dir_name:
                ds = load_from_disk(os.path.join('datasets_local/new_reasoning_datasets',dir_name))
                task_types = np.array(ds['type'])
                ds = ds.select(np.where(task_types != 'category')[0])
                pretrained_datasets.append(ds)
        train_dataset = concatenate_datasets(pretrained_datasets)
        test_dataset = train_dataset.select(range(100))
    elif "reasoning_pretrain" in dataset_dir:
        train_dataset = load_from_disk('datasets_local/new_reasoning_datasets/arxiv_graph_structure_understanding_by_title')
        train_dataset = concatenate_datasets([train_dataset, load_from_disk('datasets_local/new_reasoning_datasets/arxiv_pretrain')])
        test_dataset = train_dataset.select(range(100))
    elif 'rl_cold_start' in dataset_dir:
        train_dataset = load_from_disk('datasets_local/with_node_index/zero_shot_right_ds')
        test_dataset = train_dataset.select(range(100))
    elif 'structure_understanding' in dataset_dir:
        arxiv_dataset = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/arxiv_graph_embedding_QA')
        task_types = np.array(arxiv_dataset['task_type'])
        non_classification_train_ids = np.where(task_types != 'classification')[0]
        train_dataset = arxiv_dataset.select(non_classification_train_ids)
        np.random.shuffle(non_classification_train_ids)
        test_dataset = arxiv_dataset.select(non_classification_train_ids[0:1000])
    elif 'order_test' in dataset_dir:
        train_datasets = []
        test_datasets = []
        # zero_order_ds = load_from_disk('datasets_local/with_node_index/arxiv')
        order_changed_ds = load_from_disk('datasets_local/with_node_index/arxiv_order_changed')
        # train_list = np.where(np.array(zero_order_ds['split_set']) == 'train')[0].tolist()
        # test_list = np.where(np.array(zero_order_ds['split_set']) == 'valid')[0].tolist()
        # train_datasets.append(zero_order_ds.select(train_list))
        # test_datasets.append(zero_order_ds.select(test_list[0:200]))
        
        
        
        train_list = np.where(np.array(order_changed_ds['split_set']) == 'train')[0].tolist()
        test_list = np.where(np.array(order_changed_ds['split_set']) == 'valid')[0].tolist()[0:200]
        train_datasets.append(order_changed_ds.select(train_list))
        test_datasets.append(order_changed_ds.select(test_list))
        
        
        arxiv_dataset = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/arxiv_graph_embedding_QA')
        
        splits = np.array(arxiv_dataset['split_set'])
        task_types = np.array(arxiv_dataset['task_type'])
        
        pure_classification_train_ids = np.where((splits == 'train') & (task_types == 'classification'))[0].tolist()
        arxiv_train_classification_ds = arxiv_dataset.select(pure_classification_train_ids)
        
        non_classificaton_ids = np.where((task_types != 'classification'))[0].tolist()[:len(pure_classification_train_ids)]
        arxiv_train_non_classification_ds = arxiv_dataset.select(non_classificaton_ids)
        
        arxiv_valid_ids = np.where((splits == 'test') & (task_types == 'classification'))[0].tolist()[0:200]
        arxiv_valid_ds = arxiv_dataset.select(arxiv_valid_ids)
        
        train_datasets.append(arxiv_train_classification_ds)
        train_datasets.append(arxiv_train_non_classification_ds)
        test_datasets.append(arxiv_valid_ds)
        
        
        
        train_dataset = concatenate_datasets(train_datasets)
        test_dataset = concatenate_datasets(test_datasets)
        
    elif 'rl_full_batch' in dataset_dir:
        train_datasets = []
        test_datasets = []
        arxiv_dataset = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/arxiv_graph_embedding_QA')
        train_list = np.where((np.array(arxiv_dataset['split_set']) == 'train') & (np.array(arxiv_dataset['task_type']) == 'classification'))[0].tolist()
        test_list = np.where((np.array(arxiv_dataset['split_set']) == 'valid') & (np.array(arxiv_dataset['task_type']) == 'classification'))[0].tolist()
        train_datasets.append(arxiv_dataset.select(train_list))
        test_datasets.append(arxiv_dataset.select(test_list[0:20]))
        
        # pubmed_dataset = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/pubmed_graph_embedding_QA')
        # train_list = np.where(np.array(pubmed_dataset['split_set']) == 'train')[0].tolist()
        # test_list = np.where(np.array(pubmed_dataset['split_set']) == 'valid')[0].tolist()
        # train_datasets.append(pubmed_dataset.select(train_list))
        # test_datasets.append(pubmed_dataset.select(test_list[0:20]))
        
        # cora_dataset = load_from_disk('datasets_local/with_node_index/cora')
        # train_list = np.where(np.array(cora_dataset['split_set']) == 'train')[0].tolist()
        # test_list = np.where(np.array(cora_dataset['split_set']) == 'valid')[0].tolist()
        # train_datasets.append(cora_dataset.select(train_list))
        # test_datasets.append(cora_dataset.select(test_list[0:10]))
        
        train_dataset = concatenate_datasets(train_datasets)
        test_dataset = concatenate_datasets(test_datasets)
        
    elif 'rl_explore' in dataset_dir:
        train_datasets = [] 
        test_datasets = []
        arxiv_dataset = load_from_disk('datasets_local/with_node_index/arxiv_reasoning_based_on_llama3.2')
        train_list = np.where(np.array(arxiv_dataset['split_set']) == 'train')[0].tolist()
        test_list = train_list[0:100]
        train_datasets.append(arxiv_dataset.select(train_list))
        test_datasets.append(arxiv_dataset.select(test_list))
        
        pubmed_dataset = load_from_disk('datasets_local/with_node_index/pubmed_reasoning_based_on_llama3.2')
        train_list = np.where(np.array(pubmed_dataset['split_set']) == 'train')[0].tolist()
        test_list = train_list[0:50] 
        train_datasets.append(pubmed_dataset.select(train_list))
        test_datasets.append(pubmed_dataset.select(test_list))
        # train_datasets.append(load_from_disk('datasets_local/with_node_index/cora'))
        train_dataset = concatenate_datasets(train_datasets)
        # train_dataset.shuffle(seed = 354)
        test_dataset = concatenate_datasets(test_datasets)
    elif 'rl_sft_warmup' in dataset_dir:
        train_datasets = []
        
        train_datasets.append(load_from_disk('datasets_local/with_node_index/arxiv_reasoning_based_on_QwQ'))
        # train_datasets.append(load_from_disk('datasets_local/with_node_index/arxiv_reasoning_based_on_llama3.2'))
        train_dataset = concatenate_datasets(train_datasets)
        # train_dataset.shuffle(seed=354)
        test_dataset = train_dataset.select(np.arange(100))
    elif 'pretrain' in dataset_dir:
        pretrained_datasets = []
        for dir_name in os.listdir('datasets_local/json_texts_datasets'):
            if 'pretrain' in dir_name:
                pretrained_datasets.append(load_from_disk(os.path.join('datasets_local//json_texts_datasets',dir_name)))
        dataset = concatenate_datasets(pretrained_datasets)
        train_dataset = dataset
        test_dataset = dataset.select(range(1000))
    elif "prediction" in dataset_dir:
        arxiv_dataset = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/arxiv_graph_embedding_QA')
        
        splits = np.array(arxiv_dataset['split_set'])
        task_types = np.array(arxiv_dataset['task_type'])
        
        pure_classification_train_ids = np.where((splits == 'train') & (task_types == 'classification'))[0].tolist()
        # arxiv_train_classification_ds = arxiv_dataset.select(pure_classification_train_ids)
        
        arxiv_new_ds = load_from_disk('datasets_local/with_node_index/arxiv')
        splits = np.array(arxiv_new_ds['split_set'])
        pure_classification_train_ids = np.where((splits == 'train'))[0].tolist()
        arxiv_train_classification_ds = arxiv_new_ds.select(pure_classification_train_ids)
        
        arxiv_change_order_train_classification_ds = change_node_order(arxiv_train_classification_ds,False)
        arxiv_zero_change_order_train_classification_ds = change_node_order(arxiv_train_classification_ds, True)
        
        arxiv_train_classification_ds = concatenate_datasets([arxiv_train_classification_ds, arxiv_change_order_train_classification_ds, arxiv_zero_change_order_train_classification_ds])
        
        non_classificaton_ids = np.where((task_types != 'classification'))[0].tolist()[:len(pure_classification_train_ids)]
        arxiv_train_non_classification_ds = arxiv_dataset.select(non_classificaton_ids)
        
        # truncated = np.array(arxiv_new_ds['truncated'])
        arxiv_valid_ids = np.where((splits == 'test'))[0].tolist()
        np.random.shuffle(arxiv_valid_ids)
        arxiv_valid_ids = arxiv_valid_ids[0:3000]
        arxiv_valid_ds = arxiv_new_ds.select(arxiv_valid_ids)

        # arxiv_title_recovery_ds = load_from_disk('datasets_local/with_node_index/arxiv_title_recovery')
        # split = np.array(arxiv_title_recovery_ds['split_set'])
        # recovery_list = np.where((split == 'train'))[0]
        # arxiv_title_recovery_ds = arxiv_title_recovery_ds.select(recovery_list)
        # arxiv_zero_change_title_recovery_ds = change_node_order(arxiv_title_recovery_ds, True)
        
        # arxiv_train_non_classification_ds = concatenate_datasets([arxiv_train_classification_ds, arxiv_title_recovery_ds, arxiv_zero_change_title_recovery_ds])
        
        
        
        # arxiv_change_order_test_classification_ds = change_node_order(arxiv_valid_ds, False)
        # arxiv_zero_change_order_test_classification_ds = change_node_order(arxiv_valid_ds, True)
        
        # arxiv_valid_ds = concatenate_datasets([arxiv_valid_ds, arxiv_change_order_test_classification_ds, arxiv_zero_change_order_test_classification_ds])
        
        ### molhiv
        molhiv_positive_ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/molhiv_graph_embedding_QA_pure_nodes_with_element_type_count_cot_pure_positive')
        splits = np.array(molhiv_positive_ds['split_set'])
        molhiv_train_positive_ids = np.where(splits == 'train')[0].tolist()
        molhiv_test_positive_ids = np.where(splits == 'valid')[0].tolist()
        
        molhiv_train_ds =  molhiv_positive_ds.select(molhiv_train_positive_ids)
        molhiv_test_ds = molhiv_positive_ds.select(molhiv_test_positive_ids)
        
        molhiv_negative_ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/molhiv_graph_embedding_QA_pure_nodes_with_element_type_count_cot_pure_negative')
        splits = np.array(molhiv_negative_ds['split_set'])
        molhiv_train_negative_ids = np.where(splits == 'train')[0].tolist()[:5 * len(molhiv_train_positive_ids)]
        molhiv_test_negative_ids = np.where(splits == 'valid')[0].tolist()[: 3 * len(molhiv_test_positive_ids)]
        
        molhiv_train_ds = concatenate_datasets([molhiv_train_ds,molhiv_negative_ds.select(molhiv_train_negative_ids)])
        molhiv_test_ds = concatenate_datasets([molhiv_test_ds, molhiv_negative_ds.select(molhiv_test_negative_ids)])
        
        ### mutag 
        
        mutag_ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/mutag_graph_embedding_QA_pure_nodes_with_element_type_count_cot')
        splits = np.array(mutag_ds['split_set'])
        train_ids = np.where(splits == 'train')[0].tolist()
        test_ids = np.where(splits == 'valid')[0].tolist()
        mutag_train_ds = mutag_ds.select(train_ids)
        mutag_test_ds = mutag_ds.select(test_ids)
        ###
        cora_link_ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/cora_link_prediction')
        splits = np.array(cora_link_ds['split_set'])
        train_ids = np.where(splits == 'train')[0].tolist()
        test_ids = np.where(splits == 'valid')[0].tolist()
        cora_train_link_ds = cora_link_ds.select(train_ids)
        cora_test_link_ds = cora_link_ds.select(test_ids)
        
        
        train_dataset = concatenate_datasets([arxiv_train_classification_ds,arxiv_train_non_classification_ds, molhiv_train_ds,mutag_train_ds,cora_train_link_ds])
    
        test_dataset = concatenate_datasets([arxiv_valid_ds,molhiv_test_ds,mutag_test_ds,cora_test_link_ds])
    
    elif "value_head_finetune" in dataset_dir:
        ### molhiv
        molhiv_positive_ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/molhiv_graph_embedding_QA_pure_nodes_with_element_type_count_cot_pure_positive')
        splits = np.array(molhiv_positive_ds['split_set'])
        molhiv_train_positive_ids = np.where(splits == 'train')[0].tolist()
        molhiv_test_positive_ids = np.where(splits == 'valid')[0].tolist()
        
        molhiv_train_ds =  molhiv_positive_ds.select(molhiv_train_positive_ids)
        molhiv_test_ds = molhiv_positive_ds.select(molhiv_test_positive_ids)
        
        molhiv_negative_ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/molhiv_graph_embedding_QA_pure_nodes_with_element_type_count_cot_pure_negative')
        splits = np.array(molhiv_negative_ds['split_set'])
        molhiv_train_negative_ids = np.where(splits == 'train')[0].tolist()[:5 * len(molhiv_train_positive_ids)]
        molhiv_test_negative_ids = np.where(splits == 'valid')[0].tolist()#[: 3 * len(molhiv_test_positive_ids)]
        
        molhiv_train_ds = concatenate_datasets([molhiv_train_ds,molhiv_negative_ds.select(molhiv_train_negative_ids)])
        molhiv_test_ds = concatenate_datasets([molhiv_test_ds, molhiv_negative_ds.select(molhiv_test_negative_ids)])
        
        ### mutag 
        
        mutag_ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/mutag_graph_embedding_QA_pure_nodes_with_element_type_count_cot')
        splits = np.array(mutag_ds['split_set'])
        train_ids = np.where(splits == 'train')[0].tolist()
        test_ids = np.where(splits == 'valid')[0].tolist()
        mutag_train_ds = mutag_ds.select(train_ids)
        mutag_test_ds = mutag_ds.select(test_ids)
        
        ### link
        cora_link_ds = load_from_disk('datasets_local/json_texts_datasets/prediction_datasets/cora_link_prediction')
        splits = np.array(cora_link_ds['split_set'])
        train_ids = np.where(splits == 'train')[0].tolist()
        test_ids = np.where(splits == 'valid')[0].tolist()
        cora_train_link_ds = cora_link_ds.select(train_ids)
        cora_test_link_ds = cora_link_ds.select(test_ids)
        
        
        train_dataset = concatenate_datasets([molhiv_train_ds,mutag_train_ds,cora_train_link_ds])
    
        test_dataset = concatenate_datasets([molhiv_test_ds])#mutag_test_ds])
        
    elif 'CoraPureEmbeds2Prediction_1TokenDataset' in dataset_dir:
        dataset = load_from_disk(dataset_dir)
        label_dict = {
            'Case Based':0, 'Genetic Algorithms':0, 'Neural Networks':0,'Probabilistic Methods':0, 'Reinforcement Learning':0, 'Rule Learning':0, 'Theory':0
        }
        train_list = []
        for i in range(len(dataset)):
            if label_dict[dataset[i]['text_label']] < 20:
                train_list.append(i)
                label_dict[dataset[i]['text_label']] += 1
            if len(train_list) >= 140:
                break
        train_dataset = dataset.select(train_list)
        test_dataset = dataset.select(range(len(dataset)-1000, len(dataset)))
    elif 'ArxivPureEmbeds2Prediction_1TokenDataset' in dataset_dir:
        dataset = load_from_disk(dataset_dir)
        data = DglNodePropPredDataset('ogbn-arxiv',root='/data/sharefile/wei/dataset')
        idx_split = data.get_idx_split()
        train_dataset = dataset.select(idx_split['train'])
        test_dataset = dataset.select(idx_split['test'])
    elif 'ArxivEmbedsGraphML2Prediction_1TokenDataset_TokenizerLlama-3.2-1B' in dataset_dir:
        dataset = load_from_disk(dataset_dir)
        data = DglNodePropPredDataset('ogbn-arxiv',root='/data/sharefile/wei/dataset')
        idx_split = data.get_idx_split()
        train_dataset = dataset.select(idx_split['train'])
        test_dataset = dataset.select(idx_split['test'])
    elif 'Llama-3.2-1B-Instruct/arxiv_pretrain' in dataset_dir:
        dataset = load_from_disk(dataset_dir)
        train_dataset = dataset
        print('changed')
        test_dataset = dataset.select(torch.randperm(len(dataset))[:500])
    elif 'Llama-3.2-3B-Instruct/arxiv_pretrain' in dataset_dir:
        dataset = load_from_disk(dataset_dir)
        train_dataset = dataset
        test_dataset = dataset.select(torch.randperm(len(dataset))[:500])
    elif 'Llama-3.2-3B-Instruct/arxiv_pureEmbeds_prediction' in dataset_dir:
        dataset = load_from_disk(dataset_dir)
        splits = np.array(dataset['split_set'])
        train_ids = np.where(splits == 'train')[0].tolist()
        val_ids = np.where(splits == 'valid')[0].tolist()
        train_dataset = dataset.select(train_ids)
        test_dataset = dataset.select(val_ids)
        # for i in range(len(dataset)):
        #     if dataset[i]['split_set'] == 'train':
        #         train_ids.append(i)
        #     elif dataset[i]['split_set'] == 'valid':
        #         val_ids.append(i)
        train_dataset = dataset.select(train_ids)
        test_dataset = dataset.select(val_ids)
    elif 'Llama-3.2-3B-Instruct/arxiv_graph_embedding_QA' in dataset_dir:
        dataset = load_from_disk(dataset_dir)
        splits = np.array(dataset['split_set'])
        task_types = np.array(dataset['task_type'])
        pure_classification_train_ids = np.where((splits == 'train') & (task_types == 'classification'))[0].tolist()
        non_classification_train_ids= np.where((task_types != 'classification'))[0].tolist()
        val_ids = np.where((splits == 'valid') & (task_types == 'classification'))[0].tolist()
        # train_ids = np.where((splits == 'train') | (task_types != 'classification'))[0].tolist()
        if len(non_classification_train_ids) != 0:
            # np.random.shuffle(non_classification_train_ids)
            pure_classification_train_ids.extend(non_classification_train_ids[0:len(pure_classification_train_ids)])
        train_ids = pure_classification_train_ids
        # for i in range(len(dataset)):
        #     if dataset[i]['task_type'] != 'classification':
        #         train_ids.append(i)
        #     else:
        #         if dataset[i]['split_set'] == 'train':
        #             train_ids.append(i)
        #         elif dataset[i]['split_set'] == 'valid':
        #             val_ids.append(i)
        train_dataset = dataset.select(train_ids)
        test_dataset = dataset.select(val_ids[:2000])  
    elif 'Llama-3.2-3B-Instruct/node_combine_pretrain' in dataset_dir:
        dataset = load_from_disk(dataset_dir)
        train_dataset = dataset.shuffle(seed=42)
        test_dataset = dataset.select(torch.randperm(len(dataset))[:2000])
    elif "Llama-3.2-3B-Instruct/node_graph_combine_pretrain" in dataset_dir:
        dataset = load_from_disk(dataset_dir)
        train_dataset = dataset.shuffle(seed=42)
        node_test_set = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/node_combine_pretrain_below_500')
        node_test_set = node_test_set.select(torch.randperm(len(node_test_set))[:2000])
        graph_test_set = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/molhiv_pretrain')
        graph_test_set = graph_test_set.select(torch.randperm(len(graph_test_set))[:2000])
        test_dataset = concatenate_datasets([node_test_set,graph_test_set])
    elif "Llama-3.2-3B-Instruct/node_combine_graph_embedding_QA" in dataset_dir:
        arxiv_dataset = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/arxiv_graph_embedding_QA_pure_classification_new')
        cora_dataset = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/cora_graph_embedding_QA')
        pubmed_dataset = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/pubmed_graph_embedding_QA')
        splits = np.array(arxiv_dataset['split_set'])
        task_types = np.array(arxiv_dataset['task_type'])
        pure_classification_train_ids = np.where((splits == 'train') & (task_types == 'classification'))[0].tolist()[:2000]
        non_classification_train_ids= np.where((task_types != 'classification'))[0].tolist()[:2000]
        val_ids = np.where((splits == 'valid') & (task_types == 'classification'))[0].tolist()[:1000]
        pure_classification_train_ids.extend(non_classification_train_ids)
        arxiv_train_dataset = arxiv_dataset.select(pure_classification_train_ids)
        arxiv_test_dataset = arxiv_dataset.select(val_ids)
        
        splits = np.array(cora_dataset['split_set'])
        task_types = np.array(cora_dataset['task_type'])
        pure_classification_train_ids = np.where((splits == 'train') & (task_types == 'classification'))[0].tolist()
        val_ids = np.where((splits == 'valid') & (task_types == 'classification'))[0].tolist()[:200]
        cora_train_dataset = cora_dataset.select(pure_classification_train_ids)
        cora_test_dataset = cora_dataset.select(val_ids)
        
        splits = np.array(pubmed_dataset['split_set'])
        task_types = np.array(pubmed_dataset['task_type'])
        pure_classification_train_ids = np.where((splits == 'train') & (task_types == 'classification'))[0].tolist()
        val_ids = np.where((splits == 'valid') & (task_types == 'classification'))[0].tolist()[:200]
        pubmed_train_dataset = pubmed_dataset.select(pure_classification_train_ids)
        pubmed_test_dataset = pubmed_dataset.select(val_ids)
        train_dataset = concatenate_datasets([arxiv_train_dataset, cora_train_dataset, pubmed_train_dataset])
        test_dataset = concatenate_datasets([arxiv_test_dataset, cora_test_dataset, pubmed_test_dataset])
    elif "Llama-3.2-3B-Instruct/node_graph_combine_classification_QA" in dataset_dir:
        arxiv_dataset = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/arxiv_graph_embedding_QA_pure_classification_new')
        splits = np.array(arxiv_dataset['split_set'])
        task_types = np.array(arxiv_dataset['task_type'])
        pure_classification_train_ids = np.where((splits == 'train') & (task_types == 'classification'))[0].tolist()[:10000]#[:6000]
        non_classification_train_ids= np.where((task_types != 'classification'))[0].tolist()[:5000]
        val_ids = np.where((splits == 'valid') & (task_types == 'classification'))[0].tolist()[:3000]
        pure_classification_train_ids.extend(non_classification_train_ids)
        arxiv_train_dataset = arxiv_dataset.select(pure_classification_train_ids)
        arxiv_test_dataset = arxiv_dataset.select(val_ids)
        
      
        
        molhiv_dataset = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/molhiv_graph_embedding_QA')
        splits = np.array(molhiv_dataset['split_set'])
        task_types = np.array(molhiv_dataset['task_type'])
        val_ids = np.where((splits == 'valid') & (task_types == 'classification'))[0].tolist()
        train_ids = np.where((splits == 'train') & (task_types == 'classification'))[0].tolist()
        molhiv_train_dataset = molhiv_dataset.select(train_ids)
        molhiv_test_dataset = molhiv_dataset.select(val_ids)
        
        train_dataset = molhiv_train_dataset
        test_dataset =  molhiv_test_dataset
        
        # molhiv_dataset = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/molhiv_graph_embedding_QA_below_1200')
        # splits = np.array(molhiv_dataset['split_set'])
        # task_types = np.array(molhiv_dataset['task_type'])
        # pure_classification_train_ids = np.where((splits == 'train') & (task_types == 'classification'))[0].tolist()#[:5000]
        
        # val_ids = np.where((splits == 'valid') & (task_types == 'classification'))[0].tolist()[:500]
        # molhiv_train_dataset = molhiv_dataset.select(pure_classification_train_ids)
        # molhiv_test_dataset = molhiv_dataset.select(val_ids)
        
        # pure_positive_ds = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/molhiv_graph_embedding_QA_pure_positive')
        # splits = np.array(pure_positive_ds['split_set'])
        # train_ids = np.where((splits =='train'))[0].tolist()
        # molhiv_pure_positive_train_ds = pure_positive_ds.select(train_ids)
        # val_ids= np.where((splits == 'val'))[0].tolist()
        # molhiv_pure_positive_test_ds = pure_positive_ds.select(val_ids)
        
        # train_dataset = concatenate_datasets([arxiv_train_dataset, molhiv_train_dataset])
        # test_dataset = concatenate_datasets([arxiv_test_dataset,molhiv_pure_positive_test_ds, molhiv_test_dataset])
        train_dataset = train_dataset.shuffle(seed=42)
        test_dataset = test_dataset.shuffle(seed=42)
    elif "Llama-3.2-3B-Instruct/molhiv_graph_embedding_QA_pure_nodes_with_element_type_count_cot" in dataset_dir:
        dataset = load_from_disk(dataset_dir)
        splits = np.array(dataset['split_set'])
        pure_classification_train_ids = np.where((splits == 'train'))[0].tolist()
        valid_ids = np.where((splits == 'valid'))[0].tolist()
        
        # train_dataset = dataset.select(pure_classification_train_ids)
        # test_dataset = dataset.select(valid_ids)
        positive_ds = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/molhiv_graph_embedding_QA_pure_nodes_with_element_type_count_cot_pure_positive')
        splits = np.array(positive_ds['split_set'])
        pure_positive_train_ids = np.where((splits == 'train'))[0].tolist()
        pure_positive_non_test_ids = np.where((splits != 'test'))[0].tolist()
        
        train_dataset = concatenate_datasets([dataset.select(pure_classification_train_ids[0: len(pure_positive_train_ids) * 8]),positive_ds.select(pure_positive_train_ids)])
        
        # test_dataset = concatenate_datasets([dataset.select(valid_ids[0:3000]), positive_ds.select(pure_positive_non_test_ids)])
        test_dataset = dataset.select(valid_ids)
        
    elif "Llama-3.2-3B-Instruct/molhiv_graph_embedding_QA_pure_nodes" in dataset_dir:
        print('loading dataset')
        dataset = load_from_disk(dataset_dir)
        splits = np.array(dataset['split_set'])
        task_types = np.array(dataset['task_type'])
        pure_classification_train_ids = np.where((splits == 'train') & (task_types == 'classification'))[0].tolist()
        val_ids = np.where((splits == 'valid') & (task_types == 'classification'))[0].tolist()
        
        positive_ds = load_from_disk('datasets_local/Llama-3.2-3B-Instruct/molhiv_graph_embedding_QA_pure_nodes_pure_positive')
        splits = np.array(positive_ds['split_set'])
        pure_positive_train_ids = np.where((splits =='train'))[0].tolist()
        
        pure_positive_non_test_ids = np.where((splits !='test'))[0].tolist()
        
        
        train_dataset = concatenate_datasets([dataset.select(pure_classification_train_ids[0: len(pure_positive_train_ids) * 5]),positive_ds.select(pure_positive_train_ids)]).shuffle(seed=42)
        test_dataset = concatenate_datasets([dataset.select(val_ids[0:2000]).shuffle(seed=42), positive_ds.select(pure_positive_non_test_ids)]).shuffle(seed=42)
        
    else:
        raise ValueError('dataset: {} not found'.format(dataset_dir))
    # train_dataset.shuffle(seed=42)
    # test_dataset.shuffle(seed=42)
    return train_dataset, test_dataset


# def getCoraEmbeds2TextDataset(text_path,tokenizer:AutoTokenizer):
#     text_embeds = torch.load(text_path)
#     for i in range(len(text_embeds)):
        
        
    
# class CoraLLMDataset(Dataset):
    
    # def __init__(self, arrow_table, info = None, split = None, indices_table = None, fingerprint = None):
    #     super().__init__(arrow_table, info, split, indices_table, fingerprint)
    #     graphML_list = torch.load('datasets/cora_graphML_1degree_1edge_1hop.pt')
    #     text_generation = torch.load('datasets/cora_embedding_to_text.pt')
        
    #     self._data = 
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('/data/sharefile/wei/.llama/HF_format/Meta-Llama3.1-8B')
    PLM_model = SentenceTransformer('all-mpnet-base-v2')
    prepareCoraMultiTokenDataset(tokenizer,'datasets_local/cora_embedding_to_text.pt',1,PLM_model)