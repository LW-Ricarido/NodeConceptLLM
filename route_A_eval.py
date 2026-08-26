import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, EvalPrediction, DataCollatorForLanguageModeling, set_seed
from models.LLaMa import LLama4Graph, LLama4GraphWithValueHead
from data_utils.data_loader import getCoraEmbeds2TextDataset
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk, Dataset
import os
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from data_utils.prompt_str import (
    embedding_mask_str,
    begin_of_nodes_str,
    end_of_nodes_str,
    begin_of_edges_str,
    end_of_edges_str,
    one_edge_str,
)
from tqdm.auto import tqdm
from typing import List
import numpy as np
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
import json, re, time
from models.graphAdapter import GraphAdapter4CausalLM
from accelerate import Accelerator
from safetensors.torch import load_file as safe_load_file

# ================== 全局配置 ==================
global_ds_name = 'arxiv'
max_new_tokens = 10              # 现在不用 generate 了，这个参数不会再用到
batch_size = 64                  # 路线 A 版本我用的是单样本 loop，batch_size 这里暂时无效
use_half = True
change_order = False             # collator / 图节点顺序打乱相关，路线 A 未使用 collator
zero_modify = False
process_number = 32              # 路线 A 不再用 mp，多进程逻辑已删

# ================== 各数据集标签 ==================
if global_ds_name == 'pubmed':
    all_labels = ['Diabetes Experiments', 'Diabetes Type 1', 'Diabetes Type 2']
elif global_ds_name == 'cora':
    all_labels = ['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods',
                  'Reinforcement Learning', 'Rule Learning', 'Theory']
elif global_ds_name == 'arxiv':
    all_labels = [
        'Artificial Intelligence', 'Hardware Architecture', 'Computational Complexity',
        'Computational Engineering, Finance, and Science', 'Computational Geometry',
        'Computation and Language', 'Cryptography and Security',
        'Computer Vision and Pattern Recognition', 'Computers and Society', 'Databases',
        'Distributed, Parallel, and Cluster Computing', 'Digital Libraries',
        'Discrete Mathematics', 'Data Structures and Algorithms', 'Emerging Technologies',
        'Formal Languages and Automata Theory', 'General Literature', 'Graphics',
        'Computer Science and Game Theory', 'Human-Computer Interaction', 'Information Retrieval',
        'Information Theory', 'Machine Learning', 'Logic in Computer Science', 'Multiagent Systems',
        'Multimedia', 'Mathematical Software', 'Numerical Analysis',
        'Neural and Evolutionary Computing', 'Networking and Internet Architecture',
        'Other Computer Science', 'Operating Systems', 'Performance', 'Programming Languages',
        'Robotics', 'Symbolic Computation', 'Sound', 'Software Engineering',
        'Social and Information Networks', 'Systems and Control'
    ]
elif global_ds_name == 'products':
    all_labels = [
        "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors", "Books",
        "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl", "Cell Phones & Accessories",
        "Grocery & Gourmet Food", "Arts, Crafts & Sewing", "Clothing, Shoes & Jewelry",
        "Electronics", "Movies & TV", "Software", "Video Games", "Automotive", "Pet Supplies",
        "Office Products", "Industrial & Scientific", "Musical Instruments",
        "Tools & Home Improvement", "Magazine Subscriptions", "Baby Products", "NaN",
        "Appliances", "Kitchen & Dining", "Collectibles & Fine Art", "All Beauty",
        "Luxury Beauty", "Amazon Fashion", "Computers", "All Electronics", "Purchase Circles",
        "MP3 Players & Accessories", "Gift Cards", "Office & School Supplies",
        "Home Improvement", "Camera & Photo", "GPS & Navigation", "Digital Music",
        "Car Electronics", "Baby", "Kindle Store", "Buy a Kindle", "Furniture & Decor",
        "unknown label"
    ]
elif global_ds_name == 'WN18RR':
    all_labels = [
        "also_see", "derivationally_related_form", "has_part", "hypernym", "instance_hypernym",
        "member_meronym", "member_of_domain_region", "member_of_domain_usage", "similar_to",
        "synset_domain_topic_of", "verb_group"
    ]
else:
    # 默认二分类 Yes/No
    all_labels = ['Yes', 'No']


def is_all_label_contained(predicted_sentences: str) -> bool:
    cnt = 0
    for label in all_labels:
        if label.lower() in predicted_sentences.lower():
            cnt += 1
    return cnt == len(all_labels)


# ================ 原来的 Collator（路线 A 不再使用，但保留定义以防日后要用） ==================
class LeftPaddingPredictionCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            {
                'input_ids': [
                    np.array(example['input_ids'])[np.array(example['labels']) == -100]
                    for example in examples
                ]
            },
            return_tensors='pt',
            padding_side='left',
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch['attention_mask'] = torch.ones_like(batch['input_ids'])
        batch['attention_mask'][batch['input_ids'] == self.tokenizer.pad_token_id] = 0
        if 'unaligned_input_embeds' in examples[0].keys():
            batch['unaligned_inputs_embeds'] = [
                torch.tensor(example['unaligned_input_embeds'], device=batch['input_ids'].device)
                for example in examples
            ]
            embedding_mask_id = self.tokenizer.encode(embedding_mask_str, add_special_tokens=False)[0]
            batch['embedding_positions'] = []
            for i in range(len(examples)):
                batch['embedding_positions'].append(
                    torch.nonzero(batch['input_ids'][i] == embedding_mask_id).flatten()
                )
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
                    new_unaligned_input_embeds[j + 1] = unaligned_input_embeds[new_order[j]]
            else:
                new_order = (torch.randperm(unaligned_input_embeds.shape[0])).tolist()
                for j in range(len(new_order)):
                    order_map[str(j)] = str(new_order[j])
                    new_unaligned_input_embeds[j] = unaligned_input_embeds[new_order[j]]

            before_edge_str, after_edge_str = self.tokenizer.decode(example['input_ids']).split(begin_of_edges_str)
            digits = re.split(r'\D+', after_edge_str)
            delimiters = re.findall(r'\D+', after_edge_str)
            remapped_digits = [order_map.get(token, token) for token in digits]
            result = (
                before_edge_str
                + begin_of_edges_str
                + ''.join(t + d for t, d in zip(remapped_digits, delimiters))
                + (remapped_digits[-1] if len(remapped_digits) > len(delimiters) else '')
            )

            example['input_ids'] = self.tokenizer.encode(result, add_special_tokens=False)
            example['unaligned_input_embeds'] = new_unaligned_input_embeds
            examples[i] = example
        return super().torch_call(examples)


# ====================== 主程序 ======================
if __name__ == "__main__":
    set_seed(42)
    device = torch.device('cuda:0')

    # ====== 加载数据集，过滤 test & classification 任务 ======
    ds = load_from_disk('../../LLM4Graph_tracked/datasets_local/all_downstream_dataset_32')
    ds = ds.filter(
        lambda example: example['split_set'] == 'test'
        and example['dataset_name'] == global_ds_name
        and example['task_type'] == 'classification',
        num_proc=32
    ).select(range(5000))
    test_set_ids = np.arange(len(ds))

    print("=============length of :", len(test_set_ids))
    print("==================batch size (unused in Route A loop):", batch_size)
    print('===============global_ds_name:', global_ds_name)

    # ====== 基础模型路径 / graph connector 维度 ======
    base_model_path = 'base_model/llama3_2_1B_Instruct_resumed_from_ckpt18000'
    connect_dim = 768

    print(base_model_path)
    print(global_ds_name)
    print('change order:{} zero modify:{}'.format(change_order, zero_modify))

    # ====== Tokenizer & embedding_mask_id ======
    tokenizer = AutoTokenizer.from_pretrained(
        "../../LLM4Graph_tracked/base_models/llama3_2_1B_Instruct_pretrained_4090_ckpt_2550000_iclr"
    )
    tokenizer.pad_token = tokenizer.eos_token
    embedding_mask_id = tokenizer.encode(embedding_mask_str, add_special_tokens=False)[0]

    # ====== 把原始 question / answer 转成 chat 模板 + labels（prompt 部分为 -100） ======
    def chat_map(dp):
        QA_json = {}
        question_str = dp.pop('question')
        answer_str = dp.pop('answer')

        # 修正 key 名
        if dp['unaligned_input_embeds'] is None:
            dp['unaligned_input_embeds'] = dp['unalgined_input_embeds']

        # 一些 diabetes 专用的字符串替换（保留你的原逻辑）
        question_str = question_str.replace(
            ":Diabetes Mellitus, Experimental; Diabetes Mellitus Type 1; Diabetes Mellitus Type 2",
            ": Diabetes Type 1; Diabetes Type 2; Diabetes Experiments"
        )
        question_str = question_str.replace('Diabetes Mellitus, Experimental', 'Diabetes Experiments')
        question_str = question_str.replace('Diabetes Mellitus Type 1', 'Diabetes Type 1')
        question_str = question_str.replace('Diabetes Mellitus Type 2', 'Diabetes Type 2')
        answer_str = answer_str.replace('Diabetes Mellitus, Experimental', 'Diabetes Experiments')
        answer_str = answer_str.replace('Diabetes Mellitus Type 1', 'Diabetes Type 1')
        answer_str = answer_str.replace('Diabetes Mellitus Type 2', 'Diabetes Type 2')

        QA_json['full'] = [
            {'role': 'user', 'content': question_str},
            {'role': 'assistant', 'content': answer_str}
        ]
        QA_json['question_only'] = [
            {'role': 'user', 'content': question_str}
        ]

        full_chat = tokenizer.apply_chat_template(QA_json['full'], return_tensors='pt')[0]
        question_only = tokenizer.apply_chat_template(
            QA_json['question_only'],
            return_tensors='pt',
            add_generation_prompt=True
        )[0]

        dp['input_ids'] = full_chat.tolist()
        labels = full_chat.clone()
        prompt_length = question_only.shape[0]
        labels[:prompt_length] = -100
        dp['labels'] = labels.tolist()
        dp['length'] = len(dp['input_ids'])
        return dp

    ds = ds.map(chat_map, num_proc=16)
    ds = ds.select(test_set_ids)

    use_value_head = False

    # ====== 要评测的 ckpt 列表 ======
    ckpts_dirs = ['50000', '42000', '44000', '46000', '48000']
    count_dict = {}
    parent_checkpoint_path = 'ckpts/h100_order_change_resumed/checkpoint-44000'

    # ====== 预先把所有 label 文本 tokenize 好（防止每个样本重复 encode） ======
    label_token_ids_list = [
        tokenizer.encode(label, add_special_tokens=False)
        for label in all_labels
    ]

    for curr_ckpt in ckpts_dirs:
        torch.cuda.empty_cache()
        print("========== Eval ckpt:", curr_ckpt, "==========")

        # ===== 加载 GraphAdapter + LoRA 权重 =====
        model = GraphAdapter4CausalLM.from_pretrained(
            model_dir=base_model_path,
            connector_dim=connect_dim
        )

        if curr_ckpt != 'zero_shot':
            if not use_value_head:
                # 先加载 node_embedding_connect
                node_embed_path = os.path.join(
                    parent_checkpoint_path,
                    'checkpoint-{}'.format(curr_ckpt),
                    'node_embedding_connect.safetensors'
                )
                if os.path.exists(node_embed_path):
                    head_state = safe_load_file(node_embed_path, device="cpu")
                    model.node_embedding_connect.load_state_dict(head_state, strict=True)

                # 再加载 LoRA 适配器
                peft_model = PeftModel.from_pretrained(
                    model,
                    "{}/checkpoint-{}".format(parent_checkpoint_path, curr_ckpt)
                )
            else:
                # 你原来针对 value head 的逻辑，这里先保留但不使用
                peft_model = LLama4GraphWithValueHead.from_pretrained(base_model)
                peft_model.load_state_dict(
                    torch.load(
                        '{}/checkpoint-{}/pytorch_model.bin'.format(parent_checkpoint_path, curr_ckpt),
                        map_location='cpu'
                    ),
                    strict=False
                )
                peft_model.pretrained_model.load_adapter(
                    '{}/checkpoint-{}'.format(parent_checkpoint_path, curr_ckpt)
                )
        else:
            peft_model = model

        if use_half:
            peft_model = peft_model.half()
        peft_model = peft_model.to(device)

        peft_model.eval()

        # ============== 重要函数：对某个候选 label 计算 NLL（路线 A 核心） ==============
        def compute_candidate_nll(prompt_ids, label_token_ids, unaligned_input_embeds_np):
            """
            prompt_ids: List[int]，只包含 labels == -100 部分（question_only + assistant header）
            label_token_ids: List[int]，某个候选 label 的 token 序列
            unaligned_input_embeds_np: np.array, shape [num_nodes, dim]
            """
            # 构造 prompt + label 作为完整输入
            full_ids = torch.tensor(
                prompt_ids + label_token_ids,
                dtype=torch.long,
                device=device
            ).unsqueeze(0)  # [1, L]
            attention_mask = torch.ones_like(full_ids)
            attention_mask[len(prompt_ids):] = 0

            # labels：只在 label 部分计算 loss
            labels = full_ids.clone()
            prompt_len = len(prompt_ids)
            labels[:, :prompt_len] = -100

            # embedding_positions：embedding_mask_id 在 full_ids 中的位置
            embedding_positions = torch.nonzero(
                full_ids[0] == embedding_mask_id,
                as_tuple=False
            ).view(-1)
            embedding_positions_list = [embedding_positions]  # 和 generate 时的格式一致：list[tensor]

            # unaligned_inputs_embeds：图节点 embedding，格式为 list[tensor]
            if unaligned_input_embeds_np is not None:
                unaligned_tensor = torch.tensor(
                    unaligned_input_embeds_np,
                    device=device,
                    dtype=peft_model.node_embedding_connect[0].weight.dtype
                )
                unaligned_inputs_embeds = [unaligned_tensor]
            else:
                unaligned_inputs_embeds = None

            with torch.no_grad():
                outputs = peft_model(
                    input_ids=full_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    unaligned_inputs_embeds=unaligned_inputs_embeds,
                    embedding_positions=embedding_positions_list,
                )
                loss = outputs.loss  # 平均 NLL，越小越好

            return loss.item()

        # ================== 遍历 test_set，逐样本做多类 NLL 打分 ==================
        y_true = []
        y_pred = []
        len_count = []

        for idx in tqdm(test_set_ids, desc=f"Route A NLL scoring @ ckpt-{curr_ckpt}"):
            example = ds[idx.item()]

            input_ids_np = np.array(example['input_ids'])
            labels_np = np.array(example['labels'])

            # prompt_ids = labels == -100 的那部分（question_only + assistant header）
            prompt_mask = labels_np == -100
            prompt_ids = input_ids_np[prompt_mask].tolist()
            len_count.append(len(input_ids_np))

            # 取 gold answer 文本（labels != -100 那部分，去掉最后一个 eos）
            gold_answer_tokens = labels_np[~prompt_mask][:-1]
            gold_answer_text = tokenizer.decode(
                torch.tensor(gold_answer_tokens),
                skip_special_tokens=False
            )

            # y_true：从 gold_answer_text 里找到对应的 label（出现的那个 all_labels）
            true_label_idx = None
            gold_lower = gold_answer_text.lower()
            for li, label in enumerate(all_labels):
                if label.lower() in gold_lower:
                    true_label_idx = li
                    break

            # 如果一个都匹配不到，就跳过这个样本（非常少见，一般是数据脏）
            if true_label_idx is None:
                continue

            y_true.append(true_label_idx)

            # 图 embedding
            if example.get('unaligned_input_embeds', None) is None:
                unaligned_input_embeds_np = np.array(example['unalgined_input_embeds'])
            else:
                unaligned_input_embeds_np = np.array(example['unaligned_input_embeds'])

            # 对所有候选 label 计算 NLL
            nll_list = []
            for label_token_ids in label_token_ids_list:
                nll_val = compute_candidate_nll(
                    prompt_ids,
                    label_token_ids,
                    unaligned_input_embeds_np
                )
                nll_list.append(nll_val)

            # NLL 最小的那个 label 作为预测
            nll_tensor = torch.tensor(nll_list)
            pred_label_idx = int(torch.argmin(nll_tensor).item())
            y_pred.append(pred_label_idx)

        # ================== 计算准确率 & 宏 F1 ==================
        y_true_t = torch.tensor(y_true)
        y_pred_t = torch.tensor(y_pred)

        correct = (y_true_t == y_pred_t).sum().item()
        acc = correct / len(y_true) if len(y_true) > 0 else 0.0

        # 简单宏 F1（多分类）
        num_labels = len(all_labels)
        f1_per_label = []
        for li in range(num_labels):
            tp = torch.logical_and(y_pred_t == li, y_true_t == li).sum()
            fp = torch.logical_and(y_pred_t == li, y_true_t != li).sum()
            fn = torch.logical_and(y_pred_t != li, y_true_t == li).sum()
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1_per_label.append(f1.item())
        macro_f1 = float(np.mean(f1_per_label))

        correct_count = {
            'overall_correct_number': correct,
            'acc': acc,
            'macro_F1': macro_f1,
            'num_eval_examples': len(y_true),
        }

        print(f"[Route A] ckpt-{curr_ckpt} -- ACC: {acc:.4f}, Macro-F1: {macro_f1:.4f}, Correct: {correct}/{len(y_true)}")
        print(correct_count)

        count_dict[curr_ckpt] = correct_count

    print("======== All ckpts summary ========")
    print(count_dict)
    print(base_model_path)
    print(global_ds_name)
    print(parent_checkpoint_path)
    print('use Half: ', use_half)
    print('change order:{} zero_modify:{}'.format(change_order, zero_modify))
