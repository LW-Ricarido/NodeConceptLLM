import os
import re
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    set_seed,
)
from datasets import load_from_disk
from tqdm.auto import tqdm
from peft import PeftModel
from safetensors.torch import load_file as safe_load_file

from models.graphAdapter import GraphAdapter4CausalLM
from data_utils.prompt_str import embedding_mask_str

# ================== 全局配置 ==================
global_ds_name = 'WN18RR'
batch_size = 32          # 现在真正用于 batch eval 了
use_half = True

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
    all_labels = ['Yes', 'No']


def simple_collate(examples):
    """Eval 用的简单 collate，直接返回样本 list，不做 pad。"""
    return examples


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
    )
    test_set_ids = np.arange(len(ds))
    ds = ds.select(test_set_ids)

    print("=============length of :", len(test_set_ids))
    print('===============global_ds_name:', global_ds_name)

    # ====== 基础模型路径 / graph connector 维度 ======
    base_model_path = 'base_model/llama3_2_1B_Instruct_resumed_from_ckpt18000'
    connect_dim = 768

    print(base_model_path)

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
        if dp.get('unaligned_input_embeds', None) is None:
            dp['unaligned_input_embeds'] = dp['unalgined_input_embeds']

        # diabetes 文本替换（保留）
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
        labels[:prompt_length] = -100       # prompt 部分设成 -100，不算 loss
        dp['labels'] = labels.tolist()
        dp['length'] = len(dp['input_ids'])
        return dp

    ds = ds.map(chat_map, num_proc=16)

    # ====== 预先把所有 label 文本 tokenize 好 ======
    label_token_ids_list = [
        tokenizer.encode(label, add_special_tokens=False)
        for label in all_labels
    ]
    num_labels = len(all_labels)

    # ====== 要评测的 ckpt 列表 ======
    ckpts_dirs = ['50000', '42000', '44000', '46000', '48000']
    parent_checkpoint_path = 'ckpts/h100_order_change_resumed/checkpoint-44000'

    count_dict = {}

    for curr_ckpt in ckpts_dirs:
        torch.cuda.empty_cache()
        print("\n========== Eval ckpt:", curr_ckpt, "==========")

        # ===== 加载 GraphAdapter + LoRA 权重 =====
        base_model = GraphAdapter4CausalLM.from_pretrained(
            model_dir=base_model_path,
            connector_dim=connect_dim
        )

        # node_embedding_connect
        node_embed_path = os.path.join(
            parent_checkpoint_path,
            'checkpoint-{}'.format(curr_ckpt),
            'node_embedding_connect.safetensors'
        )
        if os.path.exists(node_embed_path):
            head_state = safe_load_file(node_embed_path, device="cpu")
            base_model.node_embedding_connect.load_state_dict(head_state, strict=True)

        # LoRA 适配器
        peft_model = PeftModel.from_pretrained(
            base_model,
            "{}/checkpoint-{}".format(parent_checkpoint_path, curr_ckpt)
        )

        if use_half:
            peft_model = peft_model.half()
        peft_model = peft_model.to(device)
        peft_model.eval()

        # ====== 计算某个 label 对当前 batch 的 NLL（批量） ======
        def compute_nll_for_label_batch(
            prompt_ids_batch,      # [B, max_prompt_len] padded
            prompt_lengths,        # [B] 各样本真实 prompt 长度
            label_token_ids,       # List[int]
            unaligned_embeds_list  # List[np.array]，长度 B
        ):
            B = prompt_ids_batch.size(0)
            label_len = len(label_token_ids)
            label_tensor = torch.tensor(label_token_ids, dtype=torch.long, device=device)

            # 为每个样本构造 prompt + label
            full_ids_list = []
            for i in range(B):
                real_len = prompt_lengths[i]
                prompt_real = prompt_ids_batch[i, :real_len]   # [real_len]
                full_i = torch.cat([prompt_real, label_tensor], dim=0)  # [real_len + label_len]
                full_ids_list.append(full_i)

            # pad 成 [B, max_seq_len]
            max_seq_len = max(x.size(0) for x in full_ids_list)
            pad_id = tokenizer.pad_token_id
            full_ids_batch = torch.full(
                (B, max_seq_len),
                pad_id,
                dtype=torch.long,
                device=device
            )
            attention_mask = torch.zeros_like(full_ids_batch)

            label_start_positions = []  # 每个样本 label 起始位置
            for i, full_i in enumerate(full_ids_list):
                L = full_i.size(0)
                full_ids_batch[i, :L] = full_i
                attention_mask[i, :L] = 1
                label_start_positions.append(prompt_lengths[i])

            # embedding_positions：list[tensor]，每个样本中 embedding_mask_id 的位置
            embedding_positions = []
            for i in range(B):
                pos = torch.nonzero(
                    full_ids_batch[i] == embedding_mask_id,
                    as_tuple=False
                ).view(-1)
                embedding_positions.append(pos)

            # unaligned_inputs_embeds：list[tensor]
            unaligned_inputs_embeds = []
            for i in range(B):
                emb_np = unaligned_embeds_list[i]
                emb_tensor = torch.tensor(
                    emb_np,
                    device=device,
                    dtype=peft_model.node_embedding_connect[0].weight.dtype
                )
                unaligned_inputs_embeds.append(emb_tensor)

            with torch.no_grad():
                outputs = peft_model(
                    input_ids=full_ids_batch,
                    attention_mask=attention_mask,
                    unaligned_inputs_embeds=unaligned_inputs_embeds,
                    embedding_positions=embedding_positions,
                )
                logits = outputs.logits  # [B, max_seq_len, vocab]

            log_probs = torch.log_softmax(logits, dim=-1)  # 同形状

            # 逐样本计算 label token 的平均 NLL
            nll_batch = []
            for i in range(B):
                start = label_start_positions[i]
                end = start + label_len
                # 如果这条样本太短，直接给个大 NLL（极端情况）
                if end > max_seq_len:
                    nll_batch.append(1e9)
                    continue
                lp_slice = log_probs[i, start:end, :]  # [label_len, vocab]
                # 对应 label token 的 log_prob
                token_log_probs = lp_slice[torch.arange(label_len, device=device), label_tensor]
                nll = - token_log_probs.mean().item()
                nll_batch.append(nll)

            return np.array(nll_batch, dtype=np.float32)

        # ============ DataLoader（按样本 batch，后续在 label 维度循环） ============
        eval_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=simple_collate,
            num_workers=4,
        )

        all_y_true = []
        all_y_pred = []

        t0 = time.time()
        for batch_examples in tqdm(eval_loader, desc=f"Route A NLL scoring @ ckpt-{curr_ckpt}"):
            B = len(batch_examples)
            if B == 0:
                continue

            # ---- 从 batch 样本里抽 prompt_ids / prompt_lengths / gold label / graph emb ----
            prompt_ids_list = []
            prompt_lengths = []
            gold_label_indices = []
            unaligned_embeds_list = []

            for ex in batch_examples:
                input_ids_np = np.array(ex['input_ids'])
                labels_np = np.array(ex['labels'])

                # prompt = labels == -100 的那部分
                prompt_mask = labels_np == -100
                prompt_ids = input_ids_np[prompt_mask].tolist()
                prompt_ids_list.append(prompt_ids)
                prompt_lengths.append(len(prompt_ids))

                # gold answer 文本
                gold_answer_tokens = labels_np[~prompt_mask][:-1]   # 去掉最后一个 eos
                gold_answer_text = tokenizer.decode(
                    torch.tensor(gold_answer_tokens),
                    skip_special_tokens=False
                )
                gold_lower = gold_answer_text.lower()

                true_idx = None
                for li, label in enumerate(all_labels):
                    if label.lower() in gold_lower:
                        true_idx = li
                        break
                gold_label_indices.append(true_idx)

                # graph emb
                if ex.get('unaligned_input_embeds', None) is None:
                    emb_np = np.array(ex['unalgined_input_embeds'])
                else:
                    emb_np = np.array(ex['unaligned_input_embeds'])
                unaligned_embeds_list.append(emb_np)

            # pad prompt_ids_list 成 batch tensor
            max_prompt_len = max(prompt_lengths)
            pad_id = tokenizer.pad_token_id
            prompt_ids_batch = torch.full(
                (B, max_prompt_len),
                pad_id,
                dtype=torch.long,
                device=device
            )
            for i, ids in enumerate(prompt_ids_list):
                L = len(ids)
                prompt_ids_batch[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
            prompt_lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.long, device=device)

            # ---- 对所有 label 计算 NLL: 得到 [B, num_labels] 的矩阵 ----
            nll_matrix = []
            for label_token_ids in label_token_ids_list:
                nll_batch = compute_nll_for_label_batch(
                    prompt_ids_batch,
                    prompt_lengths_tensor,
                    label_token_ids,
                    unaligned_embeds_list
                )  # [B]
                nll_matrix.append(nll_batch)
            # 形状变成 [B, num_labels]
            nll_matrix = np.stack(nll_matrix, axis=1)   # (B, num_labels)

            # 每个样本选 NLL 最小的 label
            pred_indices = nll_matrix.argmin(axis=1)     # [B]

            # 收集有 ground truth 的样本
            for i in range(B):
                true_idx = gold_label_indices[i]
                if true_idx is None:
                    continue
                all_y_true.append(true_idx)
                all_y_pred.append(int(pred_indices[i]))

        t1 = time.time()
        print(f"Eval time for ckpt-{curr_ckpt}: {t1 - t0:.2f} seconds")

        # ================== 计算准确率 & 宏 F1 ==================
        if len(all_y_true) == 0:
            print("No valid eval examples (no gold label matched).")
            continue

        y_true_t = torch.tensor(all_y_true)
        y_pred_t = torch.tensor(all_y_pred)

        correct = (y_true_t == y_pred_t).sum().item()
        acc = correct / len(all_y_true)

        # 宏 F1
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

        stats = {
            'overall_correct_number': correct,
            'acc': acc,
            'macro_F1': macro_f1,
            'num_eval_examples': len(all_y_true),
            'eval_time_sec': t1 - t0,
        }
        count_dict[curr_ckpt] = stats

        print(f"[Route A Batched] ckpt-{curr_ckpt} -- "
              f"ACC: {acc:.4f}, Macro-F1: {macro_f1:.4f}, "
              f"Correct: {correct}/{len(all_y_true)}")

    print("\n======== All ckpts summary (Route A Batched) ========")
    print(count_dict)
    print("base_model_path:", base_model_path)
    print("global_ds_name:", global_ds_name)
    print("parent_checkpoint_path:", parent_checkpoint_path)
    print('use Half: ', use_half)
