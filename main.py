from transformers import Trainer, AutoTokenizer,AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments,LlamaConfig, set_seed
from trl import SFTTrainer, SFTConfig
import argparse
from data_utils.data_loader import getCoraEmbeds2TextDataset, getImageTestDataset,getArxivEmbeds2TextDataset,removeTooLongDatapoint, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from data_utils.data_collators import Embeds2TextCollator, Embeds2PredictionCollator, InstructEmbedsPretrainCollator
from models.LLaMa import LLama4Graph, LLama4GraphWithValueHead
from sentence_transformers import SentenceTransformer
import torch
from accelerate import Accelerator
from data_utils.callback_saver import SaveTrainerCallBack,PredictionSaveTrainerCallBack
from datasets import load_from_disk
import torch.distributed as dist
from data_utils.prompt_str import embedding_mask_str, begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str
import os
from ogb.graphproppred import Evaluator
import numpy as np
import wandb
from accelerate import PartialState
# from trainers.wandb_wrapper import add_wandb_generation_logging

y_trues = []
y_preds = []
positive_recalled = 0
false_positive = 0
acctor =Accelerator()
PLM = None
# PLM = SentenceTransformer('all-mpnet-base-v2').to(acctor.device)
tokenizer = None
overall_prediction_num = 0
overall_correct_num = 0
global_ref = None
not_log_output = True
# def distributed_sum(value: int):
#     if dist.is_initialized():
#         tensor = torch.tensor(value, device="cuda")
#         dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
#         return tensor.item()
#     return value

def objective(args):
    
    # config = LlamaConfig.from_pretrained(args.model_dir)
    model = LLama4Graph.from_pretrained(args.model_dir)
    model.task_type = args.task
    # if config.tie_word_embeddings:
    #     model.lm_head.load_state_dict(model.model.embed_tokens.state_dict())
    #     print('tie word embeddings')
    # model = model.half()
    global tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    # tokenizer.add_tokens([embedding_mask_str])
    model.pad_token_id = tokenizer.pad_token_id
    model.embedding_mask_id = tokenizer.encode(embedding_mask_str,add_special_tokens=False)[0]
    ### fix LLM param
    for param in model.model.parameters():
        param.requires_grad = False
    if args.task == 'pretrain':
        for param in model.lm_head.parameters():
            param.requires_grad = False
        collator = InstructEmbedsPretrainCollator(tokenizer=tokenizer, mlm=False,use_fp16=args.use_fp16)
        eval_metric = sematic_measurement
        saver = SaveTrainerCallBack()
        if args.resume_path is not None:
            model.node_embedding_connect.load_state_dict(torch.load(args.resume_path,map_location='cpu').state_dict())
        global PLM
        PLM = SentenceTransformer('all-mpnet-base-v2').to(acctor.device)
    elif args.task == 'prediction':
        # model.node_embedding_connect.load_state_dict(torch.load(args.connector_pretrain_path,map_location='cpu').state_dict())
        # if args.resume_path is not None:
        #     model.lm_head.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
        for param in model.node_embedding_connect.parameters():
            param.requires_grad = False
        collator = InstructEmbedsPretrainCollator(tokenizer=tokenizer,mlm=False, use_fp16=args.use_fp16)
        eval_metric = prediction_measurement
        saver = PredictionSaveTrainerCallBack()
    # # if args.use_graph:
    # tokenizer.add_tokens([begin_of_nodes_str, end_of_nodes_str, begin_of_edges_str, end_of_edges_str, one_edge_str])
    # model.resize_token_embeddings(len(tokenizer))
    ### LoRA training
    if args.use_LoRA:
        print('getting LoRA model')
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        if args.use_value_head:
            if args.load_local_adapter is not None:
                peft_model = PeftModel.from_pretrained(model, args.load_local_adapter, is_trainable=True,).to(device=model.device)
                model = LLama4GraphWithValueHead.from_pretrained(peft_model,is_for_sft=True)
            else:
                model = LLama4GraphWithValueHead.from_pretrained(model,is_for_sft=True, peft_config=lora_config)
            model.pretrained_model.print_trainable_parameters()
        else:
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    
        
    train_dataset, test_dataset = load_dataset(args.dataset_dir,tokenizer)
    def chat_map(dp):
        QA_json = {}
        question_str = dp.pop('question')
        answer_str = dp.pop('answer')
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
        dp.pop('type')
        return dp
    with PartialState().local_main_process_first():
        train_dataset = train_dataset.map(chat_map,num_proc=16)
        test_dataset = test_dataset.map(chat_map)
    # print(test_dataset[0]['input_ids'])
    max_seq_length = np.array(train_dataset['length']).max()
    if args.use_fp16:
        model = model.half()
    
    training_arguments = SFTConfig(
        output_dir=args.model_save_path,
        # report_to='wandb',
        logging_dir=args.log_dir,
        per_device_train_batch_size=args.train_size,
        per_device_eval_batch_size=args.eval_size,
        gradient_accumulation_steps=4,
        remove_unused_columns=False,
        fp16=False,
        learning_rate=1e-2,
        # lr_scheduler_type='constant_with_warmup',
        warmup_steps=100,
        num_train_epochs=10,
        save_strategy='no',
        eval_strategy='steps',
        eval_steps=500,
        # max_grad_norm=1,
        logging_steps=100,
        optim='sgd',
        batch_eval_metrics=True,
        eval_do_concat_batches=True,
        # auto_find_batch_size=True,
        dataloader_num_workers=4,
        include_for_metrics=['loss'],
        label_names=['labels'],
        dataset_kwargs={"skip_prepare_dataset":True},
        run_name=args.run_name,
        max_seq_length=max_seq_length
        
    )
    trainer = SFTTrainer(
        model=model,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        compute_metrics=eval_metric,
        callbacks=[saver],
    )
    global global_ref
    global_ref = trainer
    trainer.train()
    
    
def sematic_measurement(eval_pred,compute_result):
    with torch.no_grad():
        torch.cuda.empty_cache()
       
        # tokenizer = AutoTokenizer.from_pretrained('/data/sharefile/wei/.llama/HF_format/Meta-Llama3.1-8B')
        global global_ref
        global not_log_output
        predictions = eval_pred.predictions.argmax(dim=-1)
        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        predictions_sentence = tokenizer.batch_decode(predictions,skip_special_tokens=True)
        input_sentence = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        p_embeds = PLM.encode(predictions_sentence)
        i_embeds = PLM.encode(input_sentence)
        similarity = PLM.similarity_pairwise(p_embeds,i_embeds)
        similarity_average = similarity.mean().item()
        
        if not_log_output and np.random.random() > 0.5 and global_ref.state.global_step % 5000 == 0 and global_ref.state.is_world_process_zero:
            not_log_output = False
            table = wandb.Table(columns=['labels', 'generated'])
            for i in range(min(10, len(predictions_sentence))):
            # check_idx = np.random.randint(0, len(input_sentence))
                table.add_data(input_sentence[i], predictions_sentence[i])
            wandb.log({'eval_check/generated_examples':table})#,step=global_ref.state.global_step)
        if compute_result:
            if not_log_output and global_ref.state.global_step % 5000 == 0 and global_ref.state.is_world_process_zero:
                table = wandb.Table(columns=['labels', 'generated'])
                for i in range(min(10, len(predictions_sentence))):
                # check_idx = np.random.randint(0, len(input_sentence))
                    table.add_data(input_sentence[i], predictions_sentence[i])
                wandb.log({'eval_check/generated_examples':table})#,step=global_ref.state.global_step)
            not_log_output == True
        return {
            'average_similarity':similarity_average  
        }

def prediction_measurement(eval_pred, compute_result):
    with torch.no_grad():
        global tokenizer
        global positive_recalled
        global false_positive
        eot_id, eos_id = tokenizer.convert_tokens_to_ids(['<|eot_id|>','<|end_of_text|>'])
        torch.cuda.empty_cache()
        
        label_ids = eval_pred.label_ids
        if isinstance(eval_pred.predictions, tuple):
            predictions = eval_pred.predictions[0].argmax(dim=-1)
            # predicted_values = eval_pred.predictions[1]
            if not isinstance(eval_pred.predictions[1],tuple):
                predicted_values = eval_pred.predictions[1]
            else:
                predicted_values = None
        else:
            predictions = eval_pred.predictions.argmax(dim=-1)
            predicted_values = None
        # new_labels = -100 * torch.ones_like(predictions,dtype=label_ids.dtype, device=label_ids.device)
        labels_length = (label_ids != -100).sum(dim=1)
        # first_eot_idx = (predictions == eot_id).type(torch.int32).argmax(dim=1)
        # first_eos_idx = (predictions == eos_id).type(torch.int32).argmax(dim=1)
        prompt_end_poses = torch.argmax((label_ids != -100).int(),dim=1)
        batch_correct_num = 0
        for i in range(predictions.shape[0]):
            current_labels = tokenizer.decode(label_ids[i,prompt_end_poses[i]+1:prompt_end_poses[i]+labels_length[i] -1])
            if ("Yes" in current_labels and "Yes" in tokenizer.decode(predictions[i])) or ("Nope" in current_labels and "Nope" in tokenizer.decode(predictions[i])):
                batch_correct_num += 1
                if "Yes" in current_labels:
                    positive_recalled += 1
            # if current_labels in tokenizer.decode(predictions[i]):
            #     batch_correct_num += 1
            #     if "Yes" in current_labels:
            #         positive_recalled += 1
            elif "Yes" in current_labels:
                false_positive += 1
        global overall_correct_num
        global overall_prediction_num
        overall_prediction_num += predictions.shape[0]
        overall_correct_num += batch_correct_num
        
        target_positive = 0
        predict_positive = 0
        
        report_prediction = overall_prediction_num
        report_correct_num = overall_correct_num
        reported_roc_auc = 0
        report_positive_recall = 0
        report_false_positive = 0
        global y_trues
        global y_preds
        if predicted_values is not None:
            y_trues.extend(predicted_values[:,0].tolist())
            y_preds.extend(predicted_values[:,1].tolist())
        if compute_result:
            overall_correct_num = 0
            overall_prediction_num = 0
            evaluator = Evaluator(name='ogbg-molhiv')
            if len(y_trues) != 0:
                print("======== get ROC auc =========")
                y_trues = np.expand_dims(np.array(y_trues),axis=1)
                y_preds = np.expand_dims(np.array(y_preds), axis=1)
                reported_roc_auc = evaluator.eval({'y_true': y_trues, 'y_pred': y_preds})['rocauc']
                target_positive = y_trues.sum()
                predict_positive = (y_preds > 0.5).sum()
            report_positive_recall = positive_recalled
            report_false_positive = false_positive
            positive_recalled = 0
            false_positive = 0
            y_trues = []
            y_preds = []
        
        return {
            'prediction_num':report_prediction,
            'correct_num':report_correct_num,
            "prediction_acc": report_correct_num / report_prediction,
            'roc_auc':reported_roc_auc,
            'traget_positive':target_positive,
            'predict_positive':predict_positive,
            "recalled_positive_by_llm":report_positive_recall,
            "false_positive_by_llm": report_false_positive, 
        }
    
    
    
       
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=['pretrain','prediction']
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model_save_path',
        type=str
    )
    parser.add_argument(
        '--log_dir',
        type=str
    )
    parser.add_argument(
        '--connector_pretrain_path',
        type=str,
        required=False
    )
    parser.add_argument(
        '--resume_path',
        type=str,
        required=False
    )
    parser.add_argument(
        '--use_LoRA',
        action='store_true'
    )
    parser.add_argument(
        '--use_graph',
        action='store_true'
    )
    parser.add_argument(
        '--use_fp16',
        action='store_true'
    )
    parser.add_argument(
        '--use_value_head',
        action='store_true'
    )
    parser.add_argument(
        '--load_local_adapter',
        type=str,
    )
    parser.add_argument(
        '--run_name',
        type=str,
        required=True
    )
    parser.add_argument(
        '--train_size',
        type=int,
        default=2
    )
    parser.add_argument(
        '--eval_size',
        type=int,
        default=8
    )
    args = parser.parse_args()
    set_seed(42)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'args.json'),'w') as fp:
        import json
        json.dump(vars(args), fp)
    objective(args)