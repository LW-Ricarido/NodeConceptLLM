import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional, Union


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from trl import PPOTrainer
import transformers
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from accelerate.utils import broadcast, gather_object
from transformers.utils import is_peft_available, is_rich_available


from trl.core import masked_mean,masked_whiten
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import(
    batch_generation,
    selective_log_softmax,
    empty_cache,
    forward,
    get_reward,
    truncate_response,
    first_true_indices,
    print_rich_table,
    log_table_to_comet_experiment,
    pad
)
from models.reward_models import realRewardFetcher
from torch.nn.parallel import DistributedDataParallel
import asyncio
INVALID_LOGPROB = 1.0

def withLabelReward(model: torch.nn.Module, responses: torch.Tensor, pad_token_id: int, context_length: int, labels: torch.Tensor,tokenizer : transformers.PreTrainedTokenizerBase, rewardFetcher: realRewardFetcher
) -> torch.Tensor:
    responses_strings = tokenizer.batch_decode(responses,skip_special_tokens=True)
    labels_strings = tokenizer.batch_decode(labels,skip_special_tokens=True)
    # responses = asyncio.run(rewardFetcher.batch_query(responses_strings, labels_strings, model.potential_label_list ))#,device=model.device
    responses = rewardFetcher.label_direct_checking(responses_strings, labels_strings, model.potential_label_list)
    rewards, answers = responses
    rewards = model(torch.FloatTensor(rewards,device=model.device))
    return rewards, answers

def forwardWithAllInput(model,query_response, pad_token_id, all_data_dict):
    if query_response.dtype is torch.float16 or query_response.dtype is torch.float32:
        attention_mask = all_data_dict['attention_mask']
        return model(
            inputs_embeds = query_response,
            attention_mask = attention_mask,
            return_dict = True,
            output_hidden_states = True
        )
    else:
        # embedding_positions = []
        # for i in range(query_response.shape[0]):
        #     embedding_positions.append(torch.nonzero(query_response[i] == model.base_model.embedding_mask_id).flatten())
        attention_mask = query_response != pad_token_id
        unaligned_inputs_embeds = []
        for i in range(len(all_data_dict['unaligned_inputs_embeds'])):
            unaligned_inputs_embeds.append(all_data_dict['unaligned_inputs_embeds'][i].to(query_response.device))
        return model(
            input_ids = query_response,
            attention_mask = attention_mask,
            return_dict = True,
            output_hidden_states=True,
            unaligned_inputs_embeds = unaligned_inputs_embeds,
        )
        
def get_reward_AllInput(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int, all_data_dict
):
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    embedding_positions = []
    for i in range(input_ids.shape[0]):
        embedding_positions.append(torch.nonzero(input_ids[i] == lm_backbone.embedding_mask_id).flatten())
    unaligned_inputs_embeds = []
    for i in range(len(all_data_dict['unaligned_inputs_embeds'])):
        unaligned_inputs_embeds.append(all_data_dict['unaligned_inputs_embeds'][i])
    output = lm_backbone(
        input_ids = input_ids,
        attention_mask = attention_mask,
        position_ids = position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
        unaligned_inputs_embeds = unaligned_inputs_embeds,
        embedding_positions = embedding_positions
    )
    reward_logits = model.score(output.hidden_states[-1])
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1),
        sequence_lengths,
    )

def generate(
    lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig, attention_mask: torch.Tensor,original_input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`GenerationConfig`):
            The configuration for the generation process.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    output = lm_backbone.generate(
        inputs_embeds = queries,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((original_input_ids, output.sequences), dim=1), logits

def batch_generation_with_embedding(
    model: torch.nn.Module,
    input_data,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: GenerationConfig,
):
    ### be careful, here padding is left, not right
    input_ids = input_data['input_ids']
    attention_mask = input_ids != pad_token_id
    input_embeds = model.base_model.get_input_embeddings()(input_ids)    
    unaligned_inputs_embeds = input_data['unaligned_inputs_embeds']
    
    embedding_positions = input_data['embedding_positions']
    aligned_embeds = []
    for i in range(len(unaligned_inputs_embeds)):
        aligned_embeds.append(model.base_model.node_embedding_connect(unaligned_inputs_embeds[i]))
    for i in range(len(aligned_embeds)):
        input_embeds[i][embedding_positions[i]] = aligned_embeds[i]
    queries = input_embeds
    query_responses = []
    logitss = []
    batch_size = queries.shape[0]
    for i in range(0, batch_size, local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
            attention_mask = attention_mask,
            original_input_ids=input_ids
        )
        query_responses.append(query_response)
        logitss.append(logits)

    # padding tensors
    padded_query_responses = pad(query_responses, padding_value=pad_token_id, padding_side="right")
    padded_logitss = pad(logitss, padding_value=0, padding_side="right")

    # reshaping
    padded_query_responses = padded_query_responses.view(-1, padded_query_responses.shape[-1])[:batch_size]
    padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]

    return padded_query_responses, padded_logitss

class MyPPOTrainer(PPOTrainer):
    
    def __init__(self,rewardFetcher: realRewardFetcher, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewardFetcher = rewardFetcher
        
    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        self.state._globalstep_last_logged = self.state.global_step
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model
        
        reported_kl = torch.tensor(0.0, device=device)
        reported_entropy = torch.tensor(0.0, device=device) 
        reported_non_score_reward = torch.tensor(0.0, device=device) 
        reported_rlhf_reward = torch.tensor(0.0, device=device) 
        reported_scores = torch.tensor(0.0, device=device) 
        reported_approxkl_avg = torch.tensor(0.0, device=device) 
        reported_policy_clipfrac_avg = torch.tensor(0.0, device=device) 
        reported_policy_avg  = torch.tensor(0.0, device=device) 
        reported_value_avg  = torch.tensor(0.0, device=device) 
        reported_val_clipfrac_avg = torch.tensor(0.0, device=device) 
        reported_entropy_avg = torch.tensor(0.0, device=device) 
        reported_ratio = torch.tensor(0.0, device=device) 
        reported_num_eos_tokens = torch.tensor(0.0, device=device) 

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                for i in range(len(data['unaligned_inputs_embeds'])):
                    data['unaligned_inputs_embeds'][i] = data['unaligned_inputs_embeds'][i].to(device)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []
                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation_with_embedding(
                        unwrapped_model.policy,
                        data,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    # ### original version
                    # query_responses, logitss = batch_generation(
                    #     unwrapped_model.policy,
                    #     queries,
                    #     args.local_rollout_forward_batch_size,
                    #     processing_class.pad_token_id,
                    #     generation_config,
                    # )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_output = forwardWithAllInput(model.policy, query_response, processing_class.pad_token_id,data)
                            # ref_output = forward(model.policy, query_response, processing_class.pad_token_id)
                    else:
                        ref_output = forwardWithAllInput(ref_policy, query_response, processing_class.pad_token_id,data)
                        # ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    
                    # full_value, _, _ = get_reward(
                    #     unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
                    # )
                    
                    full_value, _, _ = get_reward_AllInput(
                        unwrapped_value_model, query_response, processing_class.pad_token_id, context_length, data,
                    )
                    

                    value = full_value[:, context_length - 1 : -1].squeeze(-1)
                    
                    # ###The original score shape is [batch_size]
                    # _, score, _ = get_reward(
                    #     reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    # )
                    
                    
                    score, _ = withLabelReward(reward_model, postprocessed_response, processing_class.pad_token_id,context_length,data['labels'],self.processing_class,self.rewardFetcher)
                    score = score.to(value)
                    
                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                # Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
                logr = ref_logprobs - logprobs
                kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                            # output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                            mb_data = {}
                            mb_data['unaligned_inputs_embeds'] = []
                            for mb_idx in micro_batch_inds:
                                mb_data['unaligned_inputs_embeds'].append(data['unaligned_inputs_embeds'][mb_idx])
                            output, vpred_temp = forwardWithAllInput(model, mb_query_responses, processing_class.pad_token_id, mb_data)
                            
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    empty_cache()
            
            
            
            
            
            
                
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                self.state.global_step += 1
                
                reported_kl += self.accelerator.gather_for_metrics(mean_kl).mean().item()
                reported_entropy += self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                reported_non_score_reward += self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                reported_rlhf_reward += self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                reported_scores += self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                reported_approxkl_avg += self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                reported_policy_clipfrac_avg += self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                reported_policy_avg  += self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                reported_value_avg  += self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                reported_val_clipfrac_avg += self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                reported_entropy_avg += self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                reported_ratio += self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                reported_num_eos_tokens += (responses == processing_class.eos_token_id).sum().item()
                
                if self.control.should_log and self.state.global_step > self.state._globalstep_last_logged:
                    metrics = {}
                    divided_length = self.state.global_step - self.state._globalstep_last_logged
                    metrics["eps"] = eps
                    metrics["objective/kl"] = (reported_kl / divided_length).item()
                    metrics["objective/entropy"] = (reported_entropy / divided_length).item()
                    metrics["objective/non_score_reward"] = (reported_non_score_reward / divided_length).item()
                    metrics["objective/rlhf_reward"] = (reported_rlhf_reward / divided_length).item()
                    metrics["objective/scores"] = (reported_scores / divided_length).item()
                    metrics["policy/approxkl_avg"] = (reported_approxkl_avg / divided_length).item()
                    metrics["policy/clipfrac_avg"] = (reported_policy_clipfrac_avg / divided_length).item()
                    metrics["loss/policy_avg"] = (reported_policy_avg / divided_length).item()
                    metrics["loss/value_avg"] = (reported_value_avg / divided_length).item()
                    metrics["val/clipfrac_avg"] = (reported_val_clipfrac_avg / divided_length).item()
                    metrics["policy/entropy_avg"] = (reported_entropy_avg / divided_length).item()
                    metrics["val/ratio"] = (reported_ratio / divided_length).item()
                    metrics["val/num_eos_tokens"] = (reported_num_eos_tokens / divided_length).item()
                    metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                    metrics["episode"] = self.state.episode
                    self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                    self.log(metrics)
                    reported_kl -= reported_kl
                    reported_entropy -= reported_entropy
                    reported_non_score_reward -= reported_non_score_reward
                    reported_rlhf_reward -= reported_rlhf_reward
                    reported_scores -= reported_scores
                    reported_approxkl_avg -= reported_approxkl_avg
                    reported_policy_clipfrac_avg -= reported_policy_clipfrac_avg
                    reported_policy_avg -= reported_policy_avg
                    reported_value_avg -= reported_value_avg
                    reported_val_clipfrac_avg -= reported_val_clipfrac_avg 
                    reported_entropy_avg -= reported_entropy_avg 
                    reported_ratio -= reported_ratio 
                    reported_num_eos_tokens -= reported_num_eos_tokens
                    self.state._globalstep_last_logged = self.state.global_step
                    del metrics
                    
                # metrics = {}
                # metrics["eps"] = eps
                # metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                # metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                # metrics["objective/non_score_reward"] = (
                #     self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                # )
                # metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                # metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                # metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                # metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                # metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                # metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                # metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                # metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                # metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                # metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                # metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                # metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                # metrics["episode"] = self.state.episode
                # self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                # self.state.global_step += 1
                # self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, non_score_reward
            empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                for i in range(len(batch['unaligned_inputs_embeds'])):
                    batch['unaligned_inputs_embeds'][i] = batch['unaligned_inputs_embeds'][i].to(query.device)
                with torch.no_grad():
                    context_length = query.shape[1]
                    
                    # query_response, _ = batch_generation(
                    #     unwrapped_model.policy,
                    #     query,
                    #     query.shape[0],
                    #     processing_class.pad_token_id,
                    #     generation_config,
                    # )
                    
                    query_response, _ = batch_generation_with_embedding(
                        unwrapped_model.policy,
                        batch,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )
                    table["label"].extend(
                        gather_object(processing_class.batch_decode(batch['labels'], skip_special_tokens=True))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    
                    # _, score, _ = get_reward(
                    #     self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    # )
                    score, ollama_answers = withLabelReward(self.reward_model, postprocessed_response, processing_class.pad_token_id,context_length,batch['labels'],self.processing_class,self.rewardFetcher)
                    score = score.to(batch['input_ids'])

                    # table['ollama answer'].extend(
                    #     ollama_answers
                    # )
                    
                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            if is_rich_available():
                print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        # self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        if isinstance(self.model, DistributedDataParallel):
            self.model = self.model.module.policy
        else:
            self.model = self.model.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        Trainer.save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed