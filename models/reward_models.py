from trl import PPOTrainer,AutoModelForCausalLMWithValueHead
import torch
import torch.nn as nn

from ollama import AsyncClient
from pydantic import BaseModel
from typing import Optional
import asyncio
from abc import ABC
from colorama import Fore
class StructuredClassification(BaseModel):
    classification: str

class realRewardFetcher(ABC):
    
    def __init__(self):
        super().__init__()
    
    def label_direct_checking(self, response_strings, labels_strings, potential_label_list):
        rewards = []
        def contain_multi_label(answer):
            label_count = 0
            for label in potential_label_list:
                if label.lower() in answer.lower():
                    label_count += 1
                if label_count > 1:
                    return True
            return False
        def contain_label(answer):
            for label in potential_label_list:
                if label.lower() in answer.lower():
                    return True
            return False
        for i in range(len(response_strings)):
            if labels_strings[i].lower() in response_strings[i].lower():
                if contain_multi_label(labels_strings[i]):
                    rewards.append(5)
                else:
                    rewards.append(10)
            elif contain_label(response_strings[i]):
                rewards.append(1)
            else:
                rewards.append(-10)
        return rewards, None
    
    async def batch_query(self,response_strings, label_strings, potential_label_list):
        tasks = [self.single_query(response, label, potential_label_list) for response, label in zip(response_strings, label_strings)]
        responses = await asyncio.gather(*tasks)
        rewards, answers = [], []
        for i in range(len(responses)):
            rewards.append(responses[i][0])
            answers.append(responses[i][1])
        return rewards, answers
    
    async def single_query(self, response, label:str, potential_label_list):
        ollama_response = await AsyncClient(timeout=10).chat(
            messages=[
                 {
                    "role": "system",
                    "content": f"You are a helpful assistant that understands and translates text to JSON format according to the following schema. {StructuredClassification.model_json_schema()}"
                }, 
                {
                    'role': 'user',
                    'content': response,
                }
            ],
            model='Osmosis/Osmosis-Structure-0.6B',
            format=StructuredClassification.model_json_schema(),
            options={
                "seed": 2333,
            }
        )
        try:
            answer = StructuredClassification.model_validate_json(ollama_response.message.content)
        except Exception as e:
            answer = StructuredClassification(classification='no result')
        # print("Label: "+Fore.GREEN +"{}".format(label))
        # print("Respones: "+ Fore.CYAN +"{}".format(response))
        # print("Answer: " + Fore.RED + "{}".format(answer.classification))
        # print('Response: {}'.format(response))
        # print('label: {} ||| answer: {}'.format(label,answer.classification))
        # print('============================================================')
        if label.lower() in answer.classification.lower():
            return 10, answer.classification
        elif answer.classification.lower() in potential_label_list:
            return 1, answer.classification
        else:
            return -10, answer.classification

class StructureCheckRewardModel(nn.Module):
    
    def __init__(self, potential_label_list, device):
        super().__init__()
        self.potential_label_list = potential_label_list
        self.device = device    

    def forward(self,input_reward):
        return input_reward