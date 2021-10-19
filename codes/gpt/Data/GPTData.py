import pickle
import json
import random
import torch
import numpy as np
import os
from transformers import AutoTokenizer, GPT2Tokenizer

class GPTData():
    
    def __init__(self, data_dir, domain_key, eot="EOT"):
        self.eot = eot
        
        with open(data_dir+"train_input.json") as f:
            contexts_t = json.load(f)
            
        with open(data_dir+"train_tgt.json") as f:
            responses_t = json.load(f)
            
        with open(data_dir+"train_goal.json") as f:
            goals_t = json.load(f)  

        with open(data_dir+"valid_input.json") as f:
            contexts_v = json.load(f)
            
        with open(data_dir+"valid_tgt.json") as f:
            responses_v = json.load(f)
            
        with open(data_dir+"valid_goal.json") as f:
            goals_v = json.load(f)  
            

        self.train_context = []
#         for key in contexts_t:
#                 self.train_context.extend(contexts_t[key])
        for ctx in contexts_t[domain_key]:
                self.train_context.extend(ctx)
        
        self.train_response = []
#         for key in responses_t:
#                 self.train_response.extend(responses_t[key])
        for rr in responses_t[domain_key]:
                self.train_response.extend(rr)
            
        self.train_goal = []
#         for key in goals_t:
#                 self.train_goal.extend(goals_t[key])
        for gg in goals_t[domain_key]:
                self.train_goal.extend(gg)
    
        self.valid_context = []
#         for key in contexts_v:
#                 self.valid_context.extend(contexts_v[key])
        for ctx in contexts_v[domain_key]:
                self.valid_context.extend(ctx)
    
        
        self.valid_response = []
#         for key in responses_v:
#                 self.valid_response.extend(responses_v[key])
        for rr in responses_v[domain_key]:
                self.valid_response.extend(rr)
            
        self.valid_goal = []
#         for key in goals_v:
#                 self.valid_goal.extend(goals_v[key])
        for gg in goals_v[domain_key]:
                self.valid_goal.extend(gg)
            
#         self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")  
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def get_batch(self, train=True, start=-1):
        if train:
#             start = random.randint(0, len(self.train_goal)-1)
            goal = self.train_goal[start]
            context = " ".join(self.train_context[start])
#             context = self.train_context[start]
            response = self.train_response[start]
#             adverse = self.train_adverse[start]
            
        else:
            start = start % len(self.valid_goal)
            goal = self.valid_goal[start]
            context = " ".join(self.valid_context[start])
#             context = self.valid_context[start]
            response = self.valid_response[start]
#             adverse = self.valid_adverse[start]
            
        input_ids = self.tokenizer.encode(goal + 'GOAL ' + context + ' ' + response + self.tokenizer.eos_token, \
                                          return_tensors='pt')
        
        response_id = self.tokenizer.encode(' ' + response + self.tokenizer.eos_token, return_tensors='pt')
        
#         adverse_input_ids = self.tokenizer.encode(goal + 'GOAL ' + context + ' ' + adverse + self.tokenizer.eos_token, \
#                                                   return_tensors='pt')
        
#         adverse_id = self.tokenizer.encode(' ' + adverse + self.tokenizer.eos_token, return_tensors='pt')
        
        adverse_input_ids = torch.zeros(1)
        adverse_id = torch.zeros(1)
        
        
        return input_ids, response_id, adverse_input_ids, adverse_id

