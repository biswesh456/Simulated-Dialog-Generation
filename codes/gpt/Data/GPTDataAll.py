import pickle
import json
import random
import torch
import numpy as np
import os
from transformers import AutoTokenizer, GPT2Tokenizer

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

class GPTDataAll():
    
    def __init__(self, data_dir, args, eot="EOT"):
        self.eot = eot
#         set_seed(args)
        
        with open(data_dir+"train_input.json") as f:
            contexts_t = json.load(f)
            
        with open(data_dir+"train_tgt.json") as f:
            responses_t = json.load(f)
            
        with open(data_dir+"train_goal.json") as f:
            goal_t = json.load(f)
      
            
        with open(data_dir+"valid_input.json") as f:
            contexts_v = json.load(f)
            
        with open(data_dir+"valid_tgt.json") as f:
            responses_v = json.load(f)
            
        with open(data_dir+"valid_goal.json") as f:
            goal_v = json.load(f)
        
        
        self.train_context = []
        for ctx in contexts_t:
            self.train_context.extend(ctx)
            
        print(len(self.train_context))    
        
        self.train_response = []
        for rr in responses_t:
            self.train_response.extend(rr)
            
        self.train_goal = []
        for gg in goal_t:
            self.train_goal.extend(gg)
           
        
        self.valid_context = []
        for ctx in contexts_v:
            self.valid_context.extend(ctx)

        self.valid_response = []
        for rr in responses_v:
            self.valid_response.extend(rr)
            
        self.valid_goal = []
        for gg in goal_v:
            self.valid_goal.extend(gg)
    
            
#         self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")  
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # shuffle the data
        shuffle_train = list(zip(self.train_context, self.train_response,\
                                 self.train_goal))
        random.shuffle(shuffle_train, random = lambda: 0.7)
        self.train_context, self.train_response, self.train_goal = zip(*shuffle_train)
        
#         set_seed(args)
        shuffle_valid = list(zip(self.valid_context, self.valid_response,\
                                 self.valid_goal))
        random.shuffle(shuffle_valid, random = lambda: 0.7)
        self.valid_context, self.valid_response, self.valid_goal = zip(*shuffle_valid)
        
        

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





