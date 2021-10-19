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

class GPTDataTopic():
    
    def __init__(self, data_dir, domain_key, args, eot="EOT"):
        self.eot = eot
        set_seed(args)
        
        with open(data_dir+"train_input.json") as f:
            contexts_t = json.load(f)
            
        with open(data_dir+"train_query.json") as f:
            responses_t = json.load(f)
            
        with open(data_dir+"valid_input.json") as f:
            contexts_v = json.load(f)
            
        with open(data_dir+"valid_query.json") as f:
            responses_v = json.load(f)     
        
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
            

#         self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")  
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def get_batch(self, train=True, start=-1):
        if train:
#             start = random.randint(0, len(self.train_response)-1)
            context = " ".join(self.train_context[start])
            response = self.train_response[start]
            
        else:
            start = start % len(self.valid_response)
            context = " ".join(self.valid_context[start])
            response = self.valid_response[start]

        input_ids = self.tokenizer.encode(context + ' ' + response + self.tokenizer.eos_token, \
                                          return_tensors='pt')
        
        response_id = self.tokenizer.encode(response + self.tokenizer.eos_token, return_tensors='pt')

        
        return input_ids, response_id

