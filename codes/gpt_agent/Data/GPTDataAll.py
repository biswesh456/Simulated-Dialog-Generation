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
            
        with open(data_dir+"train_kb.json") as f:
            kb_t = json.load(f)
            
        with open(data_dir+"train_query.json") as f:
            query_t = json.load(f)
            
        with open(data_dir+"valid_input.json") as f:
            contexts_v = json.load(f)
            
        with open(data_dir+"valid_tgt.json") as f:
            responses_v = json.load(f)
            
        with open(data_dir+"valid_kb.json") as f:
            kb_v = json.load(f)
        
        with open(data_dir+"valid_query.json") as f:
            query_v = json.load(f)
        
        self.train_context = []
        for ctx in contexts_t:
            self.train_context.extend(ctx)
            
        print(len(self.train_context))    
        
        self.train_response = []
        for rr in responses_t:
            self.train_response.extend(rr)
            
        self.train_kb = []
        for kk in kb_t:
            self.train_kb.extend(kk)
           
        self.train_query = []
        for qq in query_t:
            self.train_query.extend(qq)
        
        self.valid_context = []
        for ctx in contexts_v:
            self.valid_context.extend(ctx)

        self.valid_response = []
        for rr in responses_v:
            self.valid_response.extend(rr)
            
        self.valid_kb = []
        for kk in kb_v:
            self.valid_kb.extend(kk)
    
        self.valid_query = []
        for qq in query_v:
            self.valid_query.extend(qq)
            
#         self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")  
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # shuffle the data
        shuffle_train = list(zip(self.train_context, self.train_response,\
                                 self.train_kb, self.train_query))
        random.shuffle(shuffle_train, random = lambda: 0.7)
        self.train_context, self.train_response, self.train_kb, self.train_query = zip(*shuffle_train)
        
#         set_seed(args)
        shuffle_valid = list(zip(self.valid_context, self.valid_response,\
                                 self.valid_kb, self.valid_query))
        random.shuffle(shuffle_valid, random = lambda: 0.7)
        self.valid_context, self.valid_response, self.valid_kb, self.valid_query = zip(*shuffle_valid)
        
        

    def get_batch(self, train=True, start=0, bs=32):
        if train:
#             start = random.randint(0, len(self.train_response)-1)
            context = " ".join(self.train_context[start])
            response = self.train_response[start]
            if start != 0:
                adverse = self.train_response[start-1]
            else:
                adverse = self.train_response[3]
            kb = self.train_kb[start]
            query = self.train_query[start]
            
        else:
            start = start % len(self.valid_kb)
            context = " ".join(self.valid_context[start])
            response = self.valid_response[start]
            if start != 0:
                adverse = self.valid_response[start-1]
            else:
                adverse = self.valid_response[3]
            kb = self.valid_kb[start]
            query = self.valid_query[start]
        
        input_ids = self.tokenizer.encode( query + ' ' + kb + ' ' + context + ' ' +  response + self.tokenizer.eos_token, \
                                          return_tensors='pt')
        
        response_id = self.tokenizer.encode(response + self.tokenizer.eos_token, return_tensors='pt')
        
        adverse_input_ids = self.tokenizer.encode(query + ' ' + kb + ' ' + context + ' ' + adverse + self.tokenizer.eos_token, \
                                                  return_tensors='pt')
        
        adverse_id = self.tokenizer.encode(adverse + self.tokenizer.eos_token, return_tensors='pt')
        
        return input_ids, response_id, adverse_input_ids, adverse_id, response




