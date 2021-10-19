import pickle
import json
import random
import torch
import numpy as np
import os
from transformers import AutoTokenizer, GPT2Tokenizer

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

class GPTData():
    
    def __init__(self, data_dir, args, eot="EOT"):
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

#         self.train_context = []
#         for ctx in contexts_t:
#             self.train_context.extend(ctx)
          
        self.train_context = []
        for key in contexts_t:
            if key != 'police' and key != 'hospital':
                for ctx in contexts_t[key]:
                    self.train_context.extend(ctx)
        
        self.train_response = []
        for key in responses_t:
            if key != 'police' and key != 'hospital':
                for rr in responses_t[key]:
                    self.train_response.extend(rr)    
              
        self.valid_context = []
        for key in contexts_v:
            if key != 'police' and key != 'hospital':
                for ctx in contexts_v[key]:
                    self.valid_context.extend(ctx)
#         self.valid_context.extend(contexts_v[domain_key])
        
        self.valid_response = []
        for key in responses_v:
            if key != 'police' and key != 'hospital':
                for rr in responses_v[key]:
                    self.valid_response.extend(rr)
                    
                    
#         self.train_response = []
#         for rr in responses_t:
#             self.train_response.extend(rr)
        
#         self.valid_context = []
#         for ctx in contexts_v:
#             self.valid_context.extend(ctx)

#         self.valid_response = []
#         for rr in responses_v:
#             self.valid_response.extend(rr)
            
#         self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")  
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # shuffle the data
        shuffle_train = list(zip(self.train_context, self.train_response))
        random.shuffle(shuffle_train, random = lambda: 0.7)
        self.train_context, self.train_response = zip(*shuffle_train)
        
        shuffle_valid = list(zip(self.valid_context, self.valid_response))
        random.shuffle(shuffle_valid, random = lambda: 0.7)
        self.valid_context, self.valid_response = zip(*shuffle_valid)
        

    def get_batch(self, train=True, start=0, batch_size=2):
        inputs = []
        context = []
        response = []
        
        if train:
#             start = random.randint(0, len(self.train_response)-1)
            for b in range(start, min(start+batch_size, len(self.train_response))):
                inputs.append(" ".join(self.train_context[b]) + ' ' + self.train_response[b] + self.tokenizer.eos_token) 
                context.append(" ".join(self.train_context[b]) + ' ')
                response.append(self.train_response[b])
  
        else:
            for b in range(start, min(start+batch_size, len(self.valid_response))):
                inputs.append(" ".join(self.valid_context[b]) + ' ' + self.train_response[b] + self.tokenizer.eos_token) 
                context.append(" ".join(self.valid_context[b]) + ' ')

        encode = self.tokenizer.batch_encode_plus(inputs, pad_to_max_length=True, return_tensors='pt')

        attention = encode['attention_mask']
        input_ids = encode['input_ids']

        context_encode = self.tokenizer.batch_encode_plus(context, pad_to_max_length=True, return_tensors='pt')
  
        context_mask = context_encode['attention_mask'] == 0
        
        mask = torch.ones(input_ids.shape)
        mask[:, :context_mask.shape[1]] = context_mask
        mask = mask * attention
        labels = mask.long() * input_ids
        
        labels[labels == 0] = -100

        return input_ids[:, :1022], labels[:, :1022]








