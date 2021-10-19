import pickle
import json
import random
import torch
import numpy as np
import os
from transformers import BertTokenizer, GPT2Tokenizer, LongformerTokenizer
from . import LoadGPT

class DataMargin():
    
    def __init__(self, data_dir, vocab_size, key, model_name='allenai/longformer-base-4096', eot="EOT"):
        self.eot = eot
        
        with open(data_dir+"train_bert_margin.json") as f:
            train_input = json.load(f)

        self.train_input = [ ctx.strip().split(' EOT ') for ctx in train_input]
        random.shuffle(self.train_input)
        
        with open(data_dir+"valid_bert_margin.json") as f:
            valid_input = json.load(f)

        self.valid_input = [ ctx.strip().split(' EOT ') for ctx in valid_input]
        random.shuffle(self.valid_input)    
            
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        
    def get_batch(self, train=True, start=-1):
        if train:
#             start = random.randint(0, len(self.train_response)-1)
            context = self.train_input[start][0]
            response = self.train_input[start][1]
            adverse = self.train_input[start][2]
            
        else:
            start = start%len(self.valid_input)
            context = self.valid_input[start][0]
            response = self.valid_input[start][1]
            adverse = self.valid_input[start][2]
        
        encode1 = self.tokenizer.batch_encode_plus([[context, response]], pad_to_max_length=True,\
                                                  return_token_type_ids=True)
        
        encode2 = self.tokenizer.batch_encode_plus([[context, adverse]], pad_to_max_length=True,\
                                                  return_token_type_ids=True)
        
        input_ids1 = torch.tensor(encode1.input_ids)
        token_type_ids1 = torch.tensor(encode1.token_type_ids)
        attention_mask1 = torch.tensor(encode1.attention_mask)
        
        input_ids2 = torch.tensor(encode2.input_ids)
        token_type_ids2 = torch.tensor(encode2.token_type_ids)
        attention_mask2 = torch.tensor(encode2.attention_mask)

        return input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2
