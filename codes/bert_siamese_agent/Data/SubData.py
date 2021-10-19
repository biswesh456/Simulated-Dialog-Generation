import pickle
import json
import random
import torch
import numpy as np
import os
from transformers import BertTokenizer, GPT2Tokenizer, LongformerTokenizer
from . import LoadGPT

class SubData():
    
    def __init__(self, data_dir, vocab_size, pretrained_model_name='allenai/longformer-base-4096'):
        
        key = 'train'
        
        with open(data_dir+"train_bert_input.json") as f:
            train_input = json.load(f)[key]

        self.train_input = [ ctx.strip().split(' EOT ') for ctx in train_input]
        
        with open(data_dir+"valid_bert_input.json") as f:
            valid_input = json.load(f)[key]

        self.valid_input = [ ctx.strip().split(' EOT ') for ctx in valid_input]
        
        
        with open(data_dir+"train_bert_label.json") as f:
            train_label = json.load(f)[key]

        self.train_label = [ ctx.strip() for ctx in train_label]
        
        with open(data_dir+"valid_bert_label.json") as f:
            valid_label = json.load(f)[key]

        self.valid_label = [ ctx.strip() for ctx in valid_label]
        
        self.tokenizer = LongformerTokenizer.from_pretrained(pretrained_model_name)    
#         self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
#         self.user_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def get_batch(self, train=True, start=-1):
        if train:
#             start = random.randint(0, len(self.train_response)-1)
            context = self.train_input[start][0]
            response = self.train_input[start][1]
            label = int(self.train_label[start])
            
        else:
            start = start%len(self.valid_input)
            context = self.valid_input[start][0]
            response = self.valid_input[start][1]
            label = int(self.valid_label[start])
        
        encode = self.tokenizer.batch_encode_plus([[context, response]], pad_to_max_length=True, return_token_type_ids=True)
        
        labels = torch.tensor([label])
        input_ids = torch.tensor(encode.input_ids)
        token_type_ids = torch.tensor(encode.token_type_ids)
        attention_mask = torch.tensor(encode.attention_mask)

        return input_ids, token_type_ids, attention_mask, labels

