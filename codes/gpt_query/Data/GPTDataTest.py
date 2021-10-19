import pickle
import json
import random
import torch
import numpy as np
import os
from transformers import AutoTokenizer, GPT2Tokenizer

class GPTDataTest():
    
    def __init__(self, data_dir):  
            
        with open(data_dir+"test_input.json") as f:
            contexts_t = json.load(f)
            
        with open(data_dir+"test_query.json") as f:
            responses_t = json.load(f)
        
        self.test_context = []
        for ctx in contexts_t:
            self.test_context.extend(ctx)
            
        print(len(self.test_context))    
        
        self.test_response = []
        for rr in responses_t:
            self.test_response.extend(rr)
            
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")  
        

    def get_batch(self, start=-1):
#         start = random.randint(0, len(self.test_response)-1)
        context = " ".join(self.test_context[start])
        response = self.test_response[start]
        kb = self.test_kb[start]
        query = self.test_query[start]
            
        input_ids = self.tokenizer.encode(context, return_tensors='pt')
        
        return input_ids, response, context





