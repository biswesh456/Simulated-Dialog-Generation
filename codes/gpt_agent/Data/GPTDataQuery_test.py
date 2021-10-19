import pickle
import json
import random
import torch
import numpy as np
import os
from transformers import AutoTokenizer, GPT2Tokenizer

class GPTDataQuery_test():
    
    def __init__(self, data_dir, Type):
        type_str = ''
        if Type == 'single':
            type_str = '_single'
        elif Type == 'multiple':
            type_str = '_multiple'
        elif Type == 'entire':
            type_str = ''
        else:
            print('Wrong TYPE !!!')
            return
        
        with open(data_dir+"test_input"+type_str+".json") as f:
            contexts_t = json.load(f)
            
        with open(data_dir+"test_tgt"+type_str+".json") as f:
            responses_t = json.load(f)
            
        with open(data_dir+"test_kb"+type_str+".json") as f:
            kb_t = json.load(f)
            
        with open(data_dir+"test_query"+type_str+".json") as f:
            query_t = json.load(f)
            
        with open(data_dir+"test_dialogue_names"+type_str+".json") as f:
            names_t = json.load(f)
        
        self.names = []
        
        self.test_context = []
        for ctx in contexts_t:
            self.test_context.extend(ctx)
            
        print(len(self.test_context))    
        
        self.test_response = []
        for i,rr in enumerate(responses_t):
            for r in rr:
                self.test_response.append(r)
                self.names.append(names_t[i])
            
        self.test_kb = []
        for kk in kb_t:
            self.test_kb.extend(kk)
           
        self.test_query = []
        for qq in query_t:
            self.test_query.extend(qq)
            
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")  
        

    def get_batch(self, start=-1):
#         start = random.randint(0, len(self.test_response)-1)
        context = " ".join(self.test_context[start])
        query_context = self.test_context[start][-1]
        response = self.test_response[start]
        kb = self.test_kb[start]
        query = self.test_query[start]
#         print('Original : ', query)
        file_name = self.names[start]
            
#         input_ids = self.tokenizer.encode(context, return_tensors='pt')
        
        return response, kb, context, file_name, query_context




