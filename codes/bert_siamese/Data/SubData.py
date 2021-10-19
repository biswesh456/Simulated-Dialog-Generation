import pickle
import json
import random
import torch
import numpy as np
import os
from transformers import BertTokenizer, GPT2Tokenizer
from . import LoadGPT

class SubData():
    
    def __init__(self, data_dir, vocab_size, bert_model_name='bert-base-uncased', eot="EOT"):
        self.eot = eot
        key = 'restaurant'
        
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
        
#         with open(data_dir+"train_input.json") as f:
#             contexts_t = json.load(f)
            
#         with open(data_dir+"train_tgt.json") as f:
#             responses_t = json.load(f)
            
#         with open(data_dir+"train_goal.json") as f:
#             goal_t = json.load(f)     
            
#         with open(data_dir+"valid_input.json") as f:
#             contexts_v = json.load(f)
            
#         with open(data_dir+"valid_tgt.json") as f:
#             responses_v = json.load(f) 
        
#         with open(data_dir+"valid_goal.json") as f:
#             goal_v = json.load(f)  
        
#         domain_key = 'train'
        
#         self.train_context = []
#         for key in contexts_t:
#             if key != 'taxi':
#                 self.train_context.extend(contexts_t[key])
        
#         self.train_response = []
#         for key in responses_t:
#             if key != 'taxi':
#                 self.train_response.extend(responses_t[key])
           
#         self.train_goal = []
#         for key in query_t:
#             if key != 'taxi':
#                 self.train_goal.extend(goal_t[key])
        
#         self.valid_context = []
#         for key in contexts_v:
#             if key != 'taxi':
#                 self.valid_context.extend(contexts_v[key])
        
#         self.valid_response = []
#         for key in responses_v:
#             if key != 'taxi':
#                 self.valid_response.extend(responses_v[key])
            
#         self.valid_goal = []
#         for key in query_t:
#             if key != 'taxi':
#                 self.valid_goal.extend(goal_t[key])
            
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
#         self.user_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#         user_gpt = self.getUserModel()
#         self.user_model = user_gpt.model
#         for param in self.user_model.parameters():
#             param.requires_grad = False
            
#         self.user_model.eval()

    def getUserModel(self):
        model_path = '/u/vineeku6/storage/gaurav/biswesh/models/multiwiz/user/gpt_train1/checkpoint_hred_user__best_128_512_0.0001_1.pth.tar'
        checkpoint = torch.load(model_path)
        load_args = checkpoint['args']
        model = LoadGPT.GPT(load_args)
        model.cuda()
        model.load_state_dict(checkpoint['state_dict'])
        return model
        
    def get_batch(self, train=True, start=-1):
        if train:
#             start = random.randint(0, len(self.train_response)-1)
            context = self.train_input[start][0]
            response = self.train_input[start][1]
            label = int(self.train_label[start])
            
        else:
            context = self.valid_input[start][0]
            response = self.valid_input[start][1]
            label = int(self.valid_label[start])
        
#         print(context, response)
        encode = self.tokenizer.batch_encode_plus([[context, response]], pad_to_max_length=True)

        labels = torch.tensor([int(label)])

        input_ids = torch.tensor(encode.input_ids)
        token_type_ids = torch.tensor(encode.token_type_ids)
        attention_mask = torch.tensor(encode.attention_mask)

        return input_ids, token_type_ids, attention_mask, labels


#     def get_batch(self, train=True, start=-1):
#             resp_no = 5
#             if train:
#                 start = random.randint(0, len(self.train_response)-1)
#                 context = " ".join(self.train_context[start])
#     #             context = self.train_context[start][-1]
#                 response = self.train_response[start]
#                 goal = self.train_goal[start]

#                 new_user_input = self.user_tokenizer.encode(goal + 'GOAL ' + " ".join(self.train_context[start]) + \
#                                                        self.user_tokenizer.eos_token, return_tensors='pt')

#                 generated = self.user_model.generate(new_user_input.cuda(), max_length=400,\
#                                                pad_token_id=self.user_tokenizer.eos_token_id, do_sample=True,\
#                                                top_k=20, top_p=0.5, num_return_sequences=resp_no, early_stopping=True)

#                 adverse = []
#                 for j in range(resp_no):
#                     resp = self.user_tokenizer.decode(generated[j][new_user_input.shape[1]:]).replace('<|endoftext|>', '')
#                     adverse.append([goal + ' ' + context, resp])

#             else:
#                 start = start % len(self.valid_goal)
#                 context = " ".join(self.valid_context[start])
#     #             context = self.valid_context[start][-1]
#                 response = self.valid_response[start]
#                 goal = self.valid_goal[start]

#                 new_user_input = self.user_tokenizer.encode(goal + 'GOAL ' + " ".join(self.valid_context[start]) + \
#                                                        self.user_tokenizer.eos_token, return_tensors='pt')

#                 generated = self.user_model.generate(new_user_input.cuda(), max_length=400,\
#                                                pad_token_id=self.user_tokenizer.eos_token_id, do_sample=True,\
#                                                top_k=20, top_p=0.5, num_return_sequences=resp_no, early_stopping=True)

#                 adverse = []
#                 for j in range(resp_no):
#                     resp = self.user_tokenizer.decode(generated[j][new_user_input.shape[1]:]).replace('<|endoftext|>', '')
#                     adverse.append([goal + ' ' + context, resp])
 
#             labels = [0]*resp_no
#             idx = random.randint(0,4)    
#             adverse[idx] = [goal + ' ' + context, response]
#             labels[idx] = 1  

#             encode = self.tokenizer.batch_encode_plus(adverse, pad_to_max_length=True)
#             labels = torch.tensor(labels)

#             input_ids = torch.tensor(encode.input_ids)
#             token_type_ids = torch.tensor(encode.token_type_ids)
#             attention_mask = torch.tensor(encode.attention_mask)

#             return input_ids, token_type_ids, attention_mask, labels

