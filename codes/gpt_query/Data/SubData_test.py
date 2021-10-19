import pickle
import json
import random
import torch
import numpy as np
import os
from tokenizers import ByteLevelBPETokenizer

class SubData_test():
    
    def __init__(self, data_dir, vocab_size, bert_model_name, eot="EOT"):
        self.eot = eot

        with open(data_dir+"test.input.txt", "r") as f:
            valid_contexts = f.readlines()
        self.valid_contexts = [[y.strip() for y in x.strip().split(eot)] for x in valid_contexts]
        with open(data_dir+"test.tgt.txt", "r") as f:
            valid_responses = f.readlines()
        self.valid_responses = [x.strip() + ' [SEP]' for x in valid_responses]
        
        with open(data_dir+"test.goal.txt", "r") as f:
            valid_goals = f.readlines()
        self.valid_goals = [x.strip() for x in valid_goals]  
        
        with open(data_dir+"test.key.txt", "r") as f:
            valid_keys = f.readlines()
        self.valid_keys = [[int(y) for y in x.strip().split()] for x in valid_keys]  
        self.valid_keys = [[(key[k], key[k+1]) for k in range(0,len(key),2)] for key in self.valid_keys]
        
        self.shuffle_te = np.arange(len(self.valid_contexts))
        
        path = data_dir+"5ByteLevelBPETokenizer" + str(vocab_size)+'-'
        self.tokenizer = ByteLevelBPETokenizer(vocab_file= path+"vocab.json",merges_file=path+"merges.txt", 
                                               lowercase=True) 
        self.tokenizer.add_special_tokens(["<pad>", "[SEP]"])
    
    
    def tensorFromSentence(self, sent, maxlen):
        indices = torch.Tensor(self.tokenizer.encode(sent).ids).long()
        ulen = len(indices)
        if ulen>maxlen:
            indices = torch.cat((indices[:maxlen-1],  indices[-1:]), dim=0)
            ulen = maxlen
        return indices, ulen
    
    def TensorFromGoal(self, sent, maxlen, g_keys):
        encoding = self.tokenizer.encode(sent)
        offset = encoding.offsets
        j = 0
        new_keys = []
        
        # map the key indices to new key indices after tokenisation
        for start,end in g_keys:
            start-=1 
            while j < len(offset) and j < maxlen:
                if offset[j][0] == start:
                    new_keys.append(j)
                    if offset[j][1] == end:
                        j += 1
                        break
                    
                    j += 1
                    while j < len(offset) and j < maxlen and offset[j][1] != end:
                        new_keys.append(j)
                        j += 1
                    
                    if j<maxlen:
                        new_keys.append(j)
                    j += 1
                    break
                    
                else:  
                    j += 1
                        
        indices = torch.Tensor(encoding.ids).long()
        ulen = len(indices)
        if ulen>maxlen:
            indices = torch.cat((indices[:maxlen-1],  indices[-1:]), dim=0)
            ulen = maxlen
        
        return indices, ulen, new_keys, len(new_keys)

    def shuffle_train(self):
        self.shuffle_te = np.random.permutation(len(self.valid_contexts))

    def get_batch(self, batch_size=10, maxlen=50, train=True, start=-1, word=None, goallen=500):
        contexts = self.valid_contexts
        responses = self.valid_responses
        shuffle = self.shuffle_te
        goal = self.valid_goals
        keys = self.valid_keys

        
        cc_plain = []
        rr_plain = []
        g_plain = []
        g_keys = []
        for i in range(batch_size):
            if word is None:
                if start==-1:
                    ind = random.randint(0, len(contexts)-1)
                else:
                    ind = start + i
                    ind = shuffle[ind]
            else:
                if start==-1:
                    x = random.randint(0, len(self.inverted_index[word])-1)
                    ind = self.inverted_index[word][x]
                else:
                    x = start + i
                    ind = self.inverted_index[word][x]
            
            cc = contexts[ind]
            rr = responses[ind]
            g = goal[ind]
            k = keys[ind]
            
            cc_plain.append(cc)
            rr_plain.append(rr)
            g_plain.append(g)
            g_keys.append(k)
            
        
        max_cutts = max([len(cc) for cc in cc_plain])
        c_utts = torch.zeros(batch_size, max_cutts, maxlen).long()
        c_ulens = torch.zeros(batch_size, max_cutts).long()
        c_clens = torch.zeros(batch_size).long()
        cind_mat = torch.zeros(batch_size, max_cutts, maxlen)

        r_utts = torch.zeros(batch_size, 1, maxlen).long()
        r_ulens = torch.zeros(batch_size, 1).long()
        r_clens = torch.zeros(batch_size).long()
        rind_mat = torch.zeros(batch_size, 1, maxlen)
        
        
        g_utts = torch.zeros(batch_size, goallen).long()
        g_ulens = torch.zeros(batch_size).long()
        g_clens = torch.zeros(batch_size).long()
        gind_mat = torch.zeros(batch_size, goallen)
        
        keys = torch.zeros(batch_size, goallen).long()
        kind_mat = torch.zeros(batch_size, goallen)
        k_ulens = torch.zeros(batch_size).long()
        
        for i,cc in enumerate(cc_plain):
            for j,utt in enumerate(cc):
                uinds, ulen = self.tensorFromSentence(utt, maxlen)
                cind_mat[i, j, :ulen] = 1
                c_utts[i,j, :ulen] = uinds
                c_ulens[i,j] = ulen
                c_clens[i] += 1
               
        for i,rr in enumerate(rr_plain):
            uinds, ulen = self.tensorFromSentence(rr, maxlen)
            rind_mat[i, 0, :ulen] = 1
            r_utts[i, 0, :ulen] = uinds
            r_ulens[i, 0] = ulen
            r_clens[i] = 1
            
        for i,gg in enumerate(g_plain):
            uinds, ulen, new_key, klen = self.TensorFromGoal(gg, goallen, g_keys[i])
            gind_mat[i, :ulen] = 1
            g_utts[i, :ulen] = uinds
            g_ulens[i] = ulen 
            keys[i, :klen] = torch.LongTensor(new_key)
            kind_mat[i, :klen] = 1
            k_ulens[i] = klen

        c_utts = c_utts[:,:,:c_ulens.max()]
        r_utts = r_utts[:,:,:r_ulens.max()]
        g_utts = g_utts[:,:g_ulens.max()]
        cind_mat = cind_mat[:,:,:c_ulens.max()]
        rind_mat = rind_mat[:,:,:r_ulens.max()]
        gind_mat = gind_mat[:,:g_ulens.max()]
        
        keys = keys[:,:k_ulens.max()]
        kind_mat = kind_mat[:,:k_ulens.max()]

        return c_utts, c_ulens, c_clens, r_utts, r_ulens, r_clens, cind_mat,\
                rind_mat, gind_mat, g_utts, g_ulens, keys, kind_mat, k_ulens


