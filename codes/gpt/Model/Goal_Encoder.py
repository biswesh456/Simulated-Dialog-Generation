import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils

class Goal_Encoder(nn.Module):
    
    def __init__(self, args):
        super(Goal_Encoder, self).__init__()
        self.vocab_size = args.vocab_size
        self.word_size = args.word_size
        self.hidden_size = args.hidden_size
        self.word2embed = nn.Embedding(self.vocab_size, self.word_size)
        self.gru = nn.GRU(input_size=self.word_size, hidden_size=int(self.hidden_size/2), batch_first=True,
                           bidirectional=True)
    
    
    def forward(self, g_utts, g_ulens, gind_mat, keys, k_ulens):
        gwembeds = self.word2embed(g_utts)
        gwembeds_packed = rnn_utils.pack_padded_sequence(gwembeds, g_ulens, batch_first=True, enforce_sorted=False)
        output,hidden = self.gru(gwembeds_packed)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)

        batch_size = output.shape[0]
        gembeds = torch.zeros(batch_size, keys.shape[1], output.shape[2]).cuda() # copying only relevent key embeddings
        g_pos = torch.zeros(batch_size, keys.shape[1]).long().cuda() # gives position ids of keys
 
        for i in range(batch_size):
            gembeds[i, :k_ulens[i], :] = output[i, keys[i], :][:k_ulens[i]]
            g_pos[i,:k_ulens[i]] = g_utts[i, keys[i]][:k_ulens[i]]
        
        return gembeds, g_pos, hidden