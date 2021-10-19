import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils

class U_Encoder(nn.Module):
    
    def __init__(self, args):
        super(U_Encoder, self).__init__()
        self.vocab_size = args.vocab_size
        self.word_size = args.word_size
        self.hidden_size = args.hidden_size
        self.word2embed = nn.Embedding(self.vocab_size, self.word_size)
        self.gru = nn.GRU(input_size=self.word_size, hidden_size=int(self.hidden_size/2), batch_first=True,
                           bidirectional=True)
    
    
    def forward(self, utts, ulens):
        uwembeds = self.word2embed(utts)
        uwembeds_packed = rnn_utils.pack_padded_sequence(uwembeds, ulens, batch_first=True, enforce_sorted=False)
        output,hidden = self.gru(uwembeds_packed)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        return output, hidden
