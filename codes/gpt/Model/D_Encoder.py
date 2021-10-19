import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils

    
class D_Encoder(nn.Module):
    
    
    def __init__(self, args):
        super(D_Encoder, self).__init__()
        self.context_size = args.hidden_size
        self.dense1 = nn.Linear(self.context_size, self.context_size)
        self.num_layers = 2
        self.num_directions = 2
        self.hidden_size = int(self.context_size/2)
        self.lstm = nn.GRU(self.context_size, self.hidden_size, batch_first=True, bidirectional=self.num_directions==2, num_layers=self.num_layers)
        
        

        
    def forward(self, uembeds, clens):
        uembeds = F.relu(self.dense1(uembeds))
        uembeds_packed = rnn_utils.pack_padded_sequence(uembeds, clens, batch_first=True, enforce_sorted=False)
        output,hidden = self.lstm(uembeds_packed)
        hidden = hidden.view(self.num_layers, self.num_directions, -1, self.hidden_size).sum(0)
        contexts = torch.cat([hidden[0], hidden[1]], dim=1).unsqueeze(1)
        return contexts

