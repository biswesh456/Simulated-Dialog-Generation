import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from transformers import GPT2LMHeadModel

class GPT(nn.Module):
    
    def __init__(self, args):
        super(GPT, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
                              

    def forward(self, input_ids, response):
        label = torch.zeros(input_ids.shape).long() - 100
        label[0, -response.shape[1]:] = response[0]
#         print(label)
#         print(response)
        label = label.cuda()
        loss, logits, _ = self.model(input_ids, labels=label)          

        return loss
        
    
    def load(self, checkpoint, args):
        self.load_state_dict(checkpoint['state_dict'])
        optimizer = torch.optim.Adam(params=self.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        return optimizer
    
    def save(self, name, optimizer, args):
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'args': args
}, name)