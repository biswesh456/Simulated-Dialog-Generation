import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from transformers import LongformerModel

class Siamese_margin(nn.Module):
    
    def __init__(self, args, model_name='allenai/longformer-base-4096'):
        super(Siamese_margin, self).__init__()
        self.longformer_model = LongformerModel.from_pretrained(model_name)
#         for param in self.bert_model.parameters():
#             param.requires_grad = False
        self.FC1 = nn.Linear(768, args.hidden_size)
        self.FC2 = nn.Linear(args.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        x = self.longformer_model(input_ids = input_ids, attention_mask = attention_mask)[1]
        x = F.relu(self.FC1(x))
        x = torch.sigmoid(self.FC2(x).squeeze(1))

        return x
    
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