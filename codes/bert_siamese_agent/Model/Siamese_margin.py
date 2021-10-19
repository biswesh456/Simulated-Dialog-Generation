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

    def forward(self, input_ids_1, attention_mask_1):
        x1 = self.longformer_model(input_ids = input_ids_1, attention_mask = attention_mask_1)[1]
        x1 = F.relu(self.FC1(x1))
        x1 = torch.sigmoid(self.FC2(x1).squeeze(1))
        
#         x2 = self.longformer_model(input_ids = input_ids_2, attention_mask = attention_mask_2)[1]
#         x2 = F.relu(self.FC1(x2))
#         x2 = torch.sigmoid(self.FC2(x2).squeeze(1))

        return x1
    
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