import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from transformers import BertModel, BertTokenizer

class Siamese(nn.Module):
    
    def __init__(self, args, model_name='bert-base-uncased'):
        super(Siamese, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         for param in self.bert_model.parameters():
#             param.requires_grad = False
        self.FC1 = nn.Linear(768, args.hidden_size)
        self.FC2 = nn.Linear(args.hidden_size, 1)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        x = self.bert_model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[1]
        x = F.relu(self.FC1(x))
        x = torch.sigmoid(self.FC2(x).squeeze(1))
        
#         print(self.tokenizer.decode(input_ids.tolist()[0]), labels[0].item(), x.item(), '\n')
        loss = F.binary_cross_entropy(x, labels.to(x))
        
        return x, loss
    
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