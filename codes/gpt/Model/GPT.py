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
        
    def compute_loss(self,worddis, r_utts, r_ulens):
        worddis = torch.log(worddis)
        r_utts_new = torch.zeros(r_utts.shape).to(r_utts).long()-100
        for i, ll in enumerate(r_ulens):
            r_utts_new[i, :ll] = r_utts[i,:ll]
        r_utts = r_utts_new
        num_utts, num_words, vocab_size = worddis.shape
        worddis = worddis.view(-1, vocab_size)
        r_utts = r_utts.view(-1)
        assert worddis.shape[0] == r_utts.shape[0]
        loss = F.nll_loss(worddis, r_utts.clone().detach(), ignore_index=-100, reduction='none')
        loss = loss.view(num_utts, num_words)
        return loss
    
    def get_context_embedding(self, c_utts, c_ulens, c_clens):
        batch_size, num_utts, num_words = c_utts.shape

        #Reshape to 2d tensor
        c_utts_2d = c_utts.contiguous().view(-1, num_words)
        c_ulens_2d = c_ulens.view(-1)
    
        #Encode the utterances
        mask = c_ulens_2d>0
        c_utts_sel = c_utts_2d[mask]
        c_ulens_sel = c_ulens_2d[mask]
        c_output_embed, c_uembeds_sel = self.u_encoder(c_utts_sel, c_ulens_sel)
        
        
        c_uembeds_2d = torch.zeros(c_utts_2d.shape[0], c_uembeds_sel.shape[1]).cuda()
        c_uembeds_2d.masked_scatter_(mask.unsqueeze(1), c_uembeds_sel)
        c_outputs_3d = torch.zeros(c_utts_2d.shape[0], num_words, c_uembeds_sel.shape[1]).cuda()
        c_outputs_3d.masked_scatter_(mask.view(-1,1,1), c_output_embed)
        
        assert c_uembeds_2d.shape[0] == batch_size*num_utts
        
        #Reshape and encode the contexts to get context embeddings
        c_uembeds = c_uembeds_2d.view(batch_size, num_utts, -1)
        
        # reshape encoder output to be used for attention
        c_uhidden = c_outputs_3d.view(batch_size, num_utts, num_words, -1) 
        c_embeds = self.c_encoder(c_uembeds, c_clens)
        
        return c_embeds, c_uhidden

    
    def get_loss_from_context_embedding(self, c_uhidden, c_embeds, r_utts, r_ulens, c_utts, gembeds, kind_mat, g_pos, g_hid):

        batch_size, num_utts, num_words = r_utts.shape
        #Reshape to 2d Tensor
        
        c_embeds_2d = c_embeds.squeeze(1)
        r_utts_2d = r_utts.contiguous().view(-1, r_utts.shape[2])
        r_ulens_2d = r_ulens.view(-1)

        assert c_embeds_2d.shape[0] == r_utts_2d.shape[0]

        worddis_sel = self.decoder(c_embeds_2d, r_utts_2d, r_ulens_2d, c_uhidden, c_utts, gembeds, kind_mat, g_pos, g_hid)
        #Compute loss
        loss = self.compute_loss(worddis_sel, r_utts_2d, r_ulens_2d).unsqueeze(1)

        return loss
                              

    def forward(self, input_ids, response):
        label = torch.zeros(input_ids.shape).long() - 100
        label[0, -response.shape[1]:] = response[0]
#         print(label)
#         print(response)
        label = label.cuda()
        loss, logits, _ = self.model(input_ids, labels=label)          

        return loss
        

    def get_greedy_samples(self, input_ids, SEP, max_length):
        response = []
        for i in range(max_length):
            logits, _ = self.model(input_ids)
            new_id = logits.argmax().item()
            response.append(new_id)
        
        return response  
        
    
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