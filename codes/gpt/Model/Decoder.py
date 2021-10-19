import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils
    

class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.word_size = args.word_size
        self.hidden_size = args.hidden_size
        self.word2embed = nn.Embedding(args.vocab_size, args.word_size)
        self.lstm = nn.GRU(input_size=self.hidden_size + self.word_size, hidden_size=self.hidden_size,
                           batch_first=True)
        self.vocab_size = args.vocab_size
        self.worder = nn.Linear(self.hidden_size * 3, args.vocab_size)
        
        # For attention
        self.W_h = nn.Linear(args.hidden_size, args.hidden_size)
        self.W_s = nn.Linear(args.hidden_size, args.hidden_size)
        self.W_v = nn.Linear(self.hidden_size, 1)
        
        # For switch probability
        self.switch1 = nn.Linear(self.hidden_size * 4 + self.word_size, self.hidden_size)
        self.switch2 = nn.Linear(self.hidden_size, 1)
        
        # For key probability
        self.key_1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.key_2 = nn.Linear(self.hidden_size, 1)
#         self.key_small = nn.Linear(self.hidden_size, self.hidden_size)
        
        # for gating
        self.gate = nn.Linear(self.hidden_size * 3, 1)


    def attention_decode(self, h_t, dec_input, c_uhidden_state, batch_size, c_utts, g_embeds, kind_mat, g_pos, g_hid):
        _, h_t = self.lstm(dec_input, h_t.unsqueeze(0))
        h_t = h_t.squeeze(0)
        
        # calculate the attention using hidden states of utterance encoder 
        c_uhidden_state_3d = c_uhidden_state.view(batch_size,-1,self.hidden_size) # concat all utterances of a conversation
        hidden_att = self.W_h(c_uhidden_state_3d)
        state_att = self.W_s(h_t).view(batch_size, 1, -1) # (batch_size, 1, hidden_size) for repeat
        state_att = state_att.repeat(1, hidden_att.shape[1], 1)  # repeat in order to add
        att_logit = self.W_v(F.relu(hidden_att + state_att)).squeeze(2)
        
        c_utts_2D = c_utts.view(batch_size, -1)
        mask = c_utts_2D == 0.0
        att_logit.masked_fill_(mask, float('-inf')) # remove attention for paddings
        attentions = F.softmax(att_logit, dim = 1)

        h_attention = torch.bmm(attentions.unsqueeze(1), c_uhidden_state_3d).squeeze(1) # calculate new hidden using attention  
        
        # calculate the attention using goal keys
        key_max_size = kind_mat.shape[1]
        
        h_state_concat = torch.cat([h_t, h_attention], dim=1).unsqueeze(1).repeat(1, g_embeds.shape[1], 1)

        key_attn_input = torch.cat([h_state_concat, g_embeds], dim=2)
        key_attn_logit = self.key_2(F.relu(self.key_1(key_attn_input))).squeeze(2)

        key_mask = kind_mat == -1
        key_attn_logit.masked_fill_(key_mask, float('-inf')) # remove attention for paddings
        key_attn_no_mask = F.softmax(key_attn_logit, dim = 1)

        # take care of case containing no keys
        no_keys_mask = torch.sum(key_mask, 1) != key_max_size
        key_attn = torch.zeros(batch_size, key_max_size).cuda()
        for b in range(batch_size):
            if no_keys_mask[b]:
                key_attn[b] = key_attn_no_mask[b]

        h_attn_key = torch.bmm(key_attn.unsqueeze(1), g_embeds).squeeze(1) # calculate new hidden using attention 
        
        # creating the state that would be used for generation
        gen_state = torch.cat([h_t, h_attention, g_hid], dim=1)
        
        # calculate p_switch
#         switch = torch.sigmoid(self.switch_h(h_attention) + self.switch_s(h_t) + self.switch_x(dec_input.squeeze(1)))
#         switch = torch.sigmoid(self.switch_h(h_attention) + self.switch_k(h_attn_key) + \
#                                self.switch_s(h_t) + self.switch_x())
        switch_state = torch.cat([h_t, h_attention, h_attn_key, dec_input.squeeze(1)], dim=1)
        switch = torch.sigmoid(self.switch2(F.relu(self.switch1(switch_state))))
        
        # calculate p_copy by adding attention of the word wherever it occurs and placing it in its index
        context_copy = torch.zeros(batch_size, self.vocab_size).cuda()
        context_copy.scatter_add_(1, c_utts_2D, attentions)
        
        # calculate the copy probability of the key in goal
        key_copy = torch.zeros(batch_size, self.vocab_size).cuda()
        key_copy.scatter_add_(1, g_pos, key_attn)
        
        # deciding among copying between key and context
        g = torch.sigmoid(self.gate(torch.cat([h_t, h_attention, h_attn_key], dim=1)))
        copy = g * key_copy + (1-g) * context_copy  

        return gen_state, switch.squeeze(1), copy, h_t, g
        

    def forward(self, cembeds, r_utts, r_ulens, c_uhidden_state, c_utts, gembeds, kind_mat, g_pos, g_hid):
        num_utts, num_words = r_utts.shape
        rwembeds = self.word2embed(r_utts)
        
        h_0 = cembeds
        # We need to provide the current sentences's embedding or "thought" at every timestep.
        cembeds = cembeds.unsqueeze(1).repeat(1, num_words, 1)  # (maxlen, u_batch_size, hidden_size)
        start_pad = torch.zeros(num_utts, 1, rwembeds.shape[2]).cuda()
        d_rwembeds = torch.cat([start_pad, rwembeds[:, :-1, :]], dim=1)
        
        # Supply predicted_embedding and delayed word embeddings for teacher forcing.
        decoder_input = torch.cat([cembeds, d_rwembeds], dim=2)
        states = torch.zeros(num_utts, num_words, self.hidden_size * 3).cuda()
        p_switch = torch.zeros(num_utts, num_words).cuda()
        p_copy = torch.zeros(num_utts, num_words, self.vocab_size).cuda()
        
        # loop to calculate attention and get output
        for t in range(num_words):
            batch_size_t = sum([l > t for l in r_ulens])
            dec_input = decoder_input[:batch_size_t, t, :].unsqueeze(1)
            state, switch, copy, h_0, _ = self.attention_decode(h_0[:batch_size_t],\
                                                                     dec_input, c_uhidden_state[:batch_size_t],\
                                                             batch_size_t, c_utts[:batch_size_t], gembeds[:batch_size_t],\
                                                             kind_mat[:batch_size_t], g_pos[:batch_size_t], g_hid[:batch_size_t]) 
            states[:batch_size_t, t, :] = state
            p_switch[:batch_size_t, t] = switch
            p_copy[:batch_size_t, t, :] = copy
        
        p_gen = F.softmax(self.worder(states), dim=2)
        p_switch = p_switch.unsqueeze(2).repeat(1, 1, self.vocab_size) # repeat to multiply to all vocabs
        p_word = p_switch * (p_copy) + (1 - p_switch) * (p_gen)
        
        return p_word

    
    def get_greedy_samples(self, c_embeds, c_utts, c_uhidden_state, gembeds, kind_mat, g_pos, g_hid, data):
        assert c_embeds.dim()==2
        nsize = c_embeds.shape[0]
        num_steps = 35
        predids_mat = []
        inp = torch.zeros(nsize,1,self.hidden_size+self.word_size).to(c_embeds)
        inp[:,0,:self.hidden_size] = c_embeds
        output, p_switch, p_copy, hidden, g = self.attention_decode(c_embeds, inp, c_uhidden_state, nsize,\
                                                                 c_utts, gembeds, kind_mat, g_pos, g_hid)
#         s = sorted([(data.tokenizer.decode([i]), p_copy[0][i].item())for i in range(p_copy.shape[1]) if p_copy[0][i] != 0], key = lambda x: x[1], reverse = True)
#         print(s)
#         print(p_switch)
        p_gen = F.softmax(self.worder(output), dim=1).squeeze().detach()
        p_word = p_switch * (p_copy) + (1 - p_switch) * (p_gen)
        p_word = p_word.squeeze(0)
        prev_wid =  p_word.argmax()
        predids_mat.append(prev_wid.item())
        
        for i in range(1, num_steps):
            dec_input = torch.zeros(nsize,1,self.hidden_size+self.word_size).to(c_embeds)
            dec_input[:,0,:self.hidden_size] = c_embeds
            dec_input[:,0,self.hidden_size:] = self.word2embed(prev_wid)
            output, p_switch, p_copy, hidden, g = self.attention_decode(hidden, dec_input, c_uhidden_state,\
                                                              nsize, c_utts, gembeds, kind_mat, g_pos, g_hid)
#             s = sorted([(data.tokenizer.decode([i]), p_copy[0][i].item())for i in range(p_copy.shape[1]) if p_copy[0][i] != 0], key = lambda x: x[1], reverse = True)
#             print(s)
#             print(p_switch)
            p_gen = F.softmax(self.worder(output), dim=1).squeeze().detach()
            p_word = p_switch * (p_copy) + (1 - p_switch) * (p_gen)
            p_word = p_word.squeeze(0)
            prev_wid =  p_word.argmax()
            predids_mat.append(prev_wid.item())
#             print(p_word.max())
            if prev_wid.item() == 1:
                break
        return predids_mat  


    def forward_first(self, c_embeds, c_utts, c_uhidden_state, gembeds, kind_mat, g_pos, g_hid):
        assert c_embeds.dim()==2
        nsize = c_embeds.shape[0]
        input = torch.zeros(nsize,1,self.hidden_size+self.word_size).to(c_embeds)
        input[:,0,:self.hidden_size] = c_embeds
       
        output, p_switch, p_copy, hidden = self.attention_decode(c_embeds, input, c_uhidden_state, nsize,\
                                                             c_utts, gembeds, kind_mat, g_pos, g_hid)
        
        p_gen = self.worder(output).squeeze()
        p_word = p_switch * (p_gen) + (1 - p_switch) * (p_copy)
        p_word = p_word.squeeze(0)
        print(p_switch)
        worddis = F.softmax(p_word, dim=0).log()

        return hidden.unsqueeze(0), worddis
    
#     def forward_next(self, c_embeds, prev_hiddens, prev_wids):
#         assert c_embeds.dim()==2
#         nsize = prev_hiddens.shape[1]
#         input = torch.zeros(nsize,1,self.hidden_size+self.word_size).to(c_embeds)
#         input[:,0,:self.hidden_size] = c_embeds
#         input[:,0,self.hidden_size:] = self.word2embed(prev_wids)
#         output, new_hiddens = self.lstm(input, prev_hiddens)
#         worddis = self.worder(output).squeeze(1)
#         worddis = F.softmax(worddis, dim=1).log().detach()
#         return new_hiddens, worddis

    def forward_next(self, c_embeds, prev_hiddens, prev_wids, c_utts, c_uhidden_state, gembeds, kind_mat, g_pos, g_hid):
        assert c_embeds.dim()==2
        nsize = prev_hiddens.shape[1]
        dec_input = torch.zeros(nsize,1,self.hidden_size+self.word_size).to(c_embeds)
        dec_input[:,0,:self.hidden_size] = c_embeds
        dec_input[:,0,self.hidden_size:] = self.word2embed(prev_wids)
        
        output, p_switch, p_copy, new_hid = self.attention_decode(prev_hiddens.squeeze(0), dec_input, c_uhidden_state,\
                                                              nsize, c_utts, gembeds, kind_mat, g_pos, g_hid)
        print(p_switch)
        p_gen = self.worder(output).squeeze()
        p_switch = p_switch.unsqueeze(1)
        p_word = p_switch * (p_gen) + (1 - p_switch) * (p_copy)
        p_word = p_word.squeeze(0)
        worddis = F.softmax(p_word, dim=0).log()
        
        return new_hid.unsqueeze(0), worddis

    
    def step(self, contexts, prev_wids=None, prev_hidden=None):
        bsize = contexts.shape[0]
        input = torch.zeros(bsize,1,self.hidden_size+self.word_size).to(contexts)
        input[:,0,:self.hidden_size] = contexts
        if prev_wids is not None:
            input[:,0,self.hidden_size:] = self.word2embed(prev_wids)
        output, hidden = self.lstm(input, prev_hidden)
        worddis = self.worder(output).squeeze(1).detach()
        return worddis, hidden
