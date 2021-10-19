import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class GreedySearch():
    
    def __init__(self, decoder, eot):
        self.decoder = decoder
        self.eot = eot
        
        
    def first_word(self, context, beam_size=5):
        hidden, worddis = self.decoder.forward_first(context)
        wdis = worddis.data.cpu().numpy()
        inds = np.argsort(wdis)[::-1].copy()
        inds = inds[:beam_size]
        probs = wdis[inds]
        new_wids = torch.from_numpy(inds).to(context).long()
        new_probs = torch.from_numpy(probs).to(context)
        return new_wids, new_probs, hidden.detach()
    

    def next_word(self, context, prev_wids, prev_lprobs, prev_hiddens, prev_sents, beam_size=5):
        new_hiddens, worddis = self.decoder.forward_next(context, prev_hiddens, prev_wids)
        new_lprobs = worddis + prev_lprobs.unsqueeze(1)
        probs_flat = new_lprobs.view(-1).data.cpu().numpy()
        inds_new = np.argsort(probs_flat)[::-1][:beam_size]
        rows = (inds_new/args.vocab_size).astype("int")
        new_wids = (inds_new%args.vocab_size).astype("int")

        new_hiddens = new_hiddens[:,rows].detach()
        new_lprobs = probs_flat[inds_new]
        new_sents = [prev_sents[x] + [y] for x,y in zip(rows, new_wids)]

        new_wids = torch.from_numpy(new_wids).to(worddis).long()
        new_probs = torch.from_numpy(new_lprobs).to(worddis)

        return new_wids, new_probs, new_hiddens, new_sents

    def next_word_greedy(self, context, prev_wids, prev_hiddens, prev_sents):
        hiddens, worddis = self.decoder.forward_next(context, prev_hiddens, prev_wids)
        probs, inds = worddis.max(1)
        sents = [x+[y.item()] for x,y in zip(prev_sents, inds)]   
        return inds, probs, hiddens, sents
    
    def get_samples(self, context, beam_size=20):
        assert context.dim()==2
        inds, probs, hiddens = self.first_word(context, beam_size=beam_size)
        sents = [[x.item()] for x in inds]
        hiddens = hiddens.repeat(1,beam_size,1)
        #inds, probs, hiddens, sents = beam_search.next_word(c_embeds[0].unsqueeze(0), inds, probs, hiddens, sents, beam_size=5*beam_size)

        finished_sents = []
        for i in range(50):
            eot = self.eot
            finished = (inds==eot).nonzero()[:,0]
            if len(finished)>0:
                finished_sents += [sents[x] for x in finished]
                #finished_probs += [probs[x] for x in finished]
            unfinished = (inds!=eot).nonzero()[:,0]
            if len(unfinished)==0:
                break
            prev_hiddens = torch.stack([hiddens[0,x] for x in unfinished], dim=0).unsqueeze(0)
            prev_wids = torch.stack([inds[x] for x in unfinished])#.squeeze()
            prev_sents = [sents[x] for x in unfinished]
            inds, probs, hiddens, sents = self.next_word_greedy(context, prev_wids, prev_hiddens, prev_sents)
        
        return finished_sents