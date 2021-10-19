import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils


class BeamSearch():

    def __init__(self, decoder, eot):
        self.decoder = decoder
        self.eot = eot
        
    def first_word(self, context, beam_size, c_utts, c_uhiddenembs, g_embeds, kind_mat, g_pos, g_hid):
        hidden, worddis = self.decoder.forward_first(context, c_utts, c_uhiddenembs, g_embeds, kind_mat, g_pos, g_hid)
        worddis = worddis.squeeze()
        inds = torch.argsort(worddis, descending=True)
        inds = inds.squeeze(0)[:beam_size]
        probs = worddis[inds]
        return inds, probs, hidden.detach()


    def next_word(self, context, prev_wids, prev_lprobs, prev_hiddens, prev_sents, beam_size,\
                  c_utts, c_uhiddenembs, g_embeds, kind_mat, g_pos, g_hid):
        
        new_hiddens, worddis = self.decoder.forward_next(context, prev_hiddens, prev_wids, c_utts,\
                                                         c_uhiddenembs, g_embeds, kind_mat, g_pos, g_hid)
        vocab_size = worddis.shape[1]
        new_lprobs = worddis + prev_lprobs.unsqueeze(1)
        probs_flat = new_lprobs.view(beam_size*vocab_size)
        _, inds_new = torch.sort(probs_flat, descending=True)
        inds_new = inds_new[:beam_size]
        rows = (inds_new/vocab_size).long()
        new_wids = (inds_new%vocab_size).long()
        new_hiddens = new_hiddens[:,rows].detach()
        new_lprobs = probs_flat[inds_new]
        new_sents = [prev_sents[x] + [y.item()] for x,y in zip(rows, new_wids)]
        return new_wids, new_lprobs, new_hiddens, new_sents

    def get_samples(self, c_embeds, c_utts, c_uhiddenembs, g_embeds, kind_mat, g_pos, g_hid, beam_size=20):
        assert c_embeds.dim()==2
        inds, probs, hiddens = self.first_word(c_embeds, beam_size, c_utts, c_uhiddenembs,\
                                               g_embeds, kind_mat, g_pos, g_hid)
        sents = [[x.item()] for x in inds]
        
        hiddens = hiddens.repeat(1,beam_size, 1)
        c_embeds = c_embeds.repeat(beam_size, 1)
        c_utts = c_utts.repeat(beam_size, 1, 1)
        c_uhiddenembs = c_uhiddenembs.repeat(beam_size, 1, 1, 1)
        g_embeds = g_embeds.repeat(beam_size, 1, 1)
        kind_mat = kind_mat.repeat(beam_size, 1)
        g_pos = g_pos.repeat(beam_size, 1)
        g_hid = g_hid.repeat(beam_size, 1)
        
        finished_sents = []
        finished_probs = []
        for iter in range(20):
#             finished = (inds==self.eot).nonzero()
            finished = (inds==torch.tensor(self.eot).cuda()).nonzero()
            finished_sents += [sents[x] for x in finished]
            finished_probs += [probs[x] for x in finished]
            probs[finished] = -100
            inds, probs, hiddens, sents = self.next_word(c_embeds, inds, probs, hiddens, sents, beam_size,\
                                                         c_utts, c_uhiddenembs, g_embeds, kind_mat, g_pos, g_hid)
        

        finished_sents += sents
        finished_probs += [x for x in probs]
        inds = np.argsort(finished_probs)[::-1]
        finished_sents = [finished_sents[x] for x in inds if finished_probs[x]>-100]
        finished_probs = [finished_probs[x] for x in inds if finished_probs[x]>-100]
        return finished_sents, finished_probs

