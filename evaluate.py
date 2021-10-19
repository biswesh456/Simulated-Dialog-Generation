import math
import json
import sqlite3
import os
import random
import logging
import argparse
from collections import Counter
from nltk.util import ngrams
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--file', type=str, default=None, help='file of prediction')
args = parser.parse_args()


class BaseEvaluator(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp):
        raise NotImplementedError

    def get_report(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            # if type(hyps[0]) is list:
            #    hyps = [hyp.split() for hyp in hyps[0]]
            # else:
            #    hyps = [hyp.split() for hyp in hyps]

            # refs = [ref.split() for ref in refs]
            hyps = [hyps]
            # Shawn's evaluation
            # refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            # hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']

            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        print(bp, s)
        return bleu


class MultiWozEvaluator(BaseEvaluator):
    def __init__(self, data_name):
        self.data_name = data_name
        self.delex_dialogues = json.load(open('createData/multiwoz21/delex.json'))
        self.db = {}
        self.db['train'] = json.load(open('createData/multiwoz21/db/train_db.json'))
        self.db['restaurant'] = json.load(open('createData/multiwoz21/db/restaurant_db.json'))
        self.db['hotel'] = json.load(open('createData/multiwoz21/db/hotel_db.json'))
        self.db['attraction'] = json.load(open('createData/multiwoz21/db/attraction_db.json'))
        self.db['taxi'] = [{
                              "taxi_colors" : ["black","white","red","yellow","blue","grey"],
                              "taxi_types":  ["toyota","skoda","bmw","honda","ford","audi",\
                                              "lexus","volvo","volkswagen","tesla"],
                              "taxi_phone": ["^[0-9]{10}$"]
                            }]

    def _parseGoal(self, goal, d, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
        if 'info' in d['goal'][domain]:
        # if d['goal'][domain].has_key('info'):
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in d['goal'][domain]:
                # if d['goal'][domain].has_key('reqt'):
                    if 'trainID' in d['goal'][domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in d['goal'][domain]:
                # if d['goal'][domain].has_key('reqt'):
                    for s in d['goal'][domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append("reference")

            goal[domain]["informable"] = d['goal'][domain]['info']
            if 'book' in d['goal'][domain]:
            # if d['goal'][domain].has_key('book'):
                goal[domain]["booking"] = d['goal'][domain]['book']

        return goal

    def _formatBeliefState(self, belief_state, domain):
        '''
        convert the belief state from string to dictionary form
        '''
        belief_state = belief_state.replace('[Q]', '').split('|')[1:]
        belief_state_new = {}
        important_keys = {'train':['leaveAt', 'destination', 'departure', 'arriveBy', 'day'],
                          'restaurant':['food', 'pricerange', 'name', 'area'],
                          'hotel':['name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type'],
                          'attraction':['type', 'name', 'area']}
        score = 0
        for q in belief_state:
            q = q.split('=')
            if len(q)>=2:
                key = q[0].strip()
                val = "".join(q[1:]).strip()
                if key in important_keys[domain] and val != '*' and key != 'name':
                    belief_state_new[key] = val

        return belief_state_new

    def getVenues(self, domain, semi, real_belief=True):
        db = self.db[domain]
        ret=[]
        for row in db:
    #         print(row)
            match = True
            for k in semi.keys():
                # normalise
                if k not in row:
                    print(k, domain)
                else:    
                    val = row[k].lower().strip()
                    semi[k] = semi[k].lower().strip()
                    if domain == 'attraction' and k == 'type':
                        semi[k] = semi[k].replace(' ', '')
                        val = val.replace(' ', '')
                        if semi[k] == 'pool':
                            semi[k] = 'swimmingpool'
                        if val == 'pool':
                            val = 'swimmingpool'
                    if k == 'parking' or k == 'internet':
                        semi[k] = semi[k].replace('free', 'yes')
                        val = val.replace('free', 'yes')
                    if k == 'food':
                        semi[k] = semi[k].replace('south indian', 'indian')
                        val = val.replace('south indian', 'indian')
                    if k == 'name':
                        val = val.replace('the', '')
                        semi[k] = semi[k].replace('the', '')
                        val = val.replace("b & b", "bed and breakfast")
                        semi[k] = semi[k].replace("b & b", "bed and breakfast")
                        val = val.replace("restaurant", "")
                        semi[k] = semi[k].replace("restaurant", "")
                        if "hotel" in val and 'gonville' not in val:
                            val = val.replace(" hotel", "")
                        if "hotel" in semi[k] and 'gonville' not in semi[k]:
                                semi[k] = semi[k].replace("hotel", "")
                        val = val.strip()
                        semi[k] = semi[k].strip()
                    
                    if(semi[k]!='not mentioned' and semi[k]!='dontcare' and semi[k]!='none' and semi[k]!=''):
                        if val!=semi[k] and k != 'leaveAt' and k != 'arriveBy':
                            match=False
                            break
                        elif val!=semi[k] and k == 'arriveBy':
                            sem_min = semi[k].split(':')
                            try:
                                sem_min = int(sem_min[0])*60+int(sem_min[1])
                                val_min = val.split(':')
                                val_min = int(val_min[0])*60+int(val_min[1])
                                if sem_min >= val_min:
                                    match=True
                                else:
                                    match=False
                                    break
                            except Exception:
                                match = False
                        elif val!=semi[k] and k == 'leaveAt':
                            try:
                                sem_min = semi[k].split(':')
                                sem_min = int(sem_min[0])*60+int(sem_min[1])
                                val_min = val.split(':')
                                val_min = int(val_min[0])*60+int(val_min[1])
                                if sem_min >= val_min:
                                    match=True
                                else:
                                    match=False
                                    break
                            except Exception:
                                match = False

            if(match):
                if domain in ['restaurant', 'hotel', 'attraction']:
                    ret.append(row['name'])
                elif domain == 'train':
                    ret.append(row['trainID'])

        return ret


    def _evaluateGeneratedDialogue(self, utt, belief_state, goal, realDialogue, real_requestables, soft_acc=False):
        """Evaluates the dialogue created by the model.
        First we load the user goal of the dialogue, then for each turn
        generated by the system we look for key-words.
        For the Inform rate we look whether the entity was proposed.
        For the Success rate we look for requestables slots"""
        # for computing corpus success
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)
        print('*'*30)
        for t, sent_t in enumerate(utt):
            for domain in goal.keys():
                # for computing success
                if '[' + domain + '_name]' in sent_t or domain + '_id' in sent_t:
#                 if domain in sent_t:
#                 if True:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        belief_state_t = self._formatBeliefState(belief_state[t], domain)

#                         print(t)
                        print('Belief : ', domain, belief_state_t)
                        print('Sent : ', sent_t)
                        venues = self.getVenues(domain, belief_state_t)
#                         print('G : ', domain, goal[domain]['informable'])
                        # if len(venues) == 0:
#                             print('V ', domain, belief_state[t], '\n', belief_state_t)

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and len(venues) != 0:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if domain + '_reference' in sent_t:
                            if 'restaurant_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'hotel_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'train_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        if domain + '_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            # if realDialogue['goal'][domain].has_key('info'):
            if 'info' in realDialogue['goal'][domain]:
                # if realDialogue['goal'][domain]['info'].has_key('name'):
                if 'name' in realDialogue['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            # the original method
            # if domain == 'train':
            #     if not venue_offered[domain]:
            #         # if realDialogue['goal'][domain].has_key('reqt') and 'id' not in realDialogue['goal'][domain]['reqt']:
            #         if 'reqt' in realDialogue['goal'][domain] and 'id' not in realDialogue['goal'][domain]['reqt']:
            #             venue_offered[domain] = '[' + domain + '_name]'

            # Wrong one in HDSA
            # if domain == 'train':
            #     if not venue_offered[domain]:
            #         if goal[domain]['requestable'] and 'id' not in goal[domain]['requestable']:
            #             venue_offered[domain] = '[' + domain + '_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and ( not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[' + domain + '_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        not_match = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.getVenues(domain, goal[domain]['informable'], real_belief=True)
                # print('SS', goal_venues)
                # print('RR', venue_offered[domain])
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                   
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1
                else:
                    print("K ", domain, venue_offered[domain])
                    print('H ', belief_state)
                    try:
                        print(utt)
                    except:
                        print('flag')
                    print('B ', goal[domain]['informable'], '\n','\n')
                    # print(utt)
                    
            else:
                if domain + '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1


            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        # rint requests, 'DIFF', requests_real, 'SUCC', success
        return success, match, stats


    def _evaluateRealDialogue(self, dialog, filename):
        """Evaluation of the real dialogue.
        First we loads the user goal and then go through the dialogue history.
        Similar to evaluateGeneratedDialogue above."""
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # get the list of domains in the goal
        domains_in_goal = []
        goal = {}
        for domain in domains:
            if dialog['goal'][domain]:
                goal = self._parseGoal(goal, dialog, domain)
                domains_in_goal.append(domain)

        # compute corpus success
        real_requestables = {}
        provided_requestables = {}
        venue_offered = {}
        for domain in goal.keys():
            provided_requestables[domain] = []
            venue_offered[domain] = []
            real_requestables[domain] = goal[domain]['requestable']

        return goal, real_requestables

    def evaluateModel(self, dialogues, real_dialogues, mode='valid'):
        """Gathers statistics for the whole sets."""
        delex_dialogues = self.delex_dialogues
        successes, matches = 0, 0
        total = 0
        corpus = []
        model_corpus = []
#         bscorer = BLEUScorer()
       
        for filename, dial in dialogues.items():
            print(filename)
            utt = dial[0]
            bs = dial[1]
            real_dialogue = dial[2]
            data = delex_dialogues[filename]
            goal, requestables = self._evaluateRealDialogue(data, filename)
            success, match, stats = self._evaluateGeneratedDialogue(utt, bs, goal, data, requestables,
                                                                    soft_acc=mode =='soft')

            successes += success
            matches += match
            total += 1
        
            model_turns, corpus_turns = [], []
            for idx, turn in enumerate(real_dialogue):
                corpus.append([turn.lower().strip().split(" ")])
            for turn in utt:
                model_corpus.append(turn.lower().strip().split(" "))

#             if len(model_turns) == len(corpus_turns):
#                 corpus.extend(corpus_turns)
#                 model_corpus.extend(model_turns)
#             else:
#                 raise('Wrong amount of turns')

        bleu_score = corpus_bleu(corpus, model_corpus, smoothing_function=SmoothingFunction().method1)*100

        report = ""
        report += '{} Corpus Matches : {:2.2f}%'.format(mode, (matches / float(total) * 100)) + "\n"
        report += '{} Corpus Success : {:2.2f}%'.format(mode, (successes / float(total) * 100)) + "\n"
        report += '{} Corpus BLEU : {:2.2f}%'.format(mode, bleu_score) + "\n"
        report += 'Total number of dialogues: %s ' % total

        print(report)

        return report, successes/float(total), matches/float(total)


if __name__ == '__main__':
    mode = "test"
    evaluator = MultiWozEvaluator(mode)
    
    with open("/NewVolume/biswesh/2.0_gol/predictions/"+args.file, "r") as f:
        generated_data = json.load(f)

#     with open("predictions/"+args.file, "r") as f:
#         generated_data = json.load(f)
    
    evaluator.evaluateModel(generated_data, True, mode=mode)
