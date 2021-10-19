from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelWithLMHead, BertTokenizer, LongformerTokenizer, LongformerModel
import torch, json, random
import numpy as np

import gpt.Model as userModel
# import gpt_agent.Data as agentData
import gpt_agent.Model as agentModel
import gpt_query.Model as queryModel
import bert_siamese.Model as userSiameseModel
import bert_siamese_agent.Model as agentSiameseModel
import io
import argparse

parser = argparse.ArgumentParser(description='Train and evaluate an HRED')
parser.add_argument('--percentage', type=str, default='None', help='Percentage')
parser.add_argument('--user-p', type=float, default=0.75, help='Top p used for sampling for user model')
parser.add_argument('--agent-p', type=float, default=0.75, help='Top p used for sampling for agent model')


args = parser.parse_args()

percentage = args.percentage
path = '../models/'
save_path = '../'
db_path = '../createData/multiwoz21/db/'
user_top_p = args.user_p
agent_top_p = args.agent_p
ratio = 1
seed = 25


d_train= json.load(open(db_path + 'train_db.json'))
d_rest = json.load(open(db_path + 'restaurant_db.json'))
d_hotel = json.load(open(db_path + 'hotel_db.json'))
d_police = json.load(open(db_path + 'police_db.json'))
d_hosp = json.load(open(db_path + 'hospital_db.json'))
d_attr = json.load(open(db_path + 'attraction_db.json'))
d_taxi = [{
  "taxi_colors" : ["black","white","red","yellow","blue","grey"],
  "taxi_types":  ["toyota","skoda","bmw","honda","ford","audi","lexus","volvo","volkswagen","tesla"],
  "taxi_phone": ["^[0-9]{10}$"]
}]
entity_db_map = {'train':d_train, 'restaurant': d_rest, 'police': d_police, 'hospital': d_hosp, 'attraction': d_attr, 'taxi':d_taxi,'hotel':d_hotel}


query_key = {'train' : list(d_train[0].keys()),
             'restaurant' : list(d_rest[0].keys()),
             'hotel' : list(d_hotel[0].keys()),
             'police' : list(d_police[0].keys()),
             'hospital' : list(d_hosp[0].keys()),
             'attraction' : list(d_attr[0].keys()),
             'taxi' : ['taxi_colors', 'taxi_types', 'taxi_phone'],
            }

bert_model_name='bert-base-uncased'
longformer_model_name='allenai/longformer-base-4096'
user_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
agent_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# agent_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
query_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
longformer_tokenizer = LongformerTokenizer.from_pretrained(longformer_model_name)

def getStringKB(kb):
    if topic == 'police':
        return '[KB] Total = 1 [KB]'
    elif topic == 'taxi':
        return '[KB] Total = 1 [KB]'

    final_str =  "[KB] " + " Total = " + str(len(kb)) + ' [KB]'
#     for k in kb:
#         final_str +=  k + " : " + str(kb[k]) + " | "

def getUserModel():
    model_path = path + '/user/'+percentage+'p/checkpoint_hred_user__best_128_512_1e-05_1.pth.tar' 
    checkpoint = torch.load(model_path)
    load_args = checkpoint['args']
    model = userModel.GPT(load_args)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def getAgentModel():
    model_path = path + '/agent/'+percentage+'p/checkpoint_hred_user__best_128_512_1.pth.tar'
    checkpoint = torch.load(model_path)
    load_args = checkpoint['args']
    model = agentModel.GPT(load_args)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    return model

def getQueryModel():
    model_path = path + '/query/'+percentage+'p/checkpoint_hred_query__best.pth.tar'
    checkpoint = torch.load(model_path)
    load_args = checkpoint['args']
    model = queryModel.GPT(load_args)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def getUserSiameseModel():
    model_path = path + '/siamese/user/'+percentage+'p/checkpoint_hred_user__best_128_128_1.pth.tar' 
    checkpoint = torch.load(model_path)
    load_args = checkpoint['args']
    model = userSiameseModel.Siamese_margin(load_args)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def getAgentSiameseModel():
    model_path = path + '/siamese/agent/'+percentage+'p/checkpoint_hred_user__best_128_128_1.pth.tar' 
    checkpoint = torch.load(model_path)
    load_args = checkpoint['args']
    model = agentSiameseModel.Siamese_margin(load_args)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def getData():
    data_path = '/u/vineeku6/storage/gaurav/models/yz/moved/goal_oriented_learning_new/data/multiwiz/user/'+percentage+'p/goals_new.json'
    with open(data_path) as f:
        goals = json.load(f)

    data_path = '/u/vineeku6/storage/gaurav/models/yz/moved/goal_oriented_learning_new/data/multiwiz/agent/'+percentage+'p/final_state.json'
    with open(data_path) as f:
        state = json.load(f)

    data_path = '/u/vineeku6/storage/gaurav/models/yz/moved/goal_oriented_learning_new/data/multiwiz/agent/'+percentage+'p/train_input.json'
    with open(data_path) as f:
        context = json.load(f)


    return goals, state


importantKeys = {}
importantKeys['train'] = ['leaveAt', 'destination', 'departure', 'arriveBy', 'day', 'people']
importantKeys['hotel'] = ['name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type', 'stay', 'day', 'people']
importantKeys['restaurant'] = ['food', 'pricerange', 'name', 'area', 'people', 'day', 'time']
importantKeys['attraction'] = ['type', 'name', 'area']
importantKeys['taxi'] = ['leaveAt', 'destination', 'departure', 'arriveBy']


delexUserKeys = {}
delexUserKeys['train'] = ['leaveAt', 'destination', 'departure', 'arriveBy']
delexUserKeys['hotel'] = ['name']
delexUserKeys['restaurant'] = ['food', 'name', 'time']
delexUserKeys['attraction'] = ['type', 'name']
delexUserKeys['taxi'] = ['leaveAt', 'destination', 'departure', 'arriveBy']
delexUserKeys['hospital'] = ['department']

def formatQuery(query, context, state, prev_query):
    for d in ['train', 'restaurant', 'hotel', 'attraction', 'police', 'taxi', 'hospital']:
        if d in query.lower():
            topic = d
            
    if topic not in state:
        if '=' in prev_query:
            query = prev_query
    query = query.lower().replace('[q]', '').split('|')
    new_query = query[0].strip()
    if topic == 'police' or topic == 'hospital':
        return 'police'
    joined_context = " ".join(context[1::2])
    intermediate_query = {}

    for q in query[1:]:
        q = q.split('=')
        if len(q)>=2:
            key = q[0].strip()
            if key == 'leaveat':
                key = 'leaveAt'
            if key == 'arriveby':
                key = 'arriveBy'
            val = "".join(q[1:]).strip()

            if val == '':
                val = '*'

            for k in delexUserKeys[topic]:
                if k.lower() == key.lower():
                    if topic not in state:
                        val = '*'
                    if val != '*' and 'info' in state[topic] and k in state[topic]['info']:
                        if state[topic]['info'][k] in joined_context:
                            key = k
                            val = state[topic]['info'][k]
                            break
                    if val != '*' and 'semi' in state[topic] and k in state[topic]['semi']:
                        if state[topic]['semi'][k] in joined_context:
                            key = k
                            val = state[topic]['semi'][k]
                            break
                    if val != '*' and 'book' in state[topic] and k in state[topic]['book']:
                        if state[topic]['book'][k] in joined_context:
                            key = k
                            val = state[topic]['book'][k]
                            break
                    if val != '*' and 'fail_info' in state[topic] and k in state[topic]['fail_info']:
                        if state[topic]['fail_info'][k] in joined_context:
                            key = k
                            val = state[topic]['fail_info'][k]
                            break
                    if val != '*' and 'fail_book' in state[topic] and k in state[topic]['fail_book']:
                        if state[topic]['fail_book'][k] in joined_context:
                            key = k
                            val = state[topic]['fail_book'][k]
                            break
            
            if key.lower() ==  'leaveat' or key.lower() == 'arriveby':
                if topic == 'train' or topic == 'taxi':
                    if 'info' in state[topic]:
                        if key in state[topic]['info']:
                            if state[topic]['info'][key] in joined_context:
                                val = state[topic]['info'][key]
                            else:
                                v = state[topic]['info'][key].replace(':','')
                                if v in joined_context:
                                    val = state[topic]['info'][key]

            if topic in val:
                val = '*'

            intermediate_query[key] = val

    if 'departure' in intermediate_query and 'destination' in intermediate_query:
            if intermediate_query['departure'] == intermediate_query['destination']:
                if 'from' in joined_context or 'depart' in joined_context:
                    intermediate_query['destination'] = '*'
                else:
                    intermediate_query['departure'] = '*'

    for key in importantKeys[topic]:
        if key in intermediate_query:
            new_query += ' | ' + key + ' = ' + intermediate_query[key]
        else:
            new_query += ' | ' + key + ' = ' + '*'

    new_query = '[Q] ' + new_query + ' [Q]'

    return new_query

seed = 25
def set_seed():
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def getKB(query, utt_no, fail_book, failed):
    not_kb = ['people', 'time', 'stay']

    query = query.lower().replace('[q]', '')
    for d in ['train', 'restaurant', 'hotel', 'attraction', 'police', 'taxi', 'hospital']:
        if d in query:
             topic = d

    if topic != 'train':
        not_kb.append('day')


    db = entity_db_map[topic]
    final_query = {}
    fail_query = {}

    for q in query.split(' | ')[1:]:
        q = q.split(' = ')
        q[0] = q[0].strip()
        q[1] = q[1].strip()

        if q[1] != 'not mentioned' and q[1]!='dontcare' and q[1]!='none' and q[1]!='' and q[1]!='*':
            fail_query[q[0]] = q[1]

        for k in query_key[topic]:
            if q[0] == k and q[1] != 'not mentioned' and q[1]!='dontcare' and q[1]!='none' and q[1]!='' and q[1]!='*':
                final_query[k] = q[1]

    if topic in fail_book:
        for k in fail_book[topic]:
            if k in fail_query and fail_book[topic][k] == fail_query[k] and failed[topic] == False:
                return '[KB] Total = 0 [KB]', {}

    if 'name' in final_query and final_query['name'] != '*':
        return '[KB] Total = 1 [KB]', {}

    ret = []
    for row in db:
        match = True
        for k in final_query.keys():
#             print(k, final_query[k], row[k])
            if(k == "arriveBy"):
                try:
                    if int(final_query[k][:2]) > int(row[k][:2]):
                        match=True
                    elif int(final_query[k][:2]) == int(row[k][:2]) and int(final_query[k][3:]) >= int(row[k][3:]):
                        match=True
                    else:
                        match=False
                        break
                except:
                    match=True
            elif(k == "leaveAt"):
                try:
                    if int(row[k][:2]) > int(final_query[k][:2]):
                        match=True
                    elif int(row[k][:2]) == int(final_query[k][:2]) and int(row[k][3:]) >= int(final_query[k][3:]):
                        match=True
                    else:
                        match=False
                        break
                except:
                    match=True
            else:
                val = row[k]
                semi = final_query
                domain = topic
                val = val.strip()
                semi[k] = semi[k].strip()
                if domain == 'attraction' and k == 'type':
                    semi[k] = semi[k].replace(' ', '')
                    val = val.replace(' ', '')
                    if semi[k] == 'pool':
                        semi[k] = 'swimmingpool'
                    if val == 'pool':
                        val = 'swimmingpool'
                    if semi[k] == 'sports' or semi[k] == 'multiplesports':
                        semi[k] = 'mutliple sports'
                    if val == 'mutliplesports' or val == 'sports':
                        val = 'mutliple sports'
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


                if k != 'name' and (val!=semi[k]):
                    match=False
                    break

        if(match):
            ret.append(row)

    if len(ret) == 0:
        return  "[KB] " + " Total = 0 [KB]", {}
    else:
#         return '[KB] Total : ' + str(len(ret)) + ' ' + str(getStringKB(ret[0])), ret[0]
         return "[KB] " + " Total = " + str(len(ret)) + ' [KB]', ret[0]

BSKeys = {}
BSKeys['hotel'] = ['name', 'area', 'parking', 'pricerange', 'stars', 'internet']
BSKeys['restaurant'] = ['food', 'pricerange', 'area']
BSKeys['attraction'] = ['type', 'area']
BSKeys['train'] = ['leaveAt', 'destination', 'departure', 'arriveBy']

def adjustScore(query, prev_query, last_context, response, kb, turn_no, entire_context):
    for d in ['train', 'restaurant', 'hotel', 'attraction', 'police', 'taxi', 'hospital']:
        if d in query.lower():
            domain_key = d
    query = query.lower().replace('[q]', '').split('|')
    query = query[1:]
    belief_state = {}
    score = 0
    for q in query:
        q = q.split('=')
        if len(q)>=2:
            key = q[0].strip()
            belief_state[key] = "".join(q[1:]).strip()

    prev_query = prev_query.lower().replace('[q]', '').split('|')[1:]
    prev_belief_state = {}
    score = 0
    for q in prev_query:
        q = q.split('=')
        if len(q)>=2:
            key = q[0].strip()
            prev_belief_state[key] = "".join(q[1:]).strip()

    requestables = ['phone', 'number', 'address', 'postcode', ' code', 'reference', 'id']
    for r in requestables:
        if r == 'number':
            r = 'phone'
        if r == ' code':
            r = 'postcode'
        if r in last_context:
            if ' ['+domain_key+'_'+r+']' in response:
                score += 0.5
        if r not in last_context:
            if ' ['+domain_key+'_'+r+']' in response:
                score += -0.1

    enquiry = ['area', 'name', 'price', 'internet', 'fee', 'travel time', 'type']
    for e in enquiry:
        if e == 'travel time':
            if 'minute' in response:
                score += 0.3
        elif e == 'fee':
            if 'value_pricerange' in response:
                score +=0.2
        elif e == 'area':
             if 'value_area' in response:
                    score +=0.2
        elif e in last_context and e in response:
            score += 0.2

    if 'name' in belief_state and '[value_count] ['+domain_key+'_name]' in response:
        return -0.5

    bs_count = 0
    if domain_key in BSKeys:
        for b in BSKeys[domain_key]:
            if b in belief_state and b in prev_belief_state:
                if belief_state[b] != '*' and prev_belief_state[b] == '*':
                    if '['+domain_key+'_'+'name]' in response and 'book' not in last_context:
                        score += 1
                    if domain_key == 'train'and '['+domain_key+'_'+'id]' in response and 'book' not in last_context:
                        score += 1
            if b in belief_state and belief_state[b] != '*':
                bs_count += 1

    if 'name' in belief_state and bs_count == len(BSKeys[domain_key]) and '0' not in kb:
        if '['+domain_key+'_'+'name]' not in entire_context and '['+domain_key+'_'+'name]' in response:
            score += 1

    if domain_key == 'train' and bs_count == len(BSKeys[domain_key]) and '0' not in kb and 'book' not in last_context:
        if '['+domain_key+'_'+'id]' not in entire_context and '['+domain_key+'_'+'id]' in response:
            score += 0.4

    if 'name' in belief_state and '1 ' in kb:
        if '['+domain_key+'_'+'name]' not in entire_context and '['+domain_key+'_'+'name]' in response:
            score += 1

    # do not ask about slots if name is already provided
    if 'name' in belief_state and (belief_state['name'] != '*' or '1' in kb):
        if '['+domain_key+'_'+'name]' not in entire_context and '['+domain_key+'_'+'name]' in response:
            if 'value_count' in response:
                return -0.2
            return 0.4
        return 0

    if '0' in kb and domain_key != 'train':
        if ' no ' in response:
            score += 0.4
        elif ' not ' in response:
            score += 0.4
        elif 'unfortunate' in response:
            score += 0.4
        elif 'unavailable' in response:
            score += 0.4
        elif 'unsuccessful' in response:
            score += 0.4
        elif 'sorry' in response:
            score += 0.4
    elif domain_key != 'train':
        if ' no ' in response:
            score += -0.7
        elif ' not ' in response:
            score += -0.7
        elif 'unfortunate' in response:
            score += -0.7
        elif 'unavailable' in response:
            score += -0.7
        elif 'unsuccessful' in response:
            score += -0.7
        elif 'sorry' in response:
            score += -0.7

    Flag = True
    if turn_no > 1:
        Flag = False

    if domain_key == 'attraction' and turn_no>0:
        Flag = False

    if domain_key == 'taxi' and turn_no>1:
        if '[taxi_type]' not in entire_context and '[taxi_type]' in response:
            score += 0.2
        Flag = False

    if '_name] -s' in response:
        score -= 0.2


    if turn_no > 1:
        if domain_key == 'train' and '['+domain_key+'_'+'id]' not in entire_context and '['+domain_key+'_'+'id]' in response and 'book' not in last_context:
            score += 0.25

    if 'when' in response:
        if ' leave' in response:
            if 'leaveat' in belief_state and belief_state['leaveat'] != '*':
                score -= 0.25
            elif 'leaveat' in belief_state and Flag:
                print('q1')
                score += 0.2

        if ' arrive' in response:
            if 'arriveby' in belief_state and belief_state['arriveby'] != '*':
                score += -0.25
            elif 'arriveby' in belief_state and Flag:
                score += 0.2
                print('q2')

    if ('what' in response and 'what about' not in response) or 'do you' in response or 'is there' in response:
        if ' time' in response and (' leave' in response or ' depart' in response):
            if 'leaveat' in belief_state and belief_state['leaveat'] != '*':
                score += -0.25
            elif 'leaveat' in belief_state and Flag:
                score += 0.2
                print('q3')

        elif ' time' in response and ' arrive' in response:
            if 'arriveby' in belief_state and belief_state['arriveby'] != '*':
                score += -0.25
            elif 'arriveby' in belief_state and Flag:
                score += 0.2
                print('q4')

        elif ' destination' in response:
            if 'destination' in belief_state and belief_state['destination'] != '*':
                score += -0.3
            elif 'destination' in belief_state and Flag:
                score += 0.4
                print('q6')

        elif ' departure' in response:
            if 'departure' in belief_state and belief_state['departure'] != '*':
                score += -0.3
            elif 'departure' in belief_state and Flag:
                score += 0.2
                print('q7')

        elif ' day' in response:
            if 'day' in belief_state and belief_state['day'] != '*':
                score += -0.25
            elif 'day' in belief_state and Flag:
                score += 0.15
                if 'book' in last_context or 'reserv' in last_context:
                    score += 0.3
                print('q8')

        elif ' area' in response or ' part of town' in response:
            if 'area' in belief_state and belief_state['area'] != '*':
                score += -0.25
            elif 'area' in belief_state and Flag:
                score += 0.1
                print('q9')

        elif ' type of food' in response or ' kind of food' in response:
            if 'food' in belief_state and belief_state['food'] != '*':
                score += -0.25
            elif 'food' in belief_state:
                score += 0.3
                print('q10')

        elif ' price range' in response:
            if 'pricerange' in belief_state and belief_state['pricerange'] != '*':
                score += -0.25
            elif 'pricerange' in belief_state and Flag:
                score += 0.2
                print('q11')

    if 'where' in response:
        if ' depart' in response or ' leaving from' in response or ' departure' in response or 'travelling from' in response or 'to and from' in response:
            if 'departure' in belief_state and belief_state['departure'] != '*':
                score += -0.3
            elif 'departure' in belief_state and Flag:
                score += 0.2
                print('q12')

        elif ' destination' in response or ' going to' or ' travelling to' in response or 'to and from' in response:
            if 'destination' in belief_state and belief_state['destination'] != '*':
                score += -0.3
            elif 'destination' in belief_state and Flag:
                score += 0.4
                print('q13')

    if 'how many' in response:
        if ' people' in response or ' ticket' in response:
            if 'people' in belief_state and belief_state['people'] != '*':
                score += -0.25
            elif 'people' in belief_state and ('thank' not in response or 'bye' not in response):
                if 'book' in last_context or 'reserv' in last_context:
                    score += 0.5

    if 'which' in response:
        if ' day' in response:
            if 'day' in belief_state and belief_state['day'] != '*':
                score += -0.25
            elif 'day' in belief_state and Flag:
                score += 0.15
                print('q14')

        elif ' area' in response or ' part of town' in response:
            if 'area' in belief_state and belief_state['area'] != '*':
                score += -0.25
            elif 'area' in belief_state and Flag:
                score += 0.1
                print('q15')

        elif ' type of food' in response or ' kind of food' in response:
            if 'food' in belief_state and belief_state['food'] != '*':
                score += -0.25
            elif 'food' in belief_state and Flag:
                score += 0.3
                print('q16')

        elif ' price range' in response:
            if 'pricerange' in belief_state and belief_state['pricerange'] != '*':
                score += -0.25
            elif 'pricerange' in belief_state and Flag:
                score += 0.15
                print('q16')

    return score

def formatResponse(agent_response):
    agent_response = agent_response.replace(',', ' , ')
    agent_response = agent_response.replace('.', ' . ')
    agent_response = agent_response.replace('?', ' ? ')
    agent_response = agent_response.replace('!', ' ! ')
    agent_response = agent_response.replace('[', ' [')
    agent_response = agent_response.replace(']', '] ')
    agent_response = agent_response.replace('  ', ' ')

    agent_response = agent_response.strip()
    return agent_response




def formatUserResponse(user_response, state, user_failed):
    user_response = user_response.strip()
    for t in ['train', 'restaurant', 'hotel', 'attraction', 'taxi']:
        if t not in state:
            if t == 'train' and 'taxi' in state and '[train_' in user_response:
                user_response = user_response.replace('[train_', '[taxi_')
            if t == 'taxi' and 'train' in state and '[taxi_' in user_response:
                user_response = user_response.replace('[taxi_', '[train_')
            if t == 'hotel' and 'restaurant' in state and '[hotel_' in user_response:
                user_response = user_response.replace('[hotel_', '[restaurant_')
            if t == 'hotel' and 'attraction' in state and '[hotel_' in user_response:
                user_response = user_response.replace('[hotel_', '[attraction_')
            if t == 'restaurant' and 'hotel' in state and '[restaurant_' in user_response:
                user_response = user_response.replace('[restaurant_', '[hotel_')
            if t == 'restaurant' and 'attraction' in state and '[restaurant_' in user_response:
                user_response = user_response.replace('[restaurant_', '[attraction_')
            if t == 'attraction' and 'hotel' in state and '[attraction_' in user_response:
                user_response = user_response.replace('[attraction_', '[hotel_')
            if t == 'attraction' and 'restaurant' in state and '[attraction_' in user_response:
                user_response = user_response.replace('[attraction_', '[restaurant_')

    for topic in ['train', 'restaurant', 'hotel', 'attraction', 'taxi']:
        for k in delexUserKeys[topic]:
            if topic+'_'+k.lower() in user_response and topic in list(state.keys()) and 'fail_info' in state[topic] and k in state[topic]['fail_info'] and user_failed[topic] == True:
                user_response = user_response.replace('['+topic+'_'+k.lower()+']', state[topic]['fail_info'][k])
                user_failed[topic] = False
            elif topic+'_'+k.lower() in user_response and topic in list(state.keys()) and 'info' in state[topic] and k in state[topic]['info']:
                user_response = user_response.replace('['+topic+'_'+k.lower()+']', state[topic]['info'][k])
            elif topic+'_'+k.lower() in user_response and topic in list(state.keys()) and 'semi' in state[topic] and k in state[topic]['semi']:
                user_response = user_response.replace('['+topic+'_'+k.lower()+']', state[topic]['semi'][k])
            elif topic+'_'+k.lower() in user_response and topic in list(state.keys()) and 'fail_book' in state[topic] and k in state[topic]['fail_book'] and user_failed[topic] == True:
                user_response = user_response.replace('['+topic+'_'+k.lower()+']', state[topic]['fail_book'][k])
                user_failed[topic] = False
            elif topic+'_'+k.lower() in user_response and topic in list(state.keys()) and 'book' in state[topic] and k in state[topic]['book']:
                user_response = user_response.replace('['+topic+'_'+k.lower()+']', state[topic]['book'][k])
            elif topic+'_'+k.lower() in user_response:
                user_response = user_response.replace('['+topic+'_'+k.lower()+']', k.lower())



    if len(user_response) > 0 and user_response[-1] != '.' and user_response[-1] != '?' and user_response[-1] != '!':
        user_response = user_response + '.'
    return user_response, user_failed

def adjustUserScore(goal, response, turn_no, entire_context):
    score = 0
    requestables = ['phone', 'number', 'address', 'postcode', 'reference', 'id']
    enquiry = ['area', 'name', 'price', 'internet', 'fee', 'travel time']
    bs_count = 0
    for r in requestables:
        if r in response:
            bs_count += 1
        if r in goal and r not in entire_context and r in response:
            score = 0.2
        if ('thank' in response or 'bye' in response) and r not in entire_context:
            score = -0.2
        if r in goal and r in entire_context and r in response:
            score = -0.1
    for e in enquiry:
        if e in response:
            bs_count += 1
        if ('thank' in response or 'bye' in response) and e not in entire_context:
            score = -0.2
    if bs_count > 2:
        score = -0.2

    return score

user_gpt = getUserModel()
user_model = user_gpt.model
user_model.eval()
user_model.cuda()

agent_gpt = getAgentModel()
agent_model = agent_gpt.model
agent_model.eval()
agent_model.cuda()

query_gpt = getQueryModel()
query_model = query_gpt.model
query_model.eval()
query_model.cuda()

user_siamese_model = getUserSiameseModel()
user_siamese_model.eval()
user_siamese_model.cuda()

agent_siamese_model = getAgentSiameseModel()
agent_siamese_model.eval()
agent_siamese_model.cuda()

goals, state = getData()

set_seed()
final_contexts = []
final_queries = []
final_kbs = []

for g in range(len(goals)):
#     print(state[domain_key][0])
    fail_book = {}
    failed = {}
    user_failed = {}
    for key in state[g]:
        if  'fail_book' in state[g][key]:
            fail_book[key] = state[g][key]['fail_book']
            failed[key] = False
            user_failed[key] = True
        elif 'fail_info' in state[g][key]:
            fail_book[key] = state[g][key]['fail_info']
            failed[key] = False
            user_failed[key] = True
            print(fail_book)
        else:
            fail_book[key] = {}
            failed[key] = False
            user_failed[key] = True
    
    flag = 0
    context = ['[st@rt]']
    queries = []
    kbs = []
    resp_no = 10
    goal = goals[g]
    police = False
    start_index = 0
    prev_query = '[Q] empty [Q]'
    try:
        for i in range(9):
            new_user_input = user_tokenizer.encode(goal + ' GOAL ' + " ".join(context), return_tensors='pt')
            if len(new_user_input)>1022:
                flag = 1
        #     response = user_model.generate(new_user_input.cuda(), max_length=500, pad_token_id=user_tokenizer.eos_token_id, num_beams=20, num_return_sequences=resp_no, early_stopping=True)
            response = user_model.generate(new_user_input[:1022].cuda(), max_length=1000, pad_token_id=user_tokenizer.eos_token_id, do_sample=True, top_k=resp_no, top_p=user_top_p, num_return_sequences=resp_no, early_stopping=True)

            max_score = 0
            user_response = ''
            for j in range(resp_no):
                response_j = user_tokenizer.decode(response[j][new_user_input.shape[1]:]).replace('<|endoftext|>', '')
                response_j = response_j.replace('[User]', '')

                if response_j.replace(' ', '') == '' or response_j.replace(' ', '') == '.' or response_j.replace(' ', '') == ',':
                    pass
                elif len(response_j.replace('?', '.').split('.')) > 4:
                    pass
                else:
                    response_j = response_j.strip()
                    if response_j[0] == ',' or response_j[0] == '!':
                        response_j = response_j[1:]
                    response_j = '[User] ' + response_j.strip()

                    input_context = " ".join(context)
                    encode = longformer_tokenizer.batch_encode_plus([[input_context, response_j]])
                    input_ids = torch.tensor(encode.input_ids[:1022]).cuda()
                    attention_mask = torch.tensor(encode.attention_mask[:1022]).cuda()

                    score = user_siamese_model(input_ids, attention_mask)
                    score += adjustUserScore(goal, response_j, i, " ".join(context))
                    if response_j == '':
                        score = -1
                    if score > max_score and len(response_j)<1023:
                        user_response = response_j
                        max_score = score

            if user_response == '':
                flag = 1

            user_final_response, user_failed = formatUserResponse(user_response, state[g], user_failed)
            context.append(user_final_response)

            if user_response.strip().lower().find('e*d') != -1 :
                break

#             query_input = query_tokenizer.encode(" ".join(context)[:1022], return_tensors='pt')
            ct = (' '.join(context[-1:]) + ' ' + prev_query)[:1000] + ' | ' 
            query_input = query_tokenizer.encode(ct, return_tensors='pt')
            query_response = query_model.generate(query_input.cuda(), max_length=1022, pad_token_id=query_tokenizer.eos_token_id, num_beams=5, num_return_sequences=5, early_stopping=True)

            query = agent_tokenizer.decode(query_response[0][query_input.shape[1]:]).replace('<|endoftext|>', '')
#             print('1 ', query, '\n')
            temp_query = query
            query = formatQuery(query, context, state[g], prev_query)
            queries.append(query)
            prev_query = query
            if query == 'police':
#                 i=10
#                 continue
                police = True
                break

            kb, kb_dict = getKB(query, i, fail_book, failed)
            kbs.append(kb)
            new_agent_input = agent_tokenizer.encode(query + " " + kb + " " + " ".join(context), return_tensors='pt')
            if len(new_agent_input)>1022:
                flag = 1
            response = agent_model.generate(new_agent_input[:1022].cuda(), max_length=1022, pad_token_id=agent_tokenizer.eos_token_id, do_sample=True, top_k=resp_no, top_p=agent_top_p, num_return_sequences=resp_no, early_stopping=True)

            max_score = 0
            agent_response = ''
            if flag == 0:
                for j in range(len(response)):
                    response_j = agent_tokenizer.decode(response[j][new_agent_input.shape[1]:]).replace('<|endoftext|>', '')
                    response_j = response_j.replace('[Agent]', '')
                    if response_j.replace(' ', '') == '':
                        pass
                    elif len(response_j.replace('?', '.').split('.')) > 3:
                        pass
                    else:
                        response_j = '[Agent]' + response_j
                        encode = longformer_tokenizer.batch_encode_plus([[query + ' ' + kb + ' ' + " ".join(context), response_j]])
                        input_ids = torch.tensor(encode.input_ids[:1022]).cuda()
                        attention_mask = torch.tensor(encode.attention_mask[:1022]).cuda()

                        score = agent_siamese_model(input_ids, attention_mask)
    #                     print(score.item())
                        score = score + adjustScore(query, prev_query, context[-1], response_j, kb, i, " ".join(context[start_index:]), )
                        if response_j == '':
                            score = -1
                        if score > max_score and len(response_j)<1023:
                            agent_response = response_j
                            max_score = score

    #                     print('response : ', j, " : ", response_j, " score ", score.item(), '\n')

            response = agent_model.generate(new_agent_input.cuda(), max_length=1000, pad_token_id=agent_tokenizer.eos_token_id, num_beams=1, num_return_sequences=1, early_stopping=True)
            for j in range(len(response)):
                response_j = agent_tokenizer.decode(response[j][new_agent_input.shape[1]:]).replace('<|endoftext|>', '')

                if response_j.replace(' ', '') == '':
                    pass
                elif len(response_j.replace('?', '.').split('.')) > 3:
                    pass
                else:
                    encode = longformer_tokenizer.batch_encode_plus([[query + ' ' + kb + ' ' + " ".join(context), response_j]])
                    input_ids = torch.tensor(encode.input_ids).cuda()
                    attention_mask = torch.tensor(encode.attention_mask).cuda()

                    score = agent_siamese_model(input_ids, attention_mask)
                    score = score + adjustScore(query, prev_query, context[-1], response_j, kb, i, " ".join(context[start_index:]), )

                    if score > max_score:
                        agent_response = response_j
                        max_score = score
    #                 print('response : ', j, " : ", response_j, " score ", score.item(), '\n')

            
            if agent_response == '':
                flag = 1
            agent_response = formatResponse(agent_response)
            if flag == 0:
                context.append(agent_response)
            if any(word in agent_response for word in [' no ', ' not ', 'unavailable', 'unfortunate', 'sorry', 'unable', 'unsuccessful']):
                    for d in ['train', 'restaurant', 'hotel', 'attraction', 'police', 'taxi', 'hospital']:
                        if d in query.lower():
                            temp_topic = d 
                    failed[temp_topic] = True
                    start_index = i

            if flag == 1 and police == False:
                if g<20:
                    print("GOAL : ", g, ' ', goal, flush=True)
                elif g%100 == 0:
                    print("GOAL : ", g, flush=True)
                final_ctx = []
                for i,ctx in enumerate(context):
                    if ctx.find('e*d') == -1:
                        final_ctx.append(ctx)
                        if g < 20:
                            print(ctx, flush=True)

                final_contexts.append(final_ctx)
                final_queries.append(queries)
                final_kbs.append(kbs)
                if g < 20:
                    print('*'*100, flush=True)
                break

        if flag == 0 and police == False:
            if g < 20:
                print("GOAL : ", g, ' ', goal, flush=True)
            elif g%100==0:
                print("GOAL : ", g, flush=True)
            final_ctx = []
            for i,ctx in enumerate(context):
                if ctx.find('e*d') == -1:
                    final_ctx.append(ctx)
                    if g < 20:
                        print(ctx, flush=True)

            final_contexts.append(final_ctx)
            final_queries.append(queries)
            final_kbs.append(kbs)
            if g < 20:
                print('*'*100, flush=True)
    except:
        continue

json.dump(final_contexts, open(save_path + '/artificial_data/data_all_'+percentage+'p.json','w'))
json.dump(final_kbs, open(save_path +'/artificial_data/kb_all_'+percentage+'p.json','w'))
json.dump(final_queries, open(save_path + '/artificial_data/query_all_'+percentage+'p.json','w'))



