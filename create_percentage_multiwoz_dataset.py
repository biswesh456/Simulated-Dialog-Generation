import torch
import json
import re
import random
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--percentage', type=str, default='None', help='Percentage')

args = parser.parse_args()

percentage = args.percentage


with open("createData/multiwoz21/delex_query_results.json") as f:
    data = json.load(f)

with open("createData/multiwoz21/valListFile.json", 'r') as f:
    valList = f.read().split('\n')[:-1]

with open("createData/multiwoz21/testListFile.json", 'r') as f:
    testList = f.read().split('\n')[:-1]

def get_goal(message):
#     message = goals['message']
    if isinstance(message, list):  
        message = ". ".join(message)
    message = " . ".join(message.split(". "))

    span_start = "class='emphasis'>"
    span_end = "</span>"
    span_start_len = len(span_start)
    span_end_len = len(span_end)
    
    words = message.split()
    msg = ""
    keys = []
    key_words = []
    i = 0
    j = 0
    while i < len(words):
        if words[i] == '<span':
            i += 1
            continue
        elif words[i].startswith(span_start):
            if span_end in words[i]:
                sind = len(msg) + 6
                end_idx = words[i].find(span_end)
                msg += '[key] ' + words[i][span_start_len:end_idx] + " [key] "
                keys.extend([sind, len(msg)-7])
                key_words.append(msg[sind:len(msg)-1])
                i += 1
            else:
                sind = len(msg) + 6
                msg += '[key] ' + words[i][span_start_len:] + " "
#                 msg.append(words[i][span_start_len:])
                keys.extend([sind, len(msg)-1])
                wk = [msg[sind:len(msg)-1]]
                i += 1
                while span_end not in words[i]:
                    sind = len(msg)
                    msg += words[i] + " "
                    keys.extend([sind, len(msg)-1])
                    wk.append(msg[sind:len(msg)-1])
                    i += 1
                sind = len(msg)  
                end_idx = words[i].find(span_end)
                msg += words[i][:end_idx] + " "
                keys.extend([sind, len(msg)-1])
                wk.append(msg[sind:len(msg)-1])
                key_words.append(" ".join(wk))
                msg += "[key] "
                i += 1
        else:
            sind = len(msg)
            msg += words[i] + " "
            i += 1   
    
    return msg, keys, key_words                             

def getMetaData(data):
    if isinstance(data, dict): 
        md = []
        for key in data:
            md.extend(getMetaData(data[key]))
        
        return md
    
    elif isinstance(data, list):
        md = []
        for d in data:
            md.extend(getMetaData(d))
        
        return md
            
    else:
        if data != "":
            return data.split()
        else:
            return []

def getTopicsFilled(metadata):
    topic_lst = []
    for m in metadata:
        if len(getMetaData(metadata[m])) != 0:
            topic_lst.append(m)
    
    return topic_lst

delexKeys = {}
delexKeys['train'] = ['leaveAt', 'destination', 'departure', 'arriveBy']
delexKeys['hotel'] = ['name']
delexKeys['restaurant'] = ['food', 'name', 'time']
delexKeys['attraction'] = ['type', 'name']
delexKeys['taxi'] = ['leaveAt', 'destination', 'departure', 'arriveBy']
delexKeys['hospital'] = ['department']

def delexicaliseUser(response, state, query):
    topic  = query.split('|')[0]
    topic = topic.replace('[Q]', '').strip()

    for key in delexKeys[topic]:
        if 'info' in state[topic] and key in state[topic]['info']:
            response = response.replace(state[topic]['info'][key], '['+topic+'_'+key.lower()+']')
        
        if 'semi' in state[topic] and key in state[topic]['semi']:
            response = response.replace(state[topic]['semi'][key], '['+topic+'_'+key.lower()+']')
    
        if 'book' in state[topic] and key == 'time':
            response = response.replace(state[topic]['book'][key], '['+topic+'_'+key.lower()+']')
    
        if 'fail_info' in state[topic] and key in state[topic]['fail_info']:
            response = response.replace(state[topic]['fail_info'][key], '['+topic+'_'+key.lower()+']')
        
        if 'fail_book' in state[topic] and key in state[topic]['fail_book']:
            response = response.replace(state[topic]['fail_book'][key], '['+topic+'_'+key.lower()+']')

#     print(response, '\n', '\n')
    return response

importantKeys = {}
importantKeys['train'] = ['leaveAt', 'destination', 'departure', 'arriveBy', 'day', 'people']
importantKeys['hospital'] = ['department']
importantKeys['hotel'] = ['name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type', 'stay', 'day', 'people']
importantKeys['restaurant'] = ['food', 'pricerange', 'name', 'area', 'people', 'day', 'time']
importantKeys['attraction'] = ['type', 'name', 'area']
importantKeys['police'] = []
importantKeys['taxi'] = ['leaveAt', 'destination', 'departure', 'arriveBy']
def importantKey(key, topic):
    if key in importantKeys[topic]:
        return True
    return False

def getTopicsFromMsg(msg):
    msg = msg.lower()
    if msg.find('train') != -1:
        return 'train'
    elif msg.find('restaurant') != -1 or msg.find('food') != -1:
        return 'restaurant'
    elif msg.find('hotel') != -1: 
        return 'hotel'
    elif msg.find('attraction') != -1:
        return 'attraction'
    elif msg.find('hospital') != -1 or msg.find('accident') != -1:
        return 'hospital'
    elif msg.find('taxi') != -1:
        return 'taxi'
    elif msg.find('police') != -1 or msg.find('robbed') != -1:
        return 'police'
    else:
        return None

def delexQuery(query):
    topic = query[0]
    query = query[1]
    for key in delexKeys[topic]:
        if key in query and query[key] != '' and query[key] != 'not mentioned':
            query[key] = '['+topic+'_'+key.lower()+']'

    return [topic,query]

def formatResults(results, query, fail_book, txt, failed):
    if len(query)!= 0:
            topic = query[0]
    else:
        print('Empty')
        return '[KB] Total = 0 [KB]'
    
    if topic == 'police':
        return '[KB] Total = 1 [KB]'
    elif topic == 'taxi':
        return '[KB] Total = 1 [KB]'
    elif len(results) == 0:
        return '[KB] Total = 0 [KB]'
    else:
        if topic not in fail_book:
            return "[KB] " + " Total = " + str(len(results)) + ' [KB]'
        for k in fail_book[topic]:
            if k in query[1] and fail_book[topic][k] == query[1][k] and failed[topic] == False:
                return '[KB] Total = 0 [KB]'
        final_str = "[KB] " + " Total = " + str(len(results)) + ' [KB]'
        return final_str

def formatQueries(query, context):
    topic = query[0]
    if topic == 'police':
        return '[Q] police [Q]'
    elif len(query) == 1:
        final_str = '[Q] ' + topic + ' | '
        for q in importantKeys[topic]:
            final_str += q + ' = ' + '*' + ' | '
        
        final_str += ' [Q]'
        return final_str    
    else:    
        query = query[1]
        final_str = '[Q] ' + topic + ' | '
        for q in importantKeys[topic]:
            if q in query and query[q] != '' and query[q] != 'not mentioned':
                final_str += q + ' = ' + str(query[q]) + ' | '
            else:
                final_str += q + ' = ' + '*' + ' | ' 
        final_str = final_str[:-3] + ' [Q]'
        return final_str    

user_context_train = []
user_context_valid = []
user_context_test = []
user_context_test_single = []
user_context_test_multiple = []
user_response_train = []
user_response_valid = []
user_response_test = []
user_response_test_single = []
user_response_test_multiple = []
agent_context_train = []
agent_context_valid = []
agent_context_test = []
agent_context_test_single = []
agent_context_test_multiple = []
agent_response_train = []
agent_response_valid = []
agent_response_test = []
agent_response_test_single = []
agent_response_test_multiple = []
goals_train = []
goals_valid = []
goals_test = []
goals_test_single = []
goals_test_multiple = []

agent_kb_train = []
agent_kb_valid = []
agent_kb_test = []
agent_kb_test_single = []
agent_kb_test_multiple = []

agent_query_train = []
agent_query_valid = []
agent_query_test = []
agent_query_test_single = []
agent_query_test_multiple = []

agent_delexquery_train = []
agent_delexquery_valid = []
agent_delexquery_test = []
agent_delexquery_test_single = []
agent_delexquery_test_multiple = []

state_train = []
state_valid = []
state_test = []
state_test_single = []
state_test_multiple = []

dialogue_names_train = []
dialogue_names_valid = []
dialogue_names_test = []
dialogue_names_test_single = []
dialogue_names_test_multiple = []

u=0
multi_goal = []

data_keys = list(data.keys())
random.shuffle(data_keys, random = lambda: 0.5)

for dd in data_keys:
#     print('dialogue : ', u)
    u+=1
    dialogue = data[dd]
    
    goal = dialogue['goal']['message']
    # indicates if the agent has already stated the failure
    failed = {}
    # from user goal stating that failure will happen
    fail_book = {}
    topic_count = 0
    v_state = {}
    te_state = {}
    tr_state = {}
    for topic in ['train', 'restaurant', 'taxi', 'hotel', 'hospital', 'police', 'attraction']:  
        if dialogue['goal'][topic]:
#             print(dd, topic)
            topic_count += 1
            if dd in valList:
                v_state[topic] = dialogue['goal'][topic]
            elif dd in testList:
                te_state[topic] = dialogue['goal'][topic]
            else:
                tr_state[topic] = dialogue['goal'][topic]            
            if 'fail_book' in dialogue['goal'][topic]:
                fail_book[topic] = dialogue['goal'][topic]['fail_book']
            else:
                fail_book[topic] = {}
                
            failed[topic] = False
    
    if dd in valList:
        state_valid.append(v_state)
    elif dd in testList:
        state_test.append(te_state) 
    else:
        state_train.append(tr_state) 
    
    if topic_count > 1:
        multi_goal.append(dd)
    
    msg, key, _ = get_goal(goal)

    utterances = ['[st@rt]']
    speakers = ['Agent']
    prev_query = None
    prev_kb = None
    
    dialogue_query = []
    dialogue_delexquery = []
    dialogue_kb = []
    
    prev_query = 'no'
    prev_kb = 'no'
    domain_topic = None
    
    for log in dialogue['log']:
        if bool(log['metadata']):
            speakers.append('Agent')
            utterances.append('[Agent] ' + log['text'].strip().lower().replace("\n", " "))
            if 'queries' in log:
                if len(log['queries']) < 1:
                    dialogue_query.append(prev_query)
                    dialogue_delexquery.append(prev_delexquery)
                    dialogue_kb.append(prev_kb)
#                     agent_modified_query.append(agent_modified_query[-1])
                else:
                    domain_topic = log['queries'][0]
                    query = formatQueries(log['queries'], utterances)
                    delex_query = formatQueries(delexQuery(log['queries']), utterances)
                    kb = formatResults(log['results'], log['queries'], fail_book, log['text'], failed)
                    dialogue_query.append(query)
                    dialogue_delexquery.append(delex_query)
                    dialogue_kb.append(kb)
#                     agent_modified_query.append(formatQueries(log['modified_queries']))
                    prev_query = query
                    prev_delexquery = delex_query
                    prev_kb = kb
                    t = log['queries'][0]
                    for l in range(len(dialogue_query)-1,-1,-1):
                        if dialogue_query[l-1] == 'no':
                            dialogue_query[l-1] = formatQueries([t], utterances)
                            dialogue_delexquery[l-1] = formatQueries([t], utterances)
                            dialogue_kb[l-1] = formatResults([], [t], fail_book, log['text'], failed)
                        else:
                            break
            else:
                dialogue_query.append(prev_query)
                dialogue_delexquery.append(prev_delexquery)
                dialogue_kb.append(prev_kb)
            
            if any(word in log['text'].strip().lower().replace("\n", " ") for word in [' no ', ' not ', 'unavailable', 'unfortunate', 'sorry', 'unable', 'unsuccessful']):
                    if domain_topic is not None:
                        failed[domain_topic] = True
            
        else:
            speakers.append('User')
            utterances.append('[User] ' + log['text'].strip().replace("\n", " "))
    
    if 'no' in dialogue_query:
        topic = getTopicsFromMsg(msg)
        if topic != None:
            dialogue_query = [formatQueries([topic], utterances)]*len(dialogue_query)
            dialogue_delexquery = [formatQueries([topic], utterances)]*len(dialogue_query)
            dialogue_kb = [formatResults([], [topic], fail_book, log['text'], failed)]*len(dialogue_kb)
     
    if 'no' in dialogue_query:
        print(dialogue_query)
        print(dialogue_kb)
        print(msg)
        
    if len(dialogue_query) != len(dialogue_delexquery):
        raise Exception('Length  Not equal')
    
    if dd in valList:
        agent_query_valid.append(dialogue_query)
        agent_delexquery_valid.append(dialogue_delexquery)
        agent_kb_valid.append(dialogue_kb)
    elif dd in testList:
        agent_query_test.append(dialogue_query)
        agent_delexquery_test.append(dialogue_delexquery)
        agent_kb_test.append(dialogue_kb)
        if dd in multi_goal:
            agent_query_test_multiple.append(dialogue_query)
            agent_kb_test_multiple.append(dialogue_kb)
        else:
            agent_query_test_single.append(dialogue_query)
            agent_kb_test_single.append(dialogue_kb)
            state_test_single.append(dialogue['goal'])
    else:
        agent_query_train.append(dialogue_query)
        agent_delexquery_train.append(dialogue_delexquery)        
        agent_kb_train.append(dialogue_kb)
    
    utterances.extend(['[User] [e*d]']) 
    speakers.extend(['User']) 
    
    dialogue_user_context = []
    dialogue_user_response = []
    dialogue_agent_context = []
    dialogue_agent_response = []
    dialogue_goals = []

    counts=0
    for i in range(1,len(utterances)):
        context = utterances[:i]
        response = utterances[i]

        if speakers[i] == 'User':
            if response.find('[e*d]') == -1 and topic != 'police':
                response = delexicaliseUser(response, dialogue['goal'], dialogue_query[counts])
#                 print(response, '\n', '\n')
            dialogue_user_context.append(context)
            dialogue_user_response.append(response)
        
            dialogue_goals.append(msg)
            counts += 1
            
        else:
            dialogue_agent_context.append(context)
            dialogue_agent_response.append(response) 
    
    if dd in valList:
        user_context_valid.append(dialogue_user_context)
        user_response_valid.append(dialogue_user_response)
        agent_context_valid.append(dialogue_agent_context)
        agent_response_valid.append(dialogue_agent_response)
        goals_valid.append(dialogue_goals)
        dialogue_names_valid.append(dd)
            
    elif dd in testList:
        user_context_test.append(dialogue_user_context)
        user_response_test.append(dialogue_user_response)
        agent_context_test.append(dialogue_agent_context)
        agent_response_test.append(dialogue_agent_response)
        goals_test.append(dialogue_goals)
        dialogue_names_test.append(dd)
        if dd in multi_goal:
            user_context_test_multiple.append(dialogue_user_context)
            user_response_test_multiple.append(dialogue_user_response)
            agent_context_test_multiple.append(dialogue_agent_context)
            agent_response_test_multiple.append(dialogue_agent_response)
            goals_test_multiple.append(dialogue_goals)
            dialogue_names_test_multiple.append(dd)
        else:
            user_context_test_single.append(dialogue_user_context)
            user_response_test_single.append(dialogue_user_response)
            agent_context_test_single.append(dialogue_agent_context)
            agent_response_test_single.append(dialogue_agent_response)
            goals_test_single.append(dialogue_goals)
            dialogue_names_test_single.append(dd)
    else:
        user_context_train.append(dialogue_user_context)
        user_response_train.append(dialogue_user_response)
        agent_context_train.append(dialogue_agent_context)
        agent_response_train.append(dialogue_agent_response)
        goals_train.append(dialogue_goals)
        dialogue_names_train.append(dd)

save_dir = "data/multiwiz/user/"+percentage+"p/"
train_length = int(float(percentage)*len(user_context_train)/100)
valid_length = int(float(percentage)*len(user_context_valid)/100)

with open(save_dir + "train_input.json", "w") as f:
    json.dump(user_context_train[:train_length],f,indent=4)
    
with open(save_dir + "rest_input.json", "w") as f:
    json.dump(user_context_train[train_length:],f,indent=4)
    
with open(save_dir + "valid_input.json", "w") as f:
    json.dump(user_context_valid[:valid_length],f,indent=4)
    
with open(save_dir + "test_input.json", "w") as f:
    json.dump(user_context_test,f,indent=4) 
    
with open(save_dir + "test_input_single.json", "w") as f:
    json.dump(user_context_test_single,f,indent=4) 
    
with open(save_dir + "test_input_multiple.json", "w") as f:
    json.dump(user_context_test_multiple,f,indent=4) 


with open(save_dir + "train_tgt.json", "w") as f:
    json.dump(user_response_train[:train_length],f,indent=4)
    
with open(save_dir + "rest_tgt.json", "w") as f:
    json.dump(user_response_train[train_length:],f,indent=4)

with open(save_dir + "valid_tgt.json", "w") as f:
    json.dump(user_response_valid[:valid_length],f,indent=4)
    
with open(save_dir + "test_tgt.json", "w") as f:
    json.dump(user_response_test,f,indent=4)
    
with open(save_dir + "test_tgt_single.json", "w") as f:
    json.dump(user_response_test_single,f,indent=4)
    
with open(save_dir + "test_tgt_multiple.json", "w") as f:
    json.dump(user_response_test_multiple,f,indent=4)
    
with open(save_dir + "train_goal.json", "w") as f:
    json.dump(goals_train[:train_length],f,indent=4)
    
with open(save_dir + "rest_goal.json", "w") as f:
    json.dump(goals_train[train_length:],f,indent=4)
    
with open(save_dir + "valid_goal.json", "w") as f:
    json.dump(goals_valid[:valid_length],f,indent=4)
    
with open(save_dir + "test_goal.json", "w") as f:
    json.dump(goals_test,f,indent=4)
    
with open(save_dir + "test_goal_single.json", "w") as f:
    json.dump(goals_test_single,f,indent=4)
    
with open(save_dir + "test_goal_multiple.json", "w") as f:
    json.dump(goals_test_multiple,f,indent=4)

save_dir = "data/multiwiz/agent/"+percentage+"p/"
train_length = int(float(percentage)*len(agent_context_train)/100)
valid_length = int(float(percentage)*len(agent_context_valid)/100)

with open(save_dir + "train_input.json", "w") as f:
    json.dump(agent_context_train[:train_length],f,indent=4)
    
with open(save_dir + "rest_input.json", "w") as f:
    json.dump(agent_context_train[train_length:],f,indent=4)
    
with open(save_dir + "valid_input.json", "w") as f:
    json.dump(agent_context_valid[:valid_length],f,indent=4)
    
with open(save_dir + "test_input.json", "w") as f:
    json.dump(agent_context_test,f,indent=4) 
    
with open(save_dir + "test_input_single.json", "w") as f:
    json.dump(agent_context_test_single,f,indent=4) 
    
with open(save_dir + "test_input_multiple.json", "w") as f:
    json.dump(agent_context_test_multiple,f,indent=4) 
    
with open(save_dir + "train_tgt.json", "w") as f:
    json.dump(agent_response_train[:train_length],f,indent=4)
    
with open(save_dir + "rest_tgt.json", "w") as f:
    json.dump(agent_response_train[train_length:],f,indent=4)
    
with open(save_dir + "valid_tgt.json", "w") as f:
    json.dump(agent_response_valid[:valid_length],f,indent=4)

with open(save_dir + "test_tgt.json", "w") as f:
    json.dump(agent_response_test,f,indent=4) 
    
with open(save_dir + "test_tgt_single.json", "w") as f:
    json.dump(agent_response_test_single,f,indent=4) 
    
with open(save_dir + "test_tgt_multiple.json", "w") as f:
    json.dump(agent_response_test_multiple,f,indent=4) 
    
with open(save_dir + "train_kb.json", "w") as f:
    json.dump(agent_kb_train[:train_length],f,indent=4)
    
with open(save_dir + "rest_kb.json", "w") as f:
    json.dump(agent_kb_train[train_length:],f,indent=4)
    
with open(save_dir + "valid_kb.json", "w") as f:
    json.dump(agent_kb_valid[:valid_length],f,indent=4)
    
with open(save_dir + "test_kb.json", "w") as f:
    json.dump(agent_kb_test,f,indent=4)
    
with open(save_dir + "test_kb_single.json", "w") as f:
    json.dump(agent_kb_test_single,f,indent=4)
    
with open(save_dir + "test_kb_multiple.json", "w") as f:
    json.dump(agent_kb_test_multiple,f,indent=4)

with open(save_dir + "train_query.json", "w") as f:
    json.dump(agent_query_train[:train_length],f,indent=4)
    
with open(save_dir + "rest_query.json", "w") as f:
    json.dump(agent_query_train[train_length:],f,indent=4)
    
with open(save_dir + "valid_query.json", "w") as f:
    json.dump(agent_query_valid[:valid_length],f,indent=4) 
    
with open(save_dir + "test_query.json", "w") as f:
    json.dump(agent_query_test,f,indent=4) 
    
with open(save_dir + "train_delex_query.json", "w") as f:
    json.dump(agent_delexquery_train[:train_length],f,indent=4)
    
with open(save_dir + "rest_delex_query.json", "w") as f:
    json.dump(agent_delexquery_train[train_length:],f,indent=4)
    
with open(save_dir + "valid_delex_query.json", "w") as f:
    json.dump(agent_delexquery_valid[:valid_length],f,indent=4) 
    
with open(save_dir + "test_delex_query.json", "w") as f:
    json.dump(agent_delexquery_test,f,indent=4) 
    
with open(save_dir + "test_query_single.json", "w") as f:
    json.dump(agent_query_test_single,f,indent=4)
    
with open(save_dir + "test_query_multiple.json", "w") as f:
    json.dump(agent_query_test_multiple,f,indent=4) 
    
with open(save_dir + "train_state.json", "w") as f:
    json.dump(state_train[:train_length],f,indent=4)
    
with open(save_dir + "rest_state.json", "w") as f:
    json.dump(state_train[train_length:],f,indent=4)
    
with open(save_dir + "valid_state.json", "w") as f:
    json.dump(state_valid[:valid_length],f,indent=4) 
    
with open(save_dir + "test_state.json", "w") as f:
    json.dump(state_test,f,indent=4) 
    
with open(save_dir + "test_state_single.json", "w") as f:
    json.dump(state_test_single,f,indent=4)
    
with open(save_dir + "test_state_multiple.json", "w") as f:
    json.dump(state_test_multiple,f,indent=4) 

with open(save_dir + "train_dialogue_names.json", "w") as f:
    json.dump(dialogue_names_train[:train_length],f,indent=4)
    
with open(save_dir + "rest_dialogue_names.json", "w") as f:
    json.dump(dialogue_names_train[train_length:],f,indent=4)
    
with open(save_dir + "valid_dialogue_names.json", "w") as f:
    json.dump(dialogue_names_valid[:valid_length],f,indent=4)
    
with open(save_dir + "test_dialogue_names.json", "w") as f:
    json.dump(dialogue_names_test,f,indent=4)
    
with open(save_dir + "test_dialogue_names_single.json", "w") as f:
    json.dump(dialogue_names_test_single,f,indent=4)
    
with open(save_dir + "test_dialogue_names_multiple.json", "w") as f:
    json.dump(dialogue_names_test_multiple,f,indent=4)


data_dir = 'data/multiwiz/user/'+percentage+'p/'
with open(data_dir+"train_input.json") as f:
    contexts_t = json.load(f)
            
with open(data_dir+"train_tgt.json") as f:
    responses_t = json.load(f)

with open(data_dir+"train_goal.json") as f:
    goal_t = json.load(f)     

with open(data_dir+"valid_input.json") as f:
    contexts_v = json.load(f)

with open(data_dir+"valid_tgt.json") as f:
    responses_v = json.load(f) 

with open(data_dir+"valid_goal.json") as f:
    goal_v = json.load(f)
    
def getUserModel(model_path):
    checkpoint = torch.load(model_path)
    load_args = checkpoint['args']
    model = LoadGPT.GPT(load_args)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model = model.model
    
    for param in model.parameters():
        param.requires_grad = False
            
    model.eval()

    return model

resp_no = 10
if percentage == '100':
    resp_no = 5
adverse_train = {}

adverse_train = []
train_context = [item for sublist in contexts_t for item in sublist]
train_response = [item for sublist in responses_t for item in sublist]
train_goal = [item for sublist in goal_t for item in sublist]

for start in range(len(train_context)):
    context = " ".join(train_context[start])
    response = train_response[start]
    goal = train_goal[start]

    adverse = []
    if percentage == '100':
        for i in range(resp_no-1):
            idx = random.randint(0,len(train_context)-3)  
            adverse.append(context + ' EOT ' + response + ' EOT ' + train_response[idx]) 

        for i in range(1):
            idx = random.randint(0,len(train_context)-3)  
            adverse.append(context + ' EOT ' + response + ' EOT ' + train_response[idx] + ' ' + train_response[idx+2].replace('[User] ', ' '))     

        if start > 1:
            idx = random.randint(0,resp_no-1)    
            adverse[idx] = context + ' EOT ' + response + ' EOT ' + train_response[start-2]      

        if start > 0:
            idx = random.randint(0,resp_no-1)    
            adverse[idx] = context + ' EOT ' + response + ' EOT ' + train_response[start-1]      


        for rr in adverse:
            rr = rr.replace('\n', '')

        adverse_train.extend(adverse)
    else: 
        for i in range(resp_no-2):
            idx = random.randint(0,len(train_context)-3)  
            adverse.append(context + ' EOT ' + response + ' EOT ' + train_response[idx]) 

        for i in range(2):
            idx = random.randint(0,len(train_context)-3)  
            adverse.append(context + ' EOT ' + response + ' EOT ' + train_response[idx] + ' ' + train_response[idx+2].replace('[User] ', ' '))     

        if start > 1:
            idx = random.randint(0,resp_no-1)    
            adverse[idx] = context + ' EOT ' + response + ' EOT ' + train_response[start-2]      

        if start > 0:
            idx = random.randint(0,resp_no-1)    
            adverse[idx] = context + ' EOT ' + response + ' EOT ' + train_response[start-1]      


        for rr in adverse:
            rr = rr.replace('\n', '')

        adverse_train.extend(adverse)
        
resp_no = 10
if percentage == '100':
    resp_no = 5
adverse_valid = {}

adverse_valid = []
valid_context = [item for sublist in contexts_v for item in sublist]
valid_response = [item for sublist in responses_v for item in sublist]
valid_goal = [item for sublist in goal_v for item in sublist]

for start in range(len(valid_context)):
    context = " ".join(valid_context[start])
    response = valid_response[start]
    goal = valid_goal[start]

    adverse = []

    if percentage == '100':
        for i in range(resp_no-1):
            idx = random.randint(0,len(valid_context)-3)  
            adverse.append(context + ' EOT ' + response + ' EOT ' + valid_response[idx])

        for i in range(1):
            idx = random.randint(0,len(valid_context)-3)  
            adverse.append(context + ' EOT ' + response + ' EOT ' + valid_response[idx] + ' ' + valid_response[idx+2].replace('[User] ', ' '))    

        if start > 1:
            idx = random.randint(0,resp_no-1)    
            adverse[idx] = context + ' EOT ' + response + ' EOT ' + valid_response[start-2]      

        if start > 0:
            idx = random.randint(0,resp_no-1)    
            adverse[idx] = context + ' EOT ' + response + ' EOT ' + valid_response[start-1]      


        for rr in adverse:
            rr = rr.replace('\n', '') 

        adverse_valid.extend(adverse) 
    else:
        for i in range(resp_no-2):
            idx = random.randint(0,len(valid_context)-3)  
            adverse.append(context + ' EOT ' + response + ' EOT ' + valid_response[idx])

        for i in range(2):
            idx = random.randint(0,len(valid_context)-3)  
            adverse.append(context + ' EOT ' + response + ' EOT ' + valid_response[idx] + ' ' + valid_response[idx+2].replace('[User] ', ' '))    

        if start > 1:
            idx = random.randint(0,resp_no-1)    
            adverse[idx] = context + ' EOT ' + response + ' EOT ' + valid_response[start-2]      

        if start > 0:
            idx = random.randint(0,resp_no-1)    
            adverse[idx] = context + ' EOT ' + response + ' EOT ' + valid_response[start-1]      


        for rr in adverse:
            rr = rr.replace('\n', '') 

        adverse_valid.extend(adverse)       
        
        
with open(data_dir + "train_bert_margin.json", "w") as f:
    json.dump(adverse_train, f)   
        
with open(data_dir + "valid_bert_margin.json", "w") as f:
    json.dump(adverse_valid, f)
    
    
data_dir = 'data/multiwiz/agent/'+percentage+'p/'
with open(data_dir+"train_input.json") as f:
    contexts_t = json.load(f)
            
with open(data_dir+"train_tgt.json") as f:
    responses_t = json.load(f)

with open(data_dir+"train_kb.json") as f:
    kb_t = json.load(f)   
    
with open(data_dir+"train_query.json") as f:
    query_t = json.load(f)      

with open(data_dir+"valid_input.json") as f:
    contexts_v = json.load(f)

with open(data_dir+"valid_tgt.json") as f:
    responses_v = json.load(f) 

with open(data_dir+"valid_kb.json") as f:
    kb_v = json.load(f)
    
with open(data_dir+"valid_query.json") as f:
    query_v = json.load(f)    
    
    
def getAgentModel(model_path):
    checkpoint = torch.load(model_path)
    load_args = checkpoint['args']
    model = LoadGPT.GPT(load_args)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model = model.model
    
    for param in model.parameters():
        param.requires_grad = False
            
    model.eval()

    return model


resp_no = 10
if percentage == '100':
    resp_no = 5
adverse_train = {}
adverse_train_bs = {}

adverse_train = []
adverse_train_bs = []
train_context = [item for sublist in contexts_t for item in sublist]
train_response = [item for sublist in responses_t for item in sublist]
train_kb = [item for sublist in kb_t for item in sublist]
train_query = [item for sublist in query_t for item in sublist]

#         model = gpt[key]

for start in range(len(train_context)):
    context = " ".join(train_context[start])
    response = train_response[start]
    kb = train_kb[start]

    query = train_query[start]

    adverse1 = []
    adverse2 = []

    if percentage == '100':
        for i in range(resp_no-1):
            idx = random.randint(0,len(train_context)-1)  
            adverse1.append(query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + train_response[idx]) 

        for i in range(1):
            idx = random.randint(0,len(train_context)-3)  
            adverse1.append(query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + train_response[idx] + ' ' + train_response[idx+2].replace('[Agent] ', ' ')) 

        if start > 1:
            idx = random.randint(0,resp_no-1)    
            adverse1[idx] = query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + train_response[start-2]  

        if start > 0:
            idx = random.randint(0,resp_no-1)    
            adverse1[idx] = query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + train_response[start-1]  


        for rr in adverse1:
            rr = rr.replace('\n', '')
    else:
        for i in range(resp_no-2):
            idx = random.randint(0,len(train_context)-1)  
            adverse1.append(query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + train_response[idx]) 

        for i in range(2):
            idx = random.randint(0,len(train_context)-3)  
            adverse1.append(query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + train_response[idx] + ' ' + train_response[idx+2].replace('[Agent] ', ' ')) 

        if start > 1:
            idx = random.randint(0,resp_no-1)    
            adverse1[idx] = query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + train_response[start-2]  

        if start > 0:
            idx = random.randint(0,resp_no-1)    
            adverse1[idx] = query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + train_response[start-1]  


        for rr in adverse1:
            rr = rr.replace('\n', '')

    adverse_train.extend(adverse1)
            
print(len(adverse_train))       

resp_no = 10
if percentage == '100':
    resp_no = 5
adverse_valid = {}
adverse_valid_bs = {}

adverse_valid = []
adverse_valid_bs = []
valid_context = [item for sublist in contexts_v for item in sublist]
valid_response = [item for sublist in responses_v for item in sublist]
valid_kb = [item for sublist in kb_v for item in sublist]
valid_query = [item for sublist in query_v for item in sublist]

for start in range(len(valid_context)):
    context = " ".join(valid_context[start])
    response = valid_response[start]
    kb = valid_kb[start]
    query = valid_query[start]

    adverse1 = []
    adverse2 = []

    if percentage == '100':
        for i in range(resp_no-2):
            idx = random.randint(0,len(valid_context)-1)  
            adverse1.append(query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + valid_response[idx]) 

        for i in range(2):
            idx = random.randint(0,len(valid_context)-3)  
            adverse1.append(query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + valid_response[idx] + ' ' + valid_response[idx+2].replace('[Agent] ', ' '))

        if start > 1:
            idx = random.randint(0,resp_no-1)    
            adverse1[idx] = query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + valid_response[start-2]  

        if start > 0:
            idx = random.randint(0,resp_no-1)    
            adverse1[idx] = query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + valid_response[start-1]  


        for rr in adverse1:
            rr = rr.replace('\n', '')


        adverse_valid.extend(adverse1) 
    else:
        for i in range(resp_no-2):
            idx = random.randint(0,len(valid_context)-1)  
            adverse1.append(query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + valid_response[idx]) 

        for i in range(2):
            idx = random.randint(0,len(valid_context)-3)  
            adverse1.append(query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + valid_response[idx] + ' ' + valid_response[idx+2].replace('[Agent] ', ' '))

        if start > 1:
            idx = random.randint(0,resp_no-1)    
            adverse1[idx] = query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + valid_response[start-2]  

        if start > 0:
            idx = random.randint(0,resp_no-1)    
            adverse1[idx] = query + ' ' + kb + ' ' + context + ' EOT ' + response + ' EOT ' + valid_response[start-1]  


        for rr in adverse1:
            rr = rr.replace('\n', '')


        adverse_valid.extend(adverse1)        
        
with open(data_dir + "train_bert_marginal_15.json", "w") as f:
    json.dump(adverse_train, f)             
        
with open(data_dir + "valid_bert_marginal_15.json", "w") as f:
    json.dump(adverse_valid, f)   
    



