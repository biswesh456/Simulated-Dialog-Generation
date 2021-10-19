from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelWithLMHead, BertTokenizer, LongformerTokenizer, LongformerModel
import torch, json, random
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--percentage', type=str, default='None', help='Percentage')

args = parser.parse_args()

percentage = args.percentage
path = '../data/multiwiz/agent/'+percentage+'p/'
db_path = '../createData/multiwoz21/'
artificial_data_path = '../artificial_data/'
save_path = '../data/multiwiz/agent/' 
# original ratio is the ratio of generate_artificial_data
original_ratio = 1
# final_ratio is what we want while training. Original should always be greater
final_ratio = 1
current_ratio = 1

with open(path + 'train_input.json') as f:
    contexts = json.load(f)
    
with open(path + 'train_tgt.json') as f:
    responses = json.load(f)
    
with open(path + 'train_query.json') as f:
    queries = json.load(f)
    
with open(path + 'train_kb.json') as f:
    kbs = json.load(f)

with open(path + 'valid_input.json') as f:
    contexts_valid = json.load(f)
    
with open(path + 'valid_tgt.json') as f:
    responses_valid = json.load(f)
    
with open(path + 'valid_query.json') as f:
    queries_valid = json.load(f)
    
with open(path + 'valid_kb.json') as f:
    kbs_valid = json.load(f)

final_contexts = contexts.copy()
final_responses=responses.copy()
final_queries=queries.copy()

final_kbs=kbs.copy()

final_contexts_valid = contexts_valid.copy()

final_responses_valid = responses_valid.copy()

final_queries_valid=queries_valid.copy()

final_kbs_valid = kbs_valid.copy()

d_train= json.load(open(db_path + 'db/train_db.json'))
d_rest = json.load(open(db_path + 'db/restaurant_db.json'))
d_hotel = json.load(open(db_path + 'db/hotel_db.json'))
d_police = json.load(open(db_path + 'db/police_db.json'))
d_hosp = json.load(open(db_path + 'db/hospital_db.json'))
d_attr = json.load(open(db_path + 'db/attraction_db.json'))
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

def getKB(query, utt_no, domain_key):
    not_kb = ['people', 'time', 'stay']
    if domain_key != 'train':
        not_kb.append('day')
    
    db = entity_db_map[domain_key]
    final_query = {}
    for q in query.split(' | ')[1:]:
        q = q.split(' = ')
        for k in query_key[domain_key]:
            if q[0].find(k) != -1 and q[1] != 'not mentioned' and q[1]!='dontcare' and q[1]!='none' and q[1]!='' and q[1]!='*':
                final_query[k] = q[1]  
    
    ret = []
    for row in db:
        match = True
        for k in final_query.keys():
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
            elif(row[k]!=final_query[k]):
                match=False
                break

        if(match):
            ret.append(row)

    if len(ret) == 0:
        return  "[KB] " + " Total = 0 [KB]", {}
    else:
#         return '[KB] Total : ' + str(len(ret)) + ' ' + str(getStringKB(ret[0])), ret[0]
         return "[KB] " + " Total = " + str(len(ret)) + ' [KB]', ret[0]

contexts_artificial = []
responses_artificial = []
queries_artificial = []
kbs_artificial = []
    
with open(artificial_data_path + 'data_all_'+percentage+'p.json') as f:
    data = json.load(f)
    print(len(data))

with open(artificial_data_path + 'query_all_'+percentage+'p.json') as f:
    query = json.load(f)
    print(len(query))

with open(artificial_data_path + 'kb_all_'+percentage+'p.json') as f:
    kb = json.load(f)
    print(len(kb))
    
for i in range(len(data)):
    conversation = data[i]
    conv_context = []
    conv_response = []
    conv_query = []
    conv_kb = []
    utt_no = 0
    for j in range(2, len(conversation), 2):
        conv_context.append(conversation[:j])
        conv_response.append(conversation[j])
        conv_query.append(query[i][utt_no])
        conv_kb.append(kb[i][utt_no])
        utt_no+=1

    if i == 0:
        print(conv_context)
        print(conv_response)
        print(conv_query)
        print(conv_kb)
    contexts_artificial.append(conv_context)
    responses_artificial.append(conv_response)
    queries_artificial.append(conv_query)
    kbs_artificial.append(conv_kb)
        
final_contexts.extend(contexts_artificial)
final_responses.extend(responses_artificial)
final_queries.extend(queries_artificial)
final_kbs.extend(kbs_artificial)

save_dir = save_path + "augmented_mixed_goal_"+percentage+"p/"

with open(save_dir + "train_input.json", "w") as f:
    json.dump(final_contexts,f,indent=4)
    
with open(save_dir + "train_tgt.json", "w") as f:
    json.dump(final_responses,f,indent=4)
    
with open(save_dir + "train_kb.json", "w") as f:
    json.dump(final_kbs,f,indent=4)
    
with open(save_dir + "train_query.json", "w") as f:
    json.dump(final_queries,f,indent=4)
    
with open(save_dir + "valid_input.json", "w") as f:
    json.dump(final_contexts_valid,f,indent=4)
    
with open(save_dir + "valid_tgt.json", "w") as f:
    json.dump(final_responses_valid,f,indent=4)
    
with open(save_dir + "valid_kb.json", "w") as f:
    json.dump(final_kbs_valid,f,indent=4)
    
with open(save_dir + "valid_query.json", "w") as f:
    json.dump(final_queries_valid,f,indent=4)
