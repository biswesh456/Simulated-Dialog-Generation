import json
import re

d_train= json.load(open('createData/multiwoz21/db/train_db.json'))
d_rest = json.load(open('createData/multiwoz21/db/restaurant_db.json'))
d_hotel = json.load(open('createData/multiwoz21/db/hotel_db.json'))
d_police = json.load(open('createData/multiwoz21/db/police_db.json'))
d_hosp = json.load(open('createData/multiwoz21/db/hospital_db.json'))
d_attr = json.load(open('createData/multiwoz21/db/attraction_db.json'))
d_taxi = [{
  "taxi_colors" : ["black","white","red","yellow","blue","grey"],
  "taxi_types":  ["toyota","skoda","bmw","honda","ford","audi","lexus","volvo","volkswagen","tesla"],
  "taxi_phone": ["^[0-9]{10}$"]
}]
entity_db_map = {'train':d_train, 'restaurant': d_rest, 'police': d_police, 'hospital': d_hosp, 'attraction': d_attr, 'taxi':d_taxi,'hotel':d_hotel}

d_data = json.load(open('createData/multiwoz21/delex.json'))

def get_results(semi,ent):
    db = entity_db_map[ent]
    ret=[]
    for row in db:
#         print(row)
        match = True
        for k in semi.keys():
            if k not in row:
                continue
            # normalise
            val = row[k].lower().strip()
            semi[k] = semi[k].lower().strip()
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
            if(val!=semi[k] and semi[k]!='not mentioned' and semi[k]!='dontcare' and semi[k]!='none' and semi[k]!=''):
                match=False
                break

        if(match):
            ret.append(row)
    
    if len(ret) == 0:
        if "leaveAt" in semi.keys() or "arriveBy" in semi.keys():
#             print(semi)
            ret1=[]
            ret2=[]
            for row in db:
#                 print(row)
                match = True
                for k in semi.keys():
                    if k not in row:
                        continue
                    row[k] = row[k].lower()
                    if(k == "arriveBy" and semi[k]!='not mentioned' and semi[k]!='dontcare'and semi[k]!='none' and semi[k]!=''):
                        if semi[k] == "afternoon" or semi[k] == "after lunch":
                            if int(row[k][:2])<=16:
                                match=True
                            else:
                                match=False
                                break
                        elif semi[k] == "morning":
                            if int(row[k][:2])<=11:
                                match=True
                            else:
                                match=False
                                break        
                        elif semi[k][0]!= ':' and int(semi[k].split(':')[0][-2:]) >= int(row[k].split(':')[0][-2:]):
                            match=True
                        else:
                            match=False
                            break
                    elif(k == "leaveAt" and semi[k]!='not mentioned' and semi[k]!='dontcare'and semi[k]!='none' and semi[k]!=''):
                        if semi[k] == "afternoon" or semi[k] == "after lunch":
                            if int(row[k][:2])<=16:
                                match=True
                            else:
                                match=False
                                break
                        elif semi[k] == "morning":
                            if int(row[k][:2])<=11:
                                match=True
                            else:
                                match=False
                                break
                
                        elif semi[k][0]!= ':' and int(semi[k].split(':')[0][-2:])-1 <= int(row[k].split(':')[0][-2:]):
                            match=True
                        else:
                            match=False
                            break        
                    elif(row[k]!=semi[k] and semi[k]!='not mentioned' and semi[k]!='dontcare'and semi[k]!='none' and semi[k]!=''):
                        match=False
                        break
             
                if match:
                    ret.append(row)
                    
    return ret

def check_query_semi(semi):
    
    for k in semi.keys():
        if((semi[k]!='not mentioned') and semi[k]!='' and semi[k]!='none'):
            return True


for dial_k in d_data.keys():
    all_results = []
    all_queries = []
    goal = d_data[dial_k]['goal']
    topics_allowed = []
    # mark the topics that are mentioned in the goal
    for t in ['train', 'attraction', 'taxi', 'restaurant', 'hotel']:
        if goal[t]:
            topics_allowed.append(t)
    current_topic = ''
    # mark the topics are finished
    topics_done = {'train' : False, 'restaurant' : False, 'attraction' : False, 'hotel':False, 'taxi':False}
    
    # go through theconversation logs and add the query and result key to it
    for utt in d_data[dial_k]['log']:
        meta = utt['metadata']
        new_meta = utt['metadata']
        text = utt['text']
        if(meta=={}):
            continue
        found = False
        
        # add the queries and results key
        utt['results']=[]
        utt['queries']=[]
        possible_topic = []
        for k in meta.keys():
            if (k in topics_allowed and k!='bus' and check_query_semi(meta[k]['semi'])):
                if(topics_done[k] != True):
                    possible_topic.append(k)
        
        # determine the current topic i.e. if 2 topics are possible then it means the current topics will be changed
        if len(possible_topic) == 1:
            current_topic = possible_topic[0]
        elif len(possible_topic) == 2:
            if current_topic == possible_topic[0]:
                topics_done[current_topic] = True
                current_topic = possible_topic[1]
            else:
                topics_done[current_topic] = True
                current_topic = possible_topic[0]        
        elif len(possible_topic) >= 3:
            print('ERROR'*100)
            print(possible_topic)
            
#                 try:
        if current_topic!='train' and current_topic!='taxi' and current_topic != '':
#                 print(current_topic)
#                 print(meta[current_topic]['semi'])
                utt['results'].extend(get_results(meta[current_topic]['semi'],current_topic))
                q = meta[current_topic]['semi']
                for b in meta[current_topic]['book']:
                    if b != 'booked':
                        q[b] = meta[current_topic]['book'][b]
                utt['queries'].extend((current_topic,q))
                all_queries.append((current_topic,q))
#                         utt['queries'].extend((k,meta[k]['semi']))
#                         all_queries.append((k,meta[k]['semi']))
                all_results.append(utt['results'])
                found=True
        elif current_topic=='taxi':
                utt['results'].extend([])
                q = meta[current_topic]['semi']
                for b in meta[current_topic]['book']:
                    if b != 'booked':
                        q[b] = meta[current_topic]['book'][b]
                utt['queries'].extend((current_topic,q))
                all_queries.append((current_topic,q))
#                         utt['queries'].extend((k,meta[k]['semi']))
#                         all_queries.append((k,meta[k]['semi']))
                all_results.append(utt['results'])
                found=True
        elif current_topic == 'train':
                text = text.split(' ')
                for i in range(len(text)):
                    if text[i].find('arriv') != -1: 
                        for cand in text[i:i+5]:
                            if cand.find(":") != -1:
                                new_meta[current_topic]['semi']['arriveBy'] = cand
                                break

                    if text[i].find('leav') != -1 or text[i].find('depart') != -1:
                        for cand in text[i:i+5]:
                            if cand.find(":") != -1:
                                new_meta[current_topic]['semi']['leaveAt'] = cand
                                break

                    if text[i][:2] == 'TR' and text[i][2:].isdecimal():
                        new_meta[current_topic]['semi']['trainID'] = text[i]
                utt['results'].extend(get_results(new_meta[current_topic]['semi'],current_topic))
                q = meta[current_topic]['semi']
                for b in meta[current_topic]['book']:
                    if b != 'booked':
                        q[b] = meta[current_topic]['book'][b]
                utt['queries'].extend((current_topic,q))
                all_queries.append((current_topic,q))
#                         utt['queries'].extend((k,meta[k]['semi']))
#                         all_queries.append((k,meta[k]['semi']))
                all_results.append(utt['results'])
                found=True

    d_data[dial_k]['all_queries'] = all_queries
    d_data[dial_k]['all_results'] = all_results
    print('Adding ', len(all_queries), 'queries')
    print('Adding ', sum(map(lambda x: len(x),all_results)), 'results')

json.dump(d_data, open('createData/multiwoz21/delex_query_results.json','w'))
