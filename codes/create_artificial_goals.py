from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelWithLMHead, BertTokenizer, LongformerTokenizer, LongformerModel
import torch, json, random
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--percentage', type=str, default='None', help='Percentage')

args = parser.parse_args()

percentage = args.percentage
data_dir = '../data/multiwiz/user/'+percentage+'p/'
agent_dir = '../data/multiwiz/agent/'+percentage+'p/'
        
if percentage == '100':
    with open(data_dir+"train_input.json") as f:
        contexts = json.load(f)

    with open(data_dir+"train_tgt.json") as f:
        responses = json.load(f)

    with open(data_dir+"train_goal.json") as f:
        goals = json.load(f)

    with open(agent_dir+"train_state.json") as f:
        final_state = json.load(f)     
else: 
    with open(data_dir+"rest_input.json") as f:
        contexts = json.load(f)

    with open(data_dir+"rest_tgt.json") as f:
        responses = json.load(f)

    with open(data_dir+"rest_goal.json") as f:
        goals = json.load(f)

    with open(agent_dir+"rest_state.json") as f:
        final_state = json.load(f)

save_path = '../data/multiwiz/user/'+percentage+'p/goals_new.json'
save_state_path = '../data/multiwiz/agent/'+percentage+'p/final_state.json'

artificial_goal = []
for i in range(len(goals)):
    artificial_goal.append(goals[i][0])

with open("../createData/multiwoz21/testListFile.json", 'r') as f:
    testList = f.read().split('\n')

with open(save_path, 'w') as f:
    json.dump(artificial_goal, f)
    
with open(save_state_path, 'w') as f:
    json.dump(final_state, f)





