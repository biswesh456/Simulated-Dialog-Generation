# Simulated-Dialog-Generation
This is the code for the paper "Simulated Chats for Building Dialog Systems: Learning to Generate Conversations from Instructions" published in the Findings of EMNLP 2021. [[Paper Link]](https://arxiv.org/pdf/2010.10216.pdf)

## Requirements
- Python 3.6
- spacy 2.1.4
- PyTorch 1.4.0
- transformers 2.11.0

## Data Preprocessing

We have provided the delexicalised MultiWoz 2.1 data which can be found by unziping createData/multiwoz21.zip. In order to run the delexicalise the data run the following command:
```
python delex.py
```

We further convert the data to a format which could be sent as an input to our models. The final data has again been provided in data.zip but could also be generated using the following command(example for a low resource setting of 10% has been provided):
```
python build_query_result.py
python create_percentage_multiwoz_dataset.py --percentage 10
```

## Training the models on various dataset sizes

We train 5 models for each dataset size i.e. siamese network for agent, siamese network for user, belief state model(delexicalised), user dialog generator and agent dialog generator. All the models can be trained in the following way:
```
python codes/bert_siamese_agent/train_margin.py --data data/multiwiz/agent/10p --model models/siamese/agent/10p --num-epoch 5
python codes/bert_siamese/train_margin.py --data data/multiwiz/user/10p --model models/siamese/user/10p --num-epoch 5
python codes/gpt_agent/train.py --data data/multiwiz/agent/10p --model models/agent/10p --aug-type 'augmented' --num-epoch 10 --lr 0.00001
python codes/gpt/train.py --data data/multiwiz/user/10p --model models/user/10p --num-epoch 6 --aug-type 'augmented' --num-epoch 10 --lr 0.00001
python codes/gpt_query/train.py --data data/multiwiz/agent/10p --model models/query/10p_mod_del  --num-epoch 10 --lr 0.000001 --delex
```


## Creating artificial data

To create artificial data in artificial_data directory and combining the data with original data, run the following commands:
```
python create_artificial_goals.py --percentage 10
python generate_artificial_data.py --percenttage 10 --user-p  0.6 --agent-p 0.7
python combine_simulated_data.py --percentage 10 
```

## Training end task model with augmented data

Here we train only 2 models i.e. belief state model and agent dialog generator. We train 2 models per dataset : 1) With augmented data 2) With original data(Note: we have already trained the agent model for generating artificial data. Although the belief state model was trained on delexicalised values which is not the case for final task. Hence we need to train belief state model again.) 

The following commands help us in training the final models with augmented data:
```
python codes/gpt_agent/train.py --data data/multiwiz/agent/augmented_mixed_goal_10p --model models/agentfinal/augmented_mixed_goal_10+90p --aug-type 'augmented' --num-epoch 10 --lr 0.000005
python codes/gpt_query/train.py --data data/multiwiz/agent/augmented_mixed_goal_10p --model models/queryfinal/augmented_mixed_goal_10+90p --num-epoch 10 --lr 0.000001
```

Run the command to train lexicalised query model without augmented data:
```
python codes/gpt_query/train.py --data data/multiwiz/agent/10p --model models/query/10p --num-epoch 10 --lr 0.000001
```

## Testing and evaluating

To test the model with augmented data with oracle belief state run the following command:
```
python3 codes/gpt_agent/test.py --model models/agentFinal/augmented_mixed_goal_10+90p --data data/multiwiz/agent/100p --mname ''  --best --prediction oracle_10+90p --type entire --prediction-dir predictions/

python evaluate.py --file oracle_10+90p.json
```

To test the model with augmented data with generated belief state run the following command:
```
python codes/gpt_agent/test_query.py --model models/agentFinal/augmented_mixed_goal_10+90p --data data/multiwiz/agent/100p --mname ''  --best --prediction generated_10+90p --query-data models/queryFinal/augmented_mixed_goal_10+90p --type entire --prediction-dir predictions/ 

python evaluate.py --file generated_10+90p.json
```

To test the model without augmented data with oracle belief state run the following command:
```
python codes/gpt_agent/test.py --model models/agent/10p --data data/multiwiz/agent/100p --mname ''  --best --prediction oracle_10p --type entire --prediction-dir predictions/

python evaluate.py --file oracle_10p.json
```

To test the model without augmented data with generated belief state run the following command:
```
python3 codes/gpt_agent/test_query.py --model models/agent/10p --data data/multiwiz/agent/100p --mname ''  --best --prediction generated_10p --query-data models/query/10p --type entire --prediction-dir predictions/

python evaluate.py --file generated_10p.json
```







