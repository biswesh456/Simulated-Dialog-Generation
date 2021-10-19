import numpy as np
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import sys
import os
import time
import argparse
import Data
import Model
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from collections import Counter
from nltk.util import ngrams
import json

parser = argparse.ArgumentParser(description='Train and evaluate an HRED')
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training (default: True)')
parser.add_argument('--vocab-size', type=int, default=2**13, metavar='V', help='Size of vocabulary (default: 20000)')
parser.add_argument('--maxlen', type=int, default=50, metavar='ML', help='Maximum Length of an utterance')
parser.add_argument('--word-size', type=int, default=128, metavar='W', help='Size of word embeddings')
parser.add_argument('--hidden-size', type=int, default=512, metavar='H', help='Size of hidden embeddings')
parser.add_argument('--resume', action='store_true', default=False, help='Resume an old model (default: False)')
parser.add_argument('--lr', type=float, default=.00001, metavar='LR', help='Learning Rate (default: .00015)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-step', type=int, default=200, metavar='LS', help='how many batches to wait before logging training status')
parser.add_argument('--skip', type=int, default=1, metavar='SK', help='How many utts to skip')
parser.add_argument('--start-depth', type=int, default=0, metavar='SD', help='How many utts to skip')
parser.add_argument('--end-depth', type=int, default=100, metavar='ED', help='How many utts to skip')

parser.add_argument('--base-dir', type=str, default='/dccstor/gpandey11/gaurav/', help='A directiory that contains a data and models folder')
parser.add_argument('--data-file', type=str, default='test', help='Prefix of data file')
parser.add_argument('--model', type=str, default=None, help='address of model with respect to model directory')
parser.add_argument('--data', type=str, default=None, help='address of data with repect to data directory')
parser.add_argument('--prediction', type=str, default=None, help='address of data with repect to prediction directory')
parser.add_argument('--query-data', type=str, default=None, help='address of query data with repect to data directory')
parser.add_argument('--prediction-dir', type=str, default=None, help='address of query data with repect to data directory')

parser.add_argument('--best', action='store_true', default=False, help='Load the best model so far')
parser.add_argument('--out', type=str, default='predicted_greedy.txt', help='File to save predictions')
parser.add_argument('--train', action='store_true', default=False, help='Get predictions for training data')
parser.add_argument('--mname', type=str, required=True, help='Name of model')
parser.add_argument('--normalize', action='store_true', default=False, help='Normalize the log-loss over utterance')
parser.add_argument('--bert-model', type=str, default='distilbert-base-uncased', help='pretrained bert model for goal')
parser.add_argument('--type', type=str, default='entire', help='type of data in test to be used for testing [single and multiple are other 2 options]')

args = parser.parse_args()
if args.data is None:
    if args.model is None:
        sys.exit("You need to specify atleast one of data and model!!!!!!")
    else:
        args.data = args.model
elif args.model is None:
    args.model = args.data

args.cuda = not args.no_cuda and torch.cuda.is_available()
# args.data_dir = args.base_dir + "data/" + args.data.lower() + "/"
# args.model_dir = args.base_dir + "models/" + args.model.lower() + "/"
args.data_dir = args.data.lower() + "/"
args.model_dir = args.model.lower() + "/"
args.query_dir = args.query_data.lower() + "/"
if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)
 
 # args.prediction_dir = args.base_dir + "predictions/" + args.model.lower() + "/"
# args.prediction_dir = "/disk1/home/biswesh/predictions/"

if not os.path.isdir(args.prediction_dir):
    os.mkdir(args.prediction_dir)
args.predicted_file = args.prediction_dir + args.out

if args.best : 
    args.load_path = args.model_dir + "checkpoint_hred_user_" + args.mname + "_best" + "_" + str(args.word_size) + "_" + str(args.hidden_size) + "_" + str(args.batch_size) + '.pth.tar'

else:
    args.load_path = args.model_dir + "checkpoint_hred_user_" + args.mname + "_" + str(args.word_size) + "_" + str(args.hidden_size) + "_" + str(args.batch_size) + '.pth.tar'

args.query_path = args.query_dir + "checkpoint_hred_query_" + args.mname + "_best" + '.pth.tar'


# data = Data.SubData_test(data_dir = args.data_dir, vocab_size=args.vocab_size, data_file=args.data_file)
# data = Data.GPTData_test(data_dir = args.data_dir, Type = args.type) 
data = Data.GPTDataQuery_test(data_dir = args.data_dir, Type = args.type) 

final_data = {}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
def init_model(args):
    checkpoint = torch.load(args.load_path)
    print(args.load_path)
    if 'args' in checkpoint:
        print("model parameters detected in checkpoint")
        load_args = checkpoint['args']
        model = Model.GPT(load_args)
    else:
        print("Model parameters not detected in checkpoin!!! ")
        model = Model.GPT(args)
    if args.cuda:
        model.cuda()
    model.load(checkpoint, args)
    return model.model

def init_query_model(args):
    checkpoint = torch.load(args.query_path)
    print(args.query_path)
    if 'args' in checkpoint:
        print("model parameters detected in checkpoint")
        load_args = checkpoint['args']
        model = Model.GPT(load_args)
    else:
        print("Model parameters not detected in checkpoin!!! ")
        model = Model.GPT(args)
    if args.cuda:
        model.cuda()
    model.load(checkpoint, args)
    return model.model

model = init_model(args).eval()
query_model = init_query_model(args).eval()

def evaluate(start, filecr, batch_size=1):
    response, kb, context, file_name, query_context = data.get_batch(start=start)
    
    if file_name not in final_data:
        final_data[file_name] = [[], [], [], []]
    
    if len(final_data[file_name][1]) == 0:
        prev_query = '[Q] empty [Q]'
    else:
        prev_query = final_data[file_name][1][-1]
        
    query_context = query_context[:1000] + ' ' + prev_query + ' | '
    
    query_input = data.tokenizer.encode(query_context[:1022], return_tensors='pt')
    query = query_model.generate(query_input.cuda(), max_length=1022, pad_token_id=data.tokenizer.eos_token_id, num_beams=1, \
                                 num_return_sequences=1, early_stopping=True)
    
    query = data.tokenizer.decode(query[0][query_input.shape[1]:]).replace('<|endoftext|>', '')
#     query = query.strip()
    idx = query.find(']')
    query = query[idx+1:]
    idx = query.find('[')
    query = '[Q] ' + (query[:idx]).strip() + ' [Q]'
#     print(query, flush = True)
    input_ids = data.tokenizer.encode(query + ' ' + kb + ' ' + context, return_tensors='pt')
    
    if args.cuda:
        input_ids = input_ids.cuda()

    predicted = model.generate(input_ids, max_length=1023, pad_token_id=data.tokenizer.eos_token_id, \
                               num_beams=1, num_return_sequences=1, early_stopping=True)
#     predicted = model.generate(input_ids, max_length=1023, pad_token_id=data.tokenizer.eos_token_id,\
#                                early_stopping=True, do_sample=True, top_p=0.5, top_k=3)
    
    predicted = data.tokenizer.decode(predicted[0][ input_ids.shape[1]:]).replace('<|endoftext|>', '')
    predicted = predicted.replace('.', ' . ')
    predicted = predicted.replace('?', ' ? ')
    predicted = predicted.replace('!', ' ! ')
    predicted = predicted.replace(',', ' , ')
    predicted = predicted.replace('[Agent]', '')
    response = response.replace('[Agent]', '')
    
    final_data[file_name][0].append(predicted)
    final_data[file_name][1].append(query)
    final_data[file_name][2].append(response)
    final_data[file_name][3].append(context)

    return predicted, response
        
    
def write_to_file(filecr, predicted, response, query, kb, context):
    
    filecr.write("Context : \n" + context + "\n")
    filecr.write("Query : \n" + query + "\n")
    filecr.write("KB : \n" + kb + "\n" +"."*50+"\n")
    
    filecr.write("Prediction : " + predicted +"\n")
    filecr.write("Gold : " + response +"\n" +"="*50+"\n")

set_seed(args)
model.eval()
t0 = time.time()
dialogs = data.test_context
import io
fcr = io.open(args.predicted_file+"crg", "w", encoding='utf-8')

num_steps_per_epoch = int(len(dialogs))
start = 0
loss = 0
matched = 0
niter = 0

total_preds = []
total_gts = []
total_gts2 = []
for step in range(0, num_steps_per_epoch):
    #Train on training data
    pred, gt = evaluate(start=start, filecr=fcr)
    total_preds.append(pred.lower().strip().split(" "))
    total_gts.append([gt.lower().strip().split(" ")])
    niter += 1
    start += 1
    
    if niter%args.log_step==0 or niter==num_steps_per_epoch-1:
        t1 = time.time()
        bleu_score = corpus_bleu(total_gts, total_preds, smoothing_function=SmoothingFunction().method1)
        print(start, "{0:.3f}".format(bleu_score*100), "{0:.3f}".format(t1-t0))
        t0 = t1
        sys.stdout.flush()
        fcr.flush()

with open(args.prediction_dir + args.prediction +'.json', 'w') as f:
    json.dump(final_data, f, indent=4)
       
fcr.close()