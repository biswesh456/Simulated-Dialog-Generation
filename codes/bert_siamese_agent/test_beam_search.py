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

parser = argparse.ArgumentParser(description='Train and evaluate an HRED')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training (default: True)')
parser.add_argument('--vocab-size', type=int, default=2**13, metavar='V', help='Size of vocabulary (default: 20000)')
parser.add_argument('--maxlen', type=int, default=50, metavar='ML', help='Maximum Length of an utterance')
parser.add_argument('--word-size', type=int, default=300, metavar='W', help='Size of word embeddings')
parser.add_argument('--hidden-size', type=int, default=1000, metavar='H', help='Size of hidden embeddings')
parser.add_argument('--resume', action='store_true', default=False, help='Resume an old model (default: False)')
parser.add_argument('--lr', type=float, default=.00015, metavar='LR', help='Learning Rate (default: .00015)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-step', type=int, default=100, metavar='LS', help='how many batches to wait before logging training status')
parser.add_argument('--skip', type=int, default=1, metavar='SK', help='How many utts to skip')
parser.add_argument('--start-depth', type=int, default=0, metavar='SD', help='How many utts to skip')
parser.add_argument('--end-depth', type=int, default=100, metavar='ED', help='How many utts to skip')

parser.add_argument('--base-dir', type=str, default='/dccstor/gpandey11/gaurav/', help='A directiory that contains a data and models folder')
parser.add_argument('--data-file', type=str, default='test', help='Prefix of data file')
parser.add_argument('--model', type=str, default=None, help='relative address of model with respect to model directory')
parser.add_argument('--data', type=str, default=None, help='relative address of data with repect to data directory')

parser.add_argument('--best', action='store_true', default=False, help='Load the best model so far')
parser.add_argument('--out', type=str, default='predicted_greedy.txt', help='File to save predictions')
parser.add_argument('--train', action='store_true', default=False, help='Get predictions for training data')
parser.add_argument('--mname', type=str, required=True, help='Name of model')
parser.add_argument('--normalize', action='store_true', default=False, help='Normalize the log-loss over utterance')
parser.add_argument('--bert-model', type=str, default='distilbert-base-uncased', help='pretrained bert model for goal')

args = parser.parse_args()
if args.data is None:
    if args.model is None:
        sys.exit("You need to specify atleast one of data and model!!!!!!")
    else:
        args.data = args.model
elif args.model is None:
    args.model = args.data

# class args:
#     batch_size = 10
#     no_cuda = False
#     vocab_size=2**13
#     maxlen=50
#     word_size = 300
#     hidden_size = 2000
#     resume = False
#     lr = .00015
#     data = 'ibm_5000'
#     val_step = 3
#     log_step = 10
#     save_step = 500
#     base_dir = '/dccstor/gpandey11/gaurav/'
#     num_epochs = 25
#     model = 'ubuntu_hred'
#     data = 'ubuntu/retrieval'
#     best = True
#     train = False
#     load_name = "h2000_best"
#     normalize = False
#     out = "predicted_greedy.txt"
#     start_depth = 1
#     end_depth = 100
#     skip = 1
#     data_file = "dev"

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.data_dir = args.base_dir + "data/" + args.data.lower() + "/"
args.model_dir = args.base_dir + "models/" + args.model.lower() + "/"
if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)
 
args.prediction_dir = args.base_dir + "predictions/" + args.model.lower() + "/"
if not os.path.isdir(args.prediction_dir):
    os.mkdir(args.prediction_dir)
args.predicted_file = args.prediction_dir + args.out

if args.best : 
    args.load_path = args.model_dir + "checkpoint_hred_user_" + args.mname + "_best" + "_" + str(args.word_size) + "_" + str(args.hidden_size)  + "_" + str(args.lr) + "_" + str(args.batch_size) + '.pth.tar'

else:
    args.load_path = args.model_dir + "checkpoint_hred_user_" + args.mname + "_" + str(args.word_size) + "_" + str(args.hidden_size)  + "_" + str(args.lr) + "_" + str(args.batch_size) + '.pth.tar'



# data = Data.SubData_test(data_dir = args.data_dir, vocab_size=args.vocab_size, data_file=args.data_file)
data = Data.SubData_test(data_dir = args.data_dir, vocab_size=args.vocab_size, bert_model_name=args.bert_model) 

args.vocab_size = data.tokenizer.get_vocab_size()
print("The vocab size is {}".format(args.vocab_size))

    
def init_model(args):
    checkpoint = torch.load(args.load_path)
    print(args.load_path)
    if 'args' in checkpoint:
        print("model parameters detected in checkpoint")
        load_args = checkpoint['args']
        model = Model.HRED(load_args)
    else:
        print("Model parameters not detected in checkpoin!!! ")
        model = Model.HRED(args)
    if args.cuda:
        model.cuda()
    model.load(checkpoint, args)
    return model
model = init_model(args)
beam_search = Model.BeamSearch(model.decoder,data.tokenizer.encode("[SEP]").ids)

print('[SEP]', data.tokenizer.encode("[SEP]").ids)

def evaluate(start, filer, filecr, filegt, batch_size=1):
    batch = data.get_batch(batch_size=batch_size, maxlen=args.maxlen, start=start)
    c_utts, c_ulens, c_clens, r_utts, r_ulens, r_clens, cind_mat, rind_mat,\
                                gind_mat, g_utts, g_ulens, keys, kind_mat, k_ulens = batch
    
    if kind_mat.shape[1] == 0:
        kind_mat = torch.zeros(1,1).long()-1
        keys = torch.zeros(1,1).long()

    if args.cuda:
        c_utts, r_utts = c_utts.cuda(), r_utts.cuda()
        c_ulens, r_ulens, k_ulens = c_ulens.cuda(), r_ulens.cuda(), k_ulens.cuda()
        c_clens, r_clens = c_clens.cuda(), r_clens.cuda()
        gind_mat, g_utts, g_ulens = gind_mat.cuda(), g_utts.cuda(), g_ulens.cuda()
        keys, kind_mat = keys.cuda(), kind_mat.cuda()

    c_embeds, c_uhiddenembs = model.get_context_embedding(c_utts, c_ulens, c_clens)
    gembeds, g_pos, g_hid = model.g_encoder(g_utts, g_ulens, gind_mat, keys, k_ulens)

#     finished_sids, finished_probs = beam_search.get_samples(c_embeds[0], c_utts, c_uhiddenembs,\
#                                                             gembeds, kind_mat, g_pos, g_hid, beam_size=5)
#     predicted = data.tokenizer.decode(finished_sids[0])
    finished_sids = model.decoder.get_greedy_samples(c_embeds[0], c_utts, c_uhiddenembs,gembeds, kind_mat, g_pos, g_hid, data)
    predicted = data.tokenizer.decode(finished_sids)
    
    gt = data.tokenizer.decode(list(r_utts[0,0]))
    gt = gt.replace("[SEP]", "").strip()
    predicted = predicted.replace("[SEP]", "").strip()
    print('P ', predicted)
    print('G ', gt)
    write_to_file(filecr, filer, filegt, c_utts, c_ulens, predicted, gt, g_utts)
    return predicted, gt
        
    
def write_to_file(filecr, filer, filegt, c_utts, c_ulens, predicted, gt, g_utts):
    context_ids = c_utts[0]
    length_ids = c_ulens[0]
    
    filecr.write("GOAL : \n" + data.tokenizer.decode(list(g_utts[0])) + "\n" +"."*50+"\n")
    
    for utt_ids, ll in zip(context_ids, length_ids):
        utt = data.tokenizer.decode(list(utt_ids))
        filecr.write(utt +"\n")


    utt = predicted
    filecr.write('GOLD : ' + gt + "\n")
    filecr.write("GENERATED : \n" + utt + "\n" +"="*50+"\n")
    filer.write(utt + "\n")

model.eval()
t0 = time.time()
dialogs = data.valid_contexts
import io
fr = io.open(args.predicted_file, "w", encoding='utf-8')
fgt = io.open(args.predicted_file+"gt", "w", encoding='utf-8')

fcr = io.open(args.predicted_file+"crg", "w", encoding='utf-8')
num_steps_per_epoch = int(len(dialogs))
start = 0
loss = 0
matched = 0
niter = 0

total_preds = []
total_gts = []
for step in range(0, num_steps_per_epoch):
    #Train on training data
    pred, gt = evaluate(start=start, filer=fr, filecr=fcr, filegt=fgt)
    total_preds.append(pred.strip().split(" "))
    total_gts.append([gt.split(" ")])
    niter += 1
    start += 1
    
    if niter%args.log_step==0:
        t1 = time.time()
        bleu_score = corpus_bleu(total_gts, total_preds, smoothing_function=SmoothingFunction().method1)
        print(start, "{0:.3f}".format(bleu_score*100), "{0:.3f}".format(t1-t0))
        t0 = t1
        sys.stdout.flush()
        fr.flush()
        fcr.flush()
        fgt.flush()

f.close()