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
from transformers import AdamW

parser = argparse.ArgumentParser(description='Train and evaluate an HRED')
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training (default: True)')
parser.add_argument('--vocab-size', type=int, default=2**13, metavar='V', help='Size of vocabulary (default: 20000)')
parser.add_argument('--maxlen', type=int, default=50, metavar='ML', help='Maximum Length of an utterance')
parser.add_argument('--word-size', type=int, default=128, metavar='W', help='Size of word embeddings')
parser.add_argument('--hidden-size', type=int, default=512, metavar='H', help='Size of hidden embeddings')
parser.add_argument('--goal-len', type=int, default=500, metavar='GL', help='Maximum Length of an utterance')
parser.add_argument('--resume', action='store_true', default=False, help='Resume an old model (default: False)')
parser.add_argument('--lr', type=float, default=.00001, metavar='LR', help='Learning Rate (default: .00015)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--val-step', type=int, default=3, metavar='ES', help='how many batches to wait before evaluating 1 test batch')
parser.add_argument('--log-step', type=int, default=40, metavar='LS', help='how many batches to wait before logging training status')
parser.add_argument('--save-step', type=int, default=120, metavar='SS', help='how many batches to wait before saving model')
parser.add_argument('--data', type=str, default=None, help='dataset')
parser.add_argument('--model', type=str, default=None, help='dataset')
parser.add_argument('--key', type=str, default=None, help='domain')
parser.add_argument('--bert-model', type=str, default='distilbert-base-uncased', help='pretrained bert model for goal')
parser.add_argument('--base-dir', type=str, default='/dccstor/gpandey11/gaurav/', help='A directiory that contains a data and models folder')
parser.add_argument('--best', action='store_true', default=False, help='Load the best model so far')
parser.add_argument('--num-epochs', type=int, default=20, metavar='E', help='Number of epochs for training the model')
parser.add_argument('--save-name', type=str, default="", help='Name of model to be saved')
parser.add_argument('--load-name', type=str, default="", help='Name of model to be loaded')
parser.add_argument('--max-steps', type=int, default=200000, help='Max steps per epoch')
parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")

args = parser.parse_args()
if args.save_name == "":
    args.save_name = args.load_name
    
if args.load_name == "":
    args.load_name = args.save_name
    
if args.data is None:
    if args.model is None:
        sys.exit("You need to specify atleast one of data-dir and model-dir!!!!!!")
    else:
        args.data = args.model
elif args.model is None:
    args.model = args.data
    
args.cuda = not args.no_cuda and torch.cuda.is_available()
# args.data_dir = args.base_dir + "data/" + args.data.lower() + "/"
# args.model_dir = args.base_dir + "models/" + args.model.lower() + "/"
args.data_dir = args.data.lower() + "/"
args.model_dir = args.model.lower() + "/"
if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)
 
args.load_path = args.model_dir + "checkpoint_hred_user_" + args.load_name + "_" + str(args.word_size) + "_" + str(args.hidden_size) + "_" + str(args.batch_size) + '.pth.tar'
args.save_path = args.model_dir + "checkpoint_hred_user_" + args.save_name + "_" + str(args.word_size) + "_" + str(args.hidden_size) + "_" + str(args.batch_size) + '.pth.tar'
args.best_path = args.model_dir + "checkpoint_hred_user_" + args.save_name + "_best" + "_" + str(args.word_size) + "_" + str(args.hidden_size)  +  "_" + str(args.batch_size) + '.pth.tar'

print("DATA LOADING......")
data = Data.GPTDataTopic(data_dir = args.data_dir, domain_key = args.key, args = args)
print("DATA LOADING DONE!!!")
print(data.tokenizer.vocab_size)
args.vocab_size = data.tokenizer.vocab_size
print("The vocab size is {}".format(args.vocab_size))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
def init_model(args, data):
    print(args.resume)
    if args.resume:
        if args.best:
            checkpoint = torch.load(args.best_path)
            print(args.best_path)
        else: 
            checkpoint = torch.load(args.load_path)
            print(args.load_path)
        if 'args' in checkpoint:
            print("model parameters detected in checkpoint")
            load_args = checkpoint['args']
            model = Model.GPTopic(load_args)
        else:
            sys.exit("Model parameters not detected in checkpoint!!! ")
        if args.cuda:
            model = model.to(device)
        optimizer = model.load(checkpoint, args)
    else:
        model = Model.GPTopic(args)
        if args.cuda:
            model = model.to(device)
        optimizer = AdamW(params=model.parameters(), lr=args.lr, correct_bias=True)

    return data, model, optimizer
    
    

def train(epoch, start=-1):
    input_ids, response = data.get_batch(start=start)

    if args.cuda:
        input_ids = input_ids.to(device) 
        response = response.to(device)
    
    model.train()      
    
    if input_ids.shape[1]<1024:
        ploss = model(input_ids, response)
        optimizer.zero_grad()
        ploss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()

        return ploss.item(), 0
    
    else:
        return 0, 0

def validate(start):
    input_ids, response = data.get_batch(start=start, train=False)
    
    if args.cuda:
        input_ids = input_ids.to(device) 
        response = response.to(device)
    
    if input_ids.shape[1]<1024:
        model.eval()  
        ploss = model(input_ids, response)

        return ploss.item(), 0
    
    else:
        return 0, 0


set_seed(args)
data, model, optimizer = init_model(args, data)
vloss_min = 200
t0 = time.time()
num_steps_per_epoch = int(len(data.train_context)/args.batch_size)
valid_len = len(data.valid_context)
if num_steps_per_epoch > args.max_steps:
    num_steps_per_epoch = args.max_steps
for epoch in range(args.num_epochs):
    print("EPOCH : ", epoch)
    start = 0
    ploss = 0
    nloss = 0
    vploss = 0
    vnloss = 0
    niter = 0
    nviter = 0

    for step in range(0, num_steps_per_epoch):
        #Train on training data
        p, n = train(epoch+1, start=start)
        ploss += p
        nloss += n
        niter += 1
        start += args.batch_size
        
        #Validate on validation data
        if niter%args.val_step==0:
            with torch.no_grad():
                p, n = validate(nviter)
                vploss += p
                vnloss += n
            nviter += 1
        
        if niter%args.log_step==0:
            t1 = time.time()
            print(epoch, start, "PLOSS : {0:.3f}".format(ploss/niter), "{0:.3f}".format(vploss/nviter),\
                  "NLOSS : {0:.3f}".format(nloss/niter), "{0:.3f}".format(vnloss/nviter), "{0:.3f}".format(t1-t0))
            t0 = t1
            sys.stdout.flush()
            
        if niter%args.save_step==0:
            model.save(args.save_path, optimizer, args)
        
    
    print("="*50)
    model.save(args.save_path, optimizer, args)
    if vploss/nviter < vloss_min:
        vloss_min = vploss/nviter
        model.save(args.best_path, optimizer, args)


