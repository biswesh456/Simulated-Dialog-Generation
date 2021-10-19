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
parser.add_argument('--val-step', type=int, default=4, metavar='ES', help='how many batches to wait before evaluating 1 test batch')
parser.add_argument('--log-step', type=int, default=1000, metavar='LS', help='how many batches to wait before logging training status')
parser.add_argument('--save-step', type=int, default=5000, metavar='SS', help='how many batches to wait before saving model')
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
parser.add_argument("--aug-type", type=str, default="none", help="plain or augmented")


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


# class args:
#     batch_size = 10
#     no_cuda = False
#     vocab_size=10000
#     maxlen=50
#     word_size = 300
#     hidden_size = 1000
#     resume = False
#     lr = .00015
#     data = 'ibm_5000'
#     val_step = 3
#     log_step = 10
#     save_step = 500
#     base_dir = '/dccstor/gpandey11/gaurav/'
#     num_epochs = 25
#     model = 'future_ibm_cr'
#     data = 'future_ibm_cr'
#     mname = 'future'
#     best = True
#     train = False
    
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.data_dir = args.data.lower() + "/"
args.model_dir = args.model.lower() + "/"
if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)
 
args.load_path = args.model_dir + "checkpoint_hred_user_" + args.load_name + "_" + str(args.word_size) + "_" + str(args.hidden_size)  + "_" + str(args.lr) + "_" + str(args.batch_size) + '.pth.tar'
args.save_path = args.model_dir + "checkpoint_hred_user_" + args.save_name + "_" + str(args.word_size) + "_" + str(args.hidden_size)  + "_" + str(args.lr) + "_" + str(args.batch_size) + '.pth.tar'
args.best_path = args.model_dir + "checkpoint_hred_user_" + args.save_name + "_best" + "_" + str(args.word_size) + "_" + str(args.hidden_size)  + "_" + str(args.lr) + "_" + str(args.batch_size) + '.pth.tar'

print("DATA LOADING......")
if args.aug_type == 'augmented':
    data = Data.GPTDataAll(data_dir = args.data_dir, args = args)
elif args.aug_type == 'plain':
    data = Data.GPTData(data_dir = args.data_dir, domain_key = args.key)

print("DATA LOADING DONE!!!")
print(data.tokenizer.vocab_size)
args.vocab_size = data.tokenizer.vocab_size
print("The vocab size is {}".format(args.vocab_size))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def init_model(args, data):
    print(args.resume)
    if args.resume:
        if args.best:
            print(args.best_path)
            checkpoint = torch.load(args.best_path)
        else:     
            print(args.load_path)
            checkpoint = torch.load(args.load_path)
        if 'args' in checkpoint:
            print("model parameters detected in checkpoint")
            load_args = checkpoint['args']
            model = Model.GPT(load_args)
        else:
            sys.exit("Model parameters not detected in checkpoint!!! ")
        if args.cuda:
            model = model.to(device)
        optimizer = model.load(checkpoint, args)
    else:
        model = Model.GPT(args)
        if args.cuda:
            model = model.to(device)
        optimizer = AdamW(params=model.parameters(), lr=args.lr, correct_bias=True)

    return data, model, optimizer
    
    

def train(epoch, start=-1):
    input_ids, response_id, adverse_input_ids, adverse_id = data.get_batch(start=start)

    if args.cuda:
        input_ids = input_ids.to(device) 
        response_id = response_id.to(device)
        adverse_input_ids = adverse_input_ids.to(device)
        adverse_id = adverse_id.to(device)
    
    model.train()      
    if input_ids.shape[1]<1024:
        ploss = model(input_ids, response_id)
        optimizer.zero_grad()
        ploss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        
        return ploss.item(), 0
    
#     nloss = model(adverse_input_ids, adverse_id)
#     if epoch > 2:
#         nloss = torch.clamp(-nloss, min=-4) 
#         optimizer.zero_grad()
#         nloss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
#         optimizer.step()
    
    return 0, 0

def validate(start):
    input_ids, response_id, adverse_input_ids, adverse_id = data.get_batch(start=start, train=False)
    
    if args.cuda:
        input_ids = input_ids.to(device) 
        response_id = response_id.to(device)
        adverse_input_ids = adverse_input_ids.to(device)
        adverse_id = adverse_id.to(device)
    
    model.eval() 
    if input_ids.shape[1]<1024:
        ploss = model(input_ids, response_id)
        return ploss.item(), 0
#     nloss = model(adverse_input_ids, adverse_id)
    
    return 0, 0



data, model, optimizer = init_model(args, data)
vloss_min = 200
t0 = time.time()
num_steps_per_epoch = int(len(data.train_context)/args.batch_size)
if num_steps_per_epoch > args.max_steps:
    num_steps_per_epoch = args.max_steps

valid_count = 0
learning_rate=args.lr
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
        p, n = train(epoch, start=start)
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
            valid_count = 0
#         else:
#             if valid_count < 6:
#                 learning_rate = 0.5 * learning_rate
#                 valid_count += 1
#                 for g in optimizer.param_groups:
#                     g['lr'] = learning_rate
#             else:
#                 break
        
    
    print("="*50)
    model.save(args.save_path, optimizer, args)
    if vploss/nviter < vloss_min:
        vloss_min = vploss/nviter
        model.save(args.best_path, optimizer, args)

