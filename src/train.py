"""
Training script of MetaChrom model
"""
__author__ = "Ben Lai"
__copyright__ = "Copyright 2020, TTIC"
__license__ = "GNU GPLv3"

import os
import sys
import torch
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import glob
import argparse
from torch.utils import data as D
from model import MetaChrom, data_loader

def train_model(train_params, epoch, model_path):
    model.train()
    torch.cuda.set_device(train_params['device'])
    running_loss = []
    pbar =  tqdm(total=len(train_params['train_loader']))

    loss_fn = train_params['loss_fn']
    train_loader = train_params['train_loader']
    val_loader = train_params['val_loader']
    optimizer = train_params['optim']

    for (id, train_batch, label_batch) in train_loader:
        train_batch, label_batch =  train_batch.cuda(), label_batch.cuda()
        out_batch = model(train_batch)
        loss = loss_fn(out_batch,label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        pbar.update()
        pbar.set_description(str(epoch) + '/'  + str(train_params['training_epoch']) + '  ' + str(np.mean(running_loss))[:6])

    tmp_loss = []
    for (id, val_batch, label_batch) in val_loader:
        val_batch, label_batch =  val_batch.cuda(), label_batch.cuda()
        out_batch = model(val_batch)
        loss = loss_fn(out_batch,label_batch) 
        tmp_loss.append(loss.item())
    pbar.set_description(str(epoch) + '/' + str(np.mean(running_loss))[:6])

    model_states = {"epoch" : epoch, "state_dict": model.state_dict(), "optimizer":optimizer.state_dict(), "loss":running_loss}
    torch.save(model_states, model_path)

if __name__ == "__main__":
    print('------------Starting MetaChrom Training------------' + '\n')

    parser = argparse.ArgumentParser(description='Training script for MetaChrom')
    parser.add_argument('--Device', type=int, help='CUDA device for training', default=0)
    parser.add_argument('--lr', type=float,  help='Learning rate for the optimizer',default=1e-3)
    parser.add_argument('--BatchSize', help='Size of the minibatch' ,type=int, default=256)
    parser.add_argument('--DataDir', help='Directory contains the training sequence and label', type=str)
    parser.add_argument('--ModelOut', help='Destination for saving the trained model' ,type=str)
    parser.add_argument('--NumTarget', help='Number of epigenomic features', type = int)
    parser.add_argument('--BaseModel', help='Pre-trained meta-data-extractor model for the MetaChrom', type = str)
    parser.add_argument('--Epoch', help='Number of training epochs', type=int, default=50)
    args = parser.parse_args()

    if args.DataDir == None:
        print('Error: Please provide data directory')
        exit(1)
    elif not os.path.isfile(os.path.join(args.DataDir,'train.seq')):
        print('Error: Train.seq not found')
        exit(1)
    elif not os.path.isfile(os.path.join(args.DataDir,'labels.pt')):
        print('Error: labels.pt not found')
        exit(1)
    else:
        pass
    
    if args.ModelOut == None:
        print('Error: Please provide model output directory')
    elif os.path.isdir(os.path.join(args.ModelOut, '')):
        pass
    else:
        cmd = 'mkdir -p ' + os.path.join(args.ModelOut, '')
        os.system(cmd)
    

    #preparing the data loader
    Dset = data_loader.seq_data(seq_path=os.path.join(args.DataDir,'train.seq'), 
                                label_path= os.path.join(args.DataDir,'labels.pt'),
                                training_mode=True)
    train_len = int(Dset.len * 0.9)
    val_len = int(Dset.len - train_len)
    train, val = D.random_split(Dset, lengths = [train_len, val_len])
    train_loader = D.DataLoader(train, batch_size = args.BatchSize, num_workers = 0)
    val_loader = D.DataLoader(val, batch_size = args.BatchSize, num_workers=0)
    #create the model
    torch.cuda.set_device(args.Device)
    model = MetaChrom.MetaChrom(num_target=args.NumTarget, load_base=True, base_path=args.BaseModel).cuda()
    #prepare the optimizer
    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    loss_fn = torch.nn.BCELoss()

    train_param_dict = {
        'model':model,
        'optim':optimizer,
        'loss_fn':loss_fn,
        'train_loader':train_loader,
        'val_loader':val_loader,
        'device':args.Device,
        'training_epoch': args.Epoch
    }

    print('DataDir: ' + args.DataDir)
    print('Number of Training Sequence: ' + str(train_len))
    print('Batch Size: ' + str(args.BatchSize))
    print('Learning Rate: ' +  str(args.lr))
    print('Number of Epochs: ' + str(args.Epoch))
    print('Saving trained model at: ' + args.ModelOut)

    for epoch in range(1,args.Epoch):
        model_path = args.ModelOut
        train_model(train_params = train_param_dict, epoch = epoch ,model_path = model_path)
