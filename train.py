import os
import time
import math
import yaml
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from utils.data import get_dataloader
from utils.model import load_model
from utils.scheduler import get_scheduler
from utils.train import seq_train, seq_eval, trans_train, trans_eval, epoch_time




class Config(object):
    def __init__(self, args):    
        with open('configs/model.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params = params[args.model]

            for p in params.items():
                setattr(self, p[0], p[1])

        self.model = args.model
        self.scheduler = args.scheduler
        self.pad_idx = 1
        self.batch_size = 128
        self.best_valid_loss = float('inf')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.scheduler == 'constant':
            self.learning_rate = 1e-3

        elif self.scheduler in ['noam', 'cosine_annealing_warm']:
            self.learning_rate = 1e-9

        elif self.scheduler in ['exponential', 'step']:
            self.learning_rate = 1e-2


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(attribute, ': ', value)




def run(config):
    #set checkpoint, record path
    chk_dir = f"checkpoints/"
    os.makedirs(chk_dir, exist_ok=True)
    
    chk_file = f"{config.model}_states.pt"
    record_file = f'{config.model}_record.json'
    
    chk_path = os.path.join(chk_dir, chk_file)
    record_path = os.path.join(chk_dir, record_file)
    
    #define training record dict
    train_record = defaultdict(list)    

    #Get dataloader
    train_dataloader = get_dataloader('train', config)
    valid_dataloader = get_dataloader('valid', config)

    #load model, criterion, optimizer, scheduler
    model = load_model(config)
    

    #Apply label_smoothing on transformer model
    if config.model == 'transformer':
        criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx, label_smoothing=0.1).to(config.device)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
	

    if config.scheduler != "constant":
        scheduler = get_scheduler(config.scheduler, optimizer)
    else:
        scheduler = None
    

    #train loop
    record_time = time.time()
    for epoch in range(config.n_epochs):
        start_time = time.time()

        print(f"Epoch {epoch}/{config.n_epoch}")
        if config.model == 'transformer':
            train_loss = trans_train(model, train_dataloader, criterion, optimizer, config)
            valid_loss = trans_eval(model, valid_dataloader, criterion, config)
        else:
            train_loss = seq_train(model, train_dataloader, criterion, optimizer, config)
            valid_loss = seq_eval(model, valid_dataloader, criterion, config)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Epoch {epoch}/{config.n_epoch} | Train Loss : {train_loss} / Eval Loss: {valid_loss} / Time: {epoch_mins}min {epoch_secs}sec")


        if scheduler is not None:
            scheduler.step()
        
        #save training records
        train_record['epoch'].append(epoch+1)
        train_record['train_loss'].append(train_loss)
        train_record['valid_loss'].append(valid_loss)
        train_record['lr'].append(optimizer.param_groups[0]['lr'])

        #save best model
        if valid_loss < config.best_valid_loss:
            config.best_valid_loss = valid_loss
            torch.save({'epochs': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss}, chk_path)

    train_mins, train_secs = epoch_time(record_time, time.time())
    train_record['train_time'].append(f"{train_mins}min {train_secs}sec")

    #save ppl score to train_record
    for (train_loss, valid_loss) in zip(train_record['train_loss'], train_record['valid_loss']):
        train_ppl = math.exp(train_loss)
        valid_ppl = math.exp(valid_loss)

        train_record['train_ppl'].append(round(train_ppl, 2))
        train_record['valid_ppl'].append(round(valid_ppl, 2))


    #save train_record to json file
    with open(record_path, 'w') as fp:
        json.dump(train_record, fp)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    parser.add_argument('-scheduler', default='constant', required=False)
    args = parser.parse_args()
    
    assert args.model in ['seq2seq', 'attention', 'transformer']
    assert args.scheduler in ['constant', 'noam', 'cosine_annealing_warm', 'exponential', 'step']
    
    config = Config(args)
    
    run(config)