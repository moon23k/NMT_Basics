import time
import argparse

import torch
import torch.nn as nn

from utils.data import get_dataloader
from utils.model import load_model
from utils.train import seq_eval, trans_eval, epoch_time
from train import Config




def run(config):
    chk_file = f"checkpoints/{config.model}_states.pt"
	test_dataloader = get_dataloader('test', config)

    model = load_model(config)
    model_state = torch.load(f'checkpoints/{config.model}_states.pt', map_location=config.device)['model_state_dict']
    model.load_state_dict(model_state)
    
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx).to(config.device)

    start_time = time.time()
    if config.model == 'transformer':
        test_loss, test_bleu = trans_eval(model, test_dataloader, criterion, config, bleu=True)
    else:
        test_loss, test_bleu = train_eval(model, test_dataloader, criterion, config, bleu=True)

    end_time = time.time()
    test_mins, test_secs = epoch_time(start_time, end_time)
    
    print(f"Test Loss: {test_loss} / Bleu Score: {test_bleu} / Time: {test_mins}min {test_secs}sec")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    args = parser.parse_args()

    assert args.model in ['seq2seq', 'attention', 'transformer']

    config = Config(args)

    run(config)