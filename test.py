import yaml
import time
import argparse

import torch
import torch.nn as nn
from torchtext.data.metrics import bleu_score

from utils.data import get_dataloader
from utils.model import load_model
from utils.train import seq_eval, trans_eval, epoch_time
from train import Config



def run(config):
    chk_file = f"checkpoints{config.model}/train_states.pt"
	test_dataloader = get_dataloader('test', config)

    model = load_model(config)
    model_state = torch.load(f'checkpoints/{config.model}_states.pt', map_location=config.device)['model_state_dict']
    model.load_state_dict(model_state)

    #Apply label_smoothing on transformer model
    if config.model == 'transformer':
        criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx, label_smoothing=0.1).to(config.device)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx).to(config.device)

    if config.model == 'transformer':
        test_loss = trans_eval(model, test_dataloader, criterion, config)

    print(f"Test Loss: {test_loss}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    args = parser.parse_args()

    assert args.model in ['seq2seq', 'attention', 'transformer']

    config = Config(args)

    run(config)