import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import yaml, random
import sentencepiece as spm
from torchtext.data.metrics import bleu_score

from models.seq2seq import Seq2Seq
from models.attention import Seq2SeqAttn
from models.transformer import Transformer


class Config(object):
    def __init__(self, args):    
        with open('configs.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params = params[args.model]
            for p in params.items():
                setattr(self, p[0], p[1])

        self.task = args.task
        self.model_name = args.model_name
        
        self.unk_idx = 0
        self.pad_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

        self.clip = 1
        self.n_epochs = 1
        self.batch_size = 128

        if self.task == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            if self.scheduler == 'constant':
                self.learning_rate = 1e-3
                self.scheduler = None
            elif self.scheduler == 'noam':
                self.learning_rate = 1e-3
                self.scheduler = optim.lr_scheduler()

        if self.task == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=0.1).to(self.device)

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")


def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_tokenizer(lang):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/{lang}_tokenizer.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')
    return tokenizer


def load_model(config):
    if config.model_name == 'seq2seq':
        model = Seq2Seq(config)
    elif config.model_name == 'attention':
        model = Seq2SeqAttn(config)
    elif config.model_name == 'transformer':
        model = Transformer(config)

    if config.task != 'train':
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)

    return model.to(config.device)