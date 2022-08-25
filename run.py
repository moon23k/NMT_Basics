from models.seq2seq import Seq2Seq
from models.attention import Seq2SeqAttn
from models.transformer import Transformer

from modules.test import Tester
from modules.train import Trainer
from modules.inference import Translator

from modules.data import load_dataloader
from modules.utils import Config, load_model, set_seed



def main(config):
    model = load_model(config)

    if config.mode == 'train':
        train_dataloader = load_dataloader('train')
        valid_dataloader = load_dataloader('valid')        
        trainer = Trainer(model, config, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader('test')
        tester = Tester(config, model, test_dataloader)
        tester.test()
    
    elif config.mode == 'inference':
        translator = Translator(model, config)
        translator.translate()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-Task', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-scheduler', required=False)
    
    args = parser.parse_args()
    assert args.task in ['train', 'test', 'inference']
    assert args.model in ['seq2seq', 'attention', 'transformer']
 
    set_seed()
    config = Config(args)
    main(config)