import torch, math, time
import torch.nn as nn
from torchtext.data.metrics import bleu_score
from modules.inference import RNNSearch, TransSearch



class Tester:
    def __init__(self, config, model, test_dataloader, tokenizer):
        super(Tester, self).__init__()
        self.model = model
        self.model_name = config.model_name
        
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.device = config.device
        self.output_dim = config.output_dim
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx, label_smoothing=0.1).to(self.device)

        if self.model_name != 'transformer':
            self.search = RNNSearch(config, self.model, tokenizer)
        elif self.model_name == 'transformer':
            self.search = TransSearch(config, self.model, tokenizer)            



    def test(self):
        self.model.eval()
        tot_len = len(self.dataloader)
        tot_loss = 0.0
        
        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                src = batch['src'].to(self.device)
                trg_input = batch['trg_input'].to(self.device)
                trg_output = batch['trg_output'].to(self.device)

                if self.model_name== 'transformer':
                    logit = self.model(src, trg_input)
                else:
                    logit = self.model(src, trg_input, teacher_forcing_ratio=0.0)

                loss = self.criterion(logit.contiguous().view(-1, self.output_dim), 
                                      trg_output.contiguous().view(-1)).item()

                tot_loss += loss
            tot_loss /= tot_len
        
        print(f'Test Results on {self.model_name} model')
        print(f">> Test Loss: {tot_loss:3f} | Test PPL: {math.exp(tot_loss):2f}")


    def get_bleu_score(self, can, ref):
        return bleu_score([self.tokenizer.Decode(can).split()], 
                          [[self.tokenizer.Decode(ref).split()]])

        
    def inference_test(self):
        self.model.eval()
        batch = next(iter(self.dataloader))
        print(f'Inference Test on {self.model_name} model')

        for i in range(10):
            input_seq = self.tokenizer.decode(batch['src'][i].tolist())
            label_seq = self.tokenizer.decode(batch['trg_input'][i].tolist())

            greedy_out = self.search.greedy_search(input_seq)
            beam_out = self.search.beam_search(input_seq)

            greedy_bleu = self.get_bleu_score(greedy_out, label_seq)
            beam_bleu = self.get_bleu_score(beam_out, label_seq)
            
            print(f"\n>> Input Sequence: {input_seq}")
            print(f">> Label Sequence: {label_seq}")
            
            print(f">> Greedy Sequence: {greedy_out}")
            print(f">> Beam   Sequence : {beam_out}")
            
            print(f">> Greedy BLEU Score: {greedy_bleu}")
            print(f">> Beam   BLEU Score : {beam_bleu}")
