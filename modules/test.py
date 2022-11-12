import torch, math, time
import torch.nn as nn
from torchtext.data.metrics import bleu_score
from modules.inference import RNNSearch, TransSearch



class Tester:
    def __init__(self, config, model, test_dataloader, tokenizer):
        super(Tester, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.device = config.device
        self.output_dim = config.output_dim
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx, label_smoothing=0.1).to(self.device)

        if config.model_name != 'transformer':
            self.search = RNNSearch(config, self.model, tokenizer)
        elif config.model_name == 'transformer':
            self.search = TransSearch(config, self.model, tokenizer)            


    def get_bleu_score(self, pred, trg):
        score = 0
        batch_size = trg.size(0)
        
        for can, ref in zip(pred, trg.tolist()):
            score += bleu_score([self.tokenizer.Decode(can).split()],
                                [[self.tokenizer.Decode(ref).split()]])
        return (score / batch_size) * 100

    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def test(self):
        self.model.eval()
        tot_len = len(self.dataloader)
        tot_loss, tot_greedy_bleu, tot_beam_bleu = 0.0, 0.0, 0.0
        start_time = time.time()
        
        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                src = batch['src'].to(self.device)
                trg_input = batch['trg_input'].to(self.device)
                trg_output = batch['trg_output'].to(self.device)

                if self.model_name== 'transformer':
                    logit = self.model(src, trg_input)
                    greedy_pred = self.search.greedy_search(src)
                    beam_pred = self.search.beam_search(src)

                else:
                    logit = self.model(src, trg_input, teacher_forcing_ratio=0.0)
                    greedy_pred = logit.argmax(-1).tolist()
                    beam_pred = self.search.beam_search(src)

                loss = self.criterion(logit.contiguous().view(-1, self.output_dim), 
                                      trg_output.contiguous().view(-1)).item()


                beam_bleu = self.get_bleu_score(beam_pred, trg)    
                greedy_bleu = self.get_bleu_score(greedy_pred, trg)

                tot_loss += loss
                tot_beam_bleu += beam_bleu
                tot_greedy_bleu += greedy_bleu

        tot_loss /= tot_len
        tot_beam_bleu /= tot_len
        tot_greedy_bleu /= tot_len
        
        print(f'Test Results on {self.model_name} model | Time: {self.measure_time(start_time, time.time())}')
        print(f">> Test Loss: {tot_loss:3f} | Test PPL: {math.exp(tot_loss):2f}")
        print(f">> Greedy BLEU: {tot_greedy_bleu:2f} | Beam BLEU: {tot_beam_bleu:2f}")


'''
Test에서도 

 General  Test 
Inference Test
'''        