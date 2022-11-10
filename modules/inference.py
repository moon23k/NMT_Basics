import torch
import sentencepiece as spm
from run import load_tokenizer
from modules.search import SeqSearch, AttnSearch, TransSearch



class Translator:
	def __init__(self, config, model, tokenizer):
		self.model = model
		self.device = config.device
		self.search = config.search
		self.bos_idx = config.bos_idx
		self.model_name == config.model_name
		self.tokenizer = tokenizer

        if self.model.training:
            self.model.eval()

        if self.model_name == 'seq2seq':
            self.beam = SeqSearch(config, self.model)
        elif self.model_name == 'attention':
            self.beam = AttnSearch(config, self.model)
        elif self.model_name == 'transformer':
            self.beam = TransSearch(config, self.model)            

	
	def tranaslate(self, config):
		print('Type "quit" to terminate Translation')
		while True:
			user_input = input('Please Type Text >> ')
			if user_input.lower() == 'quit':
				print('--- Terminate the Translation ---')
				print('-' * 30)
				break

			src = self.src_tokenizer.Encode(user_input)
			src = torch.LongTensor(src).unsqueeze(0).to(self.device)

            if self.search == 'beam':
                pred_seq = self.search.beam_search(src)
            elif self.search == 'greedy':
                pred_seq = self.search.greedy_search(src)

			print(f"Original   Sequence: {user_input}")
			print(f'Translated Sequence: {self.trg_tokenizer.Decode(pred_seq)}\n')
			
