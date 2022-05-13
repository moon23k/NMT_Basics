import argparse
import torch
import sentencepiece as spm

from train import Config
from models.transformer.module import create_src_mask, create_trg_mask



def transformer_run(config, tokenizer, max_tokens=100):
	#Load Trained Model
	model = load_model(config)
	model_state = torch.load()
	model.load_state_dict(model_state)
	model.eval()



def seq2seq_run(config, tokenizer, max_tokens=100):
	#Load Trained Model
	model = load_model(config)
	model_state = torch.load()
	model.load_state_dict(model_state)
	model.eval()



	with torch.no_grad():
		print('Type "quit" to terminate Translation')
	    while True:
	        seq = input('\nUser Input sentence >> ')
	        if seq == 'quit':
	            print('Translator terminated!')
	            print('------------------------------------')
	            break
	        
	        #process user input to model src tensor
	        src = tokenizer.EncodeAsIds(seq)
	        if config.model == 'seq2seq':
	        	src[1:-1] = seq[-2:0:-1]

	        src = torch.tensor(src, dtype=torch.long).unsqueeze(0)
	        src = model.embedding(src)
	        enc_out = model.encoder(src, src_mask)
	        trg_indice = [tokenizer.bos_id()]

	        for _ in range(max_tokens):
	            trg_tensor = torch.tensor(trg_indice, dtype=torch.long).unsqueeze(0)
	            trg = model.embedding(trg)

                dec_out, _ = model.decoder()
                out = model.fc_out(dec_out)

	            pred_token = out.argmax(2)[:, -1].item()
	            trg_indice.append(pred_token)

	            if pred_token == tokenizer.eos_id():
	                break
	        
	        pred_seq = trg_indice[1:]
	        pred_seq = tokenizer.Decode(pred_seq)

	        print(f"Translated sentence >> {pred_seq}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    args = parser.parse_args()
    
    assert args.bert in ['seq2seq', 'attention', 'transformer']
    
    config = Config(args)
    config.device = torch.device('cpu')
	
	tokenizer = spm.SentencePieceProcessor()
	tokenizer.load('data/vocab/spm.model')
	tokenizer.SetEncodeExtraOptions('bos:eos')

	if config.model == 'transformer':
		transformer_run(config, tokenizer)
	else:
		seq2seq_run(config, tokenizer)