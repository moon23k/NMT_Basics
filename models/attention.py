import random
import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.input_dim, config.emb_dim)
        self.rnn = nn.GRU(config.emb_dim, 
                          config.hidden_dim, 
                          bidirectional=True, 
                          batch_first=True)
        self.fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        out, hidden = self.rnn(embedded)
        
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        hidden = torch.tanh(self.fc(hidden))

        return out, hidden



class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.attn = nn.Linear((config.hidden_dim * 3), config.hidden_dim)
        self.v = nn.Linear(config.hidden_dim, 1, bias=False)


    def forward(self, hidden, enc_out):
        hidden = hidden.unsqueeze(1).repeat(1, enc_out.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out], dim=2)))
        attn_value = self.v(energy).squeeze(2)
        return F.softmax(attn_value, dim=1) #[batch_size, seq_len]



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.output_dim = config.output_dim
        self.attention = Attention(config)
        self.emb = nn.Embedding(config.output_dim, config.emb_dim)
        self.rnn = nn.GRU((config.hidden_dim * 2) + config.emb_dim, 
                          config.hidden_dim, 
                          batch_first=True)
        self.fc_out = nn.Linear((config.hidden_dim * 3) + config.emb_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)
        

    def forward(self, x, enc_out, hidden):
        embedded = self.dropout(self.emb(x.unsqueeze(1)))
        attn_value = self.attention(hidden, enc_out).unsqueeze(1)
        weighted = torch.bmm(attn_value, enc_out)
        rnn_input = torch.cat((embedded, weighted), dim=2)    
        out, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        hidden = hidden.squeeze(0)
        out = out.permute(1, 0, 2).squeeze(0)
        assert (out == hidden).all()
        
        embedded = embedded.permute(1, 0, 2).squeeze(0)
        weighted = weighted.permute(1, 0, 2).squeeze(0)
        
        pred = self.fc_out(torch.cat((out, weighted, embedded), dim=1))
        return pred, hidden.squeeze(0)


        
class Seq2SeqAttn(nn.Module):
    def __init__(self, config):
        super(Seq2SeqAttn, self).__init__()

        self.device = config.device
        self.output_dim = config.output_dim
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)


    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, max_len = trg.shape
        outputs = torch.zeros(max_len, batch_size, self.output_dim).to(self.device)

        dec_input = trg[:, 0]
        enc_out, hidden = self.encoder(src)

        for t in range(1, max_len):
            out, hidden = self.decoder(dec_input, enc_out, hidden)
            outputs[t] = out

            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[:, t] if teacher_force else out.argmax(1)
        
        return outputs.contiguous().permute(1, 0, 2)[:, 1:]
