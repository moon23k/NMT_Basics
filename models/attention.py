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
                          batch_first=False)
        self.fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        out, hidden = self.rnn(embedded)
        
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = torch.tanh(self.fc(hidden))

        return out, hidden



class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.attn = nn.Linear((config.hidden_dim * 3), config.hidden_dim)
        self.v = nn.Linear(config.hidden_dim, 1, bias=False)


    def forward(self, hidden, enc_out):
        src_len, batch_size, _ = enc_out.shape
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out.permute(1, 0, 2)], dim=2)))
        attn_value = self.v(energy).squeeze(2)

        return F.softmax(attn_value, dim=1) #[batch_size, seq_len]



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.output_dim = config.output_dim
        self.attention = Attention(config)
        self.emb = nn.Embedding(config.output_dim, config.emb_dim)
        self.rnn = nn.GRU((config.hidden_dim * 2) + config.emb_dim, config.hidden_dim, batch_first=False)
        self.fc_out = nn.Linear((config.hidden_dim * 3) + config.emb_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)
        

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.dropout(self.emb(x.unsqueeze(0)))
        attn_value = self.attention(hidden, encoder_outputs).unsqueeze(1)
        
        weighted = torch.bmm(attn_value, encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)    
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        return prediction, hidden.squeeze(0)


        
class Seq2SeqAttn(nn.Module):
    def __init__(self, config):
        super(Seq2SeqAttn, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.device = config.device
        self.output_dim = config.output_dim


    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        max_len, batch_size = trg.shape
        outputs = torch.ones(max_len, batch_size, self.output_dim).to(self.device)

        dec_input = trg[0, :]
        enc_out, hidden = self.encoder(src)

        for t in range(1, max_len):
            out, hidden = self.decoder(dec_input, hidden, enc_out)
            outputs[t] = out

            pred = out.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[t] if teacher_force else pred
        
        return outputs[1:, :].contiguous()

