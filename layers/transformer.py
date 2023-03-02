import torch
from torch import nn
from layers.encoder import Encoder
from layers.decoder import Decoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size_enc: int,
                 vocab_size_dec: int,
                 d_model: int,
                 n_layers: int,
                 cnt_ffn_units: int,
                 n_heads: int,
                 dropout_rate: float):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n_layers, cnt_ffn_units, n_heads, dropout_rate, vocab_size_enc, d_model)
        self.decoder = Decoder(n_layers, cnt_ffn_units, n_heads, dropout_rate, vocab_size_dec, d_model)
        self.last_linear = nn.LazyLinear(vocab_size_dec, device=device)

    def create_padding_mask(self, seq: torch.Tensor):
        mask = torch.eq(seq, 0).float()
        return mask[:, None, None, :].to(device)

    def create_look_ahead_mask(self, seq: torch.Tensor):
        seq_len = seq.size(dim=1)
        look_ahead_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
        return look_ahead_mask.to(device)

    def forward(self, enc_inputs, dec_inputs):
        enc_mask = self.create_padding_mask(enc_inputs)
        dec_mask = torch.max(self.create_padding_mask(dec_inputs), self.create_look_ahead_mask(dec_inputs)).to(device)
        dec_mask_2 = self.create_padding_mask(enc_inputs)
        enc_outputs = self.encoder(enc_inputs, enc_mask)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, dec_mask, dec_mask_2)
        outputs = self.last_linear(dec_outputs)
        return outputs
