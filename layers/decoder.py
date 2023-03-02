import torch
from torch import nn
from layers.multi_head_attention import MultiHeadAttention
from layers.positional_encoding import PositionalEncoding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DecoderLayer(nn.Module):
    def __init__(self, cnt_ffn_units: int, n_heads: int, d_model: int, dropout_rate: float):
        super(DecoderLayer, self).__init__()
        self.cnt_ffn_units = cnt_ffn_units
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.d_model = d_model
        self.multi_head_casual_attention = MultiHeadAttention(self.n_heads, d_model)
        self.dropout_1 = nn.Dropout(self.dropout_rate)
        self.norm_1 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-6, device=device)
        self.multi_head_enc_dec_attention = MultiHeadAttention(self.n_heads, d_model)
        self.dropout_2 = nn.Dropout(self.dropout_rate)
        self.norm_2 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-6, device=device)
        self.ffn_1 = nn.LazyLinear(self.cnt_ffn_units, device=device)
        self.ffn_2 = nn.LazyLinear(self.d_model, device=device)
        self.dropout_3 = nn.Dropout(self.dropout_rate)
        self.norm_3 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-6, device=device)

    def forward(self, inputs, enc_outputs, mask_1, mask_2):
        attention = self.multi_head_casual_attention(inputs, inputs, inputs, mask_1)
        attention = self.dropout_1(attention)
        attention = self.norm_1(attention + inputs)
        attention_2 = self.multi_head_enc_dec_attention(attention, enc_outputs, enc_outputs, mask_2)
        attention_2 = self.dropout_2(attention_2)
        attention_2 = self.norm_2(attention_2 + attention)
        outputs = self.ffn_1(attention_2)
        outputs = nn.functional.relu(outputs).to(device)
        outputs = self.ffn_2(outputs)
        outputs = self.dropout_3(outputs)
        outputs = self.norm_3(outputs + attention_2)
        return outputs


class Decoder(nn.Module):
    def __init__(self,
                 n_layers: int,
                 cnt_ffn_units: int,
                 n_heads: int,
                 dropout_rate: float,
                 vocab_size: int,
                 d_model: int):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_encoding = PositionalEncoding()
        self.dropout = nn.Dropout(dropout_rate)
        self.dec_layers = [DecoderLayer(cnt_ffn_units, n_heads, d_model, dropout_rate) for i in range(n_layers)]

    def forward(self, inputs, enc_outputs, mask_1, mask_2):
        outputs = self.embedding(inputs)
        d_model = torch.tensor(self.d_model, dtype=torch.float32, device=device)
        outputs *= torch.sqrt(d_model)
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs)
        for i, d in enumerate(self.dec_layers):
            outputs = d(outputs, enc_outputs, mask_1, mask_2)
        return outputs
