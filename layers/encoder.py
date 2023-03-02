import torch
from torch import nn
from layers.multi_head_attention import MultiHeadAttention
from layers.positional_encoding import PositionalEncoding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderLayer(nn.Module):
    def __init__(self, cnt_ffn_units: int, n_heads: int, d_model: int, dropout_rate: float):
        super(EncoderLayer, self).__init__()
        self.cnt_ffn_units = cnt_ffn_units
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.d_model = d_model
        self.multi_head_attention = MultiHeadAttention(self.n_heads, self.d_model)
        self.dropout_1 = nn.Dropout(self.dropout_rate)
        self.norm_1 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-6, device=device)
        self.ffn_1 = nn.LazyLinear(out_features=self.cnt_ffn_units, device=device)
        self.ffn_2 = nn.LazyLinear(out_features=self.d_model, device=device)
        self.dropout_2 = nn.Dropout(self.dropout_rate)
        self.norm_2 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-6, device=device)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        attention = self.multi_head_attention(inputs, inputs, inputs, mask)
        attention = self.dropout_1(attention)
        attention = self.norm_1(attention+inputs)
        outputs = self.ffn_1(attention)
        outputs = nn.functional.relu(outputs).to(device)
        outputs = self.ffn_2(outputs)
        outputs = self.dropout_2(outputs)
        outputs = self.norm_2(outputs + attention)

        return outputs


class Encoder(nn.Module):
    def __init__(
            self,
            cnt_layers: int,
            cnt_ffn_units: int,
            n_heads: int,
            dropout_rate: float,
            vocab_size: int,
            d_model: int):
        super(Encoder, self).__init__()
        self.cnt_layers = cnt_layers
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_encoding = PositionalEncoding()
        self.dropout = nn.Dropout(dropout_rate)
        self.enc_layers = [EncoderLayer(cnt_ffn_units, n_heads, d_model, dropout_rate) for i in range(cnt_layers)]

    def forward(self, inputs, mask):
        outputs = self.embedding(inputs).to(device)
        outputs *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=device))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs)
        for i, e in enumerate(self.enc_layers):
            outputs = e(outputs.to(device), mask.to(device))
        return outputs
