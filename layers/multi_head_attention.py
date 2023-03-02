import torch
from torch import nn
from torch.nn import functional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def scaled_dot_product(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor):
    prod = torch.matmul(queries, keys.transpose(-2, -1)).to(device)
    keys_dim = keys.size(dim=-1)
    keys_dim = torch.tensor(keys_dim, dtype=torch.float32, device=device)
    scaled_prod = prod / torch.sqrt(keys_dim)
    if mask is not None:
        scaled_prod += (mask * -1e9)
    attention = torch.matmul(functional.softmax(scaled_prod, dim=-1), values)
    return attention.to(device)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        self.query_lin = nn.LazyLinear(self.d_model, device=device)
        self.key_lin = nn.LazyLinear(self.d_model, device=device)
        self.value_lin = nn.LazyLinear(self.d_model, device=device)
        self.final_lin = nn.LazyLinear(self.d_model, device=device)

    def split_proj(self, inputs: torch.Tensor):
        batch_size = inputs.size(dim=0)
        shape = (batch_size, -1, self.n_heads, self.d_head)
        splitted_inputs = torch.reshape(inputs, shape)
        return torch.permute(splitted_inputs, dims=(0, 2, 1, 3)).to(device)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor):
        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)
        queries = self.split_proj(queries)
        # split projections between heads
        queries = self.split_proj(queries)
        keys = self.split_proj(keys)
        values = self.split_proj(values)
        # apply attention
        attention = scaled_dot_product(queries, keys, values, mask)
        # get attention scores
        attention = torch.permute(attention, dims=(0, 2, 1, 3))
        batch_size = attention.size(dim=0)
        concat_attention = torch.reshape(attention, shape=(batch_size, -1, self.d_model))
        outputs = self.final_lin(concat_attention)
        return outputs
