import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos: torch.Tensor, i: torch.Tensor, d_model: int):
        angles = 1 / torch.pow(10000., (2*(i//2)) / torch.tensor(d_model, dtype=torch.float32, device=device))
        return pos * angles

    def forward(self, inputs: torch.Tensor):
        seq_len = inputs.size(dim=-2)
        d_model = inputs.size(dim=-1)
        angles = self.get_angles(
            pos=torch.arange(seq_len)[:, None],
            i=torch.arange(d_model)[None, :],
            d_model=d_model
        )
        positional_encodings = torch.zeros_like(angles).to(device)
        positional_encodings[:, 0::2] = torch.sin(angles[:, 0::2])
        positional_encodings[:, 1::2] = torch.cos(angles[:, 1::2])

        positional_encodings = positional_encodings[None, :]

        return (inputs + positional_encodings).to(device)
