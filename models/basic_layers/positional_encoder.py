from turtle import forward
import torch.nn as nn
import torch

class TimeEncoder(nn.Module):
    def __init__(self, input_size, embed_size, max_len=31, continuous=False, *args, **kwargs):
        super(TimeEncoder, self).__init__()
        self.continuous = continuous
        self.input_size = input_size
        self.embed_size = embed_size
        if continuous == True:
            self.network = nn.Linear(input_size, embed_size//2, bias=False)
        else:
            self.embed_matrix = nn.Parameter(torch.randn(input_size, max_len, embed_size))

    def forward(self, x):
        if self.continuous:
            x = x.float()
            return torch.cat([torch.sin(self.network(x)), torch.cos(self.network(x))], dim=-1)    
        else:
            x = x.long()
            in_shape = x.shape
            embed_matrix = self.embed_matrix.reshape(*[1 for i in range(len(in_shape))[:-1]], *self.embed_matrix.shape)
            embed_matrix = embed_matrix.repeat(*in_shape[:-1], 1, 1, 1)
            x = x[..., None, None].repeat(*[1 for i in range(len(x.shape))], 1, self.embed_size)
            return torch.gather(embed_matrix, dim=-2, index=x).squeeze(dim=-2).sum(dim=-2)


class SpaEncoder(nn.Module):
    def __init__(self, input_size, embed_size, *args, **kwargs):
        super(SpaEncoder, self).__init__()

        self.network = nn.Linear(input_size, embed_size//2, bias=False)

    def forward(self, x):
        x = x.float()
        return torch.cat([torch.sin(self.network(x)), torch.cos(self.network(x))], dim=-1)

class PosEncoder(nn.Module):
    def __init__(self, spa_input_size, temp_input_size, embed_size, continuous=False, max_len=31, *args, **kwargs) -> None:
        super().__init__()
        self.space_enc = SpaEncoder(spa_input_size, embed_size)
        self.time_enc = TimeEncoder(temp_input_size, embed_size, max_len, continuous)
    
    def forward(self, x, t):
        x_enc = self.space_enc(x)
        t_enc = self.time_enc(t)
        return x_enc, t_enc.unsqueeze(dim=1)