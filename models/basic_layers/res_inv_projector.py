import imp
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Callable
from .dense import DenseNet
import torch
from .positional_encoder import *

class ResInvProjector(nn.Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]],
                 embed_size: int, ker_width_l: int=64, dim: int=3, pos_encode=False,
                bias: bool = True, **kwargs):
        super(ResInvProjector, self).__init__()
        self.output_size = output_size
        self.embed_size = embed_size
        self.dim = dim
        
        self.nn = DenseNet([embed_size, ker_width_l, output_size * embed_size], nn.GELU) \
            if pos_encode else DenseNet([dim, ker_width_l, output_size * embed_size], nn.GELU)
        self.pos_encoder = PosEncoder(dim, embed_size) if pos_encode else nn.Identity()
        
        if bias:
            self.bias = nn.Parameter(torch.randn(embed_size))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x, grid):
        shape = x.shape
        grid = self.pos_encoder(grid)
        weight = self.nn(grid)
        weight = weight.reshape(-1, self.embed_size, self.output_size)
        x = x.reshape(-1, shape[-1])
        out = torch.einsum('bi,bio->bo', x, weight).reshape(*shape[:-1], self.output_size)
        return out
            