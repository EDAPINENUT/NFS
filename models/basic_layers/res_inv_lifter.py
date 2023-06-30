import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Callable
from .dense import DenseNet
import torch
from .positional_encoder import *

class ResInvLifter(nn.Module):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 embed_size: int, ker_width_l: int=64, dim: int=3, pos_encode=False,
                bias: bool = True, **kwargs):
        super(ResInvLifter, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.dim = dim
        
        self.nn = DenseNet([embed_size, ker_width_l, input_size * embed_size], nn.GELU) \
            if pos_encode else DenseNet([dim, ker_width_l, input_size * embed_size], nn.GELU)
        self.pos_encoder = PosEncoder(dim, embed_size) if pos_encode else nn.Identity()
        
        if bias:
            self.bias = nn.Parameter(torch.randn(embed_size))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x, grid):
        shape = x.shape
        patch_size = shape[-2]
        grid = self.pos_encoder(grid)
        weight = self.nn(grid)
        weight = weight.reshape(-1, patch_size, self.input_size, self.embed_size)
        x = x.reshape(-1, patch_size, shape[-1])
        out = torch.einsum('bpi,bpio->bo', x, weight).reshape(*shape[:-2], self.embed_size)
        return out
            