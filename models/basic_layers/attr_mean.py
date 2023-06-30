import imp
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Callable
from .dense import DenseNet
import torch
from .positional_encoder import *
from torch_scatter import scatter_mean, scatter_sum

class AttrMean(nn.Module):
    def __init__(self, node_attr_size:int, embed_size: int, ker_width_l: int=64, dim: int=3, pos_encode=False,
                bias: bool = True, **kwargs):
        super(AttrMean, self).__init__()
        self.embed_size = embed_size
        self.dim = dim
        
        self.nn = DenseNet([node_attr_size, ker_width_l, embed_size * embed_size], nn.GELU) 
                
        if bias:
            self.bias = nn.Parameter(torch.randn(embed_size))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, u_l, edge_attr, grid_size, edge_from, edge_to):
        u_l = u_l[:, edge_from]
        batch_size, node_num, time_len, embed_size = u_l.shape
        # Assume the weight is not dynamic, thus decrease the computational cost
        
        weight = self.nn(edge_attr)
        weight = weight.reshape(-1, node_num, self.embed_size, self.embed_size)
 
        message = torch.einsum('bnti,bnio->bnto', u_l, weight)
        message += (1+1e-2)*u_l
        out = torch.zeros(*grid_size, embed_size).to(u_l)
        
        out = scatter_mean(message, edge_to, dim=1, out=out)
        if self.bias is not None:
            out += self.bias
        return out
            