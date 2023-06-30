
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Callable
from .dense import DenseNet
import torch
from .positional_encoder import *
from torch_scatter import scatter_mean, scatter_sum

class GaussianKer(nn.Module):
    def __init__(self, embed_size, dim, time_size) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.dim = dim - 1
        self.time_size = time_size
        self.mu = nn.Parameter(torch.randn(embed_size, self.dim))
        self.sqrt_weight = nn.Parameter(torch.randn(embed_size))
        

    def forward(self, edge_attr):
        x_attr = edge_attr[...,self.time_size:self.dim+self.time_size]
        return self.sqrt_weight**2 * \
            torch.exp(-torch.square(x_attr.unsqueeze(dim=-2).repeat(1,1,self.embed_size,1) - self.mu).sum(dim=-1))


class GaussianMean(nn.Module):
    def __init__(self, node_attr_size:int, embed_size: int, ker_width_l: int=64, dim: int=3, pos_encode=False,
                bias: bool = True, time_size=10, **kwargs):
        super(GaussianMean, self).__init__()
        self.embed_size = embed_size
        self.dim = dim
        
        self.nn = GaussianKer(embed_size, self.dim, time_size) 
                
        if bias:
            self.bias = nn.Parameter(torch.randn(embed_size))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, u_l, edge_attr, grid_size, edge_from, edge_to):
        u_l = u_l[:, edge_from]
        batch_size, node_num, time_len, embed_size = u_l.shape
        # Assume the weight is not dynamic, thus decrease the computational cost
        
        weight = self.nn(edge_attr)
        weight = weight.reshape(-1, node_num, self.embed_size)
 
        message = torch.einsum('bnti,bni->bnti', u_l, weight)
        message += (1+1e-2)*u_l
        out = torch.zeros(*grid_size, embed_size).to(u_l)
        
        out = scatter_mean(message, edge_to, dim=1, out=out)
        if self.bias is not None:
            out += self.bias
        return out
            