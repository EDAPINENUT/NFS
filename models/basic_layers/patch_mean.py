from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch
# from einops.layers.torch import Rearrange, Reduce
import numpy as np

class PatchMean(nn.Module):
    def __init__(self, patch_size, embed_size, to_patch=True) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.to_patch = to_patch
        
        rearrange_str_from = ['(x{} p{})'.format(i+1, i+1) for i in range(len(patch_size))]
        rearrange_str_from = ' '.join(rearrange_str_from)
        rearrange_str_from = 'b ' + rearrange_str_from + ' c'
        
        rearrange_str_to1 = ' '.join(['x{}'.format(i+1, i+1) for i in range(len(patch_size))])
        rearrange_str_to2 = ' '.join(['p{}'.format(i+1, i+1) for i in range(len(patch_size))])
        rearrange_str_to = 'b ' + rearrange_str_to1 + ' (' + rearrange_str_to2 +' c)' 
        
        if to_patch:
            rearrange_str = rearrange_str_from + ' -> ' + rearrange_str_to
            self.mlp = nn.Sequential(nn.Linear(np.prod(self.patch_size)*embed_size, embed_size), 
                                                nn.GELU())
        else:
            rearrange_str = rearrange_str_to + ' -> ' + rearrange_str_from
            self.mlp = nn.Sequential(nn.Linear(embed_size, np.prod(self.patch_size)*embed_size), 
                                                nn.GELU())
        
        rearrange_key = ['p{}'.format(i+1) for i in range(len(patch_size))]
        rearrange_dict = {rearrange_key[i]: patch_size[i] for i in range(len(patch_size))}
        rearrange_dict['pattern'] = rearrange_str
        
        self.rearrange = Rearrange(**rearrange_dict)
        
    def forward(self, x, *args, **kwargs):
        if self.to_patch:
            x = self.rearrange(x)
            x = self.mlp(x)
        else:
            x = self.mlp(x)
            x = self.rearrange(x)
        return x

