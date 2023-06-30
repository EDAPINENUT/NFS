import torch.nn as nn
from .token_mixer import TokenMixer
import torch 
import numpy as np

class ZTransMixer(TokenMixer):
    def __init__(self, embed_size, resolution, truncate=True, *args, **kwargs):
        super(ZTransMixer, self).__init__(embed_size, resolution, *args, **kwargs)
        self.dimension = len(resolution)
        self.truncate = truncate
        self.token_num = np.prod(resolution)
        if self.truncate:
            self.weights = nn.Parameter(torch.randn(self.token_num, self.token_num//2, dtype=torch.cfloat))
        else:
            self.weights = nn.Parameter(torch.randn(self.token_num, self.token_num, dtype=torch.cfloat))
            
        
    def forward(self, x):
        # x: batch_size, resolution1, resolution2, ..., embed_size
        batch_size, embed_size = x.shape[0], x.shape[-1]
        
        assert (torch.equal(torch.tensor(list(x.shape[1:-1]), dtype=torch.long), torch.tensor(self.resolution, dtype=torch.long)))
        x = x.transpose(1,-1)
        shape = x.shape
        x = x.reshape(batch_size, embed_size, -1).type(torch.cfloat)
        x_mix = torch.einsum('bcn, nm -> bcm', x, self.weights)
        x_mix = x_mix.reshape(*shape)
        x_mix = x_mix.transpose(1,-1)
        return x_mix