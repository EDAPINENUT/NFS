import torch.nn as nn
from .token_mixer import TokenMixer
import torch 
class FourierMixer(TokenMixer):
    def __init__(self, embed_size, resolution, truncate=True, *args, **kwargs):
        super(FourierMixer, self).__init__(embed_size, resolution, *args, **kwargs)
        self.dimension = len(resolution)
        self.fft_dim = [-(i+2) for i in range(self.dimension)]
        self.fft_dim.reverse()
        self.truncate = truncate
        
    def forward(self, x):
        # x: batch_size, resolution1, resolution2, ..., embed_size
        batch_size = x.shape[0]
        assert (torch.equal(torch.tensor(list(x.shape[1:-1]), dtype=torch.long), torch.tensor(self.resolution, dtype=torch.long)))
        if self.truncate:
            x_ft = torch.fft.rfftn(x, dim=self.fft_dim)
        else:
            x_ft = torch.fft.fftn(x, dim=self.fft_dim)
        return x_ft
    
class FourierDemixer(TokenMixer):
    def __init__(self, embed_size, resolution, *args, **kwargs):
        super(FourierDemixer, self).__init__(embed_size, resolution, *args, **kwargs)
        self.dimension = len(resolution)
        self.fft_dim = [-(i+2) for i in range(self.dimension)]
        self.fft_dim.reverse()
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.irfftn(x, dim=self.fft_dim, s=self.resolution)
        return x_ft