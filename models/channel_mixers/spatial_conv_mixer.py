import torch.nn as nn
from .channel_mixer import ChannelMixer
import torch 
class SpatialConvMixer(ChannelMixer):
    def __init__(self, embed_size, resolution, truncate=True, *args, **kwargs):
        super(SpatialConvMixer, self).__init__(embed_size, resolution, *args, **kwargs)
        self.conv_resolution = list(self.resolution)
        self.conv_resolution[-1] = self.conv_resolution[-1]//2 + 1 if truncate else self.conv_resolution[-1]
        self.weight = nn.Parameter(torch.randn(*self.conv_resolution, embed_size))
        self.dimension = len(resolution)
        
    def elewise_prod3d(self, x, weight):
        return torch.einsum("bxyti,xyti->bxyti", x, weight)
    
    def elewise_prod2d(self, x, weight):
        return torch.einsum("bxti,xti->bxti", x, weight)
    
    def forward(self, x):
        # x : shape [batch_size, resolution 1, resolution 2, ..., embed_size]
        batch_size = x.shape[0]
        if self.dimension==2:
            return self.elewise_prod2d(x, self.weight)
        
        elif self.dimension==3:
            return self.elewise_prod3d(x, self.weight)
                    
        else:
            raise NotImplementedError('The spatial dimensionality should be 2.')