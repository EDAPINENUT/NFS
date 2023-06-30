import torch.nn as nn
from .channel_mixer import ChannelMixer
import torch 
class MLPChannelMixer(ChannelMixer):
    def __init__(self, embed_size, resolution, *args, **kwargs):
        super(MLPChannelMixer, self).__init__(embed_size, resolution, *args, **kwargs)
        self.mlp = nn.Linear(embed_size, embed_size)
        
    def forward(self, x):
        # x : shape [batch_size, resolution 1, resolution 2, ..., embed_size]
        batch_size = x.shape[0]
        assert (x.shape[-1] == self.embed_size)
        
        x_mlp = self.mlp(x)
        return x_mlp