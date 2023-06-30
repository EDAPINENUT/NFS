import torch.nn as nn
from .token_mixer import TokenMixer
import torch 
class MLPTokenMixer(TokenMixer):
    def __init__(self, embed_size, resolution, *args, **kwargs):
        super(MLPTokenMixer, self).__init__(embed_size, resolution, *args, **kwargs)
        self.num_token = torch.prod(torch.tensor(resolution)).item()
        self.dimension = len(resolution)
        self.mlp = nn.Linear(self.num_token, self.num_token)
        
    def forward(self, x):
        # x : shape [batch_size, resolution 1, resolution 2, ..., embed_size]
        input_shape = x.shape
        batch_size, resolution_t, embed_size = x.shape[0], x.shape[-2], x.shape[-1]
        
        x = x.reshape(batch_size, -1, embed_size)
        x = x.transpose(1,-1)        
        x_mlp = self.mlp(x)
        x_mlp = x_mlp.transpose(1,-1)
        x_mlp = x_mlp.reshape(*input_shape)

        return x_mlp