import torch.nn as nn
from .channel_mixer import ChannelMixer
import torch 

class SpectralMulMixer(ChannelMixer):
    def __init__(self, embed_size, resolution, k=8, *args, **kwargs):
        super(SpectralMulMixer, self).__init__(embed_size, resolution, *args, **kwargs)
        self.scale = (1 / (embed_size ** 2))
        self.dimension = len(resolution)
        self.k = k
        self.weights1 = nn.Parameter(self.scale * torch.rand(*[self.k for i in range(self.dimension)], embed_size, embed_size, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(*[self.k for i in range(self.dimension)], embed_size, embed_size, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(*[self.k for i in range(self.dimension)], embed_size, embed_size, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(*[self.k for i in range(self.dimension)], embed_size, embed_size, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bxyti,xytio->bxyto", input, weights)

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y,t)
        return torch.einsum("bxyti,xyio->bxyto", input, weights)
    
    def forward(self, x):
        # x : shape [batch_size, embed_size, resolution 1, resolution 2, ...]
        batch_size = x.shape[0]
        out_ft = torch.zeros_like(x)
        
        if self.dimension==2:
            x1 = x[:, :self.k, :self.k, :, :]
            x2 = x[:, -self.k:, :self.k, :, :]
            x3 = x[:, :self.k, -self.k:, :, :]
            x4 = x[:, -self.k:, -self.k:, :, :]
            
            out_ft[:, :self.k, :self.k, :, :] = self.compl_mul2d(x1, self.weights1)
            out_ft[:, -self.k:, :self.k, :, :] = self.compl_mul2d(x2, self.weights2)
            out_ft[:, :self.k, -self.k:, :, :] = self.compl_mul2d(x3, self.weights3)
            out_ft[:, -self.k:, -self.k:, :, :] = self.compl_mul2d(x4, self.weights4)
            
            return out_ft
        
        elif self.dimension==3:  
            x1 = x[:, :self.k, :self.k, :self.k, :]
            x2 = x[:, -self.k:, :self.k, :self.k, :]
            x3 = x[:, :self.k, -self.k:, :self.k, :]
            x4 = x[:, -self.k:, -self.k:, :self.k, :]
            
            out_ft[:, :self.k, :self.k, :self.k, :] = self.compl_mul3d(x1, self.weights1)
            out_ft[:, -self.k:, :self.k, :self.k, :] = self.compl_mul3d(x2, self.weights2)
            out_ft[:, :self.k, -self.k:, :self.k, :] = self.compl_mul3d(x3, self.weights3)
            out_ft[:, -self.k:, -self.k:, :self.k, :] = self.compl_mul3d(x4, self.weights4)
            
            return out_ft
        else:
            raise NotImplementedError('The spatial dimensionality should be 2.')