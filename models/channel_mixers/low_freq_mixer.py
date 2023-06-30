import torch.nn as nn
from .channel_mixer import ChannelMixer
import torch 

class LowFreqMixer(ChannelMixer):
    def __init__(self, embed_size, resolution, k=8, time_k=None, *args, **kwargs):
        super(LowFreqMixer, self).__init__(embed_size, resolution, *args, **kwargs)
        self.scale = (1 / (embed_size ** 2))
        self.dimension = len(resolution)
        self.k = k
        self.time_k = time_k
        if time_k==None:
            self.time_k = self.resolution[-1]//2 + 1
        self.param_shape = [self.k for k in range(self.dimension-1)]
        self.param_shape.append(self.time_k)

        for i in range(2**(self.dimension-1)):
            self.register_parameter('weights{}'.format(i+1), nn.Parameter(self.scale *torch.rand(*self.param_shape, embed_size, embed_size, dtype=torch.cfloat)))

        # self.weight_dict = {'weights{}'.format(i+1): nn.Parameter(self.scale * torch.rand(*[self.k for i in range(self.dimension)], embed_size, embed_size, dtype=torch.cfloat)) for i in range(2**(self.dimension-1))}

        # if self.dimension == 3:
        #     self.weights1 = nn.Parameter(self.scale * torch.rand(*[self.k for i in range(self.dimension)], embed_size, embed_size, dtype=torch.cfloat))
        #     self.weights2 = nn.Parameter(self.scale * torch.rand(*[self.k for i in range(self.dimension)], embed_size, embed_size, dtype=torch.cfloat))
        #     self.weights3 = nn.Parameter(self.scale * torch.rand(*[self.k for i in range(self.dimension)], embed_size, embed_size, dtype=torch.cfloat))
        #     self.weights4 = nn.Parameter(self.scale * torch.rand(*[self.k for i in range(self.dimension)], embed_size, embed_size, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bxyti,xytio->bxyto", input, weights)

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y,t)
        return torch.einsum("bxti,xtio->bxto", input, weights)
    
    def forward(self, x):
        # x : shape [batch_size, embed_size, resolution 1, resolution 2, ...]
        batchsize = x.shape[0]
        out_ft = torch.zeros_like(x)
        
        if self.dimension==2:
            x1 = x[:, :self.k, :self.time_k, :]
            x2 = torch.conj(x[:, :self.k, :self.time_k, :])
            
            out_ft[:, :self.k, :self.time_k, :] = self.compl_mul2d(x1, self.weights1)
            out_ft[:, -self.k:, :self.time_k, :] = self.compl_mul2d(x2, self.weights2)
            
            return out_ft
        
        elif self.dimension==3:  
            x1 = x[:, :self.k, :self.k, :self.time_k, :]
            x2 = -x[:, :self.k, :self.k, :self.time_k, :]
            x3 = torch.conj(x[:, :self.k, :self.k, :self.time_k, :])
            x4 = -torch.conj(x[:, :self.k, :self.k, :self.time_k, :])
            
            out_ft[:, :self.k, :self.k, :self.time_k, :] = self.compl_mul3d(x1, self.weights1)
            out_ft[:, -self.k:, :self.k, :self.time_k, :] = self.compl_mul3d(x2, self.weights2)
            out_ft[:, :self.k, -self.k:, :self.time_k, :] = self.compl_mul3d(x3, self.weights3)
            out_ft[:, -self.k:, -self.k:, :self.time_k, :] = self.compl_mul3d(x4, self.weights4)
            
            return out_ft
        else:
            raise NotImplementedError('The other need to be extended by your self here.')