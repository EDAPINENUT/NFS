import torch.nn as nn

class ConvResLayer(nn.Module):
    def __init__(self, embed_size, *args, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        self.layer = nn.Identity()
    
    def forward(self, x):
        x = x.transpose(-1,1)
        return self.layer(x).transpose(1,-1)

class ConvResLayer3D(ConvResLayer):
    def __init__(self, embed_size, *args, **kwargs):
        super().__init__(embed_size)
        self.embed_size = embed_size
        self.layer = nn.Conv3d(self.embed_size, self.embed_size, 1)
    
    def forward(self, x):
        x = x.transpose(-1,1)
        return self.layer(x).transpose(1,-1)

class ConvResLayer2D(ConvResLayer):
    def __init__(self, embed_size, *args, **kwargs):
        super().__init__(embed_size)
        self.embed_size = embed_size
        self.layer = nn.Conv2d(self.embed_size, self.embed_size, 1)
    
    def forward(self, x):
        x = x.transpose(-1,1)
        return self.layer(x).transpose(1,-1)