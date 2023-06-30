import torch.nn as nn

class Identity(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        
        return x
