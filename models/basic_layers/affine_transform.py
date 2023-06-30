from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch

class AffTrans(nn.Module):
    def __init__(self, 
                 embed_size, dtype=torch.float32):
        super().__init__()
        try:
            self.layer = nn.Linear(embed_size, embed_size, bias=True, dtype=dtype)
        except:
            self.layer = nn.Linear(embed_size, embed_size, bias=True)

    
    def forward(self, x):
        return self.layer(x)