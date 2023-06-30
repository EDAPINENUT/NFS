from turtle import forward
import torch.nn as nn
import torch

class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        self.real_norm = nn.LayerNorm(normalized_shape=normalized_shape, eps=eps)
        self.imag_norm = nn.LayerNorm(normalized_shape=normalized_shape, eps=eps)
    
    def forward(self, x):
        assert x.dtype == torch.cfloat
        y = torch.zeros_like(x)
        y.real = self.real_norm(x.real)
        y.imag = self.imag_norm(x.imag)
        
        return y