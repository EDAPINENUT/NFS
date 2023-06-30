import torch
import torch.nn.functional as F

def complex_relu(x, b=0.3, out_cplx=False):
    assert x.dtype == torch.cfloat
    x_norm = x.abs()
    x_relu = F.relu(x_norm - b)
    if out_cplx:
        return x_relu * x / (x_norm + 1e-7)
    else:
        return x_relu

def complex_gelu(x, b=0.3, out_cplx=False):
    assert x.dtype == torch.cfloat
    x_norm = x.abs()
    x_gelu = F.gelu(x_norm - b)
    if out_cplx:
        return x_gelu * x / (x_norm + 1e-7)
    else:
        return x_gelu
    