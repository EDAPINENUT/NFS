import torch.nn as nn
import torch.nn.functional as F

class MLPLifter(nn.Module):
    def __init__(self, 
                 input_size,
                 embed_size,
                 inter_sizes=None,
                 dim=None,
                 activation=F.gelu):
        super().__init__()
        self.layers = []
        self.activation = activation
        if inter_sizes is not None:
            assert (isinstance(inter_sizes, tuple) or isinstance(inter_sizes, list) or isinstance(inter_sizes, int))
            if isinstance(inter_sizes, int):
                inter_sizes = [input_size, inter_sizes]
            else:
                inter_sizes.insert(0, input_size)
            for i in range(len(inter_sizes) - 1):
                self.layers.append(nn.Linear(inter_sizes[i], inter_sizes[i+1]))
            self.layers.append(nn.Linear(inter_sizes[-1], embed_size))
        else:
            self.layers.append(nn.Linear(input_size, embed_size))         
        
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x
            