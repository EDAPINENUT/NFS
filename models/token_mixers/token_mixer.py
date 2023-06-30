import torch.nn as nn

class TokenMixer(nn.Module):
    def __init__(self, embed_size, resolution, *args, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        self.resolution = resolution
        