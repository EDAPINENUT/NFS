from turtle import forward
import torch.nn as nn
from .channel_mixer import ChannelMixer
import torch 
import math
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        scores = torch.clip(scores, min=1e-9, max=1e9)

        if mask is not None:
            mask = mask.bool()
            scores = scores.masked_fill(mask==False, -1e9)
        
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in models size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, output_att=False):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model, bias=True)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)
        self.output_att = output_att

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # the same mask applies to all heads
            # unsqueeze Returns a new tensor with a dimension of size one
            # inserted at the specified position.
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention.forward(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        if self.output_att == True:
            return self.output_linear(x), attn
        return self.output_linear(x)

class AttentionMixer(ChannelMixer):
    def __init__(self, embed_size, resolution, head=8, *args, **kwargs):
        super(AttentionMixer, self).__init__(embed_size, resolution, *args, **kwargs)
        self.attention_net = MultiHeadedAttention(head, embed_size)
    
    def forward(self, x):
        input_shape = x.shape
        x = self.attention_net(x, x, x)
        return x.reshape(*input_shape)
    