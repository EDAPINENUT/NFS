import numpy as np
import torch
import torch.nn as nn
from .. import attention
from .seq2seq import Seq2SeqAttrs
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
class ATTModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, sparse_idx, attention_method, logger=None, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, sparse_idx, **model_kwargs)
        if attention_method in ['ASTGCN', 'MSTGCN']:
            self.network = getattr(attention, attention_method)(
                nb_block=self.block_num,
                nb_chev_filter=self.hidden_units,
                nb_time_filter=self.hidden_units,
                time_strides=int(self.seq_len/2),
                **model_kwargs
                )
        elif attention_method in ['STGCN']:
            self.network = getattr(attention, attention_method)(
                kernel_sizeatt=int(self.seq_len/2),
                **model_kwargs
                )
        elif attention_method == 'ALLATTN':
            self.network = getattr(attention, attention_method)(                
                nhead = 2,
                nb_block=1, 
                mlp_unit= self.mlp_units,     
                **model_kwargs
                )
        self._logger = logger
        #self.linear = nn.Linear(1,16)
    def forward(self, inputs, labels=None,tx = None,ty = None,dec_in = None, batches_seen=None): 

        ## torch.Size([32, 358, 12, 1]) b,n,t,d->          (B, N,D,T).
        inputs = inputs.permute(0,1,3,2)
        outputs =  self.network(inputs, self.sparse_idx)    
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs # t,b,n,d