import numpy as np
import torch
import torch.nn as nn
from .. import recurrent
from .seq2seq import Seq2SeqAttrs
from .encoder import EncoderModel
from .decoder import DecoderModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class RNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, sparse_idx, conv_method, logger=None,**model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, sparse_idx, **model_kwargs)

        conv = []
        
        for i in range(self.layer_num): ## 2  0，1 
                if i==0:
                    conv.append(
                        getattr(recurrent, conv_method)(
                            in_channels=self.input_dim, 
                            out_channels=self.rnn_units, 
                            **model_kwargs
                            )
                    )
                else:
                    conv.append(
                        getattr(recurrent, conv_method)(
                            in_channels=self.rnn_units, 
                            out_channels=self.rnn_units, 
                            K=self.max_view,
                            **model_kwargs
                            )
                    )
        self._logger = logger
        self.conv = nn.ModuleList(conv)
        self.encoder_model = EncoderModel(sparse_idx, self.conv, **model_kwargs)
        self.decoder_model = DecoderModel(sparse_idx, self.conv, **model_kwargs)
        
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000)) 
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None # torch.Size([12, 32, 2048, 1])
        for t in range(self.encoder_model.seq_len): # (0,12)
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state) # torch.Size([32, 2048, 1])
        #inputs t,b,n,d->inputs[t] b,n,d
        return encoder_hidden_state #torch.Size([2, 32, 2048, 64])

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
#torch.Size([2, 32, 2048, 64]),torch.Size([12, 32, 2048, 1])
        batch_size = encoder_hidden_state.size(1) #32
        go_symbol = torch.zeros((batch_size, self.node_num, self.output_dim)) #torch.Size([32, 2048, 1])
        go_symbol = go_symbol.to(encoder_hidden_state.device)#torch.Size([32, 2048, 1])
        decoder_hidden_state = encoder_hidden_state #torch.Size([2, 32, 2048, 64])
        decoder_input = go_symbol #torch.Size([32, 2048, 1])

        outputs = []

        for t in range(self.horizon):#12
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                    decoder_hidden_state)
            decoder_input = decoder_output #torch.Size([32, 2048, 1])
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs) ## torch.Size([12, 32, 2048, 1])
        return outputs


    def forward(self, inputs, labels=None,tx=None,ty=None,dec_in = None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor, input_dim)  
        :param labels: shape (horizon, batch_size, num_sensor, output_dim)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.node_num * self.output_dim)
        """
        inputs = inputs.permute(2,0,1,3) #B,N,T,D->T,B,N,D
        labels = labels.permute(2,0,1,3)
        
        encoder_hidden_state = self.encoder(inputs) # T,B,N,D
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs
