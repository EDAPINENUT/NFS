import torch

class Seq2SeqAttrs:
    def __init__(self, sparse_idx, **model_kwargs):
        self.sparse_idx = sparse_idx#torch.Size([2, 1092])
        self.max_view = int(model_kwargs.get('max_view', 2)) # 1
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000)) #2000
        self.node_num = int(model_kwargs.get('num_of_nodes', 6)) #2048
        self.layer_num = int(model_kwargs.get('depth', 2)) #2
        self.rnn_units = int(model_kwargs.get('rnn_units', 32)) #64
        self.input_dim = int(model_kwargs.get('input_size', 2)) #1
        self.output_dim = int(model_kwargs.get('output_size', 2)) #1
        self.seq_len = int(model_kwargs.get('seq_len', 12)) #12
        self.embed_dim = int(model_kwargs.get('embed_dim', 16)) #4
        self.horizon = int(model_kwargs.get('pred_len', 16)) #12
        self.block_num = int(model_kwargs.get('block_num', 2)) #1
        self.hidden_units = int(model_kwargs.get('hidden_units', 16))
