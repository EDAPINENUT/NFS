import math
import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn.conv import MessagePassing

class DConv(MessagePassing):
    r"""An implementation of the Diffusion Convolution Layer. 
    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer 
            will not learn an additive bias (default :obj:`True`).

    """

    def __init__(self, in_channels, out_channels, K, bias=True):
        super(DConv, self).__init__(aggr='add', flow="source_to_target")
        assert K > 0
        self.in_channels = in_channels ##i=0 65   ##i=1 128
        self.out_channels = out_channels ##i=0 64  ## i =1 64
        self.weight = torch.nn.Parameter(torch.Tensor(2, K, in_channels, out_channels)) #torch.Size([2, 1, 65, 64])  torch.Size([2, 1, 128, 64])

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.__reset_parameters()

    def __reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor) -> torch.FloatTensor:
        r"""Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **edge_index** (PyTorch Long Tensor) - Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional) - Edge weight vector.

        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        adj_mat = to_dense_adj(edge_index, edge_attr=edge_weight) ## torch.Size([2, 51200]) torch.Size([51200]) ->torch.Size([1, 2048, 2048])
        adj_mat = adj_mat.reshape(adj_mat.size(1), adj_mat.size(2)) # torch.Size([2048, 2048])
        deg_out = torch.matmul(adj_mat, torch.ones(size=(adj_mat.size(0), 1)).to(X.device))# torch.Size([2048, 2048]) torch.Size([2048, 1]) ->出度 torch.Size([2048, 1])
        deg_out = deg_out.flatten() # torch.Size([2048])
        deg_in = torch.matmul(torch.ones(size=(1, adj_mat.size(0))).to(X.device), adj_mat) #torch.Size([1, 2048]) torch.Size([2048, 2048])  ->入度 #torch.Size([1, 2048])
        deg_in = deg_in.flatten()

        deg_out_inv = torch.reciprocal(deg_out) # 出度矩阵的逆
        deg_in_inv = torch.reciprocal(deg_in)#入度矩阵的逆
        row, col = edge_index #51200    51200
        norm_out = deg_out_inv[row]
        norm_in = deg_in_inv[row]
        ##? 
        reverse_edge_index = adj_mat.transpose(0,1) ## 
        reverse_edge_index, vv = dense_to_sparse(reverse_edge_index) ## torch.Size([2048, 2048]) torch.Size([51200])  vv:edge_attr

        Tx_0 = X # torch.Size([32, 2048, 65])
        Tx_1 = X # torch.Size([32, 2048, 65])
        H = torch.matmul(Tx_0, (self.weight[0])[0]) + torch.matmul(Tx_0, (self.weight[1])[0])  ## self.weight torch.Size([2, 1, 65, 64])   self.weight[0] torch.Size([1, 65, 64]) ##?
## torch.matmul(Tx_0, (self.weight[0]))    torch.Size([32, 2048, 65]) torch.Size([1, 65, 64])-> torch.Size([32, 2048, 64]) ##TODO  torch.matmul(Tx_0, (self.weight[0])[0]) torch.Size([32, 2048, 64])   torch.Size([1, 65, 64])
        if self.weight.size(1) > 1:
            Tx_1_o = self.propagate(edge_index, x=X, norm=norm_out, size=None)
            Tx_1_i = self.propagate(reverse_edge_index, x=X, norm=norm_in, size=None)
            H = H + torch.matmul(Tx_1_o, (self.weight[0])[1]) + torch.matmul(Tx_1_i, (self.weight[1])[1])

        for k in range(2, self.weight.size(1)):
            Tx_2_o = self.propagate(edge_index, x=Tx_1_o, norm=norm_out, size=None)
            Tx_2_o = 2. * Tx_2_o - Tx_0
            Tx_2_i = self.propagate(reverse_edge_index, x=Tx_1_i, norm=norm_in, size=None) 
            Tx_2_i = 2. * Tx_2_i - Tx_0
            H = H + torch.matmul(Tx_2_o, (self.weight[0])[k]) + torch.matmul(Tx_2_i, (self.weight[1])[k])
            Tx_0, Tx_1_o, Tx_1_i = Tx_1, Tx_2_o, Tx_2_i
##  ###  这里好像没用到
        if self.bias is not None:
            H += self.bias

        return H

class DCRNN(torch.nn.Module):
    r"""An implementation of the Diffusion Convolutional Gated Recurrent Unit.
    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

    Args:
        in_channels (int): NUmber of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer 
            will not learn an additive bias (default :obj:`True`)

    """

    def __init__(self, in_channels: int, out_channels: int, max_view: int, bias: bool=True, **model_kwargs):
        super(DCRNN, self).__init__()

        self.in_channels = in_channels ##1  64
        self.out_channels = out_channels ## 64 64
        self.K = max_view #1
        self.bias = bias

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self): ## i = 0 更新门
        self.conv_x_z = DConv(in_channels=self.in_channels+self.out_channels, #i = 0 1+64 = 65    ##TODO 这里为什么是in+out
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 bias=self.bias)


    def _create_reset_gate_parameters_and_layers(self): ## 重置门
        self.conv_x_r = DConv(in_channels=self.in_channels+self.out_channels, ## 65 
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 bias=self.bias)


    def _create_candidate_state_parameters_and_layers(self): ## 候选状态
        self.conv_x_h = DConv(in_channels=self.in_channels+self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 bias=self.bias)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()    

    def _set_hidden_state(self, X, H): # torch.Size([32, 2048, 1])  torch.Size([32, 2048, 64])
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([X,H], dim=-1) # torch.Size([32, 2048, 1]),torch.Size([32, 2048, 64]) ##最后一个维度拼接
        Z = self.conv_x_z(Z, edge_index, edge_weight) #torch.Size([32, 2048, 64])
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H): #               torch.Size([32, 2048, 64])
        R = torch.cat([X,H], dim=-1) # torch.Size([32, 2048, 65])
        R = self.conv_x_r(R, edge_index, edge_weight)  #torch.Size([32, 2048, 64])
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([X, H * R], dim=-1)#torch.Size([32, 2048, 1])   torch.Size([32, 2048, 64])* torch.Size([32, 2048, 64]) ->torch.Size([32, 2048, 64])
        H_tilde = self.conv_x_h(H_tilde, edge_index, edge_weight) #torch.Size([32, 2048, 65])->torch.Size([32, 2048, 64])
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None, **args) -> torch.FloatTensor:
        r"""Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **edge_index** (PyTorch Long Tensor) - Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional) - Edge weight vector.
            * **H** (PyTorch Float Tensor, optional) - Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H) #  torch.Size([32, 2048, 64])
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H) # torch.Size([32, 2048, 64])
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H) #torch.Size([32, 2048, 64])
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R) #torch.Size([32, 2048, 64])
        H = self._calculate_hidden_state(Z, H, H_tilde) #torch.Size([32, 2048, 64])
        return H
