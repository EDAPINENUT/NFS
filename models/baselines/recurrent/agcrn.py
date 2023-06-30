import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros


class AVWGCN(nn.Module):
    r"""An implementation of the Node Adaptive Graph Convolution Layer.
    For details see: `"Adaptive Graph Convolutional Recurrent Network
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int, K: int, embedding_dimensions: int): ##in: 64   out:64    4
        super(AVWGCN, self).__init__()
        self.K = K # 2 
        self.weights_pool = torch.nn.Parameter(torch.Tensor(embedding_dimensions, K, in_channels, out_channels)) # torch.Size([4, 2, 64, 64])   torch.Size([4, 2, 33, 32])
        self.bias_pool = torch.nn.Parameter(torch.Tensor(embedding_dimensions, out_channels)) # torch.Size([4, 64])   torch.Size([4, 32])
        glorot(self.weights_pool)
        zeros(self.bias_pool)
        
    def forward(self, X: torch.FloatTensor, E: torch.FloatTensor) -> torch.FloatTensor:
        r"""Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **E** (PyTorch Float Tensor) - Node embeddings.
        Return types:
            * **X_G** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """

        number_of_nodes = E.shape[0] #2048
        supports = F.softmax(F.relu(torch.mm(E, E.transpose(0, 1))), dim=1) #直接得到归一化  # torch.Size([2048, 2048])
        support_set = [torch.eye(number_of_nodes).to(supports.device), supports]
        for _ in range(2, self.K):
            support = torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            support_set.append(support)
        supports = torch.stack(support_set, dim=0)
        W = torch.einsum('nd,dkio->nkio', E, self.weights_pool) #torch.Size([358, 4])  self.weights_pool torch.Size([4, 2, 65, 128]) ->torch.Size([358, 1, 65, 128])
        bias = torch.matmul(E, self.bias_pool)
        X_G = torch.einsum("knm,bmc->bknc", supports, X) ##torch.Size([2, 358, 358]) torch.Size([32, 358, 65])  torch.Size([32, 2, 358, 65])
        X_G = X_G.permute(0, 2, 1, 3)
        X_G = torch.einsum('bnki,nkio->bno', X_G, W) + bias ## 图 卷积 torch.Size([32, 358, 2, 65]) torch.Size([358, 1, 65, 128])
        return X_G


class AGCRN(nn.Module):
    r"""An implementation of the Adaptive Graph Convolutional Recurrent Unit.
    For details see: `"Adaptive Graph Convolutional Recurrent Network
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_
    Args:
        number_of_nodes (int): Number of vertices.
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """
    def __init__(self, num_of_nodes: int, in_channels: int,
                 out_channels: int, max_view: int, embed_dim: int, **model_kwargs):
        super(AGCRN, self).__init__()
        
        self.number_of_nodes = num_of_nodes ##2048
        self.in_channels = in_channels #1 32
        self.out_channels = out_channels #32 32
        self.K = max_view # 1 #2 AGCRN
        self.embedding_dimensions = embed_dim #4  4 
        self._setup_layers()
        self.node_embeddings = nn.Parameter(torch.randn(num_of_nodes, embed_dim), requires_grad=True)

    def _setup_layers(self):
        self._gate = AVWGCN(in_channels = self.in_channels + self.out_channels,
                            out_channels = 2*self.out_channels,
                            K = self.K,
                            embedding_dimensions = self.embedding_dimensions)
                           
        self._update = AVWGCN(in_channels = self.in_channels + self.out_channels,
                              out_channels = self.out_channels, ## 32
                              K = self.K,
                              embedding_dimensions = self.embedding_dimensions)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device)
        return H

    def forward(self, X: torch.FloatTensor, H: torch.FloatTensor=None, E: torch.FloatTensor=None, **args)  -> torch.FloatTensor:
        r"""Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node feature matrix. torch.Size([32, 2048, 1])
            * **H** (PyTorch Float Tensor) - Node hidden state matrix. Default is None.  shape:torch.Size([32, 2048, 32])
            * **E** (PyTorch Float Tensor) - Node embedding matrix.
        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """       
        if E is None:
            E = self.node_embeddings # shape:torch.Size([358, 4])
        H = self._set_hidden_state(X, H)    #torch.Size([32, 358, 64])
        X_H = torch.cat((X, H), dim=-1) # torch.Size([32, 358, 65])
        Z_R = torch.sigmoid(self._gate(X_H, E)) ## ?    ->torch.Size([32, 2048, 64]) DAGG
        Z, R = torch.split(Z_R, self.out_channels, dim=-1) # torch.Size([32, 2048, 32]) torch.Size([32, 2048, 32])
        C = torch.cat((X, Z*H), dim=-1)
        HC = torch.tanh(self._update(C, E))  # ht hat
        H = R*H + (1-R)*HC # ht
        return H




