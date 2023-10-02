import dgl
import torch
import torch.nn
import torch.nn.functional as F
from dgl.nn import AvgPooling, GraphConv, MaxPooling
from layer import ConvPoolBlock, SAGPool
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.conv import SAGEConv
import dgl.function as fn
from dgl.nn import GATConv
from dgl.nn import GATv2Conv
from dgl.nn.pytorch.conv import GINConv
from torch import nn
import numpy as np


class SAGNetworkHierarchical(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_convs=3,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
    ):
        super(SAGNetworkHierarchical, self).__init__()

        self.dropout = dropout
        self.num_convpools = num_convs

        convpools = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convpools.append(
                ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio)
            )
        self.convpools = torch.nn.ModuleList(convpools)

        self.lin1 = torch.nn.Linear(hid_dim * 2, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim // 2)
        self.lin3 = torch.nn.Linear(hid_dim // 2, out_dim)

    def forward(self, graph: dgl.DGLGraph):
        feat = graph.ndata["feat"]
        final_readout = None

        for i in range(self.num_convpools):
            graph, feat, readout = self.convpools[i](graph, feat)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout

        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        feat = F.log_softmax(self.lin3(feat), dim=-1)

        return feat


class SAGNetworkGlobal(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with global readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_convs=3,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
    ):
        super(SAGNetworkGlobal, self).__init__()
        self.dropout = dropout
        self.num_convs = num_convs

        convs = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convs.append(GraphConv(_i_dim, _o_dim))
        self.convs = torch.nn.ModuleList(convs)

        concat_dim = num_convs * hid_dim
        self.pool = SAGPool(concat_dim, ratio=pool_ratio)
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()

        self.lin1 = torch.nn.Linear(concat_dim * 2, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim // 2)
        self.lin3 = torch.nn.Linear(hid_dim // 2, out_dim)

    def forward(self, graph: dgl.DGLGraph):
        feat = graph.ndata["feat"]
        conv_res = []

        for i in range(self.num_convs):
            feat = self.convs[i](graph, feat)
            conv_res.append(feat)

        conv_res = torch.cat(conv_res, dim=-1)
        graph, feat, _ = self.pool(graph, conv_res)
        feat = torch.cat(
            [self.avg_readout(graph, feat), self.max_readout(graph, feat)],
            dim=-1,
        )

        feat = F.relu(self.lin1(feat))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        feat = F.log_softmax(self.lin3(feat), dim=-1)

        return feat


class GNN(torch.nn.Module):
    """
    A graph neural network (GNN) that performs graph sum pooling over all nodes in each layer and makes a prediction
    using a linear layer.

    Args:
        num_convs (int): Number of layers in the GNN
        hidden_dim (int): Hidden dimension of the GNN layers
        drop (float): Dropout probability to use during training (default: 0)

    Attributes:
        layers (nn.ModuleList): List of GNN layers
        num (int): Number of layers in the GNN
        input_dim (int): Dimension of the input feature vector
        output_dim (int): Dimension of the output prediction vector
        linear_prediction (nn.ModuleList): List of linear layers to make the prediction
        pool (SumPooling): A sum pooling module to perform graph sum pooling

    Methods:
        forward(g, h): Perform a forward pass through the GNN given a graph and input node features.

    """
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_convs=3,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
    ):
        """
        Initializes a new instance of the GNN class.

        Args:
            num_convs (int): Number of layers in the GNN
            hidden_dim (int): Hidden dimension of the GNN layers
            drop (float): Dropout probability to use during training (default: 0)

        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.num_convs = num_convs
        self.input_dim = in_dim
        self.output_dim = out_dim
        
        # Create GNN layers
        for layer in range(num_convs - 1):  # excluding the input layer
            if layer == 0:
                conv = SAGEConv(self.input_dim, hid_dim, "mean", dropout, True, torch.nn.BatchNorm1d(hid_dim), torch.nn.ReLU())
            else:
                conv = SAGEConv(hid_dim, hid_dim, "mean", dropout, True, torch.nn.BatchNorm1d(hid_dim), torch.nn.ReLU())
            self.layers.append(conv)
        
        # Create linear prediction layers
        self.linear_prediction = torch.nn.ModuleList()
        for layer in range(num_convs-1):
            self.linear_prediction.append(torch.nn.Sequential(torch.nn.Linear(hid_dim, self.output_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.BatchNorm1d(self.output_dim)))
        
        # Create sum pooling module
        self.pool = SumPooling()

        self.weights = np.arange(0.3, num_convs-1, 0.3, dtype=float)

    def forward(self, graph: dgl.DGLGraph):
        """
        Perform a forward pass through the GNN given a graph and input node features.

        Args:
            g (dgl.DGLGraph): A DGL graph
            h (torch.Tensor): Input node features

        Returns:
            score_over_layer (torch.Tensor): Output prediction

        """
        # list of hidden representation at each layer 
        feat = graph.ndata["feat"]
        hidden_rep = []
        
        # Compute hidden representations at each layer
        for i, layer in enumerate(self.layers):
            feat = layer(graph, feat)
            hidden_rep.append(feat)
        
        # Perform graph sum pooling over all nodes in each layer and weight for every representation
        output = 0.
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(graph, h) 
            output += self.linear_prediction[i](pooled_h) * self.weights[i]
        
        output = F.log_softmax(output, dim=-1)

        return output

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_convs = 5, pool_ratio=0, dropout=0.5):
    
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_convs = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_convs - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(in_dim, hid_dim, hid_dim)
            else:
                mlp = MLP(hid_dim, hid_dim, hid_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hid_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_convs):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(in_dim, out_dim))
            else:
                self.linear_prediction.append(nn.Linear(hid_dim, out_dim))
        self.drop = nn.Dropout(dropout)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g):
        # list of hidden representation at each layer (including the input layer)
        h = g.ndata["feat"]
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return  F.log_softmax(score_over_layer, dim=-1)




def get_network(net_type: str = "hierarchical"):
    if net_type == "hierarchical":
        return SAGNetworkHierarchical
    elif net_type == "global":
        return SAGNetworkGlobal
    elif net_type == 'gnn':
        return GNN
    elif net_type == 'gin':
        return GIN
    else:
        raise ValueError(
            "Network type {} is not supported.".format(net_type)
        )
