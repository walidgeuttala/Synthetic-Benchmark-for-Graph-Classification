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
from dgl.nn.pytorch.conv import GINConv, DotGatConv, GATv2Conv
from torch import nn
import numpy as np
from dgl.nn.pytorch.glob import GlobalAttentionPooling
import h5py

class SAGNetworkHierarchical(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The input node feature dimension.
        hidden_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_layers (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers=3,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        output_activation = 'log_softmax'
    ):
        super(SAGNetworkHierarchical, self).__init__()

        self.dropout = dropout
        self.num_convpools = num_layers
        self.output_activation = output_activation
        convpools = []
        for i in range(num_layers):
            _i_dim = in_dim if i == 0 else hidden_dim
            _o_dim = hidden_dim
            convpools.append(
                ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio)
            )
        self.convpools = torch.nn.ModuleList(convpools)

        self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, graph: dgl.DGLGraph, args):
        feat = graph.ndata["feat"]
        final_readout = None
        value = args.activate
        args.activate = False
        for i in range(self.num_convpools):
            if args.save_last_epoch_hidden_features_for_nodes == True and i == self.num_convpools-1 and value:
                args.activate = True
            
            graph, feat, readout = self.convpools[i](graph, feat, args)
            
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout

        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))

        return getattr(F, self.output_activation)(self.lin3(feat), dim=-1), feat

# hidden_dim is the feat output
class SAGNetworkGlobal(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with global readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The input node feature dimension.
        hidden_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_layers (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers=3,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        output_activation = 'log_softmax'
    ):
        super(SAGNetworkGlobal, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.output_activation = output_activation
        convs = []
        for i in range(num_layers):
            _i_dim = in_dim if i == 0 else hidden_dim
            _o_dim = hidden_dim
            convs.append(GraphConv(_i_dim, _o_dim, allow_zero_in_degree=True))
        self.convs = torch.nn.ModuleList(convs)

        concat_dim = num_layers * hidden_dim
        self.pool = SAGPool(concat_dim, ratio=pool_ratio)
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()

        self.lin1 = torch.nn.Linear(concat_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, graph: dgl.DGLGraph, args):
        feat = graph.ndata["feat"]
        conv_res = []

        for i in range(self.num_layers):
            feat = self.convs[i](graph, feat)
            conv_res.append(feat)
        #if args.activate == True:
        #    with h5py.File("{}/save_hidden_node_feat_test_trial{}.h5".format(args.output_path, args.current_trial), 'a') as hf:
        #        hf.create_dataset('epoch_{}_batch{}'.format(args.current_epoch, args.current_batch), data=conv_res[-1].cpu().numpy())
        conv_res = torch.cat(conv_res, dim=-1)
        graph, feat, _ = self.pool(graph, conv_res)
        feat = torch.cat(
            [self.avg_readout(graph, feat), self.max_readout(graph, feat)],
            dim=-1,
        )

        feat = F.relu(self.lin1(feat))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))

        return getattr(F, self.output_activation)(self.lin3(feat), dim=-1), feat

#hideen_feat is the output dim
class GAT(torch.nn.Module):
    """
    A graph neural network (GAT) that performs graph sum pooling over all nodes in each layer and makes a prediction
    using a linear layer.

    Args:
        num_layers (int): Number of layers in the GAT
        hidden_dim (int): Hidden dimension of the GAT layers
        drop (float): Dropout probability to use during training (default: 0)

    Attributes:
        layers (nn.ModuleList): List of GAT layers
        num (int): Number of layers in the GAT
        input_dim (int): Dimension of the input feature vector
        output_dim (int): Dimension of the output prediction vector
        linear_prediction (nn.ModuleList): List of linear layers to make the prediction
        pool (SumPooling): A sum pooling module to perform graph sum pooling

    Methods:
        forward(g, h): Perform a forward pass through the GAT given a graph and input node features.

    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers=3,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        output_activation = 'log_softmax',
    ):
        """
        Initializes a new instance of the GAT class.

        Args:
            num_layers (int): Number of layers in the GAT
            hidden_dim (int): Hidden dimension of the GAT layers
            drop (float): Dropout probability to use during training (default: 0)

        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.output_activation = output_activation
        self.ann_input_shape = num_layers * hidden_dim
        self.num_heads = 4
        self.batch_norms = []
        # Create GAT layers
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                conv = GATv2Conv(in_feats=self.input_dim,out_feats=hidden_dim, num_heads=self.num_heads, activation=nn.ReLU(), allow_zero_in_degree=True)
            else:
                conv = GATv2Conv(in_feats=hidden_dim, out_feats=hidden_dim, num_heads=self.num_heads, activation=nn.ReLU(), allow_zero_in_degree=True)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(conv)

        # Create linear prediction layers
        self.linear_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            _i_dim = hidden_dim
            _o_dim = hidden_dim
            self.linear_prediction.append(torch.nn.Sequential(torch.nn.Linear(_i_dim, _o_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.BatchNorm1d(_o_dim)))
        self.before_last_linear = torch.nn.Sequential(torch.nn.Linear(hidden_dim*num_layers, hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.BatchNorm1d(hidden_dim))
        self.last_linear = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.BatchNorm1d(out_dim))
        # Create sum pooling module

        self.pool = SumPooling()

    def forward(self, graph: dgl.DGLGraph, args):
        """
        Perform a forward pass through the GAT given a graph and input node features.

        Args:
            g (dgl.DGLGraph): A DGL graph
            h (torch.Tensor): Input node features

        Returns:
            score_over_layer (torch.Tensor): Output prediction

        """
        # list of hidden representation at each layer
        feat = graph.ndata["feat"]
        # Compute hidden representations at each layer
        pooled_h_list = []
        for i, layer in enumerate(self.layers):
            feat = layer(graph, feat).mean(1)
            self.batch_norms[i] = self.batch_norms[i].to('cuda')
            feat = self.batch_norms[i](feat)
            pooled_h = self.pool(graph, feat)
            pooled_h_list.append(self.linear_prediction[i](pooled_h))

           # hidden_rep.append(feat)

        # if args.activate == True:
        #    with h5py.File("{}/save_hidden_node_feat_test_trial{}.h5".format(args.output_path, args.current_trial), 'a') as hf:
        #        hf.create_dataset('epoch_{}_batch{}'.format(args.current_epoch, args.current_batch), data=feat.cpu().numpy())

        pooled_h = torch.cat(pooled_h_list, dim=-1)
        pooled_hh = self.before_last_linear(pooled_h)
        pooled_h = self.last_linear(pooled_hh)

        return getattr(F, self.output_activation)(pooled_h, dim=-1), pooled_hh

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
    def __init__(self, in_dim,
                 hidden_dim, 
                 out_dim, 
                 num_layers = 5, 
                 pool_ratio=0, 
                 dropout=0.5, 
                 output_activation = 'log_softmax'):
    
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.output_activation = output_activation
        
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                mlp = MLP(in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers + 1):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(in_dim, out_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, out_dim))
        self.drop = nn.Dropout(dropout)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, args):
        # list of hidden representation at each layer (including the input layer)
        h = g.ndata["feat"]
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        
        #if args.activate == True:
        #    with h5py.File("{}/save_hidden_node_feat_test_trial{}.h5".format(args.output_path, args.current_trial), 'a') as hf:
        #        hf.create_dataset('epoch_{}_batch{}'.format(args.current_epoch, args.current_batch), data=hidden_rep[-1].cpu().numpy())


        # perform graph sum pooling over all nodes in each layer
        pooled_h_list = []
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            pooled_h_list.append(pooled_h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))

        return  getattr(F, self.output_activation)(score_over_layer, dim=-1), torch.mean(torch.stack(pooled_h_list[1:]), dim=0)




def get_network(net_type: str = "hierarchical"):
    if net_type == "hierarchical":
        return SAGNetworkHierarchical
    elif net_type == "global":
        return SAGNetworkGlobal
    elif net_type == 'gat':
        return GAT
    elif net_type == 'gin':
        return GIN
    else:
        raise ValueError(
            "Network type {} is not supported.".format(net_type)
        )
