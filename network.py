import dgl
import torch
import torch.nn
import torch.nn.functional as F
from dgl.nn import AvgPooling, MaxPooling
from layer import ConvPoolBlock, SAGPool
from dgl.nn.pytorch.glob import SumPooling
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv, GATv2Conv, SAGEConv, GraphConv, ChebConv, GCN2Conv
from torch import nn
import numpy as np
from dgl.nn.pytorch.glob import GlobalAttentionPooling
import h5py

class GCNv2(nn.Module):
    def __init__(self, in_dim,
                 hidden_dim, 
                 out_dim, 
                 num_layers = 5, 
                 dropout=0.25, 
                 output_activation = 'log_softmax'):
    
        super().__init__()
        self.gcnv2layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.output_activation = output_activation
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                self.gcnv2layers.append(
                    GCN2Conv(in_dim, hidden_dim, allow_zero_in_degree=True)
                )  # set to True if learning epsilon
            else:
                self.gcnv2layers.append(
                    GCN2Conv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
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
        for i, layer in enumerate(self.gcnv2layers):
            h = layer(g, h, h)
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

class Cheb(nn.Module):
    def __init__(self, in_dim,
                 hidden_dim, 
                 out_dim, 
                 num_layers = 5, 
                 dropout=0.25, 
                 output_activation = 'log_softmax'):
    
        super().__init__()
        self.cheblayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.output_activation = output_activation
        self.k = 2
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                self.cheblayers.append(
                    ChebConv(in_dim, hidden_dim, self.k)
                )  # set to True if learning epsilon
            else:
                self.cheblayers.append(
                    ChebConv(hidden_dim, hidden_dim, self.k)
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
        for i, layer in enumerate(self.cheblayers):
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
class SAGE(nn.Module):
    def __init__(self, in_dim,
                 hidden_dim, 
                 out_dim, 
                 num_layers = 5, 
                 dropout=0.25, 
                 output_activation = 'log_softmax'):
    
        super().__init__()
        self.sagelayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.output_activation = output_activation
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                self.sagelayers.append(
                    SAGEConv(in_dim, hidden_dim)
                )  # set to True if learning epsilon
            else:
                self.sagelayers.append(
                    SAGEConv(hidden_dim, hidden_dim)
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
        for i, layer in enumerate(self.sagelayers):
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

class GCN(nn.Module):
    def __init__(self, in_dim,
                 hidden_dim, 
                 out_dim, 
                 num_layers = 5, 
                 dropout=0.25, 
                 output_activation = 'log_softmax'):
    
        super().__init__()
        self.gcnlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.output_activation = output_activation
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                self.gcnlayers.append(
                    GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
                )  # set to True if learning epsilon
            else:
                self.gcnlayers.append(
                    GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
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
        for i, layer in enumerate(self.gcnlayers):
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

class GATv2(nn.Module):
    def __init__(self, in_dim,
                 hidden_dim, 
                 out_dim, 
                 num_layers = 5, 
                 dropout=0.25, 
                 output_activation = 'log_softmax'):
    
        super().__init__()
        self.gatv2layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.output_activation = output_activation
        self.num_heads = 4
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                self.gatv2layers.append(
                    GATv2Conv(in_feats=in_dim,out_feats=hidden_dim, num_heads=self.num_heads, allow_zero_in_degree=True)
                )  # set to True if learning epsilon
            else:
                self.gatv2layers.append(
                    GATv2Conv(in_feats=hidden_dim,out_feats=hidden_dim, num_heads=self.num_heads, allow_zero_in_degree=True)
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
        for i, layer in enumerate(self.gatv2layers):
            h = layer(g, h).mean(1)
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
                 dropout=0.25, 
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
    if net_type == "gcn":
        return GCN
    elif net_type == 'gcnv2':
        return GCNv2
    elif net_type == "sage":
        return SAGE
    elif net_type == 'cheb':
        return Cheb
    elif net_type == 'gin':
        return GIN
    elif net_type == 'gatv2':
        return GATv2
    else:
        raise ValueError(
            "Network type {} is not supported.".format(net_type)
        )
