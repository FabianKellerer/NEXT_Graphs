from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, BatchNorm1d
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np




class EdgeBlock(torch.nn.Module):
    def __init__(self,inputs,hidden):
        super(EdgeBlock, self).__init__()
        self.edge_mlp = Seq(Lin(inputs*2, hidden), 
                            BatchNorm1d(hidden),
                            ReLU(),
                            Lin(hidden, hidden))

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest], 1)
        return self.edge_mlp(out)

class NodeBlock(torch.nn.Module):
    def __init__(self,inputs,hidden):
        super(NodeBlock, self).__init__()
        self.node_mlp_1 = Seq(Lin(inputs+hidden, hidden), 
                              BatchNorm1d(hidden),
                              ReLU(), 
                              Lin(hidden, hidden))
        self.node_mlp_2 = Seq(Lin(inputs+hidden, hidden), 
                              BatchNorm1d(hidden),
                              ReLU(), 
                              Lin(hidden, hidden))

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

    
class GlobalBlock(torch.nn.Module):
    def __init__(self,hidden,outputs):
        super(GlobalBlock, self).__init__()
        self.global_mlp = Seq(Lin(hidden, hidden),                               
                              BatchNorm1d(hidden),
                              ReLU(), 
                              Lin(hidden, outputs))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter_mean(x, batch, dim=0)
        return self.global_mlp(out)


class InteractionNetwork(torch.nn.Module):
    def __init__(self,hidden,dataset,inputs,outputs):
        super(InteractionNetwork, self).__init__()
        self.interactionnetwork = MetaLayer(EdgeBlock(inputs,hidden), NodeBlock(inputs,hidden), GlobalBlock(hidden,outputs))
        self.bn = BatchNorm1d(inputs)
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        
        x = self.bn(x)
        x, edge_attr, u = self.interactionnetwork(x, edge_index, edge_attr, u, batch)
        return u

    
    
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, dropout):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.BN    = BatchNorm1d(dataset.num_node_features)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.BN(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x
    
    
class GCN2(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, dropout):
        super(GCN2, self).__init__()
        torch.manual_seed(12345)
        self.BN    = BatchNorm1d(dataset.num_node_features)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.BN(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x
    
    
class GCN3(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, dropout):
        super(GCN3, self).__init__()
        torch.manual_seed(12345)
        self.BN    = BatchNorm1d(dataset.num_node_features)
        self.conv1 = GATConv(dataset.num_node_features, hidden_channels, heads=8)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=8)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=8)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings 
        x = self.BN(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x