import torch
import math
from torch_geometric.data import Batch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
from torch_scatter import scatter_mean
from torch_geometric.nn import (
    GCNConv, GraphConv, GATConv, GATv2Conv, TransformerConv, global_mean_pool, MetaLayer, GroupAddRev
)


def make_mlp(input_dim, hidden_dim, output_dim, num_layers=2):
    """Helper function to create an MLP block."""
    layers = []
    for i in range(num_layers):
        in_dim = input_dim if i == 0 else hidden_dim
        out_dim = output_dim if i == num_layers - 1 else hidden_dim
        layers.extend([Lin(in_dim, out_dim), BatchNorm1d(out_dim), ReLU()])
    return Seq(*layers)


class FixedRotation:
    def __init__(self, angle_degrees, x_indices=None):
        """
        angle_degrees: Ángulo de rotación en grados (puede ser 90, 180, 270).
        x_indices: Índices de las características en data.x que corresponden a X e Y.
        """
        self.angle_radians = math.radians(angle_degrees)
        self.x_indices = x_indices

    def __call__(self, data):
        # Asumiendo que data.pos contiene las posiciones [X, Y]
        pos = data.pos  # Tensor de tamaño [num_nodes, 2]

        # Extraer las componentes X e Y usando x_indices
        X = pos[:, self.x_indices[0]]
        Y = pos[:, self.x_indices[1]]

        # Matriz de rotación en 2D
        rotation_matrix = torch.tensor([
            [math.cos(self.angle_radians), -math.sin(self.angle_radians)],
            [math.sin(self.angle_radians),  math.cos(self.angle_radians)]
        ], dtype=pos.dtype, device=pos.device)

        XY = torch.stack([X, Y], dim=1)
        # Aplicar rotación
        rotated_XY = XY @ rotation_matrix.T  # Multiplicación de matrices


        # Actualizar las posiciones en data.pos
        pos = pos.clone()  # Clonar para evitar modificar el tensor original
        pos[:, self.x_indices[0]] = rotated_XY[:, 0]
        pos[:, self.x_indices[1]] = rotated_XY[:, 1]

        data.pos = pos
        # Actualizar las características de los nodos en data.x que corresponden a X e Y
        if self.x_indices is not None:
            x = data.x.clone()
            x[:, self.x_indices[0]] = rotated_XY[:, 0]
            x[:, self.x_indices[1]] = rotated_XY[:, 1]
            data.x = x

        return data

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms
        self.num_transforms = len(transforms)
        self.total_length = len(dataset) * self.num_transforms

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        data_idx = idx % len(self.dataset)
        transform_idx = idx // len(self.dataset)
        data = self.dataset[data_idx].clone()

        transform = self.transforms[transform_idx]
        data = transform(data)

        return data

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

class GNN(torch.nn.Module):
    def __init__(self, conv_type, hidden_dim, dataset, dropout, heads=4, edge_dim=None, num_layers=4, dropoutLayer=0.4):
        """
        Generalized GNN model supporting different convolution layers, including edge features
        conv_type: GCNConv, GraphConv, GATConv, GATv2Conv, TransformerConv
        """
        super().__init__()
        self.BN = BatchNorm1d(dataset.num_node_features)
        self.convs = torch.nn.ModuleList()
        self.dropoutLayer = torch.nn.Dropout(p=dropoutLayer) 
        # Verificar si edge_dim es necesario para conv_type
        input_dim = dataset.num_node_features
        for i in range(num_layers):
            if conv_type in [GATConv, GATv2Conv, TransformerConv]:
                self.convs.append(conv_type(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim))
                input_dim = hidden_dim * heads 
            else:
                self.convs.append(conv_type(input_dim, hidden_dim))
                input_dim = hidden_dim 

        self.lin = Lin(hidden_dim * heads if conv_type in [GATv2Conv, GATConv, TransformerConv] else hidden_dim, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.BN(x)

        # Pasar edge_attr solo si el conv_type lo soporta
        #for conv in self.convs:
        if hasattr(self.convs[0], "edge_dim") and edge_attr is not None:
            for conv in self.convs[:-1]:
                x = conv(x, edge_index, edge_attr).relu()
                x = self.dropoutLayer(x)
            x = self.convs[-1](x, edge_index, edge_attr)
        else:
            for conv in self.convs[:-1]:
                x = conv(x, edge_index).relu()
                x = self.dropoutLayer(x)
            x = self.convs[-1](x, edge_index)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

class GNNv2(torch.nn.Module):
    def __init__(self, conv_type, hidden_dim, dataset, dropout, heads=1, edge_dim=None, num_layers=4, use_groupaddrev=False, num_groups=2):
        """
        Generalized GNN model supporting different convolution layers, including edge features.
        Optionally applies GroupAddRev to make the network reversible.

        conv_type: GCNConv, GraphConv, GATConv, GATv2Conv, TransformerConv.
        """
        super().__init__()
        self.BN = BatchNorm1d(dataset.num_node_features)
        self.dropout = dropout
        self.use_groupaddrev = use_groupaddrev
        self.num_layers = num_layers
        self.num_groups = num_groups

        input_dim = dataset.num_node_features

        if self.use_groupaddrev:
            # Verificar que num_layers es par para agrupar capas en bloques
            assert num_layers % 2 == 0, "Number of layers must be even when using GroupAddRev."
            self.blocks = torch.nn.ModuleList()

            for i in range(0, num_layers, 2):
                # Crear las subfunciones F y G
                F_layers = torch.nn.ModuleList()
                G_layers = torch.nn.ModuleList()

                # Primera capa del bloque
                if conv_type in [GATConv, GATv2Conv, TransformerConv]:
                    F_layers.append(conv_type(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim))
                    F_output_dim = hidden_dim * heads
                else:
                    F_layers.append(conv_type(input_dim, hidden_dim))
                    F_output_dim = hidden_dim

                # Segunda capa del bloque
                if conv_type in [GATConv, GATv2Conv, TransformerConv]:
                    G_layers.append(conv_type(F_output_dim, hidden_dim, heads=heads, edge_dim=edge_dim))
                    G_output_dim = hidden_dim * heads
                else:
                    G_layers.append(conv_type(F_output_dim, hidden_dim))
                    G_output_dim = hidden_dim

                # Envolver las capas en un bloque reversible
                # Imprimir el valor de num_groups para verificar
                print(f"Initializing GroupAddRev with num_groups = {self.num_groups}")
                block = GroupAddRev(F_layers)
                self.blocks.append(block)
                input_dim = G_output_dim  # Actualizar input_dim para el siguiente bloque

            self.lin = Lin(input_dim, dataset.num_classes)

        else:
            # Construcción original del modelo sin reversibilidad
            self.convs = torch.nn.ModuleList()
            for i in range(num_layers):
                if conv_type in [GATConv, GATv2Conv, TransformerConv]:
                    self.convs.append(conv_type(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim))
                    input_dim = hidden_dim * heads
                else:
                    self.convs.append(conv_type(input_dim, hidden_dim))
                    input_dim = hidden_dim
            self.lin = Lin(input_dim, dataset.num_classes)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.BN(x)

        if self.use_groupaddrev:
            # Usar los bloques reversibles
            for block in self.blocks:
                x = block(x, edge_index, edge_attr)
            x = global_mean_pool(x, batch)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = self.lin(x)
        else:
            # Pasar edge_attr solo si el conv_type lo soporta
            if hasattr(self.convs[0], "edge_dim") and edge_attr is not None:
                for conv in self.convs[:-1]:
                    x = conv(x, edge_index, edge_attr).relu()
                x = self.convs[-1](x, edge_index, edge_attr)
            else:
                for conv in self.convs[:-1]:
                    x = conv(x, edge_index).relu()
                x = self.convs[-1](x, edge_index)
            x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = self.lin(x)
        return x
