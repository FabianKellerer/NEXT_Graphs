import torch
import numpy as np
import GraphDataSets as D

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce, cumsum, remove_self_loops, scatter



@functional_transform('random_node_splitting')
class RandomNodeSplit(BaseTransform):

    def __init__(self, p: float = 0.26956) -> None:
        self.p = p

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None

        N = len(data.x)
        NodeList = data.x.clone()
        A, B = data.edge_index
        E = data.edge_attr.clone()

        Splits = torch.from_numpy(np.random.binomial(1, self.p, size=N).astype(bool))
        indices = torch.where(Splits)[0]

        # Vectorized node splitting
        e = NodeList[indices, 0].clone()
        new_values = torch.rand(len(indices)) * e
        NodeList[indices, 0] = new_values
        new_nodes = torch.stack([e - new_values, NodeList[indices, 1], NodeList[indices, 2], NodeList[indices, 3]], dim=1)
        NodeList = torch.cat([NodeList, new_nodes])

        # Edge modifications
        for n, i in enumerate(indices):

            mask_B = ((B == i)).bool()# & (np.random.random(B.shape) < 0.6)).bool()

            new_A = torch.cat([B[mask_B] + N + n - i, torch.tensor([len(data.x) + n])])
            new_B = torch.cat([A[mask_B], torch.tensor([i])])
            
            A = torch.cat([A, new_A, new_B])
            B = torch.cat([B, new_B, new_A])

        E = np.reshape(NodeList[A,0]-NodeList[B,0],(len(A),1))

        data = Data(u=data.u, pos=data.pos, x=NodeList, y=data.y, edge_index=torch.tensor(np.array([np.array(A),np.array(B)])), edge_attr=E)
            
        return data


@functional_transform('random_node_splitting_slow')
class RandomNodeSplit_deprecated(BaseTransform):

    def __init__(self, p: float = 0.26956) -> None:
        self.p = p

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None

        N = data.num_nodes
        
        NodeList = data.x.clone()
        A = data.edge_index[0]
        B = data.edge_index[1]
        E = data.edge_attr.clone()

        Splits  = np.random.binomial(1,self.p,size=N).astype(bool)
        indices = np.where(Splits==1)[0]
        
        ratios = []
        for i in indices:
            node     = NodeList[i]
            e        = node[0].clone()
            node[0]  = np.random.uniform(0,e)
            nodenew  = torch.tensor([[e-node[0],node[1],node[2],node[3]]])
            NodeList = torch.cat([NodeList,nodenew])
            ratios.append(node[0]/e)
            
        for n,i in enumerate(indices):
            for j in range(len(A[np.where(B==i)[0]])):
                if np.random.rand()>0.3:
                    D = np.random.choice(np.where(B==A[np.where(B==i)[0]][j])[0])
                    A = torch.cat([A[:D],A[D+1:]])
                    B = torch.cat([B[:D],B[D+1:]])
            A = torch.cat([A,torch.cat([data.edge_index[1][data.edge_index[1]==i]+N+n-i, torch.tensor([len(data.x)+n])])])
            A = torch.cat([A,torch.cat([data.edge_index[0][data.edge_index[1]==i],torch.tensor([i])])])
            B = torch.cat([B,torch.cat([data.edge_index[0][data.edge_index[1]==i],torch.tensor([i])])])
            B = torch.cat([B,torch.cat([data.edge_index[1][data.edge_index[1]==i]+N+n-i, torch.tensor([len(data.x)+n])])])
            a = E[np.where(data.edge_index[0]==i)[0]]
            b = E[np.where(data.edge_index[1]==i)[0]]
            E[np.where(data.edge_index[0]==i)[0]] = E[np.where(data.edge_index[0]==i)[0]]*ratios[n]
            E[np.where(data.edge_index[1]==i)[0]] = E[np.where(data.edge_index[1]==i)[0]]*ratios[n]
            d = torch.tensor([[NodeList[n+N,0]-NodeList[indices[0],0]]])
            E = torch.cat([E,a*(1-ratios[n]),d])
            E = torch.cat([E,b*(1-ratios[n]),d])

        data = Data(u=data.u, pos=data.pos, x=NodeList, y=data.y, edge_index=torch.tensor(np.array([np.array(A),np.array(B)])), edge_attr=E)
            
        return data
    
    
    
@functional_transform('add_dE_to_edges') # for KNN graphs
class AddEdgeEdiff(BaseTransform):

    def __init__(self) -> None:
        super().__init__() 


    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None

        nodes_in  = data.edge_index[0]
        nodes_out = data.edge_index[1]
        edge_attr = torch.unsqueeze(data.x[nodes_in,0]-data.x[nodes_out,0],1)

        data = Data(u=data.u, pos=data.pos, x=data.x, y=data.y, edge_index=data.edge_index, edge_attr=edge_attr)
            
        return data
    
@functional_transform('random_node_deletion')
class RandomNodeDeletion(BaseTransform):
    
    def __init__(self, p: float = 0.14) -> None:
        self.p = p

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None

        N = len(data.x)
        NodeList = data.x.clone()
        A, B = data.edge_index
        E = data.edge_attr.clone()

        Splits = torch.from_numpy(np.random.binomial(1, self.p, size=N).astype(bool))
        indices = torch.where(Splits)[0]

        # Create a mask to keep track of nodes to be deleted
        mask = torch.ones(N, dtype=bool)
        mask[indices] = False

        #Redistribute energy of nodes to be deleted among their neighbours
        for idx in indices:
            result = B[A == idx][~torch.isin(B[A == idx], indices)]
            NodeList[result, 0] += NodeList[idx, 0] / len(result)

        #Delete nodes
        NodeList = NodeList[mask]

        # Update edge_index
        maskI = ~torch.isin(A, indices) & ~torch.isin(B, indices)
        A = A[maskI]
        B = B[maskI]
        unique_values = torch.unique(A, sorted=True)
        value_map = {val.item(): idx for idx, val in enumerate(unique_values)}
        A = torch.tensor([value_map[val.item()] for val in A])
        B = torch.tensor([value_map[val.item()] for val in B])

        # Update edge_attr
        E = np.reshape(NodeList[A,0]-NodeList[B,0],(len(A),1))

        data = Data(u=data.u, pos=data.pos[mask], x=NodeList, y=data.y, edge_index=torch.tensor(np.array([np.array(A),np.array(B)])), edge_attr=E)
            
        return data
    


@functional_transform('fully_connected')
class FullyConnected(BaseTransform):
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None

        N = len(data.x)

        # Create numpy arrays A and B
        A = np.repeat(np.arange(N), N - 1)
        B = np.concatenate([np.delete(np.arange(N), i) for i in range(N)])

        # Update edge_attr
        edge_attr = torch.unsqueeze(data.x[A,0]-data.x[B,0],1)

        data = Data(u=data.u, pos=data.pos, x=data.x, y=data.y, edge_index=torch.tensor(np.array([A,B])), edge_attr=edge_attr)
            
        return data