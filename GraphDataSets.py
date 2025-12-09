from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Dataset, Data
from torch.utils.data import random_split
from tqdm import tqdm
import torch.nn.functional as F
import pickle
import os
import sys
import torch
import lmdb
import pandas as pd
import numpy as np
import torch_geometric.transforms as T
import GraphTransforms as Tr


class Base(InMemoryDataset):
    
    def construct_center(self,event):
        E = sum(event.energy)
        X = sum(event.energy*event.xbin)/E
        Y = sum(event.energy*event.ybin)/E
        Z = sum(event.energy*event.zbin)/E
        return [[X,Y,Z]]

    def construct_pos(self,event):
        return np.array([event.xbin,event.ybin,event.zbin]).T

    def construct_nodes(self,event,u):
        j  = int(np.mean(event.dataset_id))
        E  = sum(event.energy)
        dX = max(abs(np.array(event.xbin-float(u[j][0][0]))))
        dY = max(abs(np.array(event.ybin-float(u[j][0][1]))))
        dZ = max(abs(np.array(event.zbin-float(u[j][0][2]))))
        if dX==0:
            dX=1
        if dY==0:
            dY=1
        if dZ==0:
            dZ=1
        n     = event.xbin.keys()[0]
        return np.array([event.energy/E,(event.xbin-u[j][0][0])/dX,(event.ybin-u[j][0][1])/dY,(event.zbin-u[j][0][2])/dZ]).T

    def construct_edge_indices(self,event,p):
        j   = int(np.mean(event.dataset_id))
        xyz = p[j]
        edge_displacements = np.array(xyz.reshape(-1,1,3) - xyz.reshape(1,-1,3))
        r = np.sqrt(np.sum(edge_displacements**2, axis=-1))
        row, col = np.where((r > 0) & (r < 3.1))
        return np.stack([row,col])
    
    def construct_edge_attr(self,event,edge_index):
        j = int(np.mean(event.dataset_id))
        E = sum(event.energy)
        edges_in  = np.array(event.xbin.keys()[edge_index[j][0]])
        edges_out = np.array(event.xbin.keys()[edge_index[j][1]])
        Ein  = event.energy[edges_in]
        Eout = event.energy[edges_out]
        dE   = np.array(Ein) - np.array(Eout)
        return np.reshape(dE/E,(len(dE),1))
    
    def process(self):
        data_list = []
        for i, file in enumerate(self.raw_file_names):
            v = pd.read_hdf(file)
            #v = v.groupby('dataset_id').apply(lambda x: x.sort_values(['xbin', 'ybin', 'zbin'])).reset_index(drop=True)
            G = v.groupby('dataset_id')
            u = G.apply(lambda x: self.construct_center(x))
            p = G.apply(lambda x: self.construct_pos(x))
            n = G.apply(lambda x: self.construct_nodes(x,u))
            y = torch.tensor(np.array(G.binclass.first()))
            e = G.apply(lambda x: self.construct_edge_indices(x,p))
            a = G.apply(lambda x: self.construct_edge_attr(x,e))
            
            ids        = v.dataset_id.unique()
            data_list += [Data(u=torch.tensor(u[i]).float(),pos=torch.tensor(p[i]).float(),x=torch.tensor(n[i]).float(),
                              y=y[list(ids).index(i)],edge_index=torch.tensor(e[i]),
                              edge_attr=torch.tensor(a[i]).float()) for i in ids]

            data, slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            
        torch.save((data, slices), self.processed_paths[0])
        
        
        
class BaseLarge(Dataset):
    
    def construct_center(self,event):
        E = sum(event.energy)
        X = sum(event.energy*event.xbin)/E
        Y = sum(event.energy*event.ybin)/E
        Z = sum(event.energy*event.zbin)/E
        return [[X,Y,Z]]

    def construct_pos(self,event):
        return np.array([event.xbin,event.ybin,event.zbin]).T

    def construct_nodes(self,event,u):
        j  = int(np.mean(event.dataset_id))
        E  = sum(event.energy)
        dX = max(abs(np.array(event.xbin-float(u[j][0][0]))))
        dY = max(abs(np.array(event.ybin-float(u[j][0][1]))))
        dZ = max(abs(np.array(event.zbin-float(u[j][0][2]))))
        if dX==0:
            dX=1
        if dY==0:
            dY=1
        if dZ==0:
            dZ=1
        n     = event.xbin.keys()[0]
        return np.array([event.energy/E,(event.xbin-u[j][0][0])/dX,(event.ybin-u[j][0][1])/dY,(event.zbin-u[j][0][2])/dZ]).T

    def construct_edge_indices(self,event,p):
        j   = int(np.mean(event.dataset_id))
        xyz = p[j]
        edge_displacements = np.array(xyz.reshape(-1,1,3) - xyz.reshape(1,-1,3))
        r = np.sqrt(np.sum(edge_displacements**2, axis=-1))
        row, col = np.where((r > 0) & (r < 2.1))
        return np.stack([row,col])
    
    def construct_edge_attr(self,event,edge_index):
        j = int(np.mean(event.dataset_id))
        E = sum(event.energy)
        edges_in  = np.array(event.xbin.keys()[edge_index[j][0]])
        edges_out = np.array(event.xbin.keys()[edge_index[j][1]])
        Ein  = event.energy[edges_in]
        Eout = event.energy[edges_out]
        dE   = np.array(Ein) - np.array(Eout)
        return np.reshape(dE/E,(len(dE),1))
    
    def process(self):
        data_list = []
        idx       = 0
        for i, file in enumerate(self.raw_file_names):
            v = pd.read_hdf(file)
            G = v.groupby('dataset_id')
            u = G.apply(lambda x: self.construct_center(x))
            p = G.apply(lambda x: self.construct_pos(x))
            n = G.apply(lambda x: self.construct_nodes(x,u))
            y = torch.tensor(np.array(G.binclass.first()))
            e = G.apply(lambda x: self.construct_edge_indices(x,p))
            a = G.apply(lambda x: self.construct_edge_attr(x,e))
            
            ids       = v.dataset_id.unique()
            data_list = [Data(u=torch.tensor(u[i]).float(),pos=torch.tensor(p[i]).float(),x=torch.tensor(n[i]).float(),
                              y=y[list(ids).index(i)],edge_index=edge_list[list(ids).index(i)].edge_index,
                              edge_attr=torch.tensor(a[i]).float()) for i in ids]

            data, slices = InMemoryDataset.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])

    
class BaseLargeMethods:
    # Copied from old BaseLarge for direct use, avoids inheritance complexity for now
    def construct_center(self,event):
        E = sum(event.energy)
        # Handle potential division by zero if E is 0
        if E == 0:
            # Return a default center or handle as appropriate
            # Using mean coordinates if energy is zero
             X = np.mean(event.xbin) if len(event.xbin) > 0 else 0
             Y = np.mean(event.ybin) if len(event.ybin) > 0 else 0
             Z = np.mean(event.zbin) if len(event.zbin) > 0 else 0
             return [[X, Y, Z]]
        X = sum(event.energy*event.xbin)/E
        Y = sum(event.energy*event.ybin)/E
        Z = sum(event.energy*event.zbin)/E
        return [[X,Y,Z]]

    def construct_pos(self,event):
        return np.array([event.xbin,event.ybin,event.zbin]).T
    
    def construct_nodes(self,event,u):
        j  = int(np.mean(event.dataset_id))
        E  = sum(event.energy)
        # Handle E=0 case for normalization
        if E == 0: E = 1.0

        # Ensure u[j] exists and has the expected structure
        center_x, center_y, center_z = 0, 0, 0
        if j in u and len(u[j]) > 0 and len(u[j][0]) == 3:
            center_x, center_y, center_z = u[j][0]
        else:
            print(f"Warning: Center 'u[{j}]' not found or invalid for dataset_id {j}. Using origin.", file=sys.stderr)
            sys.stderr.flush()


        dX = max(abs(np.array(event.xbin - center_x))) if len(event.xbin) > 0 else 1.0
        dY = max(abs(np.array(event.ybin - center_y))) if len(event.ybin) > 0 else 1.0
        dZ = max(abs(np.array(event.zbin - center_z))) if len(event.zbin) > 0 else 1.0

        # Avoid division by zero for dX, dY, dZ
        if dX == 0: dX = 1.0
        if dY == 0: dY = 1.0
        if dZ == 0: dZ = 1.0

        # Handle empty events
        if len(event.xbin) == 0:
            return np.empty((0, 4)) # Return empty array with correct feature dimension

        node_features = np.array([
            event.energy / E,
            (event.xbin - center_x) / dX,
            (event.ybin - center_y) / dY,
            (event.zbin - center_z) / dZ
        ]).T
        return node_features

    # * Notably different than previous implementation!
    def construct_edge_indices(self,event,p):
        j   = int(np.mean(event.dataset_id))
        # Ensure p[j] exists
        if j not in p or len(p[j]) == 0:
             print(f"Warning: Positions 'p[{j}]' not found or empty for dataset_id {j}. Returning empty edge index.", file=sys.stderr)
             sys.stderr.flush()
             return np.empty((2, 0), dtype=np.int64) # Return empty edge index

        xyz = p[j]
        num_nodes = xyz.shape[0]
        if num_nodes < 2:
            return np.empty((2, 0), dtype=np.int64) # No edges if fewer than 2 nodes

        # More efficient distance calculation
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(xyz, xyz)
        row, col = np.where((dist_matrix > 0) & (dist_matrix < 3.1))

        if len(row) == 0:
             return np.empty((2, 0), dtype=np.int64)

        return np.stack([row, col], axis=0).astype(np.int64) # Ensure int64 for PyG

    def construct_edge_attr(self,event,edge_index_map):
        j = int(np.mean(event.dataset_id))
        # Ensure edge_index_map[j] exists and event.energy is accessible
        if j not in edge_index_map or len(event.energy) == 0:
             print(f"Warning: Edge indices 'edge_index_map[{j}]' not found or event energy empty for dataset_id {j}. Returning empty edge attributes.", file=sys.stderr)
             sys.stderr.flush()
             # Need to know the expected shape even if empty
             return np.empty((0, 1)) # Empty with 1 feature dimension

        edge_index = edge_index_map[j]
        if edge_index.shape[1] == 0:
             return np.empty((0, 1)) # Return empty attributes if no edges

        E = sum(event.energy)
        if E == 0: E = 1.0 # Avoid division by zero

        # Ensure indices in edge_index are valid for event.energy Series
        max_node_idx = max(edge_index[0].max(), edge_index[1].max())
        if max_node_idx >= len(event.energy):
             print(f"Warning: Edge index {max_node_idx} out of bounds for event energy length {len(event.energy)} in dataset_id {j}. Returning empty edge attributes.", file=sys.stderr)
             sys.stderr.flush()
             return np.empty((0, 1))

        # Use .iloc for positional access if keys are not 0-based integers
        try:
            # Assuming event.energy keys match the node indices 0..N-1
            Ein = event.energy.iloc[edge_index[0]]
            Eout = event.energy.iloc[edge_index[1]]
            dE = np.array(Ein) - np.array(Eout)
            return np.reshape(dE / E, (len(dE), 1))
        except IndexError as e:
            print(f"Error accessing energy using edge indices for dataset_id {j}: {e}", file=sys.stderr)
            sys.stderr.flush()
            # Handle cases where edge indices might not align with Series index
            # This might require re-indexing nodes/edges if keys aren't simple ranges
            return np.empty((0, 1))








    
class Truth_all_5mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Truth_all_5mm_R1.pt']
    
    

class Truth_all_5mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Truth_all_5mm_R2.pt']
    
    
class Truth_SB50_5mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_SB50.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Truth_SB50_5mm_R1.pt']
    
    
class Truth_SB50_5mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_SB50.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Truth_SB50_5mm_R2.pt']
    
    
    
class RecoSmall_all_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoSmall_all_15mm_R1.pt']
    
    

class RecoSmall_all_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoSmall_all_15mm_R2.pt']
    
    
class RecoSmall_SB50_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoSmall_SB50_15mm_R1.pt']
    
    
class RecoSmall_SB50_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoSmall_SB50_15mm_R2.pt']
    
    

class RecoBig_all_10mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_10mm_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoBig_all_10mm_R1.pt']
    
    

class RecoBig_all_10mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_RecoBig.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoBig_all_10mm_R2.pt']
    
    
    
class RecoBig_all_10mm_R2_T(Base):
    def __init__(self, root, transform=None, pre_transform=Tr.RandomNodeSplit(), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_RecoBig.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoBig_all_10mm_R2_T.pt']
        
    
    
    
class RecoBig_SB50_10mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_10mm_SB50.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoBig_SB50_10mm_R1.pt']
    
    
class RecoBig_SB50_10mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_10mm_SB50.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoBig_SB50_10mm_R2.pt']
    
    
class RecoBig_all_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoBig_all_15mm_R2.pt']
    
    
class RecoBig_SB50_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoBig_SB50_15mm_R2.pt']
    
    
class RecoBig_all_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoBig_all_15mm_R1.pt']
    
    
class RecoBig_SB50_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoBig_SB50_15mm_R1.pt']
    
    
class RecoNew_all_10mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew/{F}' for F in os.listdir('./Input_Dataframes/RecoNew')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoNew_all_10mm_R2.pt']
    
    
class RecoNew_all_10mm_R2_hitsopt(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_HitsOpt/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_HitsOpt')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoNew_all_10mm_R2_hitsopt.pt']




class RecoNew_all_10mm_KNN_hitsopt(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=10), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_HitsOpt/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_HitsOpt')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoNew_all_10mm_KNN_hitsopt.pt']



class RecoNew_all_10mm_KNN6_hitsopt(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=6), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_HitsOpt/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_HitsOpt')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoNew_all_10mm_KNN6_hitsopt.pt']




class RecoNew_all_10mm_KNN20_hitsopt(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=20), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_HitsOpt/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_HitsOpt')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoNew_all_10mm_KNN20_hitsopt.pt']



class RecoNew_all_10mm_KNN30_hitsopt(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=30), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_HitsOpt/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_HitsOpt')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoNew_all_10mm_KNN30_hitsopt.pt']
    


class RecoNew_all_10mm_KNN40_hitsopt(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=40), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_HitsOpt/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_HitsOpt')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoNew_all_10mm_KNN40_hitsopt.pt']




class RecoNew_all_10mm_FC_hitsopt(Base):
    def __init__(self, root, transform=None, pre_transform=Tr.FullyConnected(), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_HitsOpt/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_HitsOpt')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoNew_all_10mm_FC_hitsopt.pt']


    
    
    
class RecoNew_all_10mm_R2_hitsopt_Paolina_151515(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_HitsOpt_Paolina/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_HitsOpt_Paolina')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoNew_all_10mm_R2_hitsopt_Paolina.pt']
    
    
class RecoNew_all_10mm_R2_voxopt_Paolina(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_VoxOpt_Paolina/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_VoxOpt_Paolina')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoNew_all_10mm_R2_voxopt_Paolina.pt']



class RecoNew_all_10mm_R2_hitsopt_Paolina_10105(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_HitsOpt_Paolina_10105/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_HitsOpt_Paolina_10105')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoNew_all_10mm_R2_hitsopt_Paolina_10105.pt']
    
    


class RealData_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_R1.pt']
    
    
class RealData_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_R2.pt']



class RealData_R3(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_R3.pt']
    


class RealData_KNN(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=10), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_KNN.pt']



class RealData_KNN20(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=20), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_KNN20.pt']
    


class RealData_KNN30(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=30), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_KNN30.pt']
    


class RealData_KNN40(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=40), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_KNN40.pt']
    


class RealData_Strict(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=30), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_DataStrict.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_Strict.pt']



class RealData_FC(Base):
    def __init__(self, root, transform=None, pre_transform=Tr.FullyConnected(), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_FC.pt']
    
    
    
class RealData_R2_adapted(Base):                                                     # Only number of voxels changed
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data_adapted.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_R2_adapted.pt']
    
    
    
class RealData_R2_calib(Base):               # Number of voxels changed AND energy corrections applied (z-correction and energy shift)
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data_calib.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_R2_calib.pt']
    
    
class RealData_R2_Paolina_10105(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data_Paolina10105.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_R2_Paolina_10105.pt']    
    
    
    
    
class RealData_R2_Paolina_151515(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data_Paolina151515.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RealData_R2_Paolina_151515.pt']
    



class TrueDataSignal(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/TrueDataSignal.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TrueDataSignal.pt']
    



class SingleEscapeMC(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/SingleEscapeMC.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/SingleEscapeMC.pt']
    

    
    
class Sensim(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/j/jrenner/NEXT_SPARSECONVNET/datasets/sensim_13bar_15mm/{f}' for f in os.listdir('/lhome/ific/j/jrenner/NEXT_SPARSECONVNET/datasets/sensim_13bar_15mm/') if f.startswith("sensim_") and os.path.isfile(os.path.join('/lhome/ific/j/jrenner/NEXT_SPARSECONVNET/datasets/sensim_13bar_15mm/', f))])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Sensim.pt']    
    
    def process(self):
        data_list = []
        for i, file in enumerate(self.raw_file_names):
            v = pd.read_hdf(file,'DATASET/Voxels')
            G = v.groupby('dataset_id')
            u = G.apply(lambda x: self.construct_center(x))
            p = G.apply(lambda x: self.construct_pos(x))
            n = G.apply(lambda x: self.construct_nodes(x,u))
            y = torch.tensor(np.array(G.binclass.first()))
            e = G.apply(lambda x: self.construct_edge_indices(x,p))
            a = G.apply(lambda x: self.construct_edge_attr(x,e))
            
            ids        = v.dataset_id.unique()
            data_list += [Data(u=torch.tensor(u[i]).float(),pos=torch.tensor(p[i]).float(),x=torch.tensor(n[i]).float(),
                              y=y[list(ids).index(i)],edge_index=torch.tensor(e[i]),
                              edge_attr=torch.tensor(a[i]).float()) for i in ids]

            data, slices = self.collate(data_list)
            
        torch.save((data, slices), self.processed_paths[0])
    
    
    
    
class Test(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, length=500):
        self.length = length
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_HitsOpt/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_HitsOpt')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Test.pt']
    
    def process(self):
        data_list = []
        for i, file in enumerate(self.raw_file_names):
            v = pd.read_hdf(file)
            unique_dataset_ids = v['dataset_id'].unique()
            first_five_dataset_ids = unique_dataset_ids[:self.length]
            v = v[v['dataset_id'].isin(first_five_dataset_ids)].reset_index(drop=True)
            G = v.groupby('dataset_id')
            u = G.apply(lambda x: self.construct_center(x))
            p = G.apply(lambda x: self.construct_pos(x))
            n = G.apply(lambda x: self.construct_nodes(x,u))
            y = torch.tensor(np.array(G.binclass.first()))
            e = G.apply(lambda x: self.construct_edge_indices(x,p))
            a = G.apply(lambda x: self.construct_edge_attr(x,e))
            
            ids        = v.dataset_id.unique()
            data_list += [Data(u=torch.tensor(u[i]).float(),pos=torch.tensor(p[i]).float(),x=torch.tensor(n[i]).float(),
                              y=y[list(ids).index(i)],edge_index=torch.tensor(e[i]),
                              edge_attr=torch.tensor(a[i]).float()) for i in ids]

            data, slices = self.collate(data_list)

            if len(data_list) >= self.length:
                break
            
        torch.save((data, slices), self.processed_paths[0])




class RandomSample5Mminus750k(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, fold=0):
        self.fold = fold
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'./Input_Dataframes/RecoNew_HitsOpt/{F}' for F in os.listdir('./Input_Dataframes/RecoNew_HitsOpt')])

    @property
    def processed_file_names(self):
        return [f'/lustre/ific.uv.es/ml/ific108/GNN_datasets/RandomSample5Mminus750k_fold{self.fold}.pt']
    
    def process(self):
        dataset = LargeNEWMC_LMDB(root='/lustre/ific.uv.es/ml/ific108/jrenner/GNN_datasets_largeMC_175_radial_nonStrict')
        indices = torch.arange(len(dataset))
        A, B = random_split(indices, [1500000, len(dataset)-1500000])
        data_list = []
        data_list += [dataset[i] for i in A]
        store = pd.HDFStore(f'/lustre/ific.uv.es/ml/ific108/GNN_datasets/RandomSample5Mminus750k_fold{self.fold}.h5')
        store["Indices"] = pd.DataFrame(np.array(A), columns=["Indices"])
        store.close()

        data, slices = InMemoryDataset.collate(data_list)

        torch.save((data, slices), f'/lustre/ific.uv.es/ml/ific108/GNN_datasets/RandomSample5Mminus750k_fold{self.fold}.pt')




class TestBIG(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, length=25000):
        self.length = length
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lustre/ific.uv.es/ml/ific108/jrenner/GNN_datasets/voxels/{F}' for F in os.listdir('/lustre/ific.uv.es/ml/ific108/jrenner/GNN_datasets/voxels')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TestBIG.pt']
    
    def process(self):
        data_list = []
        for i, file in enumerate(self.raw_file_names):
            v = pd.read_hdf(file)
            unique_dataset_ids = v['dataset_id'].unique()
            first_five_dataset_ids = unique_dataset_ids[:self.length]
            v = v[v['dataset_id'].isin(first_five_dataset_ids)].reset_index(drop=True)
            G = v.groupby('dataset_id')
            u = G.apply(lambda x: self.construct_center(x))
            p = G.apply(lambda x: self.construct_pos(x))
            n = G.apply(lambda x: self.construct_nodes(x,u))
            y = torch.tensor(np.array(G.binclass.first()))
            e = G.apply(lambda x: self.construct_edge_indices(x,p))
            a = G.apply(lambda x: self.construct_edge_attr(x,e))
            
            ids        = v.dataset_id.unique()
            data_list += [Data(u=torch.tensor(u[i]).float(),pos=torch.tensor(p[i]).float(),x=torch.tensor(n[i]).float(),
                              y=y[list(ids).index(i)],edge_index=torch.tensor(e[i]),
                              edge_attr=torch.tensor(a[i]).float()) for i in ids]

            data, slices = self.collate(data_list)

            if len(data_list) >= self.length:
                break
            
        torch.save((data, slices), self.processed_paths[0])






class TestBIG175(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, length=750000):
        self.length = length
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lustre/ific.uv.es/ml/ific108/jrenner/largeMC/voxels/{F}' for F in os.listdir('/lustre/ific.uv.es/ml/ific108/jrenner/largeMC/voxels')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TestBIG175.pt']
    
    def process(self):
        data_list = []
        for i, file in enumerate(self.raw_file_names):
            v = pd.read_hdf(file)
            unique_dataset_ids = v['dataset_id'].unique()
            first_five_dataset_ids = unique_dataset_ids[:self.length]
            v = v[v['dataset_id'].isin(first_five_dataset_ids)].reset_index(drop=True)
            G = v.groupby('dataset_id')
            u = G.apply(lambda x: self.construct_center(x))
            p = G.apply(lambda x: self.construct_pos(x))
            n = G.apply(lambda x: self.construct_nodes(x,u))
            y = torch.tensor(np.array(G.binclass.first()))
            e = G.apply(lambda x: self.construct_edge_indices(x,p))
            a = G.apply(lambda x: self.construct_edge_attr(x,e))
            
            ids        = v.dataset_id.unique()
            data_list += [Data(u=torch.tensor(u[i]).float(),pos=torch.tensor(p[i]).float(),x=torch.tensor(n[i]).float(),
                              y=y[list(ids).index(i)],edge_index=torch.tensor(e[i]),
                              edge_attr=torch.tensor(a[i]).float()) for i in ids]

            data, slices = self.collate(data_list)

            if len(data_list) >= self.length:
                break
            
        torch.save((data, slices), self.processed_paths[0])



class TestBIG175R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox/{F}' for F in os.listdir('/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TestBIG175R2.pt']
    


class TestBIG175R3(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox/{F}' for F in os.listdir('/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TestBIG175R3.pt']
    


class TestBIG175R3strict(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_StrictVox/{F}' for F in os.listdir('/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_StrictVox')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TestBIG175R3strict.pt']
    


class TestBIG175KNN6(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=6), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox/{F}' for F in os.listdir('/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TestBIG175KNN6.pt']
    


class TestBIG175KNN10(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=10), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox/{F}' for F in os.listdir('/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TestBIG175KNN10.pt']
    



class TestBIG175KNN20(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=20), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox/{F}' for F in os.listdir('/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TestBIG175KNN20.pt']



class TestBIG175KNN30(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=30), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox/{F}' for F in os.listdir('/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TestBIG175KNN30.pt']
    


class TestBIG175KNN40(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=40), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox/{F}' for F in os.listdir('/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TestBIG175KNN40.pt']
    




class TestBIG175FC(Base):
    def __init__(self, root, transform=None, pre_transform=Tr.FullyConnected(), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox/{F}' for F in os.listdir('/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_Varvox')])[:40]

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TestBIG175FC.pt']


        


class BIGsample_KNN30_r175_strictvox(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=30), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_StrictVox/{F}' for F in os.listdir('/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_StrictVox')])

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/BIGsample_KNN30_r175_strictvox.pt']


        
class UnorderedLarge_strict(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.KNNGraph(k=30), T.ToUndirected(), Tr.AddEdgeEdiff()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return np.sort([f'/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_StrictVox/{F}' for F in os.listdir('/lhome/ific/f/fkellere/NEXT_Graphs/Input_Dataframes/BigSample_175_StrictVox')])
    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/BIGsample_KNN30_r175_strictvox.pt']



        
class DataMCmix(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/DataMCmix.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/DataMCmix.pt']
    
    def process(self):
        MC        = RecoBig_all_10mm_R2(root='/lhome/ific/f/fkellere/NEXT_Graphs/GNN_datasets',transform=Tr.RandomNodeSplit())
        data      = RealData_R2(root='/lhome/ific/f/fkellere/NEXT_Graphs/GNN_datasets')
        labels    = MC.y
        labels[:] = 0
        MC.y      = labels
        labels    = data.y
        labels[:] = 1
        data.y    = labels
        data_list = []
        cntr      = 0
        indices   = np.linspace(0,len(MC)+len(data),len(data)).astype(int)
        for i in range(0,len(MC)+len(data)-1):
            if i in indices:
                data_list.append(data[np.where(indices==i)[0][0]])
                cntr += 1
            else:
                data_list.append(MC[i-cntr])

        data, slices = self.collate(data_list)
            
        torch.save((data, slices), self.processed_paths[0])
        
        
        
        
class DataMCmix_SB50(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/DataMCmix_SB50.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/DataMCmix_SB50.pt']
    
    def process(self):
        MC        = RecoBig_all_10mm_R2(root='/lhome/ific/f/fkellere/NEXT_Graphs/GNN_datasets',transform=Tr.RandomNodeSplit())
        data      = RealData_R2(root='/lhome/ific/f/fkellere/NEXT_Graphs/GNN_datasets')
        labels    = MC.y
        labels[:] = 0
        MC.y      = labels
        MC        = MC[0:len(data)+100]
        labels    = data.y
        labels[:] = 1
        data.y    = labels
        data_list = []
        cntr      = 0
        indices   = np.linspace(0,len(MC)+len(data),len(data)).astype(int)
        for i in range(0,len(MC)+len(data)-1):
            if i in indices:
                data_list.append(data[np.where(indices==i)[0][0]])
                cntr += 1
            else:
                data_list.append(MC[i-cntr])

        data, slices = self.collate(data_list)
            
        torch.save((data, slices), self.processed_paths[0])
    
        
        
        
        
        
class Q(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_RecoBig.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Q.pt']
    
    def construct_center(self,event):
        E = sum(event.Q)
        X = sum(event.Q*event.xbin)/E
        Y = sum(event.Q*event.ybin)/E
        Z = sum(event.Q*event.zbin)/E
        return [[X,Y,Z]]

    def construct_pos(self,event):
        return np.array([event.xbin,event.ybin,event.zbin]).T

    def construct_nodes(self,event,u):
        j  = int(np.mean(event.dataset_id))
        E  = sum(event.Q)
        dX = max(abs(np.array(event.xbin-float(u[j][0][0]))))
        dY = max(abs(np.array(event.ybin-float(u[j][0][1]))))
        dZ = max(abs(np.array(event.zbin-float(u[j][0][2]))))
        if dX==0:
            dX=1
        if dY==0:
            dY=1
        if dZ==0:
            dZ=1
        n     = event.xbin.keys()[0]
        return np.array([event.Q/E,(event.xbin-u[j][0][0])/dX,(event.ybin-u[j][0][1])/dY,(event.zbin-u[j][0][2])/dZ]).T

    def construct_edge_indices(self,event,p):
        j   = int(np.mean(event.dataset_id))
        xyz = p[j]
        edge_displacements = np.array(xyz.reshape(-1,1,3) - xyz.reshape(1,-1,3))
        r = np.sqrt(np.sum(edge_displacements**2, axis=-1))
        row, col = np.where((r > 0) & (r < 2.1))
        return np.stack([row,col])
    
    def construct_edge_attr(self,event,edge_index):
        j = int(np.mean(event.dataset_id))
        E = sum(event.Q)
        edges_in  = np.array(event.xbin.keys()[edge_index[j][0]])
        edges_out = np.array(event.xbin.keys()[edge_index[j][1]])
        Ein  = event.Q[edges_in]
        Eout = event.Q[edges_out]
        dE   = np.array(Ein) - np.array(Eout)
        return np.reshape(dE/E,(len(dE),1))
    
    
    
class Q_Data(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Q_Data.pt']
    
    def construct_center(self,event):
        E = sum(event.Q)
        X = sum(event.Q*event.xbin)/E
        Y = sum(event.Q*event.ybin)/E
        Z = sum(event.Q*event.zbin)/E
        return [[X,Y,Z]]

    def construct_pos(self,event):
        return np.array([event.xbin,event.ybin,event.zbin]).T

    def construct_nodes(self,event,u):
        j  = int(np.mean(event.dataset_id))
        E  = sum(event.Q)
        dX = max(abs(np.array(event.xbin-float(u[j][0][0]))))
        dY = max(abs(np.array(event.ybin-float(u[j][0][1]))))
        dZ = max(abs(np.array(event.zbin-float(u[j][0][2]))))
        if dX==0:
            dX=1
        if dY==0:
            dY=1
        if dZ==0:
            dZ=1
        n     = event.xbin.keys()[0]
        return np.array([event.Q/E,(event.xbin-u[j][0][0])/dX,(event.ybin-u[j][0][1])/dY,(event.zbin-u[j][0][2])/dZ]).T

    def construct_edge_indices(self,event,p):
        j   = int(np.mean(event.dataset_id))
        xyz = p[j]
        edge_displacements = np.array(xyz.reshape(-1,1,3) - xyz.reshape(1,-1,3))
        r = np.sqrt(np.sum(edge_displacements**2, axis=-1))
        row, col = np.where((r > 0) & (r < 2.1))
        return np.stack([row,col])
    
    def construct_edge_attr(self,event,edge_index):
        j = int(np.mean(event.dataset_id))
        E = sum(event.Q)
        edges_in  = np.array(event.xbin.keys()[edge_index[j][0]])
        edges_out = np.array(event.xbin.keys()[edge_index[j][1]])
        Ein  = event.Q[edges_in]
        Eout = event.Q[edges_out]
        dE   = np.array(Ein) - np.array(Eout)
        return np.reshape(dE/E,(len(dE),1))
    
    
    
class NoPos(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_RecoBig.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/NoPos.pt']
    

    def construct_nodes(self,event,u):
        E  = sum(event.energy)
        return np.array([event.energy/E]).T

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class RecoBig_all_5mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_5mm_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoBig_all_5mm_R1.pt']
    
    

class RecoBig_all_5mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_5mm_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/RecoBig_all_10mm_R2.pt']
    
    
class TruthBig_all_1mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TruthBig_all_1mm_R2.pt']
    
    
    
class TruthBig_all_1mm_1bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_1bar_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TruthBig_all_1mm_1bar_R2.pt']
    
    
    
class TruthBig_all_1mm_5bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_5bar_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TruthBig_all_1mm_5bar_R2.pt']
    
    
    
class TruthBig_all_1mm_10bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_10bar_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TruthBig_all_1mm_10bar_R2.pt']
    
    
    
class TruthBig_all_1mm_15bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_15bar_all.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/TruthBig_all_1mm_15bar_R2.pt']
    
    
    
class KrishanMC_R2(BaseLarge):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'./Input_Dataframes/KrishanFiles/{F}' for F in os.listdir('./Input_Dataframes/KrishanFiles')]

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/KrishanFiles/Graph_{F}.pt' for F in os.listdir('./Input_Dataframes/KrishanFiles')]
    
    def len(self):
        return 733394

    def get(self, i):
        store   = pd.HDFStore("Indices.h5")
        Indices = store['Indices']
        #n   = 0
        #while (Indices.fileno[n]<=i+n and n<len(Indices)):
        #    n   += 1
        for n in range(len(Indices)):
            if Indices.fileno[n]>i+n:
                n_final = n
                break
        file = torch.load(self.processed_file_names[n-1])
        j = i-(Indices.fileno[n]-len(file[1]['u']))+n-1
        
        data =  Data(u   = file[0].u[int(file[1]['u'][j])],
                     pos = file[0].pos[int(file[1]['pos'][j]):int(file[1]['pos'][j+1])],
                     x   = file[0].x[int(file[1]['x'][j]):int(file[1]['x'][j+1])],
                     y   = file[0].y[int(file[1]['y'][j])],
                     edge_index = torch.tensor([np.array(file[0].edge_index[0][file[1]['edge_index'][j]:file[1]['edge_index'][j+1]]),np.array(file[0].edge_index[1][file[1]['edge_index'][j]:file[1]['edge_index'][j+1]])]),
                     edge_attr  = file[0].edge_attr[int(file[1]['edge_attr'][j]):int(file[1]['edge_attr'][j+1])])
        
        return data
    
    

class LargestMC(BaseLarge):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/esmeralda_0.h5']

    @property
    def processed_file_names(self):
        return ['./GNN_Datasets/GraphTest.pt']

    def len(self):
        return 53439

    def get(self, i):
        return torch.load(self.processed_file_names[0])[i]

class TruthMartin_1cm_1bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/Martin_1bar.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Martin_1bar.pt']
    
    
class TruthMartin_1cm_2bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/Martin_2bar.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Martin_2bar.pt']
    
    
class TruthMartin_1cm_5bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'./Input_Dataframes/Martin_5bar/{F}' for F in os.listdir('./Input_Dataframes/Martin_5bar')]

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Martin_5bar.pt']
    
    
class TruthMartin_1cm_13bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/Martin_13bar.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Martin_13bar.pt']
    
    
class TruthMartin_1cm_20bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/Martin_20bar.h5']

    @property
    def processed_file_names(self):
        return ['/lustre/ific.uv.es/ml/ific108/GNN_datasets/Martin_20bar.pt']




class LargeNEWMC_LMDB(Dataset, BaseLargeMethods): # Inherit from Dataset and BaseLargeMethods
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.lmdb_path = os.path.join(self.processed_dir, 'data.lmdb')
        self.metadata_path = os.path.join(self.processed_dir, 'metadata.pkl')

        # These will be loaded from metadata or calculated during process
        self._num_classes = None
        self._num_node_features = None
        self._edge_dim = None
        self._total_graphs = 0
        self.db_env = None # LMDB environment handle

        # Check if processing is needed
        # Processing is needed if the processed dir doesn't exist,
        # or if the LMDB file or metadata file is missing.
        print(f"Processed files check at {self.processed_dir}...")
        processing_needed = not os.path.exists(self.processed_dir) or \
                            not os.path.exists(self.lmdb_path) or \
                            not os.path.exists(self.metadata_path)

        if processing_needed and not os.path.exists(self.raw_dir):
             raise FileNotFoundError(f"Raw data directory not found at {self.raw_dir}. Cannot process.")

        # Call PyG's Dataset constructor. It handles directory creation,
        # calling download/process if necessary, and applying transforms.
        super().__init__(root, transform, pre_transform, pre_filter)

        # Load metadata AFTER super().__init__ ensures processing is done if needed
        print("Loading metadata...")
        sys.stdout.flush()
        try:
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self._total_graphs = metadata['total_graphs']
                self._num_node_features = metadata['num_node_features']
                self._num_classes = metadata['num_classes']
                self._edge_dim = metadata.get('edge_dim', 0)
                print(f"Metadata loaded: {self._total_graphs} graphs, {self._num_node_features} features, {self._num_classes} classes, {self._edge_dim} edge_dim.")
                sys.stdout.flush()
        except FileNotFoundError:
             print(f"Error: Metadata file not found at {self.metadata_path} even after potential processing.", file=sys.stderr)
             sys.stderr.flush()
             # This indicates a problem with the process() method or file system
             raise
        except Exception as e:
             print(f"Error loading metadata pickle file: {e}", file=sys.stderr)
             sys.stderr.flush()
             raise

        # Open LMDB environment for reading AFTER processing is guaranteed done
        print(f"Opening LMDB environment at {self.lmdb_path}...")
        sys.stdout.flush()
        try:
            self.db_env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                                    readahead=False, meminit=False, max_readers=256) # Increase max_readers for DataLoader workers
        except lmdb.Error as e:
            print(f"Error opening LMDB database at {self.lmdb_path}: {e}", file=sys.stderr)
            sys.stderr.flush()
            # Provide more specific guidance if possible
            if "No such file or directory" in str(e):
                print("LMDB file is missing. Ensure the 'process' method ran correctly and created the file.", file=sys.stderr)
            raise # Reraise the exception


        print(f"Dataset initialized successfully with {self._total_graphs} graphs.")
        sys.stdout.flush()

    @property
    def raw_dir(self):
        # Override the default to point to your 'voxels' directory
        return os.path.join(self.root, 'voxels')

    @property
    def raw_file_names(self):
        """Lists raw HDF5 files from the correct raw_dir."""
        if not os.path.exists(self.raw_dir): # Use the property here
             print(f"Warning: Raw directory not found: {self.raw_dir}")
             return []
        raw_dir_content = os.listdir(self.raw_dir) # Use the property here
        raw_files = [f for f in raw_dir_content if f.endswith('.h5') and f[:-3].isdigit()]
        raw_files.sort(key=lambda x: int(x[:-3]))
        # raw_files = [f'{i}.h5' for i in range(1, 684)] # Original fixed list
        print(f"Found {len(raw_files)} raw files in {self.raw_dir}.") # Use the property here
        return raw_files

    @property
    def processed_file_names(self):
        """Specifies the processed LMDB database and metadata file."""
        return ['data.lmdb', 'metadata.pkl']

    def download(self):
        # Implement if data needs to be downloaded. Assuming data is pre-downloaded.
        pass

    def process(self):
        print(f"Processing raw data from {self.raw_dir} into LMDB at {self.lmdb_path}...")
        sys.stdout.flush()

        # --- Apply pre-transform check ---
        if self.pre_transform is None:
             print("Warning: pre_transform is None. No pre-transformation will be applied.")
        else:
             print(f"Applying pre_transform: {self.pre_transform}")
        sys.stdout.flush()

        # Estimate map_size (virtual memory space). Should be larger than the final DB size.
        # Let's estimate based on raw file sizes (e.g., 683 files * ~500MB avg?) ~ 350GB
        # Add buffer. Using 1 TB = 1099511627776 bytes
        map_size = 1099511627776

        # Create LMDB environment
        db_env = lmdb.open(self.lmdb_path, map_size=map_size)

        global_graph_index = 0
        processed_graphs_count = 0
        max_valid_class_label = -1
        first_graph_features = None
        first_graph_edge_dim = None
        expected_num_classes = 2 # Expected number of valid classes

        with db_env.begin(write=True) as txn:
            for raw_file_name in tqdm(self.raw_file_names, desc="Processing Raw Files"):
                raw_path = os.path.join(self.raw_dir, raw_file_name)
                if not os.path.exists(raw_path):
                    print(f"Warning: Raw file not found {raw_path}, skipping.", file=sys.stderr)
                    sys.stderr.flush()
                    continue

                try:
                    v = pd.read_hdf(raw_path)
                except Exception as e:
                    print(f"Error reading HDF5 file {raw_path}: {e}", file=sys.stderr)
                    sys.stderr.flush()
                    continue # Skip corrupted file

                if 'dataset_id' not in v.columns:
                     print(f"Warning: 'dataset_id' column missing in {raw_path}, skipping.", file=sys.stderr)
                     sys.stderr.flush()
                     continue

                G = v.groupby('dataset_id')
                ids = v['dataset_id'].unique()

                # Pre-calculate components for this file to avoid repeated lambda calls
                try:
                    u = G.apply(lambda x: self.construct_center(x))
                    p = G.apply(lambda x: self.construct_pos(x))
                    n = G.apply(lambda x: self.construct_nodes(x, u))
                    y_series = G.binclass.first() # Get labels for all groups in file
                    e = G.apply(lambda x: self.construct_edge_indices(x, p))  # not needed for KNN transform
                    a = G.apply(lambda x: self.construct_edge_attr(x, e))     # not needed for KNN transform
                except Exception as e:
                    print(f"Error during graph component construction for file {raw_path}: {e}", file=sys.stderr)
                    sys.stderr.flush()
                    continue # Skip file if construction fails


                for dataset_id in ids:
                    # Check if all components were successfully created for this id
                    if not all(k in comp for k in [dataset_id] for comp in [u, p, n, y_series]):
                         print(f"Warning: Missing component data for dataset_id {dataset_id} in file {raw_file_name}. Skipping graph.", file=sys.stderr)
                         sys.stderr.flush()
                         continue

                    # Construct the Data object for this specific graph
                    try:
                        # --- Check and Skip Invalid Labels ---
                        label_value = y_series[dataset_id]
                        if not isinstance(label_value, (int, float, np.number)) or int(label_value) < 0 or int(label_value) >= expected_num_classes:
                            print(f"Warning: Invalid label {label_value} for dataset_id {dataset_id} in {raw_file_name}. Skipping graph.", file=sys.stderr)
                            sys.stderr.flush()
                            continue # Skip this graph
                        # --- End Check ---

                        graph_y = torch.tensor(y_series[dataset_id], dtype=torch.long) # Ensure LongTensor for CrossEntropyLoss
                        graph_edge_index = torch.from_numpy(e[dataset_id]) # Convert edge index numpy array, not needed for KNN
                        graph_x = torch.tensor(n[dataset_id]).float()
                        graph_edge_attr = torch.tensor(a[dataset_id]).float() # not needed for KNN

                        # --- Data Validation ---
                        # num_nodes = graph_x.shape[0]
                        # if graph_edge_index.numel() > 0:
                        #    if graph_edge_index.max() >= num_nodes:
                        #        print(f"Warning: Invalid edge index {graph_edge_index.max()} for {num_nodes} nodes in dataset_id {dataset_id}, file {raw_file_name}. Skipping graph.", file=sys.stderr)
                        #        sys.stderr.flush()
                        #        continue
                        # # Basic check for edge_attr shape
                        # if graph_edge_index.shape[1] != graph_edge_attr.shape[0]:
                        #     print(f"Warning: Mismatch between edge_index ({graph_edge_index.shape[1]}) and edge_attr ({graph_edge_attr.shape[0]}) count for dataset_id {dataset_id}, file {raw_file_name}. Skipping graph.", file=sys.stderr)
                        #     sys.stderr.flush()
                        #     continue
                        # --- End Validation ---

                        data = Data(
                            u=torch.tensor(u[dataset_id]).float(),
                            pos=torch.tensor(p[dataset_id]).float(),
                            x=graph_x,
                            y=graph_y,
                            edge_index=graph_edge_index,
                            edge_attr=graph_edge_attr,
                            # Store original id for reference if needed, but not strictly necessary for LMDB key
                            original_dataset_id=torch.tensor(dataset_id)
                        )

                        # --- Apply pre_transform ---
                        if self.pre_transform is not None:
                             try:
                                 # Apply the composed transform (KNNGraph -> ToUndirected -> AddEdgeEdiff)
                                 data = self.pre_transform(data)
                             except Exception as pre_tf_err:
                                 print(f"Error applying pre_transform to graph {dataset_id} in {raw_file_name}: {pre_tf_err}", file=sys.stderr)
                                 sys.stderr.flush()
                                 continue # Skip graph if pre_transform fails

                        # --- Post-transform Validation ---
                        # Check edge indices again after KNN/ToUndirected
                        num_nodes_after_tf = data.x.shape[0] # Nodes shouldn't change for KNN
                        if data.edge_index.numel() > 0:
                           if data.edge_index.min() < 0 or data.edge_index.max() >= num_nodes_after_tf:
                               print(f"Warning: Invalid edge index after pre_transform for {dataset_id}. Skipping graph.", file=sys.stderr); sys.stderr.flush(); continue
                        # Check edge attributes shape (AddEdgeEdiff creates [N, 1])
                        if data.edge_attr is None or data.edge_index.shape[1] != data.edge_attr.shape[0]:
                           print(f"Warning: Mismatch edge/attr shape after pre_transform for {dataset_id} ({data.edge_index.shape[1]} vs {data.edge_attr.shape[0] if data.edge_attr is not None else 'None'}). Skipping graph.", file=sys.stderr); sys.stderr.flush(); continue
                        if data.edge_attr is not None and data.edge_attr.shape[1] != 1: # AddEdgeEdiff creates dim 1
                           print(f"Warning: Unexpected edge_attr dim {data.edge_attr.shape[1]} after pre_transform for {dataset_id}. Skipping graph.", file=sys.stderr); sys.stderr.flush(); continue
                        # --- End Post-transform Validation ---

                        # Store num_features from the first valid graph
                        if first_graph_features is None:
                            first_graph_features = data.num_node_features
                        if first_graph_edge_dim is None and data.edge_attr is not None:
                            # Check if edge_attr exists and has dimensions
                            if data.edge_attr.dim() > 1: # Should be 2D [num_edges, edge_dim]
                                first_graph_edge_dim = data.edge_attr.size(1)
                            elif data.edge_attr.dim() == 1 and data.num_edges > 0: # Handle case where it might be 1D for dim=1
                                first_graph_edge_dim = 1
                            elif data.num_edges == 0: # If no edges, assume edge_dim later or default
                                pass # Keep as None for now
                            else: # Unexpected shape
                                print(f"Warning: Unexpected edge_attr shape {data.edge_attr.shape} for dataset_id {dataset_id}. Cannot determine edge_dim from it.", file=sys.stderr)

                        # --- Update max_valid_class_label (only with 0 or 1) ---
                        max_valid_class_label = max(max_valid_class_label, graph_y.item())

                        # Track max class label
                        #current_max_y = data.y.max().item()
                        #if isinstance(current_max_y, (int, float)): # Check if item() returned a number
                        #    max_class_label = max(max_class_label, int(current_max_y))
                        #elif data.y.numel() > 0: # Handle cases where y might be tensor([])
                        #     print(f"Warning: Unexpected label format {data.y} for dataset_id {dataset_id}, file {raw_file_name}.", file=sys.stderr)
                        #     sys.stderr.flush()


                        # Serialize the Data object using pickle
                        serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

                        # Use the global_graph_index as the key (ASCII string)
                        key = str(global_graph_index).encode('ascii')

                        # Put into LMDB transaction
                        txn.put(key, serialized_data)

                        global_graph_index += 1
                        processed_graphs_count += 1

                    except Exception as graph_err:
                        print(f"Error processing graph for dataset_id {dataset_id} in file {raw_file_name}: {graph_err}", file=sys.stderr)
                        sys.stderr.flush()
                        # Continue to next graph in the file

        db_env.close() # Close the environment after writing

        print(f"LMDB processing complete. Processed {processed_graphs_count} graphs.")
        sys.stdout.flush()

        # Save metadata
        #final_num_classes = max_class_label + 1 if max_class_label >= 0 else 0 # Calculate num classes
        final_num_features = first_graph_features if first_graph_features is not None else 0 # Use features from first graph
        final_edge_dim = first_graph_edge_dim if first_graph_edge_dim is not None else 0
        if final_num_features == 0:
             print("Warning: Could not determine number of features or classes during processing.", file=sys.stderr)
             sys.stderr.flush()
        if first_graph_edge_dim is None:
         print("Warning: Could not determine edge dimension during processing. Assuming 0 or check data.", file=sys.stderr)

        # Check if the max label seen matches expectation
        if max_valid_class_label >= expected_num_classes:
             print(f"ERROR: Max valid label found ({max_valid_class_label}) exceeds expected number of classes ({expected_num_classes})-1. Check data processing logic.", file=sys.stderr)
             sys.stderr.flush()
             # Decide how to handle: maybe raise error, or clamp num_classes? Clamping might hide issues.
             # Forcing expected_num_classes is safer if you are SURE about the data spec.
        elif max_valid_class_label < expected_num_classes - 1 and processed_graphs_count > 0:
             print(f"Warning: Max valid label found ({max_valid_class_label}) is less than expected ({expected_num_classes - 1}). Some classes might be missing in the processed data.", file=sys.stderr)
             sys.stderr.flush()


        metadata = {
            'total_graphs': processed_graphs_count,
            'num_node_features': final_num_features,
            'num_classes': expected_num_classes,
            'edge_dim': final_edge_dim
        }

        print(f"Saving metadata: {metadata}")
        sys.stdout.flush()
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print("Metadata saved.")
        sys.stdout.flush()

    def len(self):
        """Returns the total number of graphs."""
        return self._total_graphs

    def get(self, idx):
        """Retrieves a single graph from the LMDB database."""
        if not self.db_env:
            # This might happen if accessed from a different process without re-initializing __init__
            # Or if __init__ failed to open the DB. Re-open cautiously.
            try:
                 self.db_env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                                         readahead=False, meminit=False, max_readers=256)
                 print(f"Re-opened LMDB environment in get({idx})")
                 sys.stdout.flush()
            except Exception as e:
                 print(f"Failed to re-open LMDB in get({idx}): {e}", file=sys.stderr)
                 sys.stderr.flush()
                 raise RuntimeError("LMDB environment not available in get()")


        if idx < 0 or idx >= self._total_graphs:
            raise IndexError(f"Index {idx} out of range for {self._total_graphs} graphs")

        with self.db_env.begin() as txn:
            key = str(idx).encode('ascii')
            serialized_data = txn.get(key)

        if serialized_data is None:
            # This should ideally not happen if idx is in range and processing was correct
            print(f"Warning: No data found in LMDB for key {key} (index {idx}).", file=sys.stderr)
            sys.stderr.flush()
            raise IndexError(f"Data for index {idx} not found in LMDB database.")

        # Deserialize using pickle
        try:
            data = pickle.loads(serialized_data)
        except Exception as e:
            print(f"Error deserializing data for index {idx}: {e}", file=sys.stderr)
            sys.stderr.flush()
            raise # Re-raise error

        return data

    # --- Add properties to access metadata safely ---
    @property
    def edge_dim(self):
        if self._edge_dim is None:
            # This case should ideally not happen if metadata is loaded correctly
            print("Warning: Edge dimension not loaded from metadata. Returning 0.", file=sys.stderr)
            return 0
        return self._edge_dim

    @property
    def num_classes(self):
        if self._num_classes is None:
            raise ValueError("Number of classes not loaded from metadata.")
        return self._num_classes

    @property
    def num_node_features(self):
        if self._num_node_features is None:
            raise ValueError("Number of node features not loaded from metadata.")
        return self._num_node_features

    def __del__(self):
        # Close the LMDB environment when the dataset object is destroyed
        if hasattr(self, 'db_env') and self.db_env:
            self.db_env.close()
            print("Closed LMDB environment.") # Optional: for debugging
