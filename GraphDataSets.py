from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Dataset, Data
import torch.nn.functional as F
import os
import torch
import pandas as pd
import numpy as np
import torch_geometric.transforms as T


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

    def construct_edge_attr(self,event,data,ids):
        j = list(ids).index(event.dataset_id.unique())
        E = sum(event.energy)
        edge_indices = data[j].edge_index
        edges_in  = np.array(event.xbin.keys()[edge_indices[0]])
        edges_out = np.array(event.xbin.keys()[edge_indices[1]])
        Ein  = event.energy[edges_in]
        Eout = event.energy[edges_out]
        dE   = np.array(Ein) - np.array(Eout)
        return np.reshape(dE/E,(len(dE),1))
    
    def process(self):
        data_list = []
        for i, file in enumerate(self.raw_file_names[0:5]):
            v = pd.read_hdf(file)
            G = v.groupby('dataset_id')
            u = G.apply(lambda x: self.construct_center(x))
            p = G.apply(lambda x: self.construct_pos(x))
            n = G.apply(lambda x: self.construct_nodes(x,u))
            y = torch.tensor(np.array(G.binclass.first()))
            ids = v.dataset_id.unique()
            edge_list = [Data(pos=torch.tensor(p[i]).float()) for i in ids]
            if self.pre_transform is not None:
                edge_list = [self.pre_transform(data) for data in edge_list]
            a = G.apply(lambda x: self.construct_edge_attr(x,edge_list,ids))
            data_list += [Data(u=torch.tensor(u[i]).float(),pos=torch.tensor(p[i]).float(),x=torch.tensor(n[i]).float(),
                              y=y[list(ids).index(i)],edge_index=edge_list[list(ids).index(i)].edge_index,
                              edge_attr=torch.tensor(a[i]).float()) for i in ids]

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

    def construct_edge_attr(self,event,data,ids):
        j = list(ids).index(event.dataset_id.unique())
        E = sum(event.energy)
        edge_indices = data[j].edge_index
        edges_in  = np.array(event.xbin.keys()[edge_indices[0]])
        edges_out = np.array(event.xbin.keys()[edge_indices[1]])
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
            ids = v.dataset_id.unique()
            edge_list = [Data(pos=torch.tensor(p[i]).float()) for i in ids]
            if self.pre_transform is not None:
                edge_list = [self.pre_transform(data) for data in edge_list]
            a = G.apply(lambda x: self.construct_edge_attr(x,edge_list,ids))
            data_list = [Data(u=torch.tensor(u[i]).float(),pos=torch.tensor(p[i]).float(),x=torch.tensor(n[i]).float(),
                              y=y[list(ids).index(i)],edge_index=edge_list[list(ids).index(i)].edge_index,
                              edge_attr=torch.tensor(a[i]).float()) for i in ids]

            data, slices = InMemoryDataset.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])

    
    
    
class Truth_all_5mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=1.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Truth_all_5mm_R1.pt']
    
    

class Truth_all_5mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Truth_all_5mm_R2.pt']
    
    
class Truth_SB50_5mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=1.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Truth_SB50_5mm_R1.pt']
    
    
class Truth_SB50_5mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Truth_SB50_5mm_R2.pt']
    
    
    
class RecoSmall_all_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=1.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoSmall_all_15mm_R1.pt']
    
    

class RecoSmall_all_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoSmall_all_15mm_R2.pt']
    
    
class RecoSmall_SB50_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=1.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoSmall_SB50_15mm_R1.pt']
    
    
class RecoSmall_SB50_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoSmall_SB50_15mm_R2.pt']
    
    

class RecoBig_all_10mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=1.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_10mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_10mm_R1.pt']
    
    

class RecoBig_all_10mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Input_Dataframes/cdst_voxel_RecoBig.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_10mm_R2.pt']
    
    
class RecoBig_SB50_10mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=1.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_10mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_SB50_10mm_R1.pt']
    
    
class RecoBig_SB50_10mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_10mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_SB50_10mm_R2.pt']
    
    
class RecoBig_all_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_15mm_R2.pt']
    
    
class RecoBig_SB50_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_SB50_15mm_R2.pt']
    
    
class RecoBig_all_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_15mm_R1.pt']
    
    
class RecoBig_SB50_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_SB50_15mm_R1.pt']
    
    
class RealData_R1(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=1.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RealData_R1.pt']
    
    
class RealData_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RealData_R2.pt']
    
    
    
class RealData_R2_adapted(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data_adapted.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RealData_R2_adapted.pt']
    
    
    
class RealData_R2_calib(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data_calib.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RealData_R2_calib.pt']
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class RecoBig_all_5mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=1.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_5mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_5mm_R1.pt']
    
    

class RecoBig_all_5mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_5mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_10mm_R2.pt']
    
    
class TruthBig_all_1mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/TruthBig_all_1mm_R2.pt']
    
    
    
class TruthBig_all_1mm_1bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_1bar_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/TruthBig_all_1mm_1bar_R2.pt']
    
    
    
class TruthBig_all_1mm_5bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_5bar_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/TruthBig_all_1mm_5bar_R2.pt']
    
    
    
class TruthBig_all_1mm_10bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_10bar_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/TruthBig_all_1mm_10bar_R2.pt']
    
    
    
class TruthBig_all_1mm_15bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_15bar_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/TruthBig_all_1mm_15bar_R2.pt']
    
    
    
class KrishanMC_R2(BaseLarge):
    def __init__(self, root, transform=None, pre_transform=T.Compose([T.RadiusGraph(r=2.1), T.NormalizeScale()]), pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'./Input_Dataframes/KrishanFiles/{F}' for F in os.listdir('./Input_Dataframes/KrishanFiles')]

    @property
    def processed_file_names(self):
        return [fos.getcwd()+'/'+'GNN_datasets/KrishanFiles/Graph_{F}.pt' for F in os.listdir('./Input_Dataframes/KrishanFiles')]
    
    def len(self):
        return 733394

    def get(self, i):
        store   = pd.HDFStore("Indices.h5")
        Indices = store['Indices']
        n   = 0
        while (Indices.fileno[n]<=i+n and n<len(Indices)):
            n   += 1
        file = torch.load(self.processed_file_names[n-1])
        j = i-(Indices.fileno[n]-len(file[1]['u']))+n-1
        
        data =  Data(u   = file[0].u[int(file[1]['u'][j])],
                     pos = file[0].pos[int(file[1]['pos'][j]):int(file[1]['pos'][j+1])],
                     x   = file[0].x[int(file[1]['x'][j]):int(file[1]['x'][j+1])],
                     y   = file[0].y[int(file[1]['y'][j])],
                     edge_index = torch.tensor([np.array(file[0].edge_index[0][file[1]['edge_index'][j]:file[1]['edge_index'][j+1]]),np.array(file[0].edge_index[1][file[1]['edge_index'][j]:file[1]['edge_index'][j+1]])]),
                     edge_attr  = file[0].edge_attr[int(file[1]['edge_attr'][j]):int(file[1]['edge_attr'][j+1])])
        
        return data
