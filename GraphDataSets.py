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
        for i, file in enumerate(self.raw_file_names):
            v = pd.read_hdf(file)
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

    
    
    
class Truth_all_5mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Truth_all_5mm_R1.pt']
    
    

class Truth_all_5mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Truth_all_5mm_R2.pt']
    
    
class Truth_SB50_5mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Truth_SB50_5mm_R1.pt']
    
    
class Truth_SB50_5mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_original_5mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Truth_SB50_5mm_R2.pt']
    
    
    
class RecoSmall_all_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoSmall_all_15mm_R1.pt']
    
    

class RecoSmall_all_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoSmall_all_15mm_R2.pt']
    
    
class RecoSmall_SB50_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoSmall_SB50_15mm_R1.pt']
    
    
class RecoSmall_SB50_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_official_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoSmall_SB50_15mm_R2.pt']
    
    

class RecoBig_all_10mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_10mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_10mm_R1.pt']
    
    

class RecoBig_all_10mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_RecoBig.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_10mm_R2.pt']
    
    
class RecoBig_SB50_10mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_10mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_SB50_10mm_R1.pt']
    
    
class RecoBig_SB50_10mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_10mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_SB50_10mm_R2.pt']
    
    
class RecoBig_all_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_15mm_R2.pt']
    
    
class RecoBig_SB50_15mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_SB50_15mm_R2.pt']
    
    
class RecoBig_all_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_15mm_R1.pt']
    
    
class RecoBig_SB50_15mm_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_15mm_SB50.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_SB50_15mm_R1.pt']
    
    
class RealData_R1(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RealData_R1.pt']
    
    
class RealData_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RealData_R2.pt']
    
    
    
class RealData_R2_adapted(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data_adapted.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RealData_R2_adapted.pt']
    
    
    
class RealData_R2_calib(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data_calib.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RealData_R2_calib.pt']
    
    
    
class Test(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, length=500):
        self.length = length
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/cdst_voxel_Data_calib.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Test.pt']
    
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
        
        
        
class DataMCmix(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/DataMCmix.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/DataMCmix.pt']
    
    def process(self):
        MC        = RecoBig_all_10mm_R2(root='/lhome/ific/f/fkellere/NEXT_Graphs/GNN_datasets')
        data      = RealData_R2_calib(root='/lhome/ific/f/fkellere/NEXT_Graphs/GNN_datasets')
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
        return [os.getcwd()+'/'+'GNN_datasets/DataMCmix_SB50.pt']
    
    def process(self):
        MC        = RecoBig_all_10mm_R2(root='/lhome/ific/f/fkellere/NEXT_Graphs/GNN_datasets')
        data      = RealData_R2_calib(root='/lhome/ific/f/fkellere/NEXT_Graphs/GNN_datasets')
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
        return [os.getcwd()+'/'+'GNN_datasets/Q.pt']
    
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
        return [os.getcwd()+'/'+'GNN_datasets/Q_Data.pt']
    
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
        return [os.getcwd()+'/'+'GNN_datasets/NoPos.pt']
    

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
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_5mm_R1.pt']
    
    

class RecoBig_all_5mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Marija_5mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/RecoBig_all_10mm_R2.pt']
    
    
class TruthBig_all_1mm_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/TruthBig_all_1mm_R2.pt']
    
    
    
class TruthBig_all_1mm_1bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_1bar_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/TruthBig_all_1mm_1bar_R2.pt']
    
    
    
class TruthBig_all_1mm_5bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_5bar_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/TruthBig_all_1mm_5bar_R2.pt']
    
    
    
class TruthBig_all_1mm_10bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_10bar_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/TruthBig_all_1mm_10bar_R2.pt']
    
    
    
class TruthBig_all_1mm_15bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/MC_dataset_Truth_1mm_15bar_all.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/TruthBig_all_1mm_15bar_R2.pt']
    
    
    
class KrishanMC_R2(BaseLarge):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'./Input_Dataframes/KrishanFiles/{F}' for F in os.listdir('./Input_Dataframes/KrishanFiles')]

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/KrishanFiles/Graph_{F}.pt' for F in os.listdir('./Input_Dataframes/KrishanFiles')]
    
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
    
    

class TruthMartin_1cm_1bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/Martin_1bar.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Martin_1bar.pt']
    
    
class TruthMartin_1cm_2bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/Martin_2bar.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Martin_2bar.pt']
    
    
class TruthMartin_1cm_5bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'./Input_Dataframes/Martin_5bar/{F}' for F in os.listdir('./Input_Dataframes/Martin_5bar')]

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Martin_5bar.pt']
    
    
class TruthMartin_1cm_13bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/Martin_13bar.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Martin_13bar.pt']
    
    
class TruthMartin_1cm_20bar_R2(Base):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./Input_Dataframes/Martin_20bar.h5']

    @property
    def processed_file_names(self):
        return [os.getcwd()+'/'+'GNN_datasets/Martin_20bar.pt']
