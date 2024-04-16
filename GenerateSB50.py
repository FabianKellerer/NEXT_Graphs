import sys
import random
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser('Generates the input dataframes')
parser.add_argument('-i', '--infile', required=True, type=str, help='Input file. Has to include the full path.')
parser.add_argument('-o', '--outfile', required=True, type=str, help='Output file name. May NEITHER include the path NOR the filename extension')
parser.add_argument('-l', '--emin', type=float, default=1.4, help='Lower energy cut')
parser.add_argument('-u', '--emax', type=float, default=1.8, help='Upper energy cut')
args = parser.parse_args()

def GenerateInputDataframes(infile,outfile,emin,emax):

    All = pd.read_hdf(infile,'DATASET/Voxels')

    All = All[All.binclass!=2]

    groups = [df for _, df in All.groupby('dataset_id')]
    random.shuffle(groups)
    All   = pd.concat(groups).reset_index(drop=True)
    
    eventInfo = All[['dataset_id', 'binclass']].drop_duplicates().reset_index(drop=True)
    #create new unique identifier
    dct_map   = {eventInfo.iloc[i].dataset_id : i for i in range(len(eventInfo))}
    #add dataset_id to hits and drop event_id
    All  = All.assign(dataset_id = All.dataset_id.map(dct_map))
    
    E      = All.groupby(['dataset_id']).sum().reset_index()
    Ecut   = E.loc[(E['energy'] >= emin) & (E['energy'] <= emax)]
    Res2   = All.loc[All['dataset_id'].isin(Ecut.dataset_id)]

    store3 = pd.HDFStore(f"Input_Dataframes/{outfile}_all.h5")
    store3['DATASET/Voxels'] = Res2
    store3.close()

    S = Res2[Res2.binclass==1]
    B = Res2[Res2.binclass==0]
    G = B.groupby(['dataset_id'])
    a = np.arange(G.ngroups)
    np.random.shuffle(a)
    L = len(B.dataset_id.unique())-len(S.dataset_id.unique())

    B    = B[~G.ngroup().isin(a[:L])]
    Res1 = pd.concat([S,B])
    Res1 = Res1.sort_values('dataset_id')
    Res1 = Res1.reset_index()

    store2 = pd.HDFStore(f"Input_Dataframes/{outfile}_SB50.h5")
    store2['DATASET/Voxels'] = Res1
    store2.close()
    
if __name__ == '__main__':
    GenerateInputDataframes(args.infile, args.outfile, args.emin, args.emax)
