import os
import sys
import argparse
import pandas as pd
import numpy  as np
from tqdm.notebook               import tqdm
from invisible_cities.io.dst_io  import load_dst
from invisible_cities.io         import mcinfo_io as mio
from sklearn.utils.extmath       import weighted_mode


parser = argparse.ArgumentParser('Voxelises the cdst files with additional information')
parser.add_argument('-i', '--indir', required=True, type=str, help='Input files directory.')
parser.add_argument('-o', '--outdir', required=True, type=str, help='Output files directory.')
parser.add_argument('-x', '--xbins', default='-210 210 43', type=float, nargs='+', help='X bins (xmin, xmax, Nbins).')
parser.add_argument('-y', '--ybins', default='-210 210 43', type=float, nargs='+', help='Y bins (ymin, ymax, Nbins).')
parser.add_argument('-z', '--zbins', default='20  515 100', type=float, nargs='+', help='Z bins (zmin, zmax, Nbins).')
parser.add_argument('-r', '--rmax', default=180, type=float, help='Radial fiducial cut')
parser.add_argument('-n', '--norm', default=False, type=bool, help='Broken! Do not use! Whether to normalise the hit positions prior to voxelisation')
args = parser.parse_args()


def energy_corrected(energy, z_min, z_max):
    Z_corr_factor = 2.76e-4
    return energy/(1. - Z_corr_factor*(z_max-z_min))


def get_bin_indices(hits, bins, norm, Rmax):
    segclass = 'segclass'
    binclass = 'binclass'
    fiducial_cut = (hits.x**2+hits.y**2)<Rmax**2
    binsX, binsY, binsZ = bins

    print("--- Applying cuts...")
    boundary_cut = (hits.x>=binsX.min()) & (hits.x<=binsX.max())\
                 & (hits.y>=binsY.min()) & (hits.y<=binsY.max())\
                 & (hits.z>=binsZ.min()) & (hits.z<=binsZ.max())

    hits_act = hits[fiducial_cut & boundary_cut].reset_index(drop = True)
    
    print("--- Voxelizing...")
    if norm:
        xnorm = (event.x-min(event.x))/(max(event.x)-min(event.x))
        ynorm = (event.y-min(event.y))/(max(event.y)-min(event.y))
        znorm = (event.z-min(event.z))/(max(event.z)-min(event.z))
        xbin = pd.cut(xnorm, binsX, labels = np.arange(0, len(binsX)-1)).astype(int)
        ybin = pd.cut(ynorm, binsY, labels = np.arange(0, len(binsY)-1)).astype(int)
        zbin = pd.cut(znorm, binsZ, labels = np.arange(0, len(binsZ)-1)).astype(int)
    else:
        xbin = pd.cut(hits_act.x, binsX, labels = np.arange(0, len(binsX)-1)).astype(int)
        ybin = pd.cut(hits_act.y, binsY, labels = np.arange(0, len(binsY)-1)).astype(int)
        zbin = pd.cut(hits_act.z, binsZ, labels = np.arange(0, len(binsZ)-1)).astype(int)

    hits_act = hits_act.assign(xbin=xbin, ybin=ybin, zbin=zbin)
    hits_act.event_id = hits_act.event_id.astype(np.int64)

    if segclass not in hits.columns:
        hits_act = hits_act.assign(segclass = -1)
    if binclass not in hits.columns:
        hits_act = hits_act.assign(binclass = -1)

    print("--- Aggregating...")
    agg_dict = {
        'energy': 'sum',
        'binclass': lambda x: int(x.fillna(-1).unique()[0])
    }

    # Perform the aggregation
    out = hits_act.groupby(['xbin', 'ybin', 'zbin', 'event_id']).agg(agg_dict).reset_index()
    out[binclass] = out['binclass'].astype(int)
    #out[binclass] = out[binclass].astype(int)
    return out

    #out = hits_act.groupby(['xbin', 'ybin', 'zbin', 'event_id']).apply(
    #       lambda df:pd.Series({'energy':df['energy'].sum(),
    #       binclass:int(df[binclass].unique()[0])})).reset_index()
    #out[binclass] = out[binclass].astype(int)
    #return out

    # #outputs df with bins index and energy, and optional label
    # out = hits_act.groupby(['xbin', 'ybin', 'zbin', 'event_id']).apply(
    #        lambda df:pd.Series({'energy':df['energy'].sum(),
    #        'length':df['length'].unique()[0],
    #        'numb_of_voxels':df['numb_of_voxels'].unique()[0],
    #        'numb_of_hits':df['numb_of_hits'].unique()[0],
    #        'x_min':df['x_min'].unique()[0],
    #        'y_min':df['y_min'].unique()[0],
    #        'z_min':df['z_min'].unique()[0],
    #        'r_min':df['r_min'].unique()[0],
    #        'x_max':df['x_max'].unique()[0],
    #        'y_max':df['y_max'].unique()[0],
    #        'z_max':df['z_max'].unique()[0],
    #        'r_max':df['r_max'].unique()[0],
    #        'x_ave':df['x_ave'].unique()[0],
    #        'y_ave':df['y_ave'].unique()[0],
    #        'z_ave':df['z_ave'].unique()[0],
    #        'r_ave':df['r_ave'].unique()[0],
    #        'extreme1_x':df['extreme1_x'].unique()[0],
    #        'extreme2_x':df['extreme2_x'].unique()[0],
    #        'extreme1_y':df['extreme1_y'].unique()[0],
    #        'extreme2_y':df['extreme2_y'].unique()[0],
    #        'extreme1_z':df['extreme1_z'].unique()[0],
    #        'extreme2_z':df['extreme2_z'].unique()[0],
    #        'blob1_x':df['blob1_x'].unique()[0],
    #        'blob1_y':df['blob1_y'].unique()[0],
    #        'blob1_z':df['blob1_z'].unique()[0],
    #        'blob2_x':df['blob2_x'].unique()[0],
    #        'blob2_y':df['blob2_y'].unique()[0],
    #        'blob2_z':df['blob2_z'].unique()[0],
    #        'eblob1':df['eblob1'].unique()[0],
    #        'eblob2':df['eblob2'].unique()[0],
    #        'ovlp_blob_energy':df['ovlp_blob_energy'].unique()[0],
    #        'vox_size_x':df['vox_size_x'].unique()[0],
    #        'vox_size_y':df['vox_size_y'].unique()[0],
    #        'vox_size_z':df['vox_size_z'].unique()[0],
    #        'npeak':df['npeak'].unique()[0],
    #        'Xpeak':df['Xpeak'].unique()[0],
    #        'Ypeak':df['Ypeak'].unique()[0],
    #        'Xrms':df['Xrms'].unique()[0],
    #        'Yrms':df['Yrms'].unique()[0],
    #        'Zrms':df['Zrms'].unique()[0],
    #        'X':df['x'].mean(),
    #        'Y':df['y'].mean(),
    #        'Z':df['z'].mean(),
    #        'Q':df['Q'].sum(),
    #        'Qc':df['Qc'].unique()[0],
    #        segclass:int(weighted_mode(df[segclass], df['energy'])[0][0]),
    #        binclass:int(df[binclass].unique()[0])})).reset_index()
    # out[segclass] = out[segclass].astype(int)
    # out[binclass] = out[binclass].astype(int)
    # return out

def Select_cdsts(input_dir, output_dir, xbins, ybins, zbins, rmax, norm):
    
    bins_x = np.linspace(xbins[0],xbins[1],int(xbins[2]))
    bins_y = np.linspace(ybins[0],ybins[1],int(ybins[2]))
    bins_z = np.linspace(zbins[0],zbins[1],int(zbins[2]))
    bins  = (bins_x, bins_y, bins_z)

    files = [os.path.join(root, name) for root, dirs, files in os.walk(input_dir) for name in files if name.endswith('.h5')]
    files.sort()

    data_all = pd.DataFrame()

    max_evt     = 0
    event_count = 0
    file_count  = 1
    output_file = f'{output_dir}/{file_count}.h5'
    

    for index, file in tqdm(enumerate(files),total=len(files)):
        print(f"Processing file {file}...")

        if os.path.exists(file):
            print(f"-- Reading HDF5 info...")
            voxels = pd.read_hdf(file,'CHITS/highTh')
            kdst   = pd.read_hdf(file,'DST/Events')
            trks   = pd.read_hdf(file,'Tracking/Tracks')

            kdst.event   = kdst.event   + max_evt
            trks.event   = trks.event   + max_evt
            voxels.event = voxels.event + max_evt
            
            try:
                print(f"-- Reading particles...")
                parts = mio.load_mcparticles_df(file)
                parts = parts.reset_index()[['event_id','particle_name']]
                parts   = parts.rename(columns={'event_id':'event'})
                MC      = True
                mapping = dict(zip(parts['event'].unique(), kdst['event'].unique()))
                parts['event'] = parts['event'].map(mapping).fillna(parts['event'])
            except:
                MC = False
            
            print(f"-- Filtering hits...")
            # mask = (voxels.groupby('event')['Ep'].transform('sum').between(1.4, 1.8)) & \
            #         ((voxels['X']**2 + voxels['Y']**2) < rmax**2) & \
            #         (voxels['X'].between(xbins[0], xbins[1])) & \
            #         (voxels['Y'].between(ybins[0], ybins[1])) & \
            #         (voxels['Z'].between(zbins[0], zbins[1])) & \
            #         (~voxels['track_id'].eq(1))
            # voxels = voxels[mask].reset_index(drop=True)
            voxels = voxels.groupby('event').filter(lambda x: x['Ep'].sum()>=1.4)
            voxels = voxels.groupby('event').filter(lambda x: x['Ep'].sum()<=1.8)
            voxels = voxels.groupby('event').filter(lambda x: ((x.X**2+x.Y**2)<rmax**2).all()).reset_index(drop=True)
            voxels = voxels.groupby('event').filter(lambda x: (x.X>xbins[0]).all() & (x.X<xbins[1]).all()).reset_index(drop=True)
            voxels = voxels.groupby('event').filter(lambda x: (x.Y>ybins[0]).all() & (x.Y<ybins[1]).all()).reset_index(drop=True)
            voxels = voxels.groupby('event').filter(lambda x: (x.Z>zbins[0]).all() & (x.Z<zbins[1]).all()).reset_index(drop=True)
            voxels = voxels.groupby('event').filter(lambda x: not (x['track_id']==1).any())
            voxels = voxels.drop('Xrms',axis=1)
            voxels = voxels.drop('Yrms',axis=1)
            kdst   = kdst[kdst.nS2==1]
    
            print("-- Merging tracks...")
            data   = pd.merge(trks, voxels, on='event', how='right')
            data   = pd.merge(data, kdst,   on='event', how='inner', suffixes=(None,'_kdst'))
            
            print("-- Applying labels...")
            if MC:
                clf_labels = parts.groupby('event').particle_name.apply(lambda x:sum(x=='e+')).astype(int)
                clf_labels.name = 'binclass'
                data = pd.merge(data, clf_labels, on='event', how='left')
                data = data[~pd.isna(data.binclass)]
            else:
                data['Ec'] = energy_corrected(data.Ec, data.z_min, data.z_max)*1.6/1.66757358
            
            data_all = pd.concat([data_all,data], ignore_index=True)
            
            print("-- Saving to file...")
            while len(data_all['event'].unique()) >= 10000:
                
                events_to_process = data_all['event'].unique()[:10000]
                To_save = data_all[data_all['event'].isin(events_to_process)]
                print(f'--> Saving file {output_file}...')
                process_chunk(To_save, bins, output_file, norm, rmax)

                data_all = data_all[~data_all['event'].isin(events_to_process)]

                event_count += 10000

                # Start a new file after processing 50000 events
                if event_count % 10000 == 0:
                    file_count += 1
                    output_file = f'{output_dir}/{file_count}.h5'
                        
            max_evt = voxels.event.max() + 1
                        
            
    # Process any remaining events
    if not data_all.empty:
        process_chunk(data_all, bins, output_file, norm, rmax)
        
                

def process_chunk(To_save, bins, output_file, norm, rmax):
    
    #voxelise
    To_save   = To_save.rename(columns={'energy':'trk_energy'})
    To_save   = To_save.rename(columns={'X':'x','Y':'y','Z':'z','event':'event_id','Ec':'energy'})
    To_save   = get_bin_indices(To_save, bins, norm, rmax)
    To_save   = To_save.sort_values('event_id')
    eventInfo = To_save[['event_id', 'binclass']].drop_duplicates().reset_index(drop=True)
    
    #create new unique identifier
    dct_map = {eventInfo.iloc[i].event_id : i for i in range(len(eventInfo))}
    #add dataset_id, pathname and basename to eventInfo
    eventInfo = eventInfo.assign(dataset_id = eventInfo.event_id.map(dct_map))
    #add dataset_id to data and drop event_id
    To_save = To_save.assign(dataset_id = To_save.event_id.map(dct_map))
    To_save = To_save.drop('event_id', axis=1)

    store = pd.HDFStore(output_file, "w", complib=str("zlib"), complevel=4)
    store.put('dataframe', To_save, format='table', data_columns=True)
    store.close()
    
    
if __name__ == '__main__':
    Select_cdsts(args.indir, args.outdir, args.xbins, args.ybins, args.zbins, args.rmax, args.norm)
