import pandas as pd
import awkward as ak
import numpy as np
import math
import os
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser('Convert top benchmark h5 datasets to ROOT/awkd')
parser.add_argument('-o', '--outputdir',  required=True, help='The output directory.')
parser.add_argument('-i', '--inputdir', required=True, help='The input directory')
parser.add_argument('-c', '--condition', default='all', choices=['train', 'val', 'test', 'all'], help='Create dataset for train/test/val/all.')
parser.add_argument('-t', '--transform', default=False, type=bool, help='Whether to apply random node splitting')
parser.add_argument('-m', '--mode', default='uproot', choices=['awkd', 'uproot', 'ROOT'], help='Mode to write ROOT files')
parser.add_argument('--max-event-size', type=int, default=10000, help='Maximum event size per output file.')
args = parser.parse_args()


def store_file_awkd(res_array_2d, res_array_1d, outpath):
    r"""Write .awkd files with awkward0
    """
    import awkward0 as ak0
    outpath += '.awkd'
    print('Saving to file', outpath, '...')
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))
    ak0.save(outpath, ak0.fromiter({**res_1d, **{'Part_'+k: res_2d[k] for k in res_2d}} for res_1d, res_2d in zip(res_array_1d, res_array_2d)), mode='w')


def store_file_uproot(res_array_2d, res_array_1d, outpath):
    r"""Write ROOT files with the latest feature in uproot(4)
    """
    import uproot
    def _check_uproot_version(uproot):
        v = uproot.__version__.split('.')
        v = int(v[0])*10000 + int(v[1])*100 + int(v[2])
        assert v >= 40104, "Uproot version should be >= 4.1.4 for the stable uproot-writing feature"

    _check_uproot_version(uproot)
    outpath += '.root'
    print('Saving to file', outpath, '...')
    ak_array2d = ak.from_iter(res_array_2d)
    ak_array1d = ak.from_iter(res_array_1d)
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))
    with uproot.recreate(outpath, compression=uproot.LZ4(4)) as fw:
        # Note that 2D variable names prefixed with `Part_` due to uproot storing rule of jagged arrays
        fw['Events'] = {'Part': ak.zip({k:ak_array2d[k] for k in ak.fields(ak_array2d)}), **{k:ak_array1d[k] for k in ak.fields(ak_array1d) if k != 'nPart'}}
        fw['Events'].title = 'Events'


def store_file_ROOT(res_array_2d, res_array_1d, outpath):
    r"""Write ROOT files with PyROOT
    """
    import ROOT
    from array import array
    ROOT.ROOT.EnableImplicitMT(4)
    outpath += '.root'
    print('Saving to file', outpath, '...')
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))
    f = ROOT.TFile(outpath, 'recreate')
    f.SetCompressionAlgorithm(ROOT.kLZ4)
    f.SetCompressionLevel(4)
    try:
        tree = ROOT.TTree('Events', 'Events')
        # Reserve branches for in TTree
        dic = {}
        for var in res_array_1d[0]:
            vartype = 'i' if 'int' in str(type(res_array_1d[0][var])) else 'f'
            dic[var] = array(vartype, [1])
            tree.Branch(var, dic[var], f'{var}/{vartype.upper()}')
        for var in res_array_2d[0]:
            dic['Part_' + var] = ROOT.vector('float')()
            tree.Branch('Part_' + var, 'vector<float>', dic['Part_' + var])
        # Store variables
        for res_1d, res_2d in zip(res_array_1d, res_array_2d): # loop event by event
            for var in res_1d: # loop over variable names
                dic[var][0] = res_1d[var]
            for var in res_2d:
                dic['Part_' + var].clear()
                for v in res_2d[var]:
                    dic['Part_' + var].push_back(v)
            tree.Fill()
        f.Write()
    finally:
        f.Close()


def NodeSplit(event):
    
    Splits  = np.random.binomial(1,0.26956,size=len(event)).astype(bool)
    indices = np.where(Splits==1)[0]

    for n,i in enumerate(indices):
        E = event.loc[i,'energy']
        event.loc[i,'energy'] = np.random.uniform(0,E)
        event = pd.concat([event, event.loc[[i]]])
        event = event.reset_index(drop=True)
        event.loc[i,'energy'] = E - event.loc[i,'energy']
        
    return event


def convert(input_files, output_file, store_file_func, transform):
    #  List all particle and event variables to store. Particle-level variables will be prefixed with `Part_`
    varlist_2d   = ['E', 'Xbin', 'Ybin', 'Zbin']
    varlist_1d   = ['is_signal']
    idx, ifile   = 0, 0
    cntr0, cntr1 = 0, 0
    res_array_2d, res_array_1d = [], []
    for filename in input_files:
        print('Reading table from:', filename, '...')
        df =  pd.read_hdf(os.path.join(filename))

        print('Processing events ...')

        for i in tqdm(df.dataset_id.unique()):

            if idx >= args.max_event_size:
                # Reach the max event limit per file. Store the current arrays into file
                store_file_func(res_array_2d, res_array_1d, os.path.join(args.outputdir, 'prep', f'{output_file}_{ifile}'))
                del res_array_2d, res_array_1d
                res_array_2d, res_array_1d = [], []
                ifile += 1
                idx = 0

            
            event = df[df.dataset_id==i]

            if transform:
                event = NodeSplit(event)

            y     = event.binclass[event.xbin.keys()[0]]
            
            # Define the variables
            X  = sum(event.energy*event.xbin)/sum(event.energy)
            Y  = sum(event.energy*event.ybin)/sum(event.energy)
            Z  = sum(event.energy*event.zbin)/sum(event.energy)
            dX = max(abs(np.array(event.xbin-X)))
            dY = max(abs(np.array(event.ybin-Y)))
            dZ = max(abs(np.array(event.zbin-Z)))

            # First initiate 2d  and 1d arrays
            res = {
                'E':list(event.energy/sum(event.energy)),
                'Xbin':list((event.xbin-X)/dX),
                'Ybin':list((event.ybin-Y)/dY),
                'Zbin':list((event.zbin-Z)/dZ),
                'is_signal':max(event.binclass),
                'E_tot':sum(event.energy),
                'ID':event.dataset_id.unique()[0]
            }
            
            # Split in 2d and 1d and store per event result
            res_array_2d.append({k:res[k] for k in varlist_2d})
            res_array_1d.append({k:res[k] for k in res.keys() if k not in varlist_2d})

            idx += 1

    # Save rest of events before finishing
    store_file_func(res_array_2d, res_array_1d, os.path.join(args.outputdir, 'prep', f'{output_file}_{ifile}'))


if __name__ == '__main__':

        
    h  = [args.inputdir + '/' + F for F in os.listdir(args.inputdir)]
    h.sort()

    train = h[:int(0.5*len(h))]
    val   = h[int(0.5*len(h)):int(0.75*len(h))]
    test  = h[int(0.75*len(h)):]

    store_file_func = store_file_awkd if args.mode == 'awkd' else \
        store_file_uproot if args.mode == 'uproot' else \
        store_file_ROOT if args.mode == 'ROOT' else None
    if args.condition == 'train':
        convert(input_files=train, output_file='next_train', store_file_func=store_file_func, transform=args.transform)
    elif args.condition == 'val':
        convert(input_files=val, output_file='next_val', store_file_func=store_file_func, transform=args.transform)
    elif args.condition == 'test':
        convert(input_files=test, output_file='next_test', store_file_func=store_file_func, transform=args.transform)
    elif args.condition == 'all':
        convert(input_files=train, output_file='next_train', store_file_func=store_file_func, transform=args.transform)
        convert(input_files=val  , output_file='next_val'  , store_file_func=store_file_func, transform=args.transform)
        convert(input_files=test , output_file='next_test' , store_file_func=store_file_func, transform=args.transform)

