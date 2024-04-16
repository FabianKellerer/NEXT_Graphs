#!/bin/bash

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

#Initialise software environment:
source /lhome/ific/f/fkellere/.bashrc

# might need to setup more environmental variables
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=4

#Voxelise cdsts corresponding to dataset (only if not already done):
python cdst_merge.py -i '/lustre/ific.uv.es/ml/ific108/MC/cdst/' -o 'cdst_voxel_RecoBig.h5' -x -210 210 43 -y -210 210 43 -z 20 515 100


#Train the interaction network on the selected dataset. Loss curves are stored in a separate h5 file from the trained net.
python IN.py -d 'RecoBig_all_10mm_R2' -n 400 -l 1e-3 -b 128