#!/bin/bash

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

#Initialise software environment:
source /lhome/ific/f/fkellere/.bashrc

export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=4

#Voxelise cdsts corresponding to dataset (only if not already done):
python cdst_merge.py -i '/lustre/ific.uv.es/ml/ific108/MC/cdst/' -o 'cdst_voxel_RecoBig.h5' -x -210 210 43 -y -210 210 43 -z 20 515 100


#Set dataset, specific GCN version in the Train_GCN script
python Train_GCN.py