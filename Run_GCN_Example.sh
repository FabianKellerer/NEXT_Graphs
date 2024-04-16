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


#Set dataset, specific GCN version in the Train_GCN script
python Train_GCN.py