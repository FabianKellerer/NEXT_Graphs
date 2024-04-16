#!/bin/bash

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

#Initialise software environment:
source ./setup.sh

# might need to setup more environmental variables
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=4

#Dataset name to be trained on (defined in GraphDataSets):
dataset='RecoBig_all_10mm_R2'

#Voxelise cdsts corresponding to dataset (only if not already done):
python cdst_merge.py -i '/lustre/ific.uv.es/ml/ific108/MC/cdst/' -o 'cdst_voxel_RecoBig.h5' -x -210 210 43 -y -210 210 43 -z 20 515 100

mv cdst_voxel_Data_calib.h5 Input_Dataframes/

#ParticleNet specific: Convert voxelised events to root files:
python Convert_For_PN.py -i Input_Dataframes/cdst_voxel_Data_calib.h5 -o ./GNN_datasets/$dataset


#Train and predict with ParticleNet: First train the net, then test the performance on train, validation and test samples. In a final step, the net can also be applied to real data (if a sample in the correct root format is present). The training log as well as the trained net and predictions are stored in ./weaver-benchmark/weaver/output_$dataset
cd weaver-benchmark/weaver

python train.py  --data-train ./GNN_datasets/$dataset'/prep/next_train_*.root'  --data-val ./GNN_datasets/$dataset'/prep/next_val_*.root'  --fetch-by-file --fetch-step 1 --num-workers 1  --data-config ./NEXT_Features.yaml  --network-config top_tagging/networks/particlenet_pf.py  --model-prefix output/particlenet  --gpus 0 --batch-size 128 --start-lr 5e-3 --num-epochs 100 --optimizer ranger  --log output/particlenet.train.log

python train.py --predict --data-test ./GNN_datasets/$dataset'/prep/next_test_*.root' --num-workers 3 --data-config ./NEXT_Features.yaml --network-config top_tagging/networks/particlenet_pf.py  --model-prefix ./weaver-benchmark/weaver/output/particlenet_best_epoch_state.pt --gpus 0 --batch-size 1024 --predict-output output/particlenet_predict.root

python train.py --predict --data-test ./GNN_datasets/$dataset'/prep/next_train_*.root' --num-workers 3 --data-config ./NEXT_Features.yaml --network-config top_tagging/networks/particlenet_pf.py  --model-prefix ./weaver-benchmark/weaver/output/particlenet_best_epoch_state.pt --gpus 0 --batch-size 1024 --predict-output output/particlenet_predict_train.root

python train.py --predict --data-test ./GNN_datasets/$dataset'/prep/next_val_*.root' --num-workers 3 --data-config ./NEXT_Features.yaml --network-config top_tagging/networks/particlenet_pf.py  --model-prefix ./weaver-benchmark/weaver/output/particlenet_best_epoch_state.pt --gpus 0 --batch-size 1024 --predict-output output/particlenet_predict_valid.root

python train.py --predict --data-test ./GNN_datasets/PN_RealData'/prep/next_*.root' --num-workers 3 --data-config ./NEXT_Features.yaml --network-config top_tagging/networks/particlenet_pf.py  --model-prefix ./weaver-benchmark/weaver/output/particlenet_best_epoch_state.pt --gpus 0 --batch-size 1024 --predict-output output/particlenet_predict_data.root

cp ./jobs_aux/LossFile.*.0.out ./weaver-benchmark/weaver/output/LossFile.out
mkdir ./weaver-benchmark/weaver/output_$dataset
cp -r ./weaver-benchmark/weaver/output/* ./weaver-benchmark/weaver/output_$dataset
rm -r ./weaver-benchmark/weaver/output
