# Graph Neural Networks for NEXT
# Setup:
 - IC needs to be installed (https://github.com/next-exp/IC)
 - Add all other needed dependencies to the IC conda environment that is created by the IC installation:
	- networkx
	- scipy
	- tqdm
	- pytorch
	- uproot
   and probably others. You can find respective installation instructions for each online.

You will need a folder containing simulated calibration cdst files, produced by nexus and IC, to train your neural network, for example available here: https://next-exp-sw.readthedocs.io/en/latest/production.html#next-white  

# Usage:
There are example bash scripts on how to run each network, named Run_{Network}_Example.sh. A description of the commands contained within will follow. 
To train the Graph Convolution Network (GCN) and the Interaction network (IN), the cdst files need to be modified first by using the cdst_merge script. It takes as input the folder containing the cdst files, the name of the output file which needs to be saved in the 'Input_Dataframes' folder, which might need to be created.
Once this is done, for the IN, execute the 'IN.py' macro. It needs the name of a Graph architecture (or 'dataset'), saved in the 'GraphDataSets.py' file. Take note that each dataset requires a unique input file generated in the previous step, so make sure you have the correct file in the Input_Dataframes folder for the dataset that you want to use. Furthermore, it needs to know whether to use data augmentation, the max. number of training epochs, the learning rate and the batch size.
For the GCN, simply execute the 'Train_GCN.py' macro. The settings are inside of the macro itself.
For ParticleNet (PN), you will need to execute the Convert_For_PN.py macro after running cdst_merge.py. This will create root files in the specified output directory. After this is done, look inside the 'Run_PN_Exapmle.sh' macro to see how to train the PN. There, you will find a command to train, and four to apply the trained model on the training, validation and test dataset as well as real data. You will need to modify the directories there accordingly.
After successful training, you can investigate the output using several different notebooks: The three notebooks '{Network} Analysis.ipynb' create the ROC curves and figures of merit, among other things. 'Data{Network}Ana.ipynb' apply the trained models to real data (needs to be modified by 'cdst_merge.py' as well beforehand).
