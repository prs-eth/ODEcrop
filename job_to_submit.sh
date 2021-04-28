#!/bin/bash
#BSUB -W 4:00
##BSUB -o /cluster/work/igp_psr/metzgern/Network/outputs/outputtrain_Le${object}.%J.%I.txt
##BSUB -e /cluster/work/igp_psr/metzgern/Network/outputs/train_Le${object}.%J.%I.txt
#BSUB -R "rusage[mem=24000,ngpus_excl_p=1]"
#BSUB -n 1
##BSUB -Is 
#BSUB -N
#BSUB -B
#### BEGIN #####

module load eth_proxy
module load python_gpu/3.7.4 cudnn/7.5 cuda/10.0.130 #hdf5/2.10.0 #2.7.14 #numpy/1.16.3 matplotlib/2.1.1 Keras/2.2.4

# run wandb command
#wandb agent cropteam/odecropclassification/idqkj5ta #TUM
#wandb agent cropteam/odecropclassification/zjufporp #swissdata

#wandb agent cropteam/odecropclassification/84udfiku #TUM GRU downsampling

wandb agent cropteam/odecropclassification/nvntbnu3