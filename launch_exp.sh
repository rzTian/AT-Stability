#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=v100l:2
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M        # memory per node
#SBATCH --time=00-20:00  # time (DD-HH:MM)
#SBATCH --output=./results_extra/%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=rtian081@uottawa.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-1


b=(2 6)



module load StdEnv/2020
module load python/3.8
module load scipy-stack/2020a
source $HOME/ENV-3.8.10/bin/activate  
python train_cifar10.py  --dataset 'cifar10' --model 'PreRes18_standard' --lr_max 0.1  --pgd_alpha 2.0 --random_init 1  --att_method 'norm_steep'  --beta ${b[$SLURM_ARRAY_TASK_ID]}  --seed 3