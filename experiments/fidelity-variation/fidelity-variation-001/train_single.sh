#!/bin/bash
#SBATCH --job-name=fidelity-var
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

date
module load python
# args: name, system size, batch_size, seed
python train_single.py $1 $2 $3 $4
date
