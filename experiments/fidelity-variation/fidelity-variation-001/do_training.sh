#!/bin/bash
#SBATCH --job-name=fidelity-var
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=2
#SBATCH --output=/home/btimar/slurm-logs/trset_size_scaling.out
#SBATCH --time=24:00:00

date
module load python
source ./training_config.sh
for name in "tfim_ground" "heisenberg_ground" "rydberg_ground" "ghz"; do
    for (( seed=0 ; seed<$NUMSEED ; seed=seed+1 )); do
        srun -N 1 -n 1 python train_single.py $name $SYSTEM_SIZE $BATCH_SIZE $seed &
    done
done
wait
date