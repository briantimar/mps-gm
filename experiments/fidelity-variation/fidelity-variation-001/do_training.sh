#!/bin/bash
#SBATCH --job-name=fidelity-var
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=2
#SBATCH --output=/home/btimar/slurm-logs/trset_size_scaling.out
#SBATCH --time=24:00:00

source ./training_config.sh
for name in "tfim_ground" "heisenberg_ground" "rydberg_ground" "ghz"; do
    for (( j=1 ; j <= $NUM_BATCH_SIZE; j=j+1 )); do
            BATCH_SIZE=$((j*BATCH_SIZE_STEP))
        for (( seed=0 ; seed<$NUMSEED ; seed=seed+1 )); do
            srun -N 1 -n 1 python train_single.py
        done
    done
done
