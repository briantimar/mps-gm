#!/bin/bash

OUTPUT_TEMPLATE="/home/btimar/slurm-logs/fidelity-variation-"

source ./training_config.sh
for name in "tfim_ground" "heisenberg_ground" "rydberg_ground" "ghz"; do
    for (( seed=0 ; seed<$NUMSEED ; seed=seed+1 )); do
        OUTPUT="$OUTPUT_TEMPLATE$name_$seed"
        sbatch -N 1 -n 1 --output $OUTPUT train_single.sh $name $SYSTEM_SIZE $BATCH_SIZE $seed &
    done
done
