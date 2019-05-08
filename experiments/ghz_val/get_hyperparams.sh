#!/bin/bash

OUTPATH="./data"
SETTINGSPATH="../../datasets/mps_sampled/ghz_plus_L=4_angles.npy"
SAMPLESPATH="../../datasets/mps_sampled/ghz_plus_L=4_outcomes.npy"

python ../../do_hyperparam_selection.py $SAMPLESPATH $SETTINGSPATH $OUTPATH --epochs 500 --batch_size 1024 --numseed 5 --max_sv 10 --cutoff 1e-5 --size 20000