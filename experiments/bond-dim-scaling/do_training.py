import json
import sys
sys.path.append('../..')
from mps.utils import two_phase_training
import numpy as np
import os

L=6
Dtarget=14

Dvals = list(range(2, 16, 2))

numseed = 20
N=10000
numD = len(Dvals)

metadata = dict(L=L,Dtarget=Dtarget,numseed=numseed,N=N,numD=numD,Dvals=Dvals)

target_root = "../../datasets/mps_sampled/rand_L=%d_D=%d_"%(L,Dtarget)
fname_angles = target_root + "angles.npy"
fname_outcomes = target_root + "outcomes.npy"
fname_state = target_root + "state"


trsettings = dict(epochs=10,lr_scale=1e-3,lr_timescale=1000,s2_scale=0, s2_timescale=1000,
                     batch_size=512,cutoff=1e-3, val_fraction=.2, early_stopping=False, 
                     mps_path=fname_state, samples_per_epoch=2)

data_dir = "data_varying_numpy_seed"

with open(os.path.join(data_dir, "metadata.json"), 'w') as f:
    json.dump(metadata, f)

for max_sv in Dvals:
    trsettings['max_sv']=max_sv
    for ii in range(numseed):
        seed = (max_sv*numD +ii)
        print("Training with maxsv=%d, seed=%d"%(max_sv, ii))
        __, logdict, __ = two_phase_training(fname_outcomes, fname_angles, trsettings,
                                             N=N,seed=seed, numpy_seed=seed,verbose=False)
        with open(os.path.join(data_dir, "logdict_maxsv=%d_seed=%d.json"%(max_sv,ii)), 'w') as f:
            json.dump(logdict, f)
        