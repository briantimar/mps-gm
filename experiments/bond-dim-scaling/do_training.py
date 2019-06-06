import json
import sys
sys.path.append('../..')
from mps.utils import two_phase_training
import numpy as np

L=6
Dtarget=14

Dvals = list(range(2, 20, 2))

numseed = 20
N=10000
numD = len(Dvals)

metadata = dict(L=L,Dtarget=Dtarget,numseed=numseed,N=N,numD=numD)

target_root = "../../datasets/mps_sampled/rand_L=%d_D=%d_"%(L,Dtarget)
fname_angles = target_root + "angles.npy"
fname_outcomes = target_root + "outcomes.npy"
fname_state = target_root + "state"


trsettings = dict(epochs=10,lr_scale=1e-3,lr_timescale=1000,s2_scale=0, s2_timescale=1000,
                     batch_size=512,cutoff=1e-3, val_fraction=.2, early_stopping=False, 
                     mps_path=fname_state, samples_per_epoch=2)

with open("data/metadata.json", 'w') as f:
    json.dump(metadata, f)

for max_sv in Dvals:
    trsettings['max_sv']=max_sv
    for ii in range(numseed):
        print("Training with maxsv=%d, seed=%d"%(max_sv, ii))
        __, logdict, __ = two_phase_training(fname_outcomes, fname_angles, trsettings,
                                             N=N,seed=ii,verbose=False)
        np.save("data/loss_maxsv=%d_seed=%d"%(max_sv, ii), logdict['loss'])
        np.save("data/val_loss_maxsv=%d_seed=%d"%(max_sv, ii), logdict['val_loss'])
        np.save("data/fidelity_maxsv=%d_seed=%d"%(max_sv, ii), logdict['fidelity_mps'])
        np.save("data/max_bond_dim_maxsv=%d_seed=%d"%(max_sv, ii), logdict['max_bond_dim'])
        np.save("data/s2_maxsv=%d_seed=%d"%(max_sv, ii), logdict['s2'])