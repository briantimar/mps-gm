import sys
import os
import numpy as np
import torch

DEPTH=2
DIR = os.path.dirname(os.path.abspath(__file__)).split('/')
ROOT_DIR = '/'
for i in range(len(DIR)-DEPTH):
    ROOT_DIR = os.path.join(ROOT_DIR, DIR[i])

sys.path.append(ROOT_DIR)

from mps.models import MPS, ComplexTensor
from mps.utils import select_hyperparams, get_dataset_from_settings_and_samples
from torch.utils.data import random_split
torch.manual_seed(12)

#system size
L=4
#max epochs of training
epochs = 500
#svd cutoff 
cutoff = 1e-3
# max bond dimension
max_sv = 25
#batch size
batch_size = 256

use_cache = True
early_stopping=True
verbose = True
hold_early_cutoff=True

val_split = .2

#total dataset size
N = 20000
Ntr = int( N * (1 - val_split))

DATA_ROOT = os.path.join(ROOT_DIR, 'datasets', 'mps_sampled')
fname_outcomes = os.path.join(DATA_ROOT, 'ghz_plus_L=%d_outcomes.npy'%L)
fname_angles =  os.path.join(DATA_ROOT, 'ghz_plus_L=%d_angles.npy'%L)

ds = get_dataset_from_settings_and_samples(fname_outcomes, fname_angles,numpy_seed=12,N=N)
train_ds, val_ds = random_split(ds, [Ntr, N-Ntr])

ground_truth_mps = MPS(L, 2, 2)
ground_truth_mps.load( os.path.join(DATA_ROOT, 'ghz_plus_L=%d_state'%L))

if verbose:
    print("Successfully loaded settings, samples, and mps for GHZ size L=%d"%L)
    print("total number of samples: %d"%N)


## params to select with val set
Nparam=50
lr_scale = 10**np.random.uniform(-7, -3, Nparam)
lr_timescale = np.random.uniform(.5, 10, Nparam) * epochs
s2_scale = 10**np.random.uniform(-7, 0, Nparam)
s2_timescale = np.random.uniform(.2, 10, Nparam) * epochs

seeds = range(1)

np.save(os.path.join('data', 'lr_scale'), lr_scale)
np.save(os.path.join('data', 'lr_timescale'), lr_timescale)
np.save(os.path.join('data', 's2_scale'), s2_scale)
np.save(os.path.join('data', 's2_timescale'), s2_timescale)


params, trlosses, vallosses = select_hyperparams(train_ds, val_ds, batch_size, epochs,
                                                    Nparam=Nparam,lr_scale=lr_scale, lr_timescale=lr_timescale,
                                                    s2_scale=s2_scale, s2_timescale=s2_timescale, cutoff=cutoff,
                                                    max_sv_to_keep=max_sv, use_cache=use_cache, seed=seeds,
                                                    early_stopping=early_stopping,
                                                    hold_early_cutoff=hold_early_cutoff, verbose=verbose)
print("Finished hyperparam selection")
np.save(os.path.join('data', 'validated_params'), params)
np.save(os.path.join('data', 'trlosses'), trlosses)
np.save(os.path.join('data', 'vallosses'), vallosses)