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

from models import MPS, ComplexTensor
from utils import select_hyperparams
from utils import MeasurementDataset
from qtools import pauli_exp
from torch.utils.data import random_split

#system size
L=4
#epochs of training
epochs = 500
#svd cutoff 
cutoff = 1e-5
# max bond dimension
max_sv = 10
#batch size
batch_size = 1028


use_cache = True
early_stopping=True
verbose = True

val_split = .2

#total dataset size
N = 20000
np.random.seed(0)

DATA_ROOT = os.path.join(ROOT_DIR, 'datasets', 'mps_sampled')
pauli_outcomes = np.load(os.path.join(DATA_ROOT, 'ghz_plus_L=%d_outcomes.npy'%L))
angles = np.load(os.path.join(DATA_ROOT, 'ghz_plus_L=%d_angles.npy'%L))

np.random.shuffle(pauli_outcomes)
np.random.shuffle(angles)
pauli_outcomes=pauli_outcomes[:N, ...]
angles = angles[:N, ...]

ground_truth_mps = MPS(L, 2, 2)
ground_truth_mps.load( os.path.join(DATA_ROOT, 'ghz_plus_L=%d_state'%L))

if verbose:
    print("Successfully loaded settings, samples, and mps for GHZ size L=%d"%L)
    print("total number of samples: %d"%N)

spinconfig = torch.tensor((1 -pauli_outcomes)/2, dtype=torch.long)
theta = torch.tensor(angles[..., 0],dtype=torch.float)
phi = torch.tensor(angles[..., 1], dtype=torch.float)
rotations = pauli_exp(theta, phi)

ds = MeasurementDataset(spinconfig, rotations)
Nval = int(val_split * N)
Ntr = N - Nval
train_ds, val_ds = random_split(ds, [Ntr, Nval])

## params to select with val set
Nparam=50
lr_scale = 10**np.random.uniform(-7, 0, Nparam)
lr_timescale = np.random.uniform(.5, 10, Nparam) * epochs
s2_scale = 10**np.random.uniform(-7, 0, Nparam)
s2_timescale = np.random.uniform(.2, 10, Nparam) * epochs

seeds = range(3)

params, trlosses, vallosses = select_hyperparams(train_ds, val_ds, batch_size, epochs,
                                                    Nparam=Nparam,lr_scale=lr_scale, lr_timescale=lr_timescale,
                                                    s2_scale=s2_scale, s2_timescale=s2_timescale, cutoff=cutoff,
                                                    max_sv_to_keep=max_sv, use_cache=use_cache, seed=seeds,
                                                    early_stopping=early_stopping, verbose=verbose)
print("Finished hyperparam selection")
np.save(os.path.join('data', 'validated_params'), params)
np.save(os.path.join('data', 'trlosses'), trlosses)
np.save(os.path.join('data', 'vallosses'), vallosses)