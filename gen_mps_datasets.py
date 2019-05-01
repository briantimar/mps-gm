import numpy as np
import torch
from models import MPS
from utils import build_ghz_plus, build_random_mps, draw_random
#which values of system size to try
system_sizes = np.linspace(4, 30, 2,dtype=int)

#for the random states, bond dimensions to try
bond_dims = np.linspace(2, 20, 2, dtype=int)

#where to write the data
savedir = "datasets/mps_sampled/"

#number of samples to draw per dataset
Nsamp = int(2e4)

np.random.seed(0)

for L in system_sizes:
    print("Sampling from system size %d" %L)
    print("Sampling from GHZ+ state")
    ghz_plus = build_ghz_plus(L)
    angles, outcomes = draw_random(ghz_plus, Nsamp)
    np.save(savedir + "ghz_plus_L=%d_angles"%L, angles)
    np.save(savedir + "ghz_plus_L=%d_outcomes" % L,outcomes)
    ghz_plus.save(savedir + "ghz_plus_L=%d_state"%L)

    print("sampling from random states")
    for D in bond_dims:
        psi = build_random_mps()
