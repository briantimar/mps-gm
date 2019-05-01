import sys
import numpy as np
import torch

sys.path.append('../..')
from models import MPS
from utils import do_local_sgd_training, MeasurementDataset, do_training
from qtools import pauli_exp

DATA_ROOT = "../../datasets/mps_sampled/"

def load_ghz_plus(L):
    template = "ghz_plus_L={0}_"
    angles = np.load(DATA_ROOT + template.format(L) + "angles.npy")
    outcomes = np.load(DATA_ROOT + template.format(L) + "outcomes.npy")
    psi = MPS(L, local_dim=2, bond_dim=2)
    psi.load(DATA_ROOT + template.format(L) + "state")
    return angles, outcomes, psi

def load_random(L, D):
    template = "rand_L={0}_D={1}_"
    angles = np.load(DATA_ROOT + template.format(L,D) + "angles.npy")
    outcomes = np.load(DATA_ROOT + template.format(L,D) + "outcomes.npy")
    psi = MPS(L, local_dim=2, bond_dim=D)
    psi.load(DATA_ROOT + template.format(L,D) + "state")
    return angles, outcomes, psi

def train(type, L, D, tr_set_size,
                learning_rate, batch_size, epochs, s2_schedule=None,
             cutoff=1e-10,seed=None,):
    """ Train a model on dataset of specified type, size, and bond dim.
    returns: trained MPS, logdict"""

    if type not in ['ghz', 'random']:
        raise ValueError
    #load training data and ground truth for a given state type
    if type=='ghz':
        angles, outcomes, psi_gt= load_ghz_plus(L)
    else:
        angles, outcomes, psi_gt = load_random(L, D)
    
    angles = angles[:tr_set_size]
    outcomes = outcomes[:tr_set_size]
    return do_training(angles, outcomes, learning_rate,
                            batch_size=batch_size,epochs=epochs,
                            s2_schedule=s2_schedule,cutoff=cutoff,
                            ground_truth_mps=psi_gt,seed=seed)


if __name__ == '__main__':

    system_sizes = np.arange(4, 30, 4, dtype=int)

    learning_rate = 1e-2
    batch_size = 256
    epochs = 3
    tr_set_sizes = np.linspace(1e3,1e4,10,dtype=int)

    #for the random states, bond dimensions to try
    bond_dims = np.arange(2, 20, 4, dtype=int)

    np.save("data/tr_set_sizes", tr_set_sizes)
    np.save("data/system_sizes", system_sizes)

    #fix bond dimension and scale L
    D = 2
    nseed = 1
    fidelities_scaling_L = np.empty((len(system_sizes), len(tr_set_sizes), nseed))
    print("now scaling system size, random targets")
    for i in range(len(system_sizes)):
        L = int(system_sizes[i])
        for j in range(len(tr_set_sizes)):
            N = int(tr_set_sizes[j])
            for k in range(nseed):
                seed=k
                print("training with L, N, D, seed =  ",L, N,D, seed)
                psi, logdict = train('random',L,D,N,learning_rate,batch_size,epochs,seed=seed)
                fidelities_scaling_L[i,j,k] = logdict['fidelity'][-1]

    np.save("data/rand_fidelities_scaling_L_D=%d"%D, fidelities_scaling_L)

    #fix L and scale bond dimension
    L=4
    nseed = 5
    fidelities_scaling_D = np.empty(
        (len(bond_dims), len(tr_set_sizes), nseed))
    print("now scaling bond dim, random targets")
    for i in range(len(bond_dims)):
        D = int(bond_dims[i])
        for j in range(len(tr_set_sizes)):
            N = int(tr_set_sizes[j])
            for k in range(nseed):
                seed = k
                print("training with L, N,D, seed =  ", L, N,D, seed)
                psi, logdict = train(
                    'random', L, D, N, learning_rate, batch_size, epochs, seed=seed)
                fidelities_scaling_D[i, j, k] = logdict['fidelity'][-1]
    np.save("data/rand_fidelities_scaling_D_L=%d"%L, fidelities_scaling_D)
