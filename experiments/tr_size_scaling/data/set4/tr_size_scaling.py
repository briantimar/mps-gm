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
             cutoff=1e-10, max_sv_to_keep = None,seed=None,):
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
    model, logdict= do_training(angles, outcomes, learning_rate,
                            batch_size=batch_size,epochs=epochs,
                            s2_schedule=s2_schedule,cutoff=cutoff,
                            max_sv_to_keep=max_sv_to_keep,
                            ground_truth_mps=None,seed=seed)
    f = np.abs(model.overlap(psi_gt)) / model.norm_scalar() 
    return f

if __name__ == '__main__':

    system_sizes = np.arange(4, 22, 2, dtype=int)

    learning_rate = 1e-2
    batch_size = 1028
    epochs = 10
    tr_set_sizes = np.linspace(2e3,1.5e4,40,dtype=int)
    max_sv = 20
    max_sv_to_keep = lambda ep: 2 if ep < 1 else max_sv
    cutoff=1e-2
    use_cache = True
    s2_schedule = lambda ep: np.exp(-ep)


    target_fidelity = .99

    #for the random states, bond dimensions to try
    bond_dims = np.arange(2, 20, 2, dtype=int)

    np.save("data/tr_set_sizes", tr_set_sizes)
    np.save("data/system_sizes", system_sizes)
    np.save("data/bond_dims", bond_dims)
    #fix bond dimension and scale L
    D = 2
    nseed = 5
    fidelities_scaling_L = -1 * np.ones((len(system_sizes), len(tr_set_sizes), nseed))
    print("now scaling system size, random targets")
    for i in range(len(system_sizes)):
        L = int(system_sizes[i])
        for k in range(nseed):
            seed=k
            f=-1
            j = 0
            while j < len(tr_set_sizes): #and f < target_fidelity:
                N = int(tr_set_sizes[j])
                print("training with L, N, D, seed =  ",L, N,D, seed)
                f = train('random',L,D,N,learning_rate,batch_size,epochs,
                                    s2_schedule=s2_schedule, max_sv_to_keep=max_sv_to_keep,cutoff=cutoff,seed=seed)
                fidelities_scaling_L[i,j,k] = f
                print("f = ", f)
                j += 1
        

    np.save("data/rand_fidelities_scaling_L_D=%d"%D, fidelities_scaling_L)

    #fix L and scale bond dimension
    L=8
    nseed = 5
    fidelities_scaling_D = -1 * np.ones(
        (len(bond_dims), len(tr_set_sizes), nseed))
    print("now scaling bond dim, random targets")
   
    for i in range(len(bond_dims)):
        D = int(bond_dims[i])
        for k in range(nseed):
            seed=k
            f=-1
            j = 0
            while j < len(tr_set_sizes): #and f < target_fidelity:
                N = int(tr_set_sizes[j])
                print("training with L, N, D, seed =  ",L, N,D, seed)
                f = train('random',L,D,N,learning_rate,batch_size,epochs,
                                    s2_schedule=s2_schedule, max_sv_to_keep=max_sv_to_keep,cutoff=cutoff,seed=seed)
                fidelities_scaling_D[i,j,k] = f
                print("f = ", f)
                j += 1
    np.save("data/rand_fidelities_scaling_D_L=%d"%L, fidelities_scaling_D)
