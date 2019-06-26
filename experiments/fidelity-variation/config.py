import os
import numpy as np
import qutip as qt

#where exact qutip states are stored
STATES_ROOT = "/Users/btimar/Dropbox/data/states/qutip"
#where the corresponding random-unitary data are located
DATA_ROOT = "/Users/btimar/Dropbox/data/random_unitary_data/from_qutip_states/seed_3"

state_names = ["ghz", "heisenberg_ground", "rydberg_ground", "tfim_ground"]
system_sizes = [2,4,6,8]

def get_qutip_path(name, system_size):
    """ Returns path to the qutip ground-truth state for a particular system size."""
    if name not in state_names or system_size not in system_sizes:
        raise ValueError("state does not exist")
    state_path = os.path.join(STATES_ROOT, "{0}_L={1}".format(name, system_size))
    return state_path

def get_data_paths(name, system_size):
    """ Returns paths to measurement settings and samples (pauli eigenvalues) for a particular state
    type and system size."""
    if name not in state_names or system_size not in system_sizes:
        raise ValueError("state does not exist")
    base_path = os.path.join(DATA_ROOT, "{0}_L={1}".format(name, system_size))
    settings_path = base_path + "_settings.npy"
    samples_path = base_path + "_samples.npy"
    return settings_path, samples_path
