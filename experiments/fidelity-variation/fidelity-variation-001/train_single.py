from config import get_data_paths, get_qutip_path
import sys
import json
import os
sys.path.append('../../..')
from mps.utils import two_phase_training

def get_training_settings(batch_size, qutip_path):
    with open("settings.json") as f:
        settings = json.load(f)
    settings['batch_size'] = batch_size
    settings['qutip_path'] = qutip_path
    return settings

def get_base_path(name, system_size, batch_size):
    """ Base path for saving the trained model data."""
    from config import SAVEDIR
    return os.path.join(SAVEDIR, "{0}_L={1}_batch_size={2}".format(name, system_size, batch_size))

def get_model_path(name, system_size, batch_size):
    return get_base_path(name, system_size, batch_size) + "_model"

def get_logdict_path(name, system_size, batch_size):
    return get_base_path(name, system_size, batch_size) + "_logdict.json"


def train(name, system_size, batch_size, seed=0, numpy_seed=0):
    """ Train MPS on a particular state type and system size, using the batch size specified
    and the other training settings specified in `settings.json`.

    The trained model, as well as the training logdict, are written into the SAVEDIR specified in config.
    """
    qutip_path = get_qutip_path(name, system_size)
    settings_path, samples_path = get_data_paths(name, system_size)
    training_settings = get_training_settings(batch_size, qutip_path)
    model, logdict, meta = two_phase_training(samples_path, settings_path, training_settings, 
                                                seed=seed,
                                                numpy_seed=numpy_seed)
    
    model.save(get_model_path(name, system_size, batch_size))
    with open(get_logdict_path(name, system_size, batch_size), 'w') as f:
        json.dump(logdict, f)


if __name__ == '__main__':
    name = sys.argv[1]
    system_size = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    seed = int(sys.argv[4])
    print("Now training on {0}, system size {1}, batch size {2}, seed {3}".format(name, system_size, batch_size, seed))
    train(name, system_size, batch_size, seed=seed)
