"""Do training on a particular numpy dataset. """

import sys
import os
import numpy as np
import argparse
import json
#depth, relative to the root, of this file
DEPTH=0
DIR = os.path.dirname(os.path.abspath(__file__)).split('/')
ROOT_DIR = '/'
for i in range(len(DIR)-DEPTH):
    ROOT_DIR = os.path.join(ROOT_DIR, DIR[i])

sys.path.append(ROOT_DIR)

from mps.utils import train_from_filepath


#some default settings
RECORD_EIGS = False
RECORD_S2 = False
COMPUTE_OVERLAPS = False
USE_CACHE = True

parser = argparse.ArgumentParser(description="""Trains an MPS on the specified dataset and saves it. Training hyperparameters should be specified in a single json
                                                file; other training options are available as command line flags.""")
parser.add_argument('outcomes_path', help="Filepath to numpy array holding Pauli measurement outcomes")
parser.add_argument('angles_path', help="Filepath to numpy array holding local-rotation angles")
parser.add_argument('training_settings', help="Path to json file holding training metadata.")
parser.add_argument('output_dir', help="Directory to write validated params and losses")


parser.add_argument('--size', help="Size of the total dataset (tr + val) to use. If not provided, full dataset is used", 
                                default=None,
                                type=int)

parser.add_argument('--npseed', help="Numpy seed for shuffling the dataset, if desired.", 
                                default=None,
                                type=int)

parser.add_argument('--seed', help="Torch seed for the MPS, if desired.", 
                                default=None,
                                type=int)

parser.add_argument('--record_eigs', help="Whether or not to record some MPS singular values during training.", 
                                type=bool,
                                default=RECORD_EIGS,
                                action='store_true')

parser.add_argument('--record_s2', help="Whether or not to record Renyi entropies during training.", 
                                type=bool,
                                default=RECORD_S2,
                                action='store_true')

parser.add_argument('--compute_overlaps', help="Whether to compute random-unitary estimates of state overlap with data.", 
                                type=bool,
                                default=COMPUTE_OVERLAPS,
                                action='store_true')
                               
parser.add_argument('--verbose', help="Whether to print training information", action='store_true',
                                default=True)
if __name__ == '__main__': 
    args = parser.parse_args()
    outcomes_path = args.outcomes_path
    angles_path = args.angles_path
    output_dir = args.output_dir
    training_settings = args.training_settings

    N = args.size
    numpy_seed =args.npseed
    seed = args.seed
    record_eigs = args.record_eigs
    record_s2 = args.record_s2
    compute_overlaps = args.compute_overlaps
    use_cache = USE_CACHE
    verbose = args.verbose
    
    model, logdict, meta = train_from_filepath(outcomes_path, angles_path, training_settings, 
                                                numpy_seed=numpy_seed,N=N,seed=seed,
                                                record_eigs=record_eigs,record_s2=record_s2,
                                                compute_overlaps=compute_overlaps,use_cache=use_cache,
                                                verbose=verbose)
    model.save(os.path.join(output_dir, 'trained_model'))
    with open(os.path.join(output_dir, 'logdict'), 'w') as f:
        json.dump(logdict, f)
    with open(os.path.join(output_dir, 'metadata'), 'w') as f:
        json.dump(meta, f)