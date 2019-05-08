""" Select hyperparams using out-of-sample performance on specified dataset """

import sys
import os
import numpy as np
import argparse

#depth, relative to the root, of this file
DEPTH=0
DIR = os.path.dirname(os.path.abspath(__file__)).split('/')
ROOT_DIR = '/'
for i in range(len(DIR)-DEPTH):
    ROOT_DIR = os.path.join(ROOT_DIR, DIR[i])

sys.path.append(ROOT_DIR)

from utils import select_hyperparams_from_filepath

# some default settings
Nparam=50
EPOCHS = 500
lr_scale = 10**np.random.uniform(-7, 0, Nparam)
lr_timescale = np.random.uniform(.5, 10, Nparam) * EPOCHS
s2_scale = 10**np.random.uniform(-7, 0, Nparam)
s2_timescale = np.random.uniform(.2, 10, Nparam) * EPOCHS

NUMSEED=5
EARLY_STOPPING=True
MAX_SV=50
BATCH_SIZE=1024
CUTOFF=1e-5
VAL_SPLIT=.2

parser = argparse.ArgumentParser(description="Selects hyperparameters for MPS training ")
parser.add_argument('outcomes_path', help="Filepath to numpy array holding Pauli measurement outcomes")
parser.add_argument('angles_path', help="Filepath to numpy array holding local-rotation angles")
parser.add_argument('output_dir', help="Directory to write validated params and losses")
parser.add_argument('--epochs', help="Number of epochs to train.", 
                                default=EPOCHS)
parser.add_argument('--batch_size', help="Batch size for training", 
                                default=BATCH_SIZE)

parser.add_argument('--numseed', help="How many different MPS models to average over for a given hyperparameter setting", 
                                default=NUMSEED)
parser.add_argument('--early_stopping', help="Whether or not to stop training early based on val performance", 
                                default=EARLY_STOPPING)
parser.add_argument('--max_sv', help="Max number of singular values to keep at truncation steps", 
                                default=MAX_SV)
parser.add_argument('--cutoff', help="Truncation threshold for singular values", 
                                default=CUTOFF)
parser.add_argument('--size', help="Size of the total dataset (tr + val) to use. If not provided, full dataset is used", 
                                default=None)
parser.add_argument('--val_split', help="Fraction of the data to use for validation", 
                                default=VAL_SPLIT)
parser.add_argument('--verbose', help="Whether to print training information", 
                                default=True)
if __name__ == '__main__': 
    args = parser.parse_args()
    select_hyperparams_from_filepath(args.outcomes_path, args.angles_path, args.output_dir,
                                      lr_scale, lr_timescale, s2_scale, s2_timescale, 
                                       numpy_seed=0, 
                                    N=args.size, val_split=args.val_split, 
                                    Nparam=Nparam, nseed=args.numseed, 
                                    epochs=args.epochs, cutoff=args.cutoff, max_sv=args.max_sv, batch_size=args.batch_size, 
                                    use_cache=True, early_stopping=args.early_stopping, verbose=args.verbose)