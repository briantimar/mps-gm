from torch.utils.data import TensorDataset
import numpy as np
import scipy as sp
import qutip as qt
import scipy.linalg
from .zgesvd import svd_zgesvd
import torch
import warnings
import json

def svd(A, cutoff=1e-8, max_sv_to_keep=None):
    """ Perform singular value decomp of complex matrix A.
        Singular values below cutoff will be dropped.
        max_sv_to_keep: if not None, max number of singular values to keep
        (the largest will be kept)

        Returns:
        u, s, v
        where A \approx usv, A.shape=(N,M)
        u.shape = (N,k)
        s.shape = (k,)
        v.shape = (k,M)
        and k is the number of singular values retained.
        """
    try:
        if A.dtype in (np.complex128, np.complex64):
            u,s,v = svd_zgesvd(A, full_matrices=False, compute_uv=True)
        else:
            u,s,v = sp.linalg.svd(A, full_matrices=False)
    except Exception as e:
        print(e)

    singular_vals = np.diag(s)
    #keep only singular values above cutoff
    if cutoff is not None:
        singular_vals = singular_vals[singular_vals>cutoff]
    #if a max number of eigvalues is requested, enforce that too
    if max_sv_to_keep is not None:
        singular_vals = singular_vals[:max_sv_to_keep]
    #number of singular values
    k=len(singular_vals)
    #update the u and v matrices accordingly
    u = u[:, :k]
    v = v[:k, :]
    return u, singular_vals, v

def svd_push_right(Aleft, Aright, cutoff=1e-16, max_sv_to_keep=None):
    """Perform SVD on the left site matrix, and push singular values into the right site matrix.
        This results in the left site-matrix being left-normalized.
        Aleft: complex numpy array, shape (local_dim, bond_dim1, bond_dim2)
        Aright: complex numpy array, shape (local_dim, bond_dim2, bond_dim3).
        
        Singular values below cutoff will be dropped.
        max_sv_to_keep: if not None, max number of singular values to keep"""

    local_dim, D1, D2 = Aleft.shape
    # pull both indices over to the left side
    Aleft = np.reshape(Aleft, (local_dim * D1, D2))
    #svd on the D2 index
    u,singular_vals, v = svd(Aleft, cutoff=cutoff,max_sv_to_keep=max_sv_to_keep)
    k = len(singular_vals)
    #the unitary u defines the new left site matrix
    Aleft_new = np.reshape(u, (local_dim,D1,k))
    #contract s and v with the old right-matrix to define the new right-matrix
    Aright_new = np.einsum('q,qi,sij->sqj',singular_vals,v,Aright)
    return Aleft_new, Aright_new
    
def svd_push_left(Aleft, Aright, cutoff=1e-16, max_sv_to_keep=None):
    """Perform SVD on the right site matrix, and push singular values into the left site matrix.
        This results in the right site-matrix being right-normalized.
        Aleft: complex numpy array, shape (local_dim, bond_dim1, bond_dim2)
        Aright: complex numpy array, shape (local_dim, bond_dim2, bond_dim3)"""

    local_dim, D2, D3 = Aright.shape
    #pull spin index to the right
    Aright = np.reshape(np.swapaxes(Aright, 0,1), (D2, local_dim * D3))
    #perform SVD on the D2-index
    u,singular_vals,v = svd(Aright, cutoff=cutoff,max_sv_to_keep=max_sv_to_keep)
    k=len(singular_vals)
  
    #new right-matrix, with shape local_dim, k, D3
    Aright_new = np.swapaxes(v.reshape( (k, local_dim, D3)), 0, 1)
    #push u and s to the left to get new left-matrix
    Aleft_new = np.einsum('sij,jq,q->siq', Aleft,u,singular_vals)

    return Aleft_new, Aright_new

def split_two_site(A, normalize='left', cutoff=1e-8, max_sv_to_keep=None):
    """Perform SVD on two-site tensor A, and normalize either the left or right tensor 
    of the result.
    A: two-site complex array, shape (local_dim, local_dim, D1, D2) 
        (left and right spin indices, and left and right bond dim resp.) 
    
        Returns: single-site tensors Aleft, Aright
        shape(Aleft) = (local_dim, D1, k)
        shape(Aright) = (local_dim, k, D2)
        k = number of singular values retained."""

    _, local_dim, D1, D2 = A.shape
    if A.shape[0] != local_dim:
        raise ValueError("invalid shape {0} for two-site tensor".format(A.shape))
    if normalize not in ['left', 'right']:
        raise ValueError("Invalid normalization")
    #bend the spin indices over
    A = np.swapaxes(A, 1,2)
    A = np.reshape(A, (local_dim * D1, local_dim * D2))
    u, s, v = svd(A,cutoff=cutoff,max_sv_to_keep=max_sv_to_keep)
    k = len(s)
    if normalize == 'left':
        Aleft = np.reshape(u, (local_dim, D1, k))
        Aright = np.reshape(np.einsum('q,qi->qi',s,v), (k, local_dim, D2))
        Aright = np.swapaxes(Aright, 0,1)
    else:
        Aright = np.swapaxes(np.reshape(v, (k, local_dim, D2)), 0,1)
        Aleft = np.reshape(np.einsum('iq,q->iq',u,s), (local_dim, D1, k))
    
    return Aleft, Aright

def get_singular_vals(twosite_tensor, cutoff=1e-16, max_sv_to_keep=None):
    """ Extract singular values from a twosite tensor
        twosite_tensor: complex numpy array shape (local dim, localdim, bond dim1, bond dim2)
        returns: numpy array of singular values.
        
        cutoff: singular values below this cutoff will be discarded.
        max_sv_to_keep: if not None, the max number of SV's to keep.
        """
    _, local_dim, D1, D2 = twosite_tensor.shape
    if twosite_tensor.shape[0] != local_dim:
        raise ValueError("invalid shape {0} for two-site tensor".format(twosite_tensor.shape))
    A = np.swapaxes(twosite_tensor, 1,2)
    A = np.reshape(A, (local_dim * D1, local_dim * D2))
    __, singular_vals, __, = svd(A, cutoff=cutoff, max_sv_to_keep=max_sv_to_keep)
    return singular_vals**2

def make_onehot(int_tensor, n):
    """Return one-hot encoding of specified tensor.
        n = max integer value. Assumed that int_tensor only takes values in 0...n-1
        (right now this is not checked)
        int_tensor: (N,) integer tensor of labels
        Returns: floattensor, shape (N, n), of one-hot encoding"""
    onehot = torch.FloatTensor(int_tensor.size(0),n)
    onehot.zero_()
    dim=1
    indices = int_tensor.to(dtype=torch.long).view(-1,1)
    onehot.scatter_(dim,indices, 1)
    return onehot


### helper functions for building a few simple GHZ states

def build_ghz_plus(L):
    """ Return normalized MPS representing a GHZ+ state of length L"""
    from .models import MPS, ComplexTensor
    psi = MPS(L, local_dim=2, bond_dim=2)
    with torch.no_grad():
        A0r = torch.tensor([[0, 1], [0, 0]], dtype=torch.float)
        A1r = torch.tensor([[0, 0], [1, 0]], dtype=torch.float)
        Ar = torch.stack([A0r, A1r], 0)
        Ai = torch.zeros_like(Ar)
        ## Bulk tensor
        A = ComplexTensor(Ar, Ai)

        l0r = torch.tensor([[0, 1]], dtype=torch.float)
        l1r = torch.tensor([[1, 0]], dtype=torch.float)
        lr = torch.stack([l0r, l1r], 0)
        rr = torch.stack([l1r.view(2, 1)/np.sqrt(2),
                          l0r.view(2, 1)/np.sqrt(2)], 0)
        li = torch.zeros_like(lr)
        ri = torch.zeros_like(rr)
        #left edge tensor
        l = ComplexTensor(lr, li)
        #right edge tensor
        r = ComplexTensor(rr, ri)

    psi.set_local_tensor(0, l)
    psi.set_local_tensor(L-1, r)
    for i in range(1, L-1):
        psi.set_local_tensor(i, A)
    psi.gauge_index = None
    return psi


def build_uniform_product_state(L, theta, phi):
    """ Return uniform product state where qubit is in eigenstate of n \cdot \sigma, 
        n being the unit vector defined by polar angles (theta, phi) """
    from .models import MPS, ComplexTensor
    Ar = torch.tensor([np.cos(theta/2), np.sin(theta/2)
                       * np.cos(phi)]).view(2, 1, 1)
    Ai = torch.tensor([0., np.sin(phi)]).view(2, 1, 1)
    A = ComplexTensor(Ar, Ai)
    psi = MPS(L, local_dim=2, bond_dim=1)
    for i in range(L):
        psi.set_local_tensor(i, A)
    psi.gauge_index = None
    return psi

def build_random_mps(L, bond_dim):
    """ Build an mps with uniform bond dimension and random tensors."""
    from .models import MPS
    return MPS(L,local_dim=2,bond_dim=bond_dim)

def do_local_sgd_training(mps_model, dataloader, epochs, 
                            learning_rate, 
                            val_ds = None, 
                            s2_penalty=None,nstep=1,
                            cutoff=1e-10,max_sv_to_keep=None, 
                            ground_truth_mps = None, ground_truth_qutip = None,
                             verbose=False, use_cache=True, 
                            record_eigs=True, record_s2=True, early_stopping=True, 
                            compute_overlaps=True, spinconfig_all=None, rotations_all=None):
    """Perform SGD local-update training on an MPS model using measurement outcomes and rotations
    from provided dataloader.
        mps_model: an MPS
        dataloader: pytorch dataloader which yields dictionaries holding batches of spin configurations
        and local unitaires.
        epochs: int, number of epochs to train
        val_ds: if not None, validation dataset on which NLL will be computed after each epoch.
        learning_rate : lr for gradient descent. Scalar, or function of epoch which returns scalar.
        s2_penalty: coefficient of S2 regularization term. if not None, scalar or function of epoch which returns scalar
        nstep: how many gradient descent updates to make at each bond.
        cutoff: threshold below which to drop singular values
        max_sv_to_keep: if not None, max number of singular vals to keep at each bond, or function of batch number 
        that returns such
        ground_truth_mps: if not None, an MPS against which the model's fidelity will be checked after
        every sweep.
        ground_truth_qutip: if not None, qutip state against which the model's fidelity is checked after each epoch
        use_cache: whether to cache partial amplitudes during the sweeps.
        record_eigs: whether to record eigenvalues of the half-chain density op.
        record_s2: whether to record the Renyi-2 entropy of the half-chain density op.
        early_stopping: if True, halt training when val loss fails to decrease by more than 1e-3 in 5 epochs.
        compute_overlaps: whether to compute overlap estimates onto the target state.
            if true: spinconfig_all, rotations_all are used to compute the overlap estimates.
        Returns: dictionary, mapping:
                    'loss' -> batched loss function during training
                    'fidelity' -> if ground truth state was provided, array of fidelties during training.
                    'max_bond_dim' -> array of max bond dimensions
                    'eigenvalues' -> eigenvalues when partitioning at chain center, if requested
                    's2' -> renyi-2 entropy, if requested
            These quantities are recorded at the end of each epoch.
        """
    from .models import ComplexTensor
    import time
    if s2_penalty is None:
        s2_penalty = np.zeros(len(dataloader) * epochs)
    if compute_overlaps and ( spinconfig_all is None or rotations_all is None):
        raise ValueError(""" Provide spinconfig and rotation datasets to enable overlap estimation """)
    #system size
    L = mps_model.L
    #logging the loss function
    losses = []
    #if ground-truth MPS is known, compute fidelity at each step
    fidelities_mps = []
    #same, for ground_truth qutip state
    fidelities_qutip = []
    # record max bond_dim
    max_bond_dim = []
    #record eigenspectrum cutting across chain center
    eigenvalues = []
    #Renyi-2 entropy cutting across chain center
    s2 = []
    #losses on validation set
    val_loss = []
    #overlap estimates on the source state
    overlap = []
    overlap_err = []
    overlap_converged = []

    #how many epochs must val score fail to improve before we stop?
    NUM_EP_EARLY_STOP=5
    #val score must decrease by at least this fraction to count as improvement.
    REL_VAL_EARLY_STOP=1e-3

    for ep in range(epochs):
        t0=time.time()
        #load lr, s2 penalty, and max sv's for the epoch
        try:
            _s2_penalty = s2_penalty(ep)
        except TypeError:
            _s2_penalty = s2_penalty
        try:
            max_sv = max_sv_to_keep(ep)
        except TypeError:
            max_sv = max_sv_to_keep
        try:
            lr = learning_rate(ep)
        except TypeError:
            lr = learning_rate
            
        for step, inputs in enumerate(dataloader):
            #get torch tensors representing measurement outcomes, and corresponding local unitaries
            spinconfig = inputs['samples']
            rot = inputs['rotations']
            rotations = ComplexTensor(rot['real'], rot['imag'])

            #forward sweep across the chain
            if use_cache:
                mps_model.init_sweep('right', spinconfig,rotation=rotations)
            for i in range(L-1):
                if i == L//2 - 1:
                    if record_eigs:
                        eigs = mps_model.get_eigenvalues(i)
                        eigenvalues.append(list(eigs))
                    if record_s2:
                        s2.append(mps_model.renyi2_entropy(i))
                for __ in range(nstep):
                
                    #computes merged two-site tensor at bond i, and the gradient of NLL cost function
                    # with respect to this 'blob'; updates merged tensor accordingly, then breaks back to local tensors
                   
                    mps_model.do_sgd_step(i, spinconfig,
                                    rotation=rotations, cutoff=cutoff, direction='right',
                                    max_sv_to_keep=max_sv,
                                    learning_rate=lr, s2_penalty=_s2_penalty,use_cache=use_cache)
                
            #backward sweep across the chain
            if use_cache:
                mps_model.init_sweep('left', spinconfig, rotation=rotations)
            #need to gauge the MPS here so that cached partial amplitudes on the right are accurate
            mps_model.gauge_to(L-2)
            for i in range(L-3, 0, -1):
                for __ in range(nstep):
                    mps_model.do_sgd_step(i, spinconfig,
                                    rotation=rotations, cutoff=cutoff, direction='left',
                                     max_sv_to_keep=max_sv,
                                    learning_rate=lr, s2_penalty=_s2_penalty, use_cache=use_cache)


        with torch.no_grad():
            #record batched loss functions
            losses.append(mps_model.nll_loss(spinconfig, rotation=rotations).item())
            if ground_truth_mps is not None:
                fidelities_mps.append(np.abs(mps_model.overlap(ground_truth_mps)) / mps_model.norm_scalar() )
            if ground_truth_qutip is not None:
                fidelities_qutip.append( qt.fidelity(ground_truth_qutip, mps_model.to_qutip_ket()))
            max_bond_dim.append(mps_model.max_bond_dim)
            if compute_overlaps:
                mu, sig, convergence_acheived = estimate_overlap(mps_model, spinconfig_all, rotations_all, eps=1e-2, Nsample=10)
                overlap.append(mu)
                overlap_err.append(sig)
                overlap_converged.append(convergence_acheived)

            if val_ds is not None:
                val_loss.append(compute_NLL(val_ds, mps_model))
                if early_stopping and ep>NUM_EP_EARLY_STOP:
                    rel_val_loss_diff = (np.diff(val_loss)/val_loss[1:])[-5:]
                    if not (rel_val_loss_diff< - REL_VAL_EARLY_STOP).any():
                        if verbose:
                            print("Val score not decreasing, halting training")
                        break
        if verbose:
            print("Finished epoch {0} in {1:.3f} sec".format(ep, time.time() - t0))
            print("Model shape: ", mps_model.shape)
        
    return dict(loss=(losses),
                fidelity_mps=(fidelities_mps),
                fidelity_qutip = (fidelities_qutip),
                max_bond_dim=max_bond_dim, 
                eigenvalues=eigenvalues,
                s2=(s2), 
                val_loss=(val_loss), 
                overlap=dict(mean=(overlap), 
                            err=(overlap_err), 
                            converged=(overlap_converged)))
                
def draw_random(mps, N):
    """ Draw N samples from mps, each taken in a random basis.
        Returns: angles, outcomes
        where angles = (N, L, 2) tensor holding theta, phi angles 
        outcomes = (N, L) tensor of pauli eigenvalue outcomes."""
    from .qutip_utils import sample_random_angles
    from .qtools import pauli_exp
    angles = torch.tensor(sample_random_angles((N, mps.L)),dtype=torch.float)
    rotations = pauli_exp(angles[..., 0], angles[..., 1])
    index_outcomes = mps.sample(N, rotations=rotations).numpy()
    #convert to pauli eigenvalues
    pauli_eig_outcomes = 1 - 2 * index_outcomes
    return angles, pauli_eig_outcomes

def train_from_dataset(meas_ds,
                learning_rate, batch_size, epochs,
                val_ds=None,
                 s2_penalty=None, cutoff=1e-10,
                 max_sv_to_keep = None,
                ground_truth_mps=None, ground_truth_qutip=None, use_cache=True, seed=None, 
                record_eigs=False, record_s2=False, verbose=False, early_stopping=True,
                compute_overlaps=True):
    """ Given a MeasurementDataset ds, create and train an MPS on it.
        val_ds: if not None, validation dataset on which NLL will be computed after each epoch"""
    from torch.utils.data import DataLoader
    from .models import MPS
    if seed is not None:
        torch.manual_seed(seed)
    L = meas_ds[0]['samples'].size(0)
    N = len(meas_ds)
    #check format of rotations
    r0 = meas_ds[0]['rotations']
    if verbose:
        print("Training on system size %d with %d samples"%(L, N))
    spinconfig_all, rotations_all= None, None
    if compute_overlaps:
        #overlaps computed on full dataset
        spinconfig_all, rotations_all = meas_ds.unpack()
    dl = DataLoader(meas_ds, batch_size=batch_size, shuffle=True)
    #train a model
    model = MPS(L, local_dim=2, bond_dim=2)
    logdict = do_local_sgd_training(model, dl, epochs, learning_rate,
                                    val_ds=val_ds,
                                    s2_penalty=s2_penalty,cutoff=cutoff,
                                    max_sv_to_keep=max_sv_to_keep,
                                    use_cache=use_cache,
                                    ground_truth_mps=ground_truth_mps, ground_truth_qutip=ground_truth_qutip,
                                    record_eigs=record_eigs, record_s2=record_s2,
                                    early_stopping=early_stopping,verbose=verbose,
                                   compute_overlaps=compute_overlaps, spinconfig_all=spinconfig_all, rotations_all=rotations_all)
    return model, logdict

def do_training(angles, pauli_outcomes, 
                learning_rate, batch_size, epochs,
                 s2_penalty=None, cutoff=1e-10,
                 max_sv_to_keep = None,
                ground_truth_mps=None, use_cache=True, seed=None, 
                record_eigs=False, record_s2=False, verbose=False):
    """ Train MPS on given angles and outcomes.
        angles: (N, L, 2) numpy array of theta, phi angles
        pauli_outcomes: (N, L) numpy array of corresponding pauli eigenvalue outcomes.
        returns: trained mps and logdict holding loss and fidelity from training."""

   
   
    from .qtools import pauli_exp
    if seed is not None:
        torch.manual_seed(seed)
    
    if angles.shape[:2] != pauli_outcomes.shape:
        raise ValueError("angle and outcome arrays are incompatible")
    L = angles.shape[1]
    angles = torch.tensor(angles,
                          dtype=torch.float, requires_grad=False)
    
    spin_config_outcomes = torch.tensor(
        (1 - pauli_outcomes)/2, dtype=torch.long, requires_grad=False)

    #generate local unitaries from angles
    rotations = pauli_exp(angles[..., 0], angles[..., 1])
    ds = MeasurementDataset(spin_config_outcomes, rotations)
    return train_from_dataset(ds, learning_rate,batch_size, epochs, 
                                s2_penalty=s2_penalty, cutoff=cutoff, 
                                max_sv_to_keep=max_sv_to_keep, ground_truth_mps=ground_truth_mps, 
                                use_cache=use_cache, seed=seed,record_eigs=record_eigs, 
                                record_s2=record_s2, verbose=verbose)

def compute_NLL(meas_ds,
                model):
    """ Compute the NLL loss function using the given MPS model"""
    
    from .models import ComplexTensor
    spins_all = meas_ds[:]['samples']
    Ui = meas_ds[:]['rotations']['imag']
    Ur = meas_ds[:]['rotations']['real']
    U = ComplexTensor(Ur, Ui)
    return model.nll_loss(spins_all, rotation=U).item()


def do_validation(train_ds, val_ds, 
                 batch_size, epochs,
                 params, 
                 cutoff=1e-10,
                 max_sv_to_keep = None,
                 use_cache=True, seed=None, 
                 early_stopping=True,
                 verbose=False):
    """ Validate a set of parameters on held-out dataset.   
        params: list of dicts holding hyperparams. Each must have keys:
            'learning_rate'
            's2_penalty'
        seed: if not None, int, or list of ints. In the latter case, scores will be averaged over seeds.
        Returns: val scores (Nparam),  training losses , val losses. """
    Nparam = len(params)
    trlosses = []
    vallosses = []
    scores = np.empty(Nparam)

    try:
        nseed = len(seed)
        seeds = seed
    except TypeError:
        nseed = 1
        seeds = [seed]
    if verbose:
        print("Training on %d different param sets, with %d seeds each" % (Nparam, nseed))
    for i in range(Nparam):
        learning_rate = params[i]['learning_rate']
        s2_penalty = params[i]['s2_penalty']
        _scores = []
        if verbose:
                print("Training with lr = {0}, s2 penalty = {1}".format(learning_rate, s2_penalty))
        _trlosses = []
        _vallosses = []
        for j in range(nseed):
            
            seed = seeds[j]

            _trloss = np.inf * np.ones(epochs)
            _valloss = np.inf * np.ones(epochs)
            try:
                __, logdict = train_from_dataset(train_ds,learning_rate, batch_size, epochs, 
                                        val_ds = val_ds,
                                        s2_penalty=s2_penalty, cutoff=cutoff, max_sv_to_keep=max_sv_to_keep, 
                                        use_cache=use_cache, seed=seed, early_stopping=early_stopping, verbose=False,
                                        compute_overlaps=False)
                _trloss = logdict['loss']
                _valloss = logdict['val_loss']
                _score = _valloss[-1]
                
            except Exception as e:
                print("Training failed:")
                print(e)
                _score = np.inf
            _scores.append(_score)
        score = np.mean(_scores)
        if verbose:
            print("Acheived val score: {0}".format(score))
        
        trlosses.append(_trloss)
        vallosses.append(_valloss)
        scores[i] = score
        
    return scores, trlosses, vallosses

def select_hyperparams(train_ds, val_ds, batch_size, epochs,
                 Nparam=20, 
                lr_scale=None, lr_timescale=None,s2_scale=None, s2_timescale=None, 
                 cutoff=1e-10,
                 max_sv_to_keep = None,
                 use_cache=True, seed=None, 
                 early_stopping=True,
                 verbose=False
                 ):
    """ Obtain hyperparams by validation.
        hyperparams validated are lr and s2_penalty. """
    
    if s2_scale is None:
        s2_scale = 10**np.random.uniform(-4, 0, Nparam)
    if s2_timescale is None:
        s2_timescale = np.random.uniform(.2, 1,Nparam) * epochs
    if lr_scale is None:
        lr_scale = 10**np.random.uniform(-6, 0, Nparam)
    if lr_timescale is None:
        lr_timescale = np.random.uniform(.5, 10, Nparam) * epochs
    
    for param_arr in (lr_scale, lr_timescale, s2_scale, s2_timescale):
        if len(param_arr) != Nparam:
            raise ValueError("param array length does not match param number")


    lr = [make_exp_schedule(A, tau) for (A, tau) in zip(lr_scale, lr_timescale)]
    s2 = [make_exp_schedule(A, tau) for (A, tau) in zip(s2_scale, s2_timescale)]
    params = [dict(learning_rate=lr[i],s2_penalty=s2[i]) for i in range(Nparam)]

    scores, trlosses, vallosses = do_validation(train_ds, val_ds, 
                            batch_size, epochs,
                            params, 
                            cutoff=cutoff,
                            max_sv_to_keep=max_sv_to_keep,
                            use_cache=use_cache, seed=seed, 
                            early_stopping=early_stopping,
                            verbose=verbose)
    best_index = np.argmin(scores)
    best_params = dict(lr_scale=lr_scale[best_index],
                        lr_timescale=lr_timescale[best_index],
                        s2_scale=s2_scale[best_index],
                        s2_timescale=s2_timescale[best_index])

    return best_params, trlosses[best_index], vallosses[best_index]

def select_hyperparams_and_train(ds,
                             batch_size, epochs,
                            val_split=.1,Nparam=20,
                            cutoff=1e-10,
                            max_sv_to_keep = None,
                            ground_truth_mps=None, ground_truth_qutip=None, use_cache=True,val_seeds=None, seed=None, 
                            record_eigs=False, record_s2=False, verbose=False, early_stopping=True,
                           compute_overlaps=True):
    """ Select hyperparams using single validation split, then train on full dataset.
        Returns: trained model, logdict, best params, trloss from val, valloss from val"""
    N = len(ds)
    Nval = int(val_split * N)
    from torch.utils.data import random_split
    train_ds, val_ds = random_split(ds,[N-Nval,Nval])

    params, trloss, valloss = select_hyperparams(train_ds, val_ds, batch_size, epochs,
                                                Nparam=Nparam, 
                                                cutoff=cutoff,
                                                max_sv_to_keep=max_sv_to_keep,
                                                use_cache=use_cache, seed=val_seeds, 
                                                early_stopping=early_stopping,
                                                verbose=verbose
                                                )
    #if early stopping has been used, may find that we want to train for fewer epochs, 
    # as indicated by where training stopped in best validation run.
    epochs = len(trloss)

    lr = make_exp_schedule(params['lr_scale'], params['lr_timescale'])
    s2_penalty = make_exp_schedule(params['s2_scale'], params['s2_timescale'])
    #train on full dataset
    model, logdict = train_from_dataset(ds,lr, batch_size,epochs,s2_penalty=s2_penalty,
                                        cutoff=cutoff,max_sv_to_keep=max_sv_to_keep,
                                        ground_truth_mps=ground_truth_mps,ground_truth_qutip=ground_truth_qutip,
                                        use_cache=use_cache,seed=seed,record_eigs=record_eigs,
                                        record_s2=record_s2,verbose=verbose,compute_overlaps=compute_overlaps)
    return model, logdict, params, trloss, valloss


def get_dataset_from_settings_and_samples(fname_outcomes, fname_angles, numpy_seed=None, N=None, verbose=True):
    """ Returns MeasurementDataset corresponding to the pauli outcomes and local rotation angles in the specified files.
        N : if not None, how many samples to load. None-> load all samples
    """
    
    from .qtools import pauli_exp

    pauli_outcomes = np.load(fname_outcomes)
    angles = np.load(fname_angles)

    if numpy_seed is not None:
        np.random.seed(numpy_seed)
        perm = np.random.permutation(angles.shape[0])
        angles = angles[perm, ...]
        pauli_outcomes = pauli_outcomes[perm, ...]

    if N is not None:
        pauli_outcomes=pauli_outcomes[:N, ...]
        angles = angles[:N, ...]

    N = angles.shape[0]
    L = angles.shape[1]
    if verbose:
        print("Successfully loaded %d settings, samples for system of size L=%d"%(N,L))

    spinconfig = torch.tensor((1 -pauli_outcomes)/2, dtype=torch.long)
    theta = torch.tensor(angles[..., 0],dtype=torch.float)
    phi = torch.tensor(angles[..., 1], dtype=torch.float)
    rotations = pauli_exp(theta, phi)

    return MeasurementDataset(spinconfig, rotations)

def select_hyperparams_from_filepath(fname_outcomes, fname_angles, output_dir, 
                                    lr_scale, lr_timescale, s2_scale, s2_timescale,
                                numpy_seed=0, 
                                N=None, val_split=.2, 
                                Nparam=50, nseed=1, 
                                epochs=500, cutoff=1e-5, max_sv=25, batch_size=1024, 
                                use_cache=True, early_stopping=True, verbose=True):
    """ Select hyperparams by training on the given datasets. 
        fname_outcomes: file path to numpy array holding measurement outcomes.
        fname_angles: filepath to numpy array holding angles.
        output dir: directory to write validation results to.
        The learning rate and s2 penalty and parameterized with exponential decay schedules, 
            f(epoch) = A * exp(-epoch / timescale)
        lr_scale, s2_scale: the 'A' coefficient for learning rate and S2 penalty respectively
        lr_timescale, s2_timescale: the 'timescale' coefficient for learning rate and S2 penalty """
    import os
    from torch.utils.data import random_split
    import json


    ds = get_dataset_from_settings_and_samples(fname_outcomes,fname_angles,numpy_seed=numpy_seed,N=N,verbose=verbose)
    N=len(ds)
    Nval = int(val_split * N)
    Ntr = N - Nval
    train_ds, val_ds = random_split(ds, [Ntr, Nval])

    Nparam = len(lr_scale)
    for param_array in (lr_timescale, s2_scale, s2_timescale):
        if len(param_array) != Nparam:
            raise ValueError("param array has inconsistent length.")

    seeds = range(nseed)
    metadata = dict(lr_scale_samples=list(lr_scale), lr_timescale_samples=list(lr_timescale), 
                    s2_scale_samples=list(s2_scale), s2_timescale_samples=list(s2_timescale),
                    Ntotal=N,val_split=val_split,Nparam=Nparam,
                    nseed=nseed,epochs=epochs,cutoff=cutoff,
                        max_sv=max_sv, batch_size=batch_size,
                        use_cache=use_cache,early_stopping=early_stopping)


    params, trlosses, vallosses = select_hyperparams(train_ds, val_ds, batch_size, epochs,
                                                        Nparam=Nparam,lr_scale=lr_scale, lr_timescale=lr_timescale,
                                                        s2_scale=s2_scale, s2_timescale=s2_timescale, cutoff=cutoff,
                                                        max_sv_to_keep=max_sv, use_cache=use_cache, seed=seeds,
                                                        early_stopping=early_stopping, verbose=verbose)
    #record the number of epochs used in val params (will be less than specified in case of early stopping)
    params['epochs'] = len(trlosses)
    print("Finished hyperparam selection")
    with open(os.path.join(output_dir, 'validated_params.json'), 'w') as f:
        json.dump(params, f)
    for k in params.keys():
        metadata[k] = params[k]
    metadata['epochs_validated'] = len(trlosses)
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    np.save(os.path.join(output_dir, 'trlosses'), trlosses)
    np.save(os.path.join(output_dir, 'vallosses'), vallosses)

def to_json(d):
    for k, v in enumerate(d):
        if isinstance(v, np.ndarray):
            d[k] = list(v)

def train_from_filepath(fname_outcomes, fname_angles, 
                                    fname_training_metadata,
                                    numpy_seed=0, 
                                    N=None,seed=None,
                                    record_eigs=False, record_s2=True,
                                    compute_overlaps=True, use_cache=True,
                                    verbose=True):
    """ Train mps on given dataset.
        fname_outcomes: file path to numpy array holding measurement outcomes.
        fname_angles: filepath to numpy array holding angles.
        fname_training_metadata: path to json file holding training metadata.
            This should hold all of the necessary training hyperparameters, as well as 
        paths to the ground truth states if desired.
        output dir: directory to write validation results to.
        The learning rate and s2 penalty and parameterized with exponential decay schedules, 
            f(epoch) = A * exp(-epoch / timescale)
        lr_scale, s2_scale: the 'A' coefficient for learning rate and S2 penalty respectively
        lr_timescale, s2_timescale: the 'timescale' coefficient for learning rate and S2 penalty """

    #load a measurement dataset from the spec'd numpy files
    ds = get_dataset_from_settings_and_samples(fname_outcomes,fname_angles,numpy_seed=numpy_seed,N=N,verbose=verbose)
    L=ds[0]['samples'].size(0)

    #load training hyperparams from json
    print("Loading training settings from", fname_training_metadata)
    with open(fname_training_metadata) as f:
        training_metadata = json.load(f)

    #training hyperparameters
    lr_scale = training_metadata['lr_scale']
    lr_timescale = training_metadata['lr_timescale']
    s2_scale = training_metadata['s2_scale']
    s2_timescale = training_metadata['s2_timescale']
    epochs = training_metadata['epochs']
    cutoff = training_metadata['cutoff']
    max_sv= training_metadata['max_sv']
    batch_size = training_metadata['batch_size']

    if verbose:
        print("Loaded the following settings:")
        for setting in ['lr_scale', 'lr_timescale', 's2_scale', 's2_timescale', 'epochs', 'cutoff', 'max_sv', 'batch_size']:
            print("{0} = {1:3e}".format(setting, training_metadata[setting]))

    #other settings for training...
    ground_truth_mps_path = training_metadata.get('mps_path', None)
    ground_truth_qutip_path = training_metadata.get('qutip_path', None)

    if ground_truth_mps_path is not None:
        print("loading ground truth MPS from ", ground_truth_mps_path)
        from .models import MPS
        ground_truth_mps = MPS(L, 2, 2)
        ground_truth_mps.load(ground_truth_mps_path)
    else:
        ground_truth_mps = None
    if ground_truth_qutip_path is not None:
        print("Loading ground truth qutip state from ", ground_truth_qutip_path)
        import qutip as qt
        ground_truth_qutip = qt.qload(ground_truth_qutip_path)
    else:
        ground_truth_qutip = None

    metadata = dict(fname_outcomes=fname_outcomes, fname_angles=fname_angles, 
                    ground_truth_mps_path=ground_truth_mps_path,
                    ground_truth_qutip_path=ground_truth_qutip_path,
                    lr_scale=lr_scale, lr_timescale=lr_timescale, 
                    s2_scale=s2_scale, s2_timescale=s2_timescale,
                    Ntotal=N,
                    seed=seed,
                    epochs=epochs,cutoff=cutoff,
                        max_sv=max_sv, batch_size=batch_size,
                        use_cache=use_cache)
                        
    learning_rate = make_exp_schedule(lr_scale, lr_timescale)
    s2_penalty = make_exp_schedule(s2_scale, s2_timescale)

    model, logdict = train_from_dataset(ds,
                            learning_rate, batch_size, epochs,
                            val_ds=None,
                            s2_penalty=s2_penalty, cutoff=cutoff,
                            max_sv_to_keep=max_sv,
                            ground_truth_mps=ground_truth_mps, ground_truth_qutip=ground_truth_qutip, 
                            use_cache=use_cache, seed=seed, 
                            record_eigs=record_eigs, record_s2=record_s2, verbose=verbose,
                            compute_overlaps=compute_overlaps)
    

    print("Finished training")
    return model, logdict, metadata

def hamming_distance(s1, s2):
    """ The Hamming distance between two (N, L) spin configurations."""
    return (s1!=s2).sum(1)

def random_unitary_overlap_estimate(mps, spin_config, rotations):
    """ Estimate the overlap between an MPS and the quantum state which produced a particular set of data.
        mps: an MPS model
        spin_config: (N, L) tensor of measurement outcomes, as indices of basis states.
        rotations: corresponding (N, L, 2) random unitary ComplexTensor
        Nsamp: how many samples to drawn from the MPS
        returns: estimate of the overlap fidelity according to eqn (34) of https://arxiv.org/pdf/1812.02624.pdf
        """
    N = spin_config.size(0)
    samples = mps.sample(N, rotations=rotations)
    #(N)
    D = hamming_distance(spin_config, samples).to(dtype=torch.float)
    d = mps.local_dim
    return (torch.pow(-1, D) * torch.pow(d, mps.L - D)).mean()


def scale_overlap_estimate(mps, spin_config, rotations, eps=1e-2, Nsample=10):
    """ Compute overlap estimates on successively larger training set sizes until
        convergence within eps is obtained.
        spin_config: (N, L) tensor of basis-state outcomes
        rotations: (N, L, 2) complextensor of corresponding local unitaries:
        eps: convergence tolerance. When the mean overlap changes between sample sizes by less than this fraction, 
        scaling in system size is halted.
        Nsample: number of datasets of each size to sample.
        
        
        Returns: numpy arrays N, mean_overlap, err_overlap, convergence_acheived
             N, mean_overlap, err_overlap: hold the sample sizes tried, the mean overlap estimate at each, 
        and the standard error of the mean.
            convergence_acheived: bool, whether convergence was acheived within specified tolerance."""
    Ntot = spin_config.size(0)
    N = Ntot //10
    dN = N
    diff = 1
    prev_est = None
    estimates_by_size = []
    stat_errs_by_size = []
    sample_sizes = []
    while diff > eps and N < Ntot:
        sample_sizes.append(N)
        estimates = []
        for j in range(Nsample):
            perm = torch.randperm(Ntot)
            s = spin_config[perm, ...][:N]
            rot = rotations[perm, ...][:N]
            estimates.append(random_unitary_overlap_estimate(mps, s, rot))
        #overlap estimate and SEM for a fixed sample size.
        mu, sig = np.mean(estimates), np.std(estimates)/ np.sqrt(Nsample)
        if prev_est is not None:
            diff = np.abs(prev_est - mu) / prev_est
        estimates_by_size.append(mu)
        stat_errs_by_size.append(sig)
        prev_est = mu
        N += dN
    convergence_acheived = True
    if diff > eps:
        warnings.warn("overlap estimate failed to converge within tolerance {0:.2e}".format(eps))
        convergence_acheived = False
    #return: mean overlap estimate at terminating sample size, relative diff, statistical error
    return np.asarray(sample_sizes), np.asarray(estimates_by_size), np.asarray(stat_errs_by_size), convergence_acheived

def estimate_overlap(mps, spin_config, rotations, eps=1e-2, Nsample=10):
    """ Estimate the overlap tr(rho1 rho2) between the mps and the state that produced a given dataset.
        mps: MPS pure state model. 
        spin_config: (N, L) tensor of basis state outcomes
        rotations: (N, L, 2) complextensor of corresponding local rotations.
        eps: relative convergence criterion for the overlap, default 1e-2
        Nsample: how many times to sample each dataset size when scaling up the overlap estimates.
        Returns: overlap estimate, err, convergence_acheived (bool)
            error is defined by the larger of: statistical error at the final subset size sampled, 
             change in fidelity when increasing system size to final value.
        """
    __, mean_by_size, stat_err_by_size, convergence_acheived = scale_overlap_estimate(mps, spin_config, rotations, 
                                                                            eps=eps, Nsample=Nsample)
    overlap_est = mean_by_size[-1]
    overlap_err = max(stat_err_by_size[-1], np.abs(overlap_est - mean_by_size[-2]))
    return overlap_est, overlap_err, convergence_acheived

def make_linear_schedule(start, finish, epochs):
    def f(ep):
        return start + ep * (finish - start) / (epochs-1)
    return f

def make_exp_schedule(start, timescale):
    def f(ep):
        return start * np.exp(- ep / timescale)
    return f

class MeasurementDataset(TensorDataset):
    """ Holds local unitaries (key: rotations) and corresponding outcomes (key: samples)"""
    def __init__(self, samples, rotations):
        super().__init__()
        if samples.shape[0] != rotations.shape[0]:
            raise ValueError
        if tuple(rotations.shape[-2:])!= (2,2):
            raise ValueError("Not a valid rotation")
        self.samples = TensorDataset(samples)
        self.rotations = TensorDataset(rotations)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        samples = self.samples[i][0]
        rot = self.rotations[i][0]
        return dict(samples=samples, rotations=dict(real=rot.real, imag=rot.imag))

    def unpack(self, start_index=0, stop_index=None):
        """ Returns samples, rotations tensors holding data corresponding to the given slice."""
        if stop_index is None:
            stop_index = len(self)
        spinconfig = self.samples[start_index:stop_index, ...][0]
        U = self.rotations[start_index:stop_index, ...][0]
        return spinconfig, U