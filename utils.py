from torch.utils.data import TensorDataset
import numpy as np
import scipy as sp
import scipy.linalg
import zgesvd
import torch

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
            u,s,v = zgesvd.svd_zgesvd(A, full_matrices=False, compute_uv=True)
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
    from models import MPS, ComplexTensor
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
    from models import MPS, ComplexTensor
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
    from models import MPS
    return MPS(L,local_dim=2,bond_dim=bond_dim)

def do_local_sgd_training(mps_model, dataloader, epochs, 
                            learning_rate, s2_penalty=None,nstep=1,
                            cutoff=1e-10,max_sv_to_keep=None, 
                            ground_truth_mps = None, verbose=False, use_cache=True, 
                            record_eigs=True, record_s2=True):
    """Perform SGD local-update training on an MPS model using measurement outcomes and rotations
    from provided dataloader.
        mps_model: an MPS
        dataloader: pytorch dataloader which yields dictionaries holding batches of spin configurations
        and local unitaires.
        epochs: int, number of epochs to train
        learning_rate : lr for gradient descent. Scalar, or function of epoch which returns scalar.
        s2_penalty: coefficient of S2 regularization term. if not None, scalar or function of epoch which returns scalar
        nstep: how many gradient descent updates to make at each bond.
        cutoff: threshold below which to drop singular values
        max_sv_to_keep: if not None, max number of singular vals to keep at each bond, or function of batch number 
        that returns such
        ground_truth_mps: if not None, an MPS against which the model's fidelity will be checked after
        every sweep.
        use_cache: whether to cache partial amplitudes during the sweeps.
        record_eigs: whether to record eigenvalues of the half-chain density op.
        record_s2: whether to record the Renyi-2 entropy of the half-chain density op.
        Returns: dictionary, mapping:
                    'loss' -> batched loss function during training
                    'fidelity' -> if ground truth state was provided, array of fidelties during training.
                    'max_bond_dim' -> array of max bond dimensions
                    'eigenvalues' -> eigenvalues when partitioning at chain center, if requested
                    's2' -> renyi-2 entropy, if requested
            These quantities are recorded at the end of each epoch.
        """
    from models import ComplexTensor
    import time
    if s2_penalty is None:
        s2_penalty = np.zeros(len(dataloader) * epochs)
    #system size
    L = mps_model.L
    #logging the loss function
    losses = []
    #if ground-truth MPS is known, compute fidelity at each step
    fidelities = []
    # record max bond_dim
    max_bond_dim = []
    #record eigenspectrum cutting across chain center
    eigenvalues = []
    #Renyi-2 entropy cutting across chain center
    s2 = []

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
                        eigenvalues.append(eigs)
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
            losses.append(mps_model.nll_loss(spinconfig, rotation=rotations))
            if ground_truth_mps is not None:
                fidelities.append(np.abs(mps_model.overlap(ground_truth_mps)) / mps_model.norm_scalar() )
            max_bond_dim.append(mps_model.max_bond_dim)
            
        if verbose:
            print("Finished epoch {0} in {1:.3f} sec".format(ep, time.time() - t0))
            print("Model shape: ", mps_model.shape)
    return dict(loss=np.asarray(losses),
                fidelity=np.asarray(fidelities),
                max_bond_dim=max_bond_dim, 
                eigenvalues=eigenvalues,
                s2=s2)
                
def draw_random(mps, N):
    """ Draw N samples from mps, each taken in a random basis.
        Returns: angles, outcomes
        where angles = (N, L, 2) tensor holding theta, phi angles 
        outcomes = (N, L) tensor of pauli eigenvalue outcomes."""
    from qutip_utils import sample_random_angles
    from qtools import pauli_exp
    angles = torch.tensor(sample_random_angles((N, mps.L)),dtype=torch.float)
    rotations = pauli_exp(angles[..., 0], angles[..., 1])
    index_outcomes = mps.sample(N, rotations=rotations).numpy()
    #convert to pauli eigenvalues
    pauli_eig_outcomes = 1 - 2 * index_outcomes
    return angles, pauli_eig_outcomes

def train_from_dataset(meas_ds,
                learning_rate, batch_size, epochs,
                 s2_penalty=None, cutoff=1e-10,
                 max_sv_to_keep = None,
                ground_truth_mps=None, use_cache=True, seed=None, 
                record_eigs=False, record_s2=False, verbose=False):
    """ Given a MeasurementDataset ds, create and train an MPS on it."""
    if seed is not None:
        torch.manual_seed(seed)
    L = meas_ds[0]['samples'].size(0)
    N = len(meas_ds)
    if verbose:
        print("Training on system size %d with %d samples"%(L, N))
    dl = DataLoader(meas_ds, batch_size=batch_size, shuffle=True)
    #train a model
    model = MPS(L, local_dim=2, bond_dim=2)
    logdict = do_local_sgd_training(model, dl, epochs, learning_rate,
                                    s2_penalty=s2_penalty,cutoff=cutoff,
                                    max_sv_to_keep=max_sv_to_keep,
                                    use_cache=use_cache,
                                    ground_truth_mps=ground_truth_mps, 
                                    record_eigs=record_eigs, record_s2=record_s2,verbose=verbose)
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

    from torch.utils.data import DataLoader
    from models import MPS
    from qtools import pauli_exp
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
    
    from models import ComplexTensor
    spins_all = meas_ds[:]['samples']
    Ui = meas_ds[:]['rotations']['imag']
    Ur = meas_ds[:]['rotations']['real']
    U = ComplexTensor(Ur, Ui)
    return model.nll_loss(spins_all, rotation=U)


def evaluate(train_ds, val_ds, 
                learning_rate, batch_size, epochs,
                 s2_penalty=None, cutoff=1e-10,
                 max_sv_to_keep = None,
                 use_cache=True, seed=None, 
                 verbose=False):
    """ Train a model on the given training MeasurementDataset, then compute its NLL cost function on 
    the held-out validation set.
    Returns: trained model, validation loss."""
    model, __ =  train_from_dataset(train_ds,
                        learning_rate, batch_size, epochs,
                        s2_penalty=s2_penalty, cutoff=cutoff,
                        max_sv_to_keep=max_sv_to_keep,
                        ground_truth_mps=None, use_cache=use_cache, seed=seed, 
                        record_eigs=False, record_s2=False, verbose=verbose)
    val_loss = compute_NLL(val_ds, model)
    return model, val_loss


class MeasurementDataset(TensorDataset):
    """ Holds local unitaries (key: rotations) and corresponding outcomes (key: samples)"""
    def __init__(self, samples, rotations):
        super().__init__()
        if samples.shape[0] != rotations.shape[0]:
            raise ValueError
        self.samples = TensorDataset(samples)
        self.rotations = TensorDataset(rotations)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        samples = self.samples[i][0]
        rot = self.rotations[i][0]
        return dict(samples=samples, rotations=dict(real=rot.real, imag=rot.imag))
