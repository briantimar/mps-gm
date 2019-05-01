from torch.utils.data import TensorDataset
import numpy as np
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
    u,s,v=np.linalg.svd(A, full_matrices=False)
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
                            learning_rate, s2_schedule=None,nstep=1,
                            cutoff=1e-10,max_sv_to_keep=None, 
                            ground_truth_mps = None):
    """Perform SGD local-update training on an MPS model using measurement outcomes and rotations
    from provided dataloader.
        mps_model: an MPS
        dataloader: pytorch dataloader which yields dictionaries holding batches of spin configurations
        and local unitaires.
        epochs: int, number of epochs to train
        learning_rate : lr for gradient descent
        s2_schedule: if not None, iterable of s2 penalty coefficients, of length epochs * len(dataloader)
        nstep: how many gradient descent updates to make at each bond.
        cutoff: threshold below which to drop singular values
        max_sv_to_keep: if not None, max number of singular vals to keep at each bond.
        ground_truth_mps: if not None, an MPS against which the model's fidelity will be checked after
        every sweep.

        Returns: dictionary, mapping:
                    'loss' -> batched loss function during training
                    'fidelity' -> if ground truth state was provided, array of fidelties during training.
        """
    from models import ComplexTensor
    if s2_schedule is None:
        s2_schedule = np.zeros(len(dataloader) * epochs)
    #system size
    L = mps_model.L
    #logging the loss function
    losses = []
    fidelities = []
    for ep in range(epochs):
        for step, inputs in enumerate(dataloader):
            #get torch tensors representing measurement outcomes, and corresponding local unitaries
            spinconfig = inputs['samples']
            rot = inputs['rotations']
            rotations = ComplexTensor(rot['real'], rot['imag'])

            s2_penalty = s2_schedule[ep*len(dataloader) + step]
            #forward sweep across the chain
            for i in range(L-1):
                for __ in range(nstep):
                    #computes merged two-site tensor at bond i, and the gradient of NLL cost function
                    # with respect to this 'blob'; updates merged tensor accordingly, then breaks back to local tensors
                    mps_model.do_sgd_step(i, spinconfig,
                                    rotation=rotations, cutoff=cutoff, normalize='left', 
                                    max_sv_to_keep=max_sv_to_keep,
                                    learning_rate=learning_rate, s2_penalty=s2_penalty)
            #backward sweep across the chain
            for i in range(L-3, 0, -1):
                for __ in range(nstep):
                    mps_model.do_sgd_step(i, spinconfig,
                                    rotation=rotations, cutoff=cutoff, normalize='right',
                                     max_sv_to_keep=max_sv_to_keep,
                                    learning_rate=learning_rate, s2_penalty=s2_penalty)
            with torch.no_grad():
                #record batched loss functions
                losses.append(mps_model.nll_loss(spinconfig, rotation=rotations))
                if ground_truth_mps is not None:
                    fidelities.append(np.abs(mps_model.overlap(ground_truth_mps)) / mps_model.norm().numpy())
    return dict(loss=np.asarray(losses),
                fidelity=np.asarray(fidelities))
                



class MeasurementDataset(TensorDataset):
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
