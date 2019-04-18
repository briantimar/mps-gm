import numpy as np


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
    if cutoff:
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

def svd_push_right(Aleft, Aright, cutoff=1e-8, max_sv_to_keep=None):
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
    
def svd_push_left(Aleft, Aright, cutoff=1e-8, max_sv_to_keep=None):
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
    A: two-site complex tensor, shape (local_dim, local_dim, D1, D2) 
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

    
