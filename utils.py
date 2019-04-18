import numpy as np

def svd_push_right(Aleft, Aright):
    """Perform SVD on the left site matrix, and push singular values into the right site matrix.
        This results in the left site-matrix being left-normalized.
        Aleft: complex numpy array, shape (local_dim, bond_dim1, bond_dim2)
        Aright: complex numpy array, shape (local_dim, bond_dim2, bond_dim3)"""

    local_dim, D1, D2 = Aleft.shape
    # pull both indices over to the left side
    Aleft = np.reshape(Aleft, (local_dim * D1, D2))
    #svd on the D2 index
    u,s,v=np.linalg.svd(Aleft, full_matrices=False)
    #number of singular values
    k=s.shape[0]
    #the unitary u defines the new left site matrix
    Aleft_new = np.reshape(u, (local_dim,D1,k))
    #contract s and v with the old right-matrix to define the new right-matrix
    Aright_new = np.einsum('qi,sij->sqj',np.dot(np.diag(s), v),Aright)
    return Aleft_new, Aright_new
    
def svd_push_left(Aleft, Aright):
     """Perform SVD on the right site matrix, and push singular values into the left site matrix.
        This results in the right site-matrix being right-normalized.
        Aleft: complex numpy array, shape (local_dim, bond_dim1, bond_dim2)
        Aright: complex numpy array, shape (local_dim, bond_dim2, bond_dim3)"""

    local_dim, D2, D3 = Aright.shape
    #pull spin index to the right
    Aright = np.reshape(np.swapaxes(Aright, 0,1), (D2, local_dim * D3))
    #perform SVD on the D2-index
    u,s,v = np.linalg.svd(B, full_matrices=False)
    #number of singular values
    k=s.shape[0]
    #new right-matrix, with shape local_dim, k, D3
    Aright_new = np.swapaxes(v.reshape( (k, local_dim, D3)), 0, 1)
    #push u and s to the left to get new left-matrix
    Aleft_new = np.einsum('sij,jq->siq', Aleft,np.dot(u, np.diag(s)))

    return Aleft_new, Aright_new
