""" Miscellaneous tools -- for eg sampling synthetic data """

import numpy as np

SORT_CONVENTIONS = {'sort': lambda x,ns: x, 'revsort': lambda x,ns: ns-1-x}
def to_int(b,sort='revsort'):
    """ Given a binary string b, return its index in the occupation number basis.
        revsort = standard order for quspin/qutip, most excited states first"""

    x=sum([(2**j) * b[len(b)-1-j] for j in range(len(b)) ])
    x=SORT_CONVENTIONS[sort](x,2**len(b))
    return int(x)

def to_binary(x,L, sort='revsort'):
    """Given an integer x, return its binary representation."""
    x = SORT_CONVENTIONS[sort](x, 2**L)
    b=np.zeros(L,dtype=int)
    for i in range(L):
        x,r=divmod(x,2)
        b[L-i-1]=r
    return b

def draw(probs,  occ_rep, Nsamp):
    """Given array of probabilities probs in the standard basis, and the array of corresponding
    occupation number representations occ_rep, draw Nsamp occ-rep samples from the distribution.

    Returns: array (Nsamp, L), L being the number of atoms """
    Ns, L = occ_rep.shape
    shots= np.empty((Nsamp,L))
    results = np.random.choice(list(range(len(probs))), Nsamp, p=probs)
    for ii in range(Nsamp):
        shots[ii, : ] = occ_rep[results[ii], :]
    return shots

def do_bitflip(zshots, p01, p10):
    """ Run the z-basis data zshots through a classical bit-flip channel.
            With probability p01 (read: '0 given 1')
                    1---->0
            and with probability p10
                    0 ---> 1
            """
    zshots = zshots.astype(int)
    r = np.random.rand(*zshots.shape)
    f01 = r<p01
    f10 = r<p10
    return (zshots==1)* (f01*np.logical_not(zshots) + (1-f01)*zshots) + (zshots==0) * (f10 * np.logical_not(zshots) + (1-f10) * zshots)

def generate_binary_space(L,order='revsort'):
    """Generates all possible L-bit strings (representing computational basis states)
    By default, uses quspin convention for state ordering -- that is, the most highly excited states come first.
    Returns: (2^L, L) dimensional numpy array

    """

    Ns=2**L
    sorts = {'sort': lambda x: x, 'revsort': lambda x: Ns-1-x}
    sorter = sorts[order]
    space = np.zeros((Ns, L)
                       )
    for i in range(Ns):
        d=sorter(i)
        for j in range(L):
            d, r = divmod(d, 2)
            space[i, L - j - 1] = int(r)

    return space


def get_conditional_probs_asbf( p01, p10, v):
    """ Matrix of transition probabilities between all 2^L states in standard basis.
        p10 = probability of 0->1
        p01 = probability of 1->0
        v = (2^L, L) array which holds the position space reps of basis states.
        Returns: matrix Pi of conditional probs, such that Pi[a, b] is the probability
        of transitioning *to* a *from* b

        Note, this is an extremely inefficient implementation and you should be
        ashamed to use it."""

    ns,L =v.shape
    if ns != 2**L:
        raise ValueError
    #matrix of conitional probs.
    pi = np.zeros((ns,ns))
    p00 = 1-p10
    p11 = 1-p01
    for i in range(ns):
        #final state
        si = v[i,:]
        for j in range(i,ns):
            #initial state
            sj = v[j, :]
            #number of sites where 1->1 transition occurs, etc
            n11 = np.sum((si==1)*(sj==1))
            n10 = np.sum((si==1)*(sj==0))
            n01 = np.sum((si==0)*(sj==1))
            n00 = np.sum((si==0)*(sj==0))
            pi[i, j] = (p11**n11)*(p10**n10)*(p01**n01)*(p00**n00)
            pi[j,i] = (p11**n11)*(p10**n01)*(p01**n10)*(p00**n00)
    return pi

def get_n_bysite_from_pdist(pdist,v):
    """v = (N, L) array of occupation-number data points.
            return the average excitation per site."""
    return np.tensordot(pdist,v,axes=([0],[0]))

def get_zz_correlation_matrix_from_pdist(pdist, v):
    """v = (N, L) array of occupation-number data points.
            return the two-point spin-spin connected correlator matrix."""
    if np.min(v)<0:
        raise ValueError("expecting binary input")
    z = 2 * v-1
    zizj_bar = np.einsum('i,ij,ik->jk',pdist, z, z)
    zi_barzj_bar = np.einsum('i,ij,l,lk->jk',pdist, z,pdist,z)
    return zizj_bar - zi_barzj_bar

def get_avg_correlation_from_matrix(zz):
    """Given array of correlations zz, returns the spatial average
        correlation values.
        zz shape = (L, L, ...)
        return shape = (L-1, ...)"""
    L=zz.shape[0]
    ns=L-1
    zzbar = np.zeros((ns, *zz.shape[2:]))
    for i in range(ns):
        s=i+1
        zzbar[i, ...] = np.mean(np.asarray([zz[ii, ii+s, ...] for ii in range(L-s)]), axis=0)
    return zzbar

def make_experimental_distribution(zshots):
    """Given (N, L) array of observations, return the corresponding frequency
        distribution."""
    N,L = zshots.shape
    pvals = np.zeros(2**L)
    for j in range(N):
        m = to_int(zshots[j, :])
        pvals[m] += 1/N
    return pvals

def safelog(x,eps=1e-12):
    xsf = (x>eps) *x + (x <=eps) * eps
    return np.log(xsf)

def Hshannon(p,eps=1e-12,axis=None):
    p = np.asarray(p)
    return - np.sum(p * safelog(p,eps=eps),axis=axis)

def NLL(p, q,eps=1e-12):
    if (q<eps).any():
        print("Warning: q<eps values in the log of NLL can lead to ill-defined results!")
    return - np.sum(p * safelog(q,eps=eps))

def DKL(p, q,eps=1e-12):
    """Pay attention to the order -- it matters in a practical sense if one distribution as a lot of zeros.
        You want that one to be 'p' and the other one (hopefully nonzero everywhere) to be q"""
    return -Hshannon(p,eps=eps) + NLL(p, q,eps=eps)

def fidelity(p, q):
    return np.sum(np.sqrt(p*q))

def trace_dist(p, q):
    return .5 * np.sum(np.abs(p-q))
