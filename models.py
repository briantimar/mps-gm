import numpy as np
import torch
import torch.nn as nn
import warnings

from utils import make_onehot
#tools for SVD -- normalization and breaking two-site tensors
from utils import svd_push_left, svd_push_right, split_two_site


class ComplexTensor:

    def __init__(self, real, imag):
        assert real.shape == imag.shape
        self.real = real
        self.imag = imag

    def apply_mul(self, other, mul_op):
        """Apply a multiplication op.
            other: a ComplexTensor
            mul_op: a function which takes two real tensors and returns some sort of contraction, 
            bilinear in its args.
            Returns: a ComplexTensor representing the product of self with other."""
        r = mul_op(self.real, other.real) - mul_op(self.imag, other.imag)
        i = mul_op(self.real, other.imag) + mul_op(self.imag, other.real)
        return ComplexTensor(r,i)

    @property
    def shape(self):
        return self.real.shape

    def conj(self):
        """ Returns the complex conjugate tensor"""
        return ComplexTensor(self.real, -self.imag)

    def __getitem__(self, slice):
        return ComplexTensor(self.real[slice], self.imag[slice])

    def view(self, *shape):
        return ComplexTensor(self.real.view(*shape), self.imag.view(*shape))
    def squeeze(self):
        return ComplexTensor(self.real.squeeze(), self.imag.squeeze())

    def __add__(self, other):
        return ComplexTensor(self.real + other.real, 
                                self.imag + other.imag)

    def __mul__(self, other):
        if isinstance(other, ComplexTensor):
            mul_op = lambda x, y: x * y
            return self.apply_mul(other, mul_op)
        return ComplexTensor(self.real * other, self.imag * other)
        
    def div(self, other):
        return ComplexTensor(self.real/other, self.imag/other)

    def norm(self):
        return (self * self.conj()).real

    def abs(self):
        return (self * self.conj()).real.sqrt()

    def __repr__(self):
        return "ComplexTensor shape {0}".format(self.shape)

    def size(self, i):
        return self.real.size(i)

    def numpy(self):
        return self.real.detach().numpy() + 1j * self.imag.detach().numpy()
    
    def mean(self,dim):
        return ComplexTensor(self.real.mean(dim), self.imag.mean(dim))
    
    def display(self, items=10):
        return self.numpy()[:items]



class MPS(nn.Module):
    """ MPS with complex amplitudes """

    def __init__(self, L, local_dim, bond_dim):
        """ L =  size of the physical system
        local_dim = dimensionality of each local hilbert space
        bond_dimension = uniform bond dimension"""
        super().__init__()
        self.L = L
        self.local_dim = local_dim
        self.bond_dim = bond_dim
        self.build_tensors()
        self.init_tensors()
        #the site to which the mps is gauged, if any
        self.gauge_index = None

        self.normalize()

    def build_tensors(self):
        """Create the site tensors which define the MPS"""
        def build(shape):
            return nn.Parameter(torch.randn(*shape, requires_grad=True))

        #shape of tensors in the bulk and at the edges of the MPS
        bulk_shape = (self.local_dim, self.bond_dim, self.bond_dim)
        left_shape = (self.local_dim, 1, self.bond_dim)
        right_shape = (self.local_dim, self.bond_dim, 1)

        self.tensors = []
        self.real_tensors = []

        #build edge tensors
        left_r = build(left_shape)
        left_i = build(left_shape)
        right_r = build(right_shape)
        right_i = build(right_shape)

        left_tensor = ComplexTensor(left_r, left_i)
        right_tensor = ComplexTensor(right_r, right_i)

        self.tensors.append(left_tensor)
        self.real_tensors.append(dict(real=left_r,imag=left_i))
        #each site has its own tensor
        for __ in range(self.L-2):
            bulk_r, bulk_i = build(bulk_shape), build(bulk_shape)
            self.tensors.append( ComplexTensor(bulk_r, bulk_i))
            self.real_tensors.append(dict(real=bulk_r, imag=bulk_i))
        self.tensors.append(right_tensor)
        self.real_tensors.append(dict(real=right_r, imag=right_i))

    def init_tensors(self):
        """ Initialize all tensors, with normally-distributed values
            """
        init = 1.0 / np.power(self.bond_dim**2 * self.local_dim, .25)
        for t in self.tensors:
            with torch.no_grad():
                t.real.normal_(0, init)
                t.imag.normal_(0, init)

    def sanitize_spin_config(self, spin_config):
        if len(spin_config.shape) == 1:
            spin_config = spin_config.unsqueeze(0)
        if spin_config.size(1) != self.L:
            raise ValueError("spin configuration does not match MPS size")
        return spin_config

    def get_local_tensor(self, site_index):
        """ Returns the (three-index) tensor living at a particular physical index."""
        if (site_index <0) or (site_index >= self.L):
            raise ValueError("Invalid site index")
        return self.tensors[site_index]

    def rescale_site_tensor(self, site_index):
        """Divides site tensor at spec'd index by square root of its contraction"""
        N = self.site_contraction(site_index).item()
        A = self.get_local_tensor(site_index)
        self.set_local_tensor(site_index, ComplexTensor(A.real/np.sqrt(N), 
                                                        A.imag/np.sqrt(N)))

    def norm_full(self):
        """Compute the norm <psi|psi> of the MPS by contraction of the full tensor"""
        #contracts the left edge of the MPS
        init_contractor = lambda x, y: torch.einsum('sij,sik->jk', x, y)
        #contract a single physical index between accumulated tensor and site tensor
        #contract the upper site tensor...
        upper_bond_contractor = lambda acc, site: torch.einsum('ij,sik->skj',acc, site)
        #and then the lower
        lower_bond_contractor = lambda acc, site: torch.einsum('skj,sjl->kl',acc,site)

        #begin contraction with the left edge of the mps
        left_tensor = self.get_local_tensor(0)
        cont = left_tensor.apply_mul(left_tensor.conj(), init_contractor)

        for site in range(1,self.L):
            bulk_tensor = self.get_local_tensor(site)
            #absorb upper site tensor
            cont = cont.apply_mul(bulk_tensor, upper_bond_contractor)
            #absorb lower site tensor, contract over spin
            cont = cont.apply_mul(bulk_tensor.conj(), lower_bond_contractor)

        return cont.squeeze().real
    
    def norm(self):
        """Compute the state norm as tensor."""
        if self.gauge_index is None:
            return self.norm_full()
        return self.site_contraction(self.gauge_index)

    def guarantee_is_gauged(self):
        """ Ensures that a gauge index exists"""
        if self.gauge_index is None:
            self.gauge_to(0)

    def normalize(self):
        """normalize the MPS"""
        self.guarantee_is_gauged()
        self.rescale_site_tensor(self.gauge_index)
    
    def set_local_tensor(self, site_index, A):
        """Set tensor at specifed site index equal to A.
            A: a ComplexTensor"""
        self.tensors[site_index].real.data = A.real.data
        self.tensors[site_index].imag.data = A.imag.data

    def set_local_tensor_from_numpy(self, site_index, A):
        ct = ComplexTensor(torch.tensor(A.real), torch.tensor(A.imag))
        self.set_local_tensor(site_index, ct)

    def left_normalize_at(self, site_index, cutoff=1e-16, max_sv_to_keep=None):
        """ Apply a single SVD at the bond (site_index, site_index +1), resulting 
            in left-normalization at site_index."""
        if site_index <0 or site_index>= self.L-1:
            raise ValueError("invalid index %d for left-normalization"%site_index)
        Aleft = self.get_local_tensor(site_index).numpy()
        Aright = self.get_local_tensor(site_index+1).numpy()
        Aleft, Aright = svd_push_right(Aleft, Aright,
                        cutoff=cutoff, max_sv_to_keep=max_sv_to_keep)
        self.set_local_tensor_from_numpy(site_index, Aleft)
        self.set_local_tensor_from_numpy(site_index+1, Aright)
    
    def right_normalize_at(self, site_index, cutoff=1e-16, max_sv_to_keep=None):
        """ Apply a single SVD at the bond (site_index-1, site_index), resulting 
            in right-normalization at site_index ."""
        if site_index == 0 or site_index > self.L-1:
            raise ValueError("invalid index %d for right-normalization"%site_index)
        Aleft = self.get_local_tensor(site_index-1).numpy()
        Aright = self.get_local_tensor(site_index).numpy()
        Aleft, Aright = svd_push_left(Aleft, Aright,
                        cutoff=cutoff, max_sv_to_keep=max_sv_to_keep)
        self.set_local_tensor_from_numpy(site_index-1, Aleft)
        self.set_local_tensor_from_numpy(site_index, Aright)
        
    def gauge_to(self, site_index, cutoff=1e-16, max_sv_to_keep=None):
        """Left-normalizes all tensors to the left of site_index; right-normalizes all
        tensors to the right."""
        if site_index < 0 or site_index > self.L-1:
            raise ValueError("invalid index %d" % site_index)

        if self.gauge_index is None:
            left_norm_range = range(0,site_index)
            right_norm_range = range(self.L-1, site_index,-1)
           
        elif self.gauge_index < site_index:
            # in this case, right normalization is already taken care of
            left_norm_range = range(self.gauge_index, site_index)
            right_norm_range = range(-1)

        elif self.gauge_index > site_index:
            #in this case, left norm is already taken care of
            left_norm_range = range(-1)
            right_norm_range = range(self.gauge_index, site_index, -1)
        else:
            left_norm_range = range(-1)
            right_norm_range = range(-1)

        for i in left_norm_range:
            self.left_normalize_at(i, cutoff=cutoff,max_sv_to_keep=max_sv_to_keep)
        for i in right_norm_range:
            self.right_normalize_at(i, cutoff=cutoff, max_sv_to_keep=max_sv_to_keep)
    
        self.gauge_index = site_index
    
    def site_contraction(self, site_index):
        """ Return the tensor contraction of single site tensor with its conjugate."""
        A = self.get_local_tensor(site_index)
        contractor = lambda a, aconj: torch.einsum('sij,sij->',a, aconj)
        return A.apply_mul(A.conj(), contractor).real

    def get_local_matrix(self, site_index, spin_index, rotation=None):
        """spin_index = an (N,) tensor of spin configurations.
            rotation: if not None, (N,d,d,) complex tensor
             defining single-qubit unitaries to contract
            with local tensors before returning. (d= localdim)
            returns: (N,D,D) tensor holding local matrices for each spin
            configuration. (D = bonddim)"""
        #shape (local_dim, D, D)
        local_tensor = self.get_local_tensor(site_index)

        if rotation is not None:
            if spin_index.size(0) != rotation.size(0):
                raise ValueError("Index tensor incompatible with rotations")
            N = spin_index.size(0)
            contractor = lambda x, y: torch.einsum('ast,tij->asij', x, y)
            #shape (N, local_dim, D, D)
            local_tensor = rotation.apply_mul(local_tensor, contractor)
            #shape (N, D, D)
            local_matrix = local_tensor[range(N), spin_index, ...]
        else:
            local_matrix = local_tensor[spin_index, ...]
        return local_matrix

    def contract_interval(self, spin_config, start_index, stop_index, rotation=None):
        """Contract a specific configuration of local tensors on the interval [start_index, stop_index)
             spin_config = (N, L) tensor listing spin configurations.
            rotations: (N, L, d, d) complextensor of local unitaries applied.
            Returns: (N,D1, D2) tensor of amplitudes, D1 and D2 being the dangling bond dimensions
            of the tensors at the edges of the interval."""

        spin_config = self.sanitize_spin_config(spin_config)

        def contractor(x, y):
            return torch.einsum('sij,sjk->sik', x, y)

        def rotated_local_matrix(site_index):
            rot = None if rotation is None else rotation[:, site_index, ...]
            return self.get_local_matrix(site_index, spin_config[:, site_index],
                                         rotation=rot)

        m = rotated_local_matrix(start_index)
        for site_index in range(start_index+1, stop_index):
            m = m.apply_mul( rotated_local_matrix(site_index), 
                                contractor)
        return m

    def trace_rho_squared(self, site_index):
        """ Compute the trace of the reduced partial density matrix squared, obtained
        by partitioning the system between sites (site_index, site_index + 1)
        
        Returns: real scalar."""
        if site_index < 0 or site_index >= self.L-1:
            raise ValueError("not a valid bond for partitioning the system")
        if self.gauge_index != site_index:
            warnings.warn("MPS should be gauged before computing reduced density ops")
            self.gauge_to(site_index)

        A = self.get_local_tensor(site_index)
        contractor_inner = lambda a, astar: torch.einsum('sij,sik->jk', a, astar)
        contractor_spatial = lambda a1, a2: torch.einsum('ij,ij->', a1, a2)
        Ainner = A.apply_mul(A.conj(), contractor_inner)
        return Ainner.apply_mul(Ainner.conj(), contractor_spatial).numpy().real

    def renyi2_entropy(self, site_index):
        """ Compute the Renyi-2 entropy for the density matrix defined on the subsystem
            [0, ... site_index] (inclusive)"""
        return -np.log(self.trace_rho_squared(site_index))
        

    def amplitude(self, spin_config, rotation=None):
        """ spin_config= (N, L) tensor listing spin configurations.
        rotations: (N, L,d, d) complextensor of local unitaries applied.
            Returns: (N,) tensor of amplitudes, one for each spin config.
            """
        spin_config = self.sanitize_spin_config(spin_config)
        N = spin_config.shape[0]
        return self.contract_interval(spin_config, 0, self.L, rotation=rotation).view(N)

    def merge(self, site_index):
        """ Contract the two local tensors at site_index, site_index + 1
        and return the corresponding four-index object, with shape
        (spin1, spin2, bond1, bond2)"""
        contractor = lambda al, ar: torch.einsum('sij,tjl->stil', al,ar)
        al = self.get_local_tensor(site_index)
        ar = self.get_local_tensor(site_index+1)
        return al.apply_mul(ar, contractor)

    def prob_unnormalized(self, x,rotation=None):
        """Compute the unnormalized probability associated with spin configuration x"""
        a = self.amplitude(x, rotation=rotation)
        return (a * a.conj()).real

    def nll_loss(self, x, rotation=None):
        """ Compute the negative-log-likelihood loss function for a batch of spin
        configurations x, and corresponding local rotations."""
        return - self.prob_unnormalized(x,rotation=rotation).log().mean() + self.norm().log()

    def prob_normalized(self, x, rotation=None):
        """ Compute the normalized probability of a spin configuration x.
            If provided, local rotations are applied to the state before computing the
            probability.
            """
        return self.prob_unnormalized(x,rotation=rotation) / self.norm()

    def amplitude_normalized(self, x, rotation=None):
        """ Compute the normalized amplitude of a spin configuration x.
            If provided, local rotations are applied to the state first."""
        return self.amplitude(x, rotation=rotation).div(self.norm().sqrt())

    def overlap(self, other):
        """ Compute the overlap <other|self> onto another MPS.
            Returns: complex scalar"""
        if self.L != other.L:
            raise ValueError("other state does not have the same length")
        contractor_upper = lambda blob, upper: torch.einsum('ijkl,skm->ijmls', blob, upper)
        contractor_lower = lambda blob, lower: torch.einsum('ijmls,sln->ijmn', blob, lower)
        spin_contractor = lambda upper, lower: torch.einsum('sik,sjl->ijkl',upper, lower)
        A0 = self.get_local_tensor(0)
        blob = A0.apply_mul(other.get_local_tensor(0).conj(),spin_contractor)
        for i in range(1, self.L):
            blob = blob.apply_mul(self.get_local_tensor(i),contractor_upper)
            blob = blob.apply_mul(other.get_local_tensor(i).conj(), contractor_lower)
        return blob.numpy().item()

    ### methods for computing various gradients
    def partial_deriv_twosite_psi(self, site_index, spin_config, 
                                            rotation=None):
        """Compute the gradient of Psi(spin_config) (with the given local
        unitaries applied) with respect to the two-site merged tensor at (site_index, site_index + 1)
        Returns: complex numpy array, indexing as: (batch, spin1, spin2, bond1, bond2), shape
                    (N, local_dim, local_dim, bond1, bond2) 
            spin_config: (N, L) int tensor of spin configurations
            rotation:(N, L, d, d) complextensor of local unitaries applied."""
        with torch.no_grad():
            spin_config = self.sanitize_spin_config(spin_config)
            N = spin_config.shape[0]
            if site_index < 0 or site_index >= self.L-1:
                raise ValueError("Invalid index for twosite gradient")

            if site_index == 0:
                left_contracted = ComplexTensor(torch.ones((N,1,1)), torch.zeros((N,1,1)))
            else:
                #shape (N, 1, D1)
                left_contracted = self.contract_interval(spin_config,0,site_index,
                                                                rotation=rotation)
            if site_index == self.L-2:
                right_contracted = ComplexTensor(
                    torch.ones((N, 1, 1)), torch.zeros((N, 1, 1)))
            else:
                #shape (N, D2, 1)
                right_contracted = self.contract_interval(spin_config, site_index +2, self.L,
                                                                rotation=rotation)
            
            D1 = left_contracted.shape[-1]
            D2 = right_contracted.shape[-2]
            # grad_shape = (N, self.local_dim, self.local_dim, D1, D2 )
            left_contracted = left_contracted.view(N,1, 1, D1, 1)
            right_contracted = right_contracted.view(N,1, 1, 1, D2)
            if rotation is None:
                #in this case the contracted unitary is a delta function in the 
                # spin index, ie one-hot encoding
                U1_contracted = make_onehot(spin_config[:,site_index],self.local_dim)
                U2_contracted = make_onehot(spin_config[:,site_index+1],self.local_dim)
                
            else:
                U1_contracted = rotation[range(N), site_index, 
                            spin_config[:, site_index], :]
                U2_contracted = rotation[range(N), site_index+1,
                              spin_config[:, site_index+1], :]

            U1_contracted = U1_contracted.view(N, self.local_dim, 1, 1, 1)
            U2_contracted = U2_contracted.view(N, 1, self.local_dim, 1, 1)
           
            return left_contracted * right_contracted * (U1_contracted * U2_contracted)

    def partial_deriv_twosite_norm(self, site_index):
        """ Compute the grad of the norm WRT two-site blob at specified index. 
        First checks that mps is gauged to the relevant site.
        Returns: (loc_dim, loc_dim, D1, D2) ComplexTensor"""
        if self.gauge_index != site_index:
            warnings.warn("MPS should be gauged to blob site before calling norm gradient")
            self.gauge_to(site_index)
        return self.merge(site_index).conj()

    def partial_deriv_twosite_nll(self, site_index, spin_config, rotation=None):
        """ Compute the partial derivative of the negative-log-likelihood cost function
           WRT complex-valued twosite blob at the specd site.
             averaged over batch dimension."""
        spin_config=self.sanitize_spin_config(spin_config)
        N = spin_config.shape[0]
        with torch.no_grad():
            #paritial of the amplitude WRT blob, shape (N, d, d, D1, D2)
            grad_psi = self.partial_deriv_twosite_psi(site_index, spin_config,rotation=rotation)
            #gradient of the WF normalization
            grad_norm = self.partial_deriv_twosite_norm(site_index)
            #amplitudes of the spin configurations
            amplitude = self.amplitude(spin_config,rotation=rotation).view(N, 1, 1, 1, 1)
            return ((grad_psi * amplitude.conj()).div( amplitude.norm())).mean(0)* -1.0  + grad_norm

    def grad_twosite_nll(self, site_index, spin_config,
                         rotation=None, cutoff=1e-10, normalize='left',
                         max_sv_to_keep=None):
        """Computes the update tensor defined by the gradient of the negative log-likelihood cost function
         with respect to the real and imaginary parts of the two-site blob at site_index.
         
            site_index: spatial index for the left edge of the two-site blob.
            spin_config: (batch_size, L) tensor of integer indices of observed spin configurations.
            rotation: local unitaries to apply to the MPS
            cutoff: singular values below cutoff will be dropped when blob is split.
            normalize = 'left', 'right': how to normalize the blob after splitting.
            max_sv_to_keep: if not None, max number of singular values to keep at splitting.

            Note: psi will be gauged to site_index.

            Returns: gradient array of shape (local_dim, local_dim, bond_dim, bond_dim) holding the gradients
            of the NLL WRT the real and imag parts of the two-site blob."""

        self.gauge_to(site_index)
        #gradient of the log-prob WRT that complex matrix
        #note that A has to updated from the conjugate!
        g = self.partial_deriv_twosite_nll(site_index, spin_config, rotation=rotation).numpy().conj()
        return 2 * g

    def partial_deriv_twosite_trace_rho_squared(self, site_index):
        """ Compute the partial derivative of the purity, defined by partitioning
        the system at bond (site_index, site_index +1), with respect to the blob tensor at the bond.
        Returns: (local_dim, local_dim, bond_dim, bond_dim) ComplexTensor"""
        if self.gauge_index != site_index:
            warnings.warn("MPS should be gauged to blob site before computing grads")
            self.gauge_to(site_index)
        A = self.merge(site_index)
        inner_contractor = lambda a, astar: torch.einsum('stij,stik->jk',a,astar)
        edge_contractor = lambda astar, a: torch.einsum('stil,lk->stik',astar,a)
        inner_blob = A.apply_mul(A.conj(), inner_contractor)
        return A.conj().apply_mul(inner_blob, edge_contractor) * 2


    def partial_deriv_twosite_renyi2_entropy(self, site_index):
        """ Compute the partial derivative of the renyi-2 entropy, defined by partitioning
        the system at bond (site_index, site_index +1), with respect to the blob tensor at the bond.
        Returns: (local_dim, local_dim, bond_dim, bond_dim) ComplexTensor"""
        partial_tr = self.partial_deriv_twosite_trace_rho_squared(site_index)
        purity = self.trace_rho_squared(site_index)
        return partial_tr.div(purity) * (-1.0)

    def grad_twosite_renyi2_entropy(self, site_index):
        """ Compute the gradient of the renyi-2 entropy defined by paritioning at (site_index, 
        site_index +1) WRT the blob matrix there.
        Returns: (local_dim, local_dim, bond_dim, bond_dim) complex numpy array holding gradients
        of renyi-2 WRT real and imag parts of the blob."""
        return 2 * self.partial_deriv_twosite_renyi2_entropy(site_index).numpy().conj()

    def set_sites_from_twosite(self, site_index, twosite,
                                    cutoff=1e-16, max_sv_to_keep=None, 
                                    normalize='left'):
        """Update the MPS local tensors at site_index, site_index +1 from twosite blob provided.
        The blob is SVD'd with spec truncation parameters, and either the left or right tensor is normalized.
        Local tensors are then overwritten with the SVD results.
        twosite: (local_dim, local_dim, bond_dim, bond_dim) complex numpy array.
        
        Note that definite gauge is enforced: psi will be gauged to either site_index or
        site_index + 1 depending on which normalization is selected."""
        Aleft, Aright = split_two_site(twosite,normalize=normalize,
                                            cutoff=cutoff,max_sv_to_keep=max_sv_to_keep)

        self.set_local_tensor_from_numpy(site_index, Aleft)
        self.set_local_tensor_from_numpy(site_index+1, Aright)
        self.gauge_to(site_index+1 if normalize=='left' else site_index)
    
    def do_sgd_step(self, site_index, spin_config, 
                    rotation=None,cutoff=1e-10,normalize='left',max_sv_to_keep=None,
                    learning_rate=1e-3, s2_penalty=None):
        """Perform a gradient-descent step by varying only the two-site blob with left edge at site_index.
           site_index: spatial index for the left edge of the two-site blob.
            spin_config: (batch_size, L) tensor of integer indices of observed spin configurations.
            rotation: local unitaries to apply to the MPS
            cutoff: singular values below cutoff will be dropped when blob is split.
            normalize = 'left', 'right': how to normalize the blob after splitting.
            max_sv_to_keep: if not None, max number of singular values to keep at splitting.
            learning_rate: for SGD update
            s2_penalty: if not None, penalty term corresponding to coefficient of the Renyi-2 entropy in 
            the cost function. Positive values will discourage high entropy.
        """
        #gradient of the NLL cost function with respect to real and imag parts of blob
        nll_grad = self.grad_twosite_nll(site_index, spin_config,  
                                                rotation=rotation,cutoff=cutoff,normalize=normalize,
                                                max_sv_to_keep=max_sv_to_keep)
        
        #gradient array of the renyi-2 entropy at site_index WRT blob
        if s2_penalty is not None:
            s2_grad = self.grad_twosite_renyi2_entropy(site_index)
            nll_grad = nll_grad + s2_penalty * s2_grad
            
        #two-site blob matrix at the site
        blob = self.merge(site_index).numpy()
        blob = blob - learning_rate * nll_grad
        #update the mps with new blob
        self.set_sites_from_twosite(site_index, blob,
                                   cutoff=cutoff, max_sv_to_keep=max_sv_to_keep,
                                   normalize=normalize)



    @property
    def shape(self):
        """ Representation of MPS 'shape' as defined by its bond dimensions"""
        shapes = [tuple(t.shape[1:]) for t in self.tensors]
        return shapes

### helper functions for building a few simple GHZ states

def build_ghz_plus(L):
    """ Return normalized MPS representing a GHZ+ state of length L"""
    psi = MPS(L, local_dim=2,bond_dim=2)
    with torch.no_grad():
        A0r = torch.tensor([[0,1],[0,0]],dtype=torch.float)
        A1r = torch.tensor([[0,0],[1,0]],dtype=torch.float)
        Ar = torch.stack([A0r, A1r], 0)
        Ai = torch.zeros_like(Ar)
        ## Bulk tensor
        A = ComplexTensor(Ar, Ai)

        l0r = torch.tensor([[0,1]],dtype=torch.float) 
        l1r = torch.tensor([[1,0]],dtype=torch.float) 
        lr = torch.stack([l0r, l1r],0)
        rr = torch.stack([l1r.view(2,1)/np.sqrt(2), l0r.view(2,1)/np.sqrt(2)],0)
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
    Ar = torch.tensor([np.cos(theta/2), np.sin(theta/2)*np.cos(phi)]).view(2,1,1)
    Ai = torch.tensor([0., np.sin(phi)]).view(2,1,1)
    A = ComplexTensor(Ar, Ai)
    psi = MPS(L, local_dim=2,bond_dim=1)
    for i in range(L):
        psi.set_local_tensor(i,A)
    psi.gauge_index=None
    return psi

from torch.utils.data import TensorDataset

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

