import numpy as np
import torch
import torch.nn as nn
import warnings

#tools for SVD -- normalization and breaking two-site tensors
from utils import svd_push_left, svd_push_right, split_two_site

class SingleBasisMPS(nn.Module):
    """ A matrix product state ..."""

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

    def build_tensors(self):
        """Create and store local tensors.
        Local tensor has shape (bond_dim, bond_dim, L)"""

        self.bulk_tensor = nn.Parameter(torch.randn(self.bond_dim, self.bond_dim,
                                        self.local_dim,requires_grad=True))
        self.left_tensor = nn.Parameter(torch.randn(1, self.bond_dim, self.local_dim,
                                                requires_grad=True))
        self.right_tensor = nn.Parameter(torch.randn(self.bond_dim, 1, self.local_dim,
                                                requires_grad = True))

    def init_tensors(self):
        for t in self.parameters():
            with torch.no_grad():
                t.normal_(0, 1.0 / np.sqrt(self.bond_dim * self.local_dim))


    def norm(self):
        cont = torch.einsum('ijk,ilk->jl', self.left_tensor, self.left_tensor)
        for site in range(self.L-2):
            cont = torch.einsum('jl,jms,lks->mk', cont,
                                                self.bulk_tensor,
                                                self.bulk_tensor)
        cont = torch.einsum('ij,iks,jls->kl', cont, self.right_tensor,
                                                    self.right_tensor)
        return cont

    def get_local_tensor(self, site_index):
        if (site_index <0) or (site_index >= self.L):
            raise ValueError("Invalid site index")
        if site_index == 0:
            return self.left_tensor
        elif site_index == self.L-1:
            return self.right_tensor
        return self.bulk_tensor

    def get_local_matrix(self, site_index, spin_index):
        """spin_index = an (N,) tensor of spin configurations.
            returns: (D,D,N) tensor holding local matrices for each spin
            configuration."""
        local_tensor = self.get_local_tensor(site_index)
        return local_tensor[..., spin_index]

    def amplitude(self, x):
        """ x= (N, L) tensor listing spin configurations.
            Returns: (N,) tensor of amplitudes"""
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        N = x.shape[0]
        m = torch.einsum('ijb,jkb->ikb',self.get_local_matrix(0, x[:,0]),
                            self.get_local_matrix(1,x[:,1]))
        for ii in range(self.L-3):
            m = torch.einsum('ijb,jkb->ikb',m, self.get_local_matrix(ii+2, x[:,ii+2]))
        a = torch.einsum('ijb,jkb->ikb', m, self.get_local_matrix(self.L-1, x[:,self.L-1]))
        return a.view(N)

    def prob_unnormalized(self, x):
        a = self.amplitude(x)
        return a * a

    def nll_cost(self, x):
        return - self.prob_unnormalized(x).log().mean() + self.norm().log()

    def prob_normalized(self, x):
        return self.prob_unnormalized(x) / self.norm()


class ComplexTensor:

    def __init__(self, real, imag):
        assert real.shape == imag.shape
        self.real = real
        self.imag = imag

    def apply_mul(self, other, mul_op):
        r = mul_op(self.real, other.real) - mul_op(self.imag, other.imag)
        i = mul_op(self.real, other.imag) + mul_op(self.imag, other.real)
        return ComplexTensor(r,i)

    @property
    def shape(self):
        return self.real.shape

    def conj(self):
        return ComplexTensor(self.real, -self.imag)

    def __getitem__(self, slice):
        return ComplexTensor(self.real[slice], self.imag[slice])

    def view(self, *shape):
        return ComplexTensor(self.real.view(*shape), self.imag.view(*shape))
    def squeeze(self):
        return ComplexTensor(self.real.squeeze(), self.imag.squeeze())

    def __mul__(self, other):
        mul_op = lambda x, y: x * y
        return self.apply_mul(other, mul_op)

    def abs(self):
        return (self * self.conj()).real.sqrt()

    def __repr__(self):
        return "ComplexTensor shape {0}".format(self.shape)

    def size(self, i):
        return self.real.size(i)

    def numpy(self):
        return self.real.detach().numpy() + 1j * self.imag.detach().numpy()



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

        if len(spin_config.shape) == 1:
            spin_config = spin_config.unsqueeze(0)

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

    def amplitude(self, spin_config, rotation=None):
        """ spin_config= (N, L) tensor listing spin configurations.
        rotations: (N, L,d, d) complextensor of local unitaries applied.
            Returns: (N,) tensor of amplitudes, one for each spin config.
            """
        if len(spin_config.shape) == 1:
            spin_config = spin_config.unsqueeze(0)
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
        a = self.amplitude(x, rotation=rotation)
        return (a * a.conj()).real

    def nll_loss(self, x, rotation=None):
        return - self.prob_unnormalized(x,rotation=rotation).log().mean() + self.norm().log()

    def prob_normalized(self, x, rotation=None):
        return self.prob_unnormalized(x,rotation=rotation) / self.norm()

    ### methods for computing various gradients
    def grad_twosite_psi(self, site_index, spin_config, 
                                            rotation=None):
        """Compute the gradient of Psi(spin_config) (with the given local
        unitaries applied) with respect to the two-site merged tensor at (site_index, site_index + 1)
        Returns: complex numpy array, indexing as: (batch, spin1, spin2, bond1, bond2), shape
                    (N, local_dim, local_dim, bond1, bond2) 
            spin_config: (N, L) int tensor of spin configurations
            rotation:(N, L, d, d) complextensor of local unitaries applied."""
        with torch.no_grad():
            if len(spin_config.shape) == 1:
                spin_config = spin_config.unsqueeze(0)
            N = spin_config.shape[0]
            if site_index < 0 or site_index >= self.L-1:
                raise ValueError("Invalid index for twosite gradient")

            if site_index == 0:
                left_contracted = ComplexTensor(torch.ones((N,1,1)), torch.zeros((N,1,1)))
            else:
                #shape (N, 1, D1)
                left_contracted = self.contract_interval(spin_config,0,site_index)
            if site_index == self.L-2:
                right_contracted = ComplexTensor(
                    torch.ones((N, 1, 1)), torch.zeros((N, 1, 1)))
            else:
                #shape (N, D2, 1)
                right_contracted = self.contract_interval(spin_config, site_index +1, self.L)
            
            D1 = left_contracted.shape[-1]
            D2 = right_contracted.shape[-2]
            # grad_shape = (N, self.local_dim, self.local_dim, D1, D2 )
            left_contracted = left_contracted.view(N,1, 1, D1, 1)
            right_contracted = right_contracted.view(N,1, 1, 1, D2)
            if rotation is None:
                U1r = torch.stack([torch.eye(self.local_dim) for __ in range(N)],0)
                U2r = torch.stack([torch.eye(self.local_dim) for __ in range(N)],0)
                U1i = torch.zeros_like(U1r)
                U2i = torch.zeros_like(U2r)
                U1 = ComplexTensor(U1r, U1i)
                U2 = ComplexTensor(U2r, U2i)
            else:
                U1 = rotation[range(N), site_index, 
                            spin_config[:, site_index], :]
                U2 = rotation[range(N), site_index+1,
                              spin_config[:, site_index+1], :]
            U1 = U1.view(N, self.local_dim, self.local_dim, 1, 1)
            U2 = U2.view(N, self.local_dim, self.local_dim, 1, 1)
           
            return left_contracted * right_contracted * U1 * U2

    def grad_twosite_norm(self, site_index):
        """ Compute the grad of the norm WRT two-site blob at specified index. 
        First checks that mps is gauged to the relevant site.
        Returns: (loc_dim, loc_dim, D1, D2) ComplexTensor"""
        if self.gauge_index != site_index:
            warnings.warn("MPS should be gauged to blob site before calling norm gradient")
            self.gauge_to(site_index)
        return self.merge(site_index).conj()

    def grad_twosite_logprob(self, site_index, spin_config, rotation=None):
        """ Compute the gradient of the log probability WRT twosite blob at the specd site.
            Gradient is averaged over batch dimension."""
        #gradient of the amplitude WRT blob
        grad_psi = self.grad_twosite_psi(site_index, spin_config)
        #gradient of the WF normalization
        grad_norm = self.grad_twosite_norm(site_index)
        #amplitudes of the spin configurations
        amplitude = self.amplitude(spin_config,rotation=rotation)
        


    def set_sites_from_twosite(self, site_index, twosite,
                                    cutoff=1e-16, max_sv_to_keep=None, 
                                    normalize='left'):
        """Update the MPS local tensors at site_index, site_index +1 from twosite blob provided.
        The blob is SVD'd with spec truncation parameters, and either the left or right tensor is normalized.
        Local tensors are then overwritten with the SVD results.
        twosite: (local_dim, local_dim, bond_dim, bond_dim) complex numpy array"""
        Aleft, Aright = split_two_site(twosite,normalize=normalize,
                                            cutoff=cutoff,max_sv_to_keep=max_sv_to_keep)
        self.set_local_tensor_from_numpy(site_index, Aleft)
        self.set_local_tensor_from_numpy(site_index, Aright)
        
        


class UniformMPS(nn.Module):
    """ MPS with complex amplitudes, and uniform bulk matrix """

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

    def build_tensors(self):
        def build(shape):
            return nn.Parameter(torch.randn(*shape, requires_grad=True))

        bulk_shape = (self.local_dim, self.bond_dim, self.bond_dim)
        left_shape = (self.local_dim, 1, self.bond_dim)
        right_shape = (self.local_dim, self.bond_dim, 1)

        self.bulk_r = build(bulk_shape)
        self.bulk_i = build(bulk_shape)
        self.left_r = build(left_shape)
        self.left_i = build(left_shape)
        self.right_r = build(right_shape)
        self.right_i = build(right_shape)

        self.bulk_tensor = ComplexTensor(self.bulk_r, self.bulk_i)
        self.left_tensor = ComplexTensor(self.left_r, self.left_i)
        self.right_tensor = ComplexTensor(self.right_r, self.right_i)

    def init_tensors(self):
        for t in self.parameters():
            with torch.no_grad():
                t.normal_(0, 1.0 / np.sqrt(self.bond_dim * self.local_dim))

    def norm(self):
        init_contractor = lambda x, y: torch.einsum('sij,sik->jk', x, y)
        spin_contractor = lambda x, y: torch.einsum('sik,sjl->ijkl', x, y)
        bulk_contractor = lambda x, y: torch.einsum('ij,ijkl->kl', x, y)

        cont = self.left_tensor.apply_mul(self.left_tensor.conj(), init_contractor)
        bc = self.bulk_tensor.apply_mul(self.bulk_tensor.conj(), spin_contractor)
        rc = self.right_tensor.apply_mul(self.right_tensor.conj(),spin_contractor)

        for site in range(self.L-2):
            cont = cont.apply_mul(bc, bulk_contractor)

        cont = cont.apply_mul(rc, bulk_contractor)
        return cont.squeeze().real

    def get_local_tensor(self, site_index):
        if (site_index <0) or (site_index >= self.L):
            raise ValueError("Invalid site index")
        if site_index == 0:
            return self.left_tensor
        elif site_index == self.L-1:
            return self.right_tensor
        return self.bulk_tensor

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
                raise ValueError("Index tensor incompatible with rotation")
            N = spin_index.size(0)
            contractor = lambda x, y: torch.einsum('ast,tij->asij', x, y)
            #shape (N, local_dim, D, D)
            local_tensor = rotation.apply_mul(local_tensor, contractor)
            #shape (N, D, D)
            local_matrix = local_tensor[range(N), spin_index, ...]
        else:
            local_matrix = local_tensor[spin_index, ...]
        return local_matrix

    def amplitude(self, x, rotation=None):
        """ x= (N, L) tensor listing spin configurations.
        rotation: (N, L,d, d) complextensor of local unitaries applied.
            Returns: (N,) tensor of amplitudes.
            """
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        N = x.shape[0]
        contractor = lambda x, y: torch.einsum('sij,sjk->sik', x, y)

        def rotated_local_matrix(site_index):
            rot=None if rotation is None else rotation[:, site_index, ...]
            return self.get_local_matrix(site_index, x[:, site_index],
                                            rotation=rot)
        m0 = rotated_local_matrix(0)
        m1 = rotated_local_matrix(1)
        m = m0.apply_mul( m1,contractor)

        if self.L > 2:
            for ii in range(self.L-3):
                mbulk = rotated_local_matrix(ii+2)
                m = m.apply_mul(mbulk, contractor)

            a = m.apply_mul(rotated_local_matrix(self.L-1),contractor)
        return a.view(N)

    def prob_unnormalized(self, x,rotation=None):
        a = self.amplitude(x, rotation=rotation)
        return (a * a.conj()).real

    def nll_loss(self, x, rotation=None):
        return - self.prob_unnormalized(x,rotation=rotation).log().mean() + self.norm().log()

    def prob_normalized(self, x, rotation=None):
        return self.prob_unnormalized(x,rotation=rotation) / self.norm()
