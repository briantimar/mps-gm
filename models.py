import numpy as np
import torch
import torch.nn as nn


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

    def get_local_matrix(self, site_index, spin_index):
        """spin_index = an (N,) tensor of spin configurations.
            returns: (N,D,D) tensor holding local matrices for each spin
            configuration."""
        local_tensor = self.get_local_tensor(site_index)
        return local_tensor[spin_index,...]

    def amplitude(self, x):
        """ x= (N, L) tensor listing spin configurations.
            Returns: (N,) tensor of amplitudes"""
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        N = x.shape[0]
        contractor = lambda x, y: torch.einsum('sij,sjk->sik', x, y)
        m = self.get_local_matrix(0, x[:,0]).apply_mul(
                                            self.get_local_matrix(1,x[:,1]),
                                            contractor)
        for ii in range(self.L-3):
            m = m.apply_mul(self.get_local_matrix(ii+2, x[:,ii+2]),
                                            contractor)

        a = m.apply_mul(self.get_local_matrix(self.L-1, x[:,self.L-1]),
                                        contractor)
        return a.view(N)

    def prob_unnormalized(self, x):
        a = self.amplitude(x)
        return (a * a.conj()).real

    def nll_loss(self, x):
        return - self.prob_unnormalized(x).log().mean() + self.norm().log()

    def prob_normalized(self, x):
        return self.prob_unnormalized(x) / self.norm()
