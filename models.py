import numpy as np
import torch
import torch.nn as nn

def dag(tensor):
    return torch.transpose(conj(tensor), 0, 1)

def conj(tensor):
    return tensor

class SingleBasisMPS(nn.Module):
    """ A matrix product state ..."""

    def __init__(self, L, local_dim, bond_dim):
        """ L =  size of the physical system
        local_dim = dimensionality of each local hilbert space
        bond_dimension = uniform bond dimension"""
        super().__init__()
        assert L >= 3
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
        return self.prob_unnormalized(x) / self.norm().sqrt()
