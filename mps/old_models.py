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
                                                    self.local_dim, requires_grad=True))
        self.left_tensor = nn.Parameter(torch.randn(1, self.bond_dim, self.local_dim,
                                                    requires_grad=True))
        self.right_tensor = nn.Parameter(torch.randn(self.bond_dim, 1, self.local_dim,
                                                     requires_grad=True))

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
        if (site_index < 0) or (site_index >= self.L):
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
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        N = x.shape[0]
        m = torch.einsum('ijb,jkb->ikb', self.get_local_matrix(0, x[:, 0]),
                         self.get_local_matrix(1, x[:, 1]))
        for ii in range(self.L-3):
            m = torch.einsum('ijb,jkb->ikb', m,
                             self.get_local_matrix(ii+2, x[:, ii+2]))
        a = torch.einsum('ijb,jkb->ikb', m,
                         self.get_local_matrix(self.L-1, x[:, self.L-1]))
        return a.view(N)

    def prob_unnormalized(self, x):
        a = self.amplitude(x)
        return a * a

    def nll_cost(self, x):
        return - self.prob_unnormalized(x).log().mean() + self.norm().log()

    def prob_normalized(self, x):
        return self.prob_unnormalized(x) / self.norm()



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

    def sanitize_spin_config(self, spin_config):
        if len(spin_config.shape) == 1:
            spin_config = spin_config.unsqueeze(0)
        if spin_config.size(1) != self.L:
            raise ValueError("spin configuration does not match MPS size")
        return spin_config

    def norm(self):
        def init_contractor(x, y): return torch.einsum('sij,sik->jk', x, y)
        def spin_contractor(x, y): return torch.einsum('sik,sjl->ijkl', x, y)
        def bulk_contractor(x, y): return torch.einsum('ij,ijkl->kl', x, y)

        cont = self.left_tensor.apply_mul(
            self.left_tensor.conj(), init_contractor)
        bc = self.bulk_tensor.apply_mul(
            self.bulk_tensor.conj(), spin_contractor)
        rc = self.right_tensor.apply_mul(
            self.right_tensor.conj(), spin_contractor)

        for __ in range(self.L-2):
            cont = cont.apply_mul(bc, bulk_contractor)

        cont = cont.apply_mul(rc, bulk_contractor)
        return cont.squeeze().real

    def get_local_tensor(self, site_index):
        if (site_index < 0) or (site_index >= self.L):
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
            def contractor(x, y): return torch.einsum('ast,tij->asij', x, y)
            #shape (N, local_dim, D, D)
            local_tensor = rotation.apply_mul(local_tensor, contractor)
            #shape (N, D, D)
            local_matrix = local_tensor[range(N), spin_index, ...]
        else:
            local_matrix = local_tensor[spin_index, ...]
        return local_matrix

    def amplitude(self, spin_config, rotation=None):
        """ spin_config= (N, L) tensor listing spin configurations.
        rotation: (N, L,d, d) complextensor of local unitaries applied.
            Returns: (N,) tensor of amplitudes.
            """
        spin_config = self.sanitize_spin_config(spin_config)
        N = spin_config.size(0)

        def contractor(x, y): return torch.einsum('sij,sjk->sik', x, y)

        def rotated_local_matrix(site_index):
            rot = None if rotation is None else rotation[:, site_index, ...]
            return self.get_local_matrix(site_index, spin_config[:, site_index],
                                         rotation=rot)
        m0 = rotated_local_matrix(0)
        m1 = rotated_local_matrix(1)
        m = m0.apply_mul(m1, contractor)

        if self.L > 2:
            for ii in range(self.L-3):
                mbulk = rotated_local_matrix(ii+2)
                m = m.apply_mul(mbulk, contractor)

            a = m.apply_mul(rotated_local_matrix(self.L-1), contractor)
        return a.view(N)

    def prob_unnormalized(self, x, rotation=None):
        a = self.amplitude(x, rotation=rotation)
        return (a * a.conj()).real

    def nll_loss(self, x, rotation=None):
        return - self.prob_unnormalized(x, rotation=rotation).log().mean() + self.norm().log()

    def prob_normalized(self, x, rotation=None):
        return self.prob_unnormalized(x, rotation=rotation) / self.norm()
