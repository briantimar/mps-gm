import unittest
from unittest import TestCase
from utils import *
import numpy as np
from models import MPS
from models import build_uniform_product_state
from tools import generate_binary_space


class TestUtils(TestCase):

    def test_push_right_normalization(self):
        al = np.random.normal(size=(2,4,5)) + 1j*np.random.normal(size=(2,4,5))
        ar = np.random.normal(size=(2,5,6))
        al_new, ar_new = svd_push_right(al,ar)
       
        al_contracted = np.einsum('sij,sik->jk',al_new, np.conj(al_new))
        self.assertAlmostEqual(np.sum((al_contracted - np.identity(5))),0)
    
    def test_push_right_truncation(self):
        al = np.random.normal(size=(2,4,5)) + 1j*np.random.normal(size=(2,4,5))
        ar = np.random.normal(size=(2,5,6))
        al_new, ar_new = svd_push_right(al,ar,max_sv_to_keep=2)
        self.assertEqual(al_new.shape, (2, 4, 2))
        self.assertEqual(ar_new.shape, (2, 2, 6))
       
        al_contracted = np.einsum('sij,sik->jk',al_new, np.conj(al_new))
        self.assertAlmostEqual(np.sum((al_contracted - np.identity(2))),0)


    def test_push_left(self):
        al = np.random.normal(size=(2,4,5)) + 1j*np.random.normal(size=(2,4,5))
        ar = np.random.normal(size=(2,5,6)) + 1j*np.random.normal(size=(2,5,6))
        
        al_new, ar_new = svd_push_left(al,ar)
        ar_contracted = np.einsum('sik,sjk->ij',ar_new, np.conj(ar_new))
        self.assertAlmostEqual(np.sum((ar_contracted - np.identity(5))),0)


    def test_split_two_site(self):
        local_dim, D1, D2 = 2, 6, 10
        sv_to_keep = 3
        A = ( np.random.normal(size=(local_dim, local_dim, D1, D2))
                + 1j * np.random.normal(size=(local_dim, local_dim, D1, D2)))
        Aleft, Aright= split_two_site(A, normalize='left', 
                                        max_sv_to_keep=sv_to_keep)

        self.assertEqual(Aleft.shape, (local_dim, D1, sv_to_keep))
        self.assertEqual(Aright.shape, (local_dim, sv_to_keep, D2))

    def test_split_two_site_left_normalized(self):
        local_dim, D1, D2 = 2, 6, 10
        sv_to_keep = 3
        A = ( np.random.normal(size=(local_dim, local_dim, D1, D2))
                + 1j * np.random.normal(size=(local_dim, local_dim, D1, D2)))
        Aleft, Aright= split_two_site(A, normalize='left', 
                                        max_sv_to_keep=sv_to_keep)

        Alc = np.einsum('sij,sik->jk', Aleft, np.conj(Aleft))
        self.assertAlmostEqual( np.sum( Alc - np.identity(sv_to_keep)),0)


    def test_split_two_site_right_normalized(self):
        local_dim, D1, D2 = 2, 6, 10
        sv_to_keep = 3
        A = ( np.random.normal(size=(local_dim, local_dim, D1, D2))
                + 1j * np.random.normal(size=(local_dim, local_dim, D1, D2)))
        Aleft, Aright= split_two_site(A, normalize='right', 
                                        max_sv_to_keep=sv_to_keep)

        Arc = np.einsum('sik,sjk->ij', Aright, np.conj(Aright))
        self.assertAlmostEqual( np.sum( Arc - np.identity(sv_to_keep)),0)


class TestQTools(TestCase):

    def test_pauli_exp(self):
        from qtools import pauli_exp
        theta = torch.tensor([0, np.pi/2, np.pi/2])
        phi = torch.tensor([0, 0, np.pi/2])
        U = pauli_exp(theta, phi)

        target1 = np.identity(2)
        target2 = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], 
                            [-1/np.sqrt(2), 1/np.sqrt(2)]])
        target3 = (1.0 / np.sqrt(2)) * np.array([[ np.exp(1j * np.pi/4), np.exp(-1j * np.pi/4)],
                                                [-np.exp(1j * np.pi/4), np.exp(-1j * np.pi/4)]])
        targets = np.stack([target1, target2, target3], axis=0)

        U = pauli_exp(theta, phi).numpy()

        self.assertAlmostEqual( np.abs(np.sum(U - targets)),0,places=6)

class TestMPS(TestCase):

    def test_normalization(self):
        L=2
        psi = MPS(L,local_dim=2,bond_dim=5)
        
        with torch.no_grad():
            basis = torch.tensor(generate_binary_space(L),dtype=torch.long)
            probs = psi.prob_normalized(basis)
            self.assertAlmostEqual(probs.sum().item(), 1.0,places=6)

    def test_uniform_product_state(self):
        L=2
        theta, phi = 0,0
        psi = build_uniform_product_state(L, theta, phi)
        with torch.no_grad():
            self.assertAlmostEqual(psi.norm().item(), 1.0)

            n_eigenvals = torch.tensor(generate_binary_space(L), dtype=torch.long)
            basis = 1-n_eigenvals
            target = np.asarray([1,0,0,0])
            
            self.assertAlmostEqual(np.sum(np.abs(target - 
                                                psi.prob_normalized(basis).numpy())),0,places=6)

            psix = build_uniform_product_state(L, np.pi/2, 0)
            targetx = .25 * np.ones(4)
            self.assertAlmostEqual(np.sum(np.abs(targetx -
                                                 psix.prob_normalized(basis).numpy())), 0, places=6)

    def test_amplitudes(self):
        from qtools import pauli_exp
        L = 2
        psi = build_uniform_product_state(L, 0, 0)
        with torch.no_grad():
           
            n_eigenvals = torch.tensor(
                generate_binary_space(L), dtype=torch.long)
            basis = 1-n_eigenvals
            amp = psi.amplitude_normalized(basis).numpy()
            target = np.asarray([1, 0, 0, 0])
            self.assertAlmostEqual(np.sum(np.abs(target - amp)),0,places=6)

            theta, phi = (np.pi /2 * torch.ones_like(basis,dtype=torch.float),
                         0.0 * torch.ones_like(basis,dtype=torch.float))
            U = pauli_exp(theta, phi)
            
            amp = psi.amplitude_normalized(basis, rotation=U).numpy()
            target = .5 * np.asarray([1, -1, -1, 1])
            self.assertAlmostEqual(np.sum(np.abs(target - amp)), 0, places=6)



    def test_overlap(self):
        psi = MPS(2,2,5)
        with torch.no_grad():
            self.assertAlmostEqual(psi.norm().numpy(), psi.overlap(psi), places=6)
    
    def test_build_ghz(self):
        from models import build_ghz_plus
        L = 4
        psi = build_ghz_plus(L)
        self.assertAlmostEqual(np.sum(np.abs(
                                        psi.amplitude_normalized(torch.tensor([1,0,1,0])).numpy()
                                        - 1.0/np.sqrt(2))),
                                0,places=6)
        self.assertAlmostEqual(np.sum(np.abs(
                                psi.amplitude_normalized(torch.tensor([0,1,0,1])).numpy()
                                    - 1.0/np.sqrt(2))),
                                0,places=6)
    


if __name__=='__main__':
    unittest.main()
