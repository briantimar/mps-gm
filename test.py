import unittest
from unittest import TestCase
from utils import *
import numpy as np

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



if __name__=='__main__':
    unittest.main()