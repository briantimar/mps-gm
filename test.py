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



if __name__=='__main__':
    unittest.main()