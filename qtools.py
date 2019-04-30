""" tools for interfacing gen model outputs with quantum objects """
import qutip as qt
import numpy as np
import torch

def to_meas_setting(pauli_code, ms_type='integer'):
    """Convert a particular pauli operator 'X', 'Y', or 'Z' into a measurement
    specification.
        ms_type: the format of the measurement setting.
        'integer' --> returns an integer code
            --> return tensor has shape (L, 1)
        'angles' --> returns a pair of angles (theta, phi)

         """
    pauli_code = pauli_code.upper()

    if ms_type not in ['integer', 'angles']:
        raise ValueError("ms_type can be 'integer', 'angles'")

    if len(pauli_code)==1:
        pauli_list = ['X','Y','Z']
        if pauli_code not in pauli_list:
            raise ValueError("Valid pauli codes: {0}. received: {1}".format(
                                                    pauli_list, pauli_code
            ))
        if ms_type=='integer':
            return np.asarray([[pauli_list.index(pauli_code)]])
        elif ms_type =='angles':
            if pauli_code == 'X':
                ms= np.asarray((np.pi/2, 0.0))
            elif pauli_code == 'Y':
                ms= np.asarray((np.pi/2, np.pi/2))
            else:
                ms= np.asarray((0.0,0.0))
            return np.reshape(ms, (1,2))

    else:
        L = len(pauli_code)
        lastdim = 1 if ms_type == 'integer' else 2
        settings = np.empty((L, lastdim))
        for ii in range(L):
            settings[ii, :] = to_meas_setting(pauli_code[ii],ms_type=ms_type)
        return settings

def int_ms_to_pauli_code(ms):
    """measurement code --> Pauli strings
        0 -> x
        1 -> y
        2 -> z
        3 -> I"""
    try:
        s = ''
        for i in ms:
            s += int_ms_to_pauli_code(i)
        return s
    except TypeError:
        pauli_list = ['X','Y','Z','I']
        return pauli_list[ms]

def int_basis_indx_to_ms(indx, L, nsetting=3):
    ms = np.empty(L,dtype=int)
    for i in range(L):
        r = indx % nsetting
        ms[L-i-1] = r
        indx = (indx - r)//nsetting
    return ms

def tolabel(pauli_code):
    s = ''
    for p in pauli_code:
        s+=p
    return s

def int_basis_indx_to_paulis(indx, L,include_identity=True):
    """ converts int in the range((3 or 4)^L) to a Pauli string
        include_identity: whether or not to include the identity operator.
        indx: integer
        L: int, system size

        returns: length-L Pauli string"""

    nsetting = 4 if include_identity else 3
    if indx <0 or indx >= nsetting**L:
        raise ValueError("index %d not a valid basis specification"%indx)
    int_ms = int_basis_indx_to_ms(indx, L, nsetting=nsetting)
    return int_ms_to_pauli_code(int_ms)

def to_cartesian(angles):
    """ Convert an array of angles to cartesian coordinates."""
    theta, phi = angles[...,0], angles[...,1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    settings = np.stack([x,y,z], axis=-1)
    return settings

from models import ComplexTensor

def pauli_exp(theta, phi):
    """ Return a ComplexTensor representing local unitaries specified by the
    given angles.s
        Returns: (*angles.shape, 2,2) complexTensor"""
    if theta.shape != phi.shape:
        raise ValueError

    rU = torch.empty((*theta.shape,2,2),dtype=theta.dtype)
    iU = torch.empty((*theta.shape,2,2),dtype=theta.dtype)
    ct, st = (theta/2).cos(), (theta/2).sin()
    cp, sp = (phi/2).cos(), (phi/2).sin()

    rU[...,0,0] = ct * cp
    rU[...,0,1] = st*cp
    rU[...,1,0] = -st * cp
    rU[...,1,1] = ct * cp

    iU[...,0,0] = ct * sp
    iU[...,0,1] = -st * sp
    iU[...,1,0] = -st * sp
    iU[...,1,1] = -ct * sp

    return ComplexTensor(rU, iU)
