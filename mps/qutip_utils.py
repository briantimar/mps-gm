""" Helper functions for working with qutip states"""

import qutip as qt
import numpy as np

PAULIS = ['X', 'Y', 'Z']
CODES = [0,1,2]

def to_code(pauli):
    if pauli in CODES:
        return pauli
    return PAULIS.index(pauli)
def to_pauli(code):
    return PAULIS[code]

def n():
    return 0.5 * ( 1 + qt.sigmaz())

def embed(ops, L, indices):
    """ embeds local operators O at sites <indices> into the hilbert space
        of an L-site chain"""
    d = ops[0].dims[0][0]
    if len(ops[0].dims[0])>1:
        raise ValueError("operator O has tensor product structure")
    if len(ops)!=len(indices):
        raise ValueError("number of ops does not match number of indices")
    ids = [ qt.identity(d) for __ in range(L)]

    for i in range(len(ops)):
        o, site = ops[i], indices[i]
        if o.dims != ops[0].dims:
            raise ValueError("operator inputs have different dimension")
        ids[site] = o
    return qt.tensor(ids)

def get_L(rho):
    """returns the physical system size for rho with tensor product structure"""
    return len(rho.dims[0])

def to_qutip_ket(numpy_vector, L):
    """Converts quspin state in the standard basis to a qutip ket with spin-1/2 tensor product structure.
        Change at your own risk.
        L = the number of physical sites. Assumed to be qubits."""
    if len(numpy_vector) != 2**L:
        raise ValueError("State does not have qubit dims matching L")
    dims_pure_state = [[2]*L, [1]*L]
    return qt.qobj.Qobj(numpy_vector, dims=dims_pure_state)

def get_all_onebody(onebody_op, L):
    """ Get all embeddings of a single-site operator onebody_op.
        L = number of sites."""
    return [embed([onebody_op], L, [i]) for i in range(L)]

def get_onebody_expt_vals(rho, onebody_op):
    """ Returns array of the expectation values of single-site operator
        onebody_op in the state rho.
            onebody_op should be defined on a single qubit hilbert space."""
    L = get_L(rho)
    ops = get_all_onebody(onebody_op, L)
    return np.asarray([qt.expect(o, rho) for o in ops])

def fetch_n_expt_vals(rho):
    """Return the expectation values of Rydberg projectors for each site."""
    return get_onebody_expt_vals(rho, n())

def fetch_x_expt_vals(rho):
    """Return the expectation values of the Pauli x matrix per site."""
    return get_onebody_expt_vals(rho, qt.sigmax())

def fetch_n_spatial_avg(rho):
    """Spatial avg of excitation density"""
    return np.mean(fetch_n_expt_vals(rho))

def fetch_x_spatial_avg(rho):
    return np.mean(fetch_x_expt_vals(rho))

def fetch_y_expt_vals(rho):
    """Return the expectation values of the Pauli y matrix per site."""
    return get_onebody_expt_vals(rho, qt.sigmay())

def get_zbasis_probs(rho):
    if not rho.isoper:
        rho = qt.ket2dm(rho)
    return rho.diag()

def get_bloch_state(n):
    """Returns the state rho for a bloch vector n.
    n = a (3,) numpy array specifying Bloch sphere coordinates """
    n = np.asarray(n)
    nm = np.sum(n**2)
    if (nm-1)>1e-8:
        raise ValueError("Not a valid Bloch state")
    return .5 * (qt.identity(2) + n[0] * qt.sigmax() + n[1] * qt.sigmay() + n[2] * qt.sigmaz())

def get_ghz_state(L, phase=0):
    """ Returns ghz pure state with given phase"""
    a = np.asarray([0, 1]*(L//2) + [0]*(L%2))
    b=1-a
    psi1 = qt.tensor([qt.basis(2, x) for x in a])
    psi2 = qt.tensor([qt.basis(2, x) for x in b])
    
    return (1.0/np.sqrt(2)) * (psi1 + np.exp(1j*phase)*psi2)

# Projectors for synthetic experiments.
def get_pauli(which):
    """Single Pauli operator.
        which = either a pauli string,
            'X' (0), 'Y' (1), 'Z' (2), 'I'
        or a pair of angles theta, phi which define a unit vector n.
        In the latter case, the pauli operator n \cdot \sigma is returned
    """
    if which=='I':
        return qt.identity(2)
    paulis = {'X': qt.sigmax, 'Y': qt.sigmay, 'Z':qt.sigmaz}
    if which in CODES:
        return paulis[PAULIS[which]]()
    if which in PAULIS:
        return paulis[which]()
    try:
        theta, phi = which
        z = np.cos(theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        return x * qt.sigmax() + y * qt.sigmay() + z * qt.sigmaz()
    except TypeError:
        raise ValueError("{0} is not a valid Pauli spec".format(which))


def get_pauli_proj(which, eval):
    """Projector onto the eigenstate of specified pauli operator with specified eigenvalue.
        which: a pauli spec (see get_pauli)
        eval: the eigenvalue, +- 1"""

    if eval not in (1, -1):
        raise ValueError("Not a Pauli eigenvalue")
    return .5*(qt.identity(2) + eval * get_pauli(which))

def get_tensor_pauli(pauli_list):
    """Returns the tensor product of the specified Pauli operators.
        pauli_list: a list of 'X', 'Y', or 'Z' """
    return qt.tensor([get_pauli(which) for which in pauli_list])

def get_tensor_pauli_proj(pauli_list, eval_list):
    """Returns the tensor product of the projectors onto the eigenvalues specified.
        pauli_list: a list of pauli specs. See get_pauli for allowed values.
        eval_list : a list of +/-1, the eigenvalue of each site operator.
         """

    return qt.tensor([get_pauli_proj(op, eval) for op, eval in zip(pauli_list, eval_list)])

def construct_joint_distribution(pauli_list, rho):
    """Construct the full joint distribution over Pauli outcomes.
        pauli_list: a list specifying the pauli to measure on each site
        each entry should be one of the pauli specs accepted
        by get_pauli, namely:
            one of 'X', 'Y', or 'Z'
            one of 0,1,2
            a pair of angles theta, phi
        rho: the qutip state

    Returs: (probs, outcomes)
        probs = (2^L,) array whose kth entry is the probability of observing the kth
    bit pattern.
        outcomes = (2^L, L) array holding the corresponding +-1 bit patterns"""
    try:
        from datagen.tools import generate_binary_space
    except ImportError:
        from tools import generate_binary_space
    L = get_L(rho)
    if len(pauli_list) != L:
        raise ValueError("Expecting one pauli per site, got: %d" % len(pauli_list))
    #array holding all possible measurement outcomes
    #shape (2**L, L), +-1 entries
    outcomes = 2 * generate_binary_space(L)-1
    # array holding the correspdonding probabilities
    probs = np.empty(2**L)

    #careful :O
    for i in range(2**L):
        probs[i] = qt.expect(get_tensor_pauli_proj(pauli_list, outcomes[i, :]), rho)
    return probs, outcomes

#simulating measurements...
def measure_in_pauli_basis(pauli_list, rho, N):
    """Perform a simulated measurement of the tensor-product operator specified by the
    list of pauli strings in pauli_list.

    pauli_list: an L-length list providing a Pauli spec at each site. See
    get_pauli for allowed entries.
    rho: the qutip state
    N: the number of measurements to take
    Returns: an (N, L) array whose rows are the measurement outcomes, +-1. """
    L = get_L(rho)
    if L != len(pauli_list):
        raise ValueError("Expecting one pauli for each qubit")

    #probs and all the possible measurement outcomes in the spec'd basis.
    probs, outcomes = construct_joint_distribution(pauli_list, rho)
    #holds the synthetic measurement data
    measurements = np.empty((N, L))
    #labels of the outcome of each measurement
    outcome_labels = np.random.choice(np.arange(2**L), size=(N,), p=probs)
    for ii in range(N):
        measurements[ii, :] = outcomes[outcome_labels[ii], :]
    return measurements



def measure_single_site_paulis(pauli_list, rho, N, return_probs=False):
    """
    Note: this is computing single-site expectation values only! Not useful for POVM based
    tomography. An earlier version of measure_in_pauli_basis() used this code incorrectly.

    Perform a simulated measurement of the set of specified Pauli operators.
        pauli_list: a list of pauli operators ('X', 'Y', or 'Z'), one for each qubit.
        rho: the state, a qutip QObj.
        N: the number of samples to draw.
        Returns: (N, L) numpy array of samples. The rows are outcomes (+1 or -1) for each site for a particular measurement.
            If return_probs: returns samples,probs, where probs holds probabilities for measurement outcomes, in a (2, L) array.
            The first row is the probabilty of a +1 outcome at each site."""
    L = get_L(rho)
    if L != len(pauli_list):
        raise ValueError("Expecting one pauli for each qubit")
    # +1 projectors for each site
    plus_proj_by_site = [embed([get_pauli_proj(pauli_list[i], +1)],L,[i]) for i in range(L)]
    #+1 prob per measurement per site
    plus_prob_by_site = [qt.expect(proj, rho) for proj in plus_proj_by_site]
    #Pauli measurement distributions per site
    probs_by_site = np.transpose(np.asarray([ [p, 1-p] for p in plus_prob_by_site]))
    outcomes = np.empty((N, L))
    #sample the synthetic data.
    for i in range(L):
        outcomes[:, i] = np.random.choice([+1, -1], size=(N,),p=probs_by_site[:, i])
    if return_probs:
        return outcomes, probs_by_site
    return outcomes

def get_all_bases(L):
    """returns a (3**L, L) array of all possible L-length measurement bases"""
    if L<1:
        raise ValueError
    if L==1:
        return np.asarray([[0,1,2]]).reshape((3,1)).astype(np.int32)
    subbases =get_all_bases(L-1)
    bases = np.empty((3**L,L))
    stp=subbases.shape[0]
    for ii in range(3):
        bases[ii*stp: (ii+1)*stp, 0] = ii
        bases[ii*stp: (ii+1)*stp, 1:] = subbases
    return bases.astype(np.int32)

def get_measurements_from_bases(rho,bases, N_per_basis):
    """ Measure a specified number of times, in each of a specified set of bases.
        bases: (Nbasis, L) integer array, where each row specifies a basis to measure in.
        N_per_basis: integer, number of measurements to draw from each basis.

        returns: settings, samples
        settings: (Nbasis * N_per_basis, L) array holding measurement settings used.
        samples: (Nbasis * N_per_basis, L) array holding measurement outcomes.

        The ith row of settings is a list of integer codes indicating the basis used
        for the ith measurement. The ith row of samples is the list of outcomes from that measurement (+/- 1)"""

    Nbasis, L = bases.shape
    samples = np.empty((N_per_basis *Nbasis, L),dtype=int)
    settings = np.empty_like(samples,dtype=int)
    for i in range(Nbasis):
        i0, i1 = i * N_per_basis, (i+1) * N_per_basis
        settings[i0:i1, ...] = bases[i, :]
        samples[i0:i1, ...] = measure_in_pauli_basis(bases[i,:], rho, N_per_basis)
    return settings, samples

def get_measurements_from_angles(rho, angles, N_per_basis):
    """ Measure a specified number of times, in each of a specified set of bases.
        bases: (Nbasis, L,2) float array
            each row specifies a pair of angles at each physical site.
        N_per_basis: integer, number of measurements to draw from each basis.

        returns: settings, samples
        settings: (Nbasis * N_per_basis, L,2) array holding measurement settings used.
        samples: (Nbasis * N_per_basis, L) array holding measurement outcomes.

        The ith row of settings specifies the angles used in the ith measurement.
        The ith row of samples is the list of outcomes from that measurement (+/- 1)"""

    Nbasis, L, _ = angles.shape
    if angles.shape[2] != 2:
        raise ValueError("Expecting 2 angles per site. Received shape: {0}".format(
                                angles.shape
        ))
    samples = np.empty((N_per_basis *Nbasis, L),dtype=int)
    settings = np.empty((N_per_basis*Nbasis, L, 2), dtype=float)
    for i in range(Nbasis):
        i0, i1 = i * N_per_basis, (i+1) * N_per_basis
        settings[i0:i1, ...] = angles[i, ...]
        pauli_list = [ tuple(angles[i,j,:]) for j in range(L)]
        samples[i0:i1, ...] = measure_in_pauli_basis(pauli_list, rho, N_per_basis)
    return settings, samples

def get_measurements_from_settings(rho, settings, N_per_setting, how='discrete'):
    """ Return measurements from a state using per-site pauli specs
        rho = the state
        settings = array of measurement settings.
            if how == 'discrete':
                integer array, shape (Nsetting, L)
            if how == 'angles':
                float array, shape (Nsetting, L, 2)
        N_per_setting: how many samples to draw per setting.

        Returns: measurement settings and samples drawn, N_per_setting * Nsetting
        in total """
    if how == 'discrete':
        return get_measurements_from_bases(rho, settings, N_per_setting)
    elif how == 'angles':
        return get_measurements_from_angles(rho, settings, N_per_setting)
    raise ValueError("%s not a valid setting type"%how)


def sample_random_angles(shape):
    """Returns (*shape, 2) numpy array of angles which are uniformly distributed
    over the unit sphere.
        Last dimension index 0 --> polar angle theta
        Last dimension index 1 --> azimuthal angle phi
        """
    phi = np.random.uniform(low=0.0, high = 2 * np.pi, size=(*shape,1))
    theta = np.arccos( np.random.uniform(low=-1., high=1., size=(*shape,1)))
    return np.concatenate([theta, phi], axis=-1)

def get_random_local_measurements(rho, N):
    """ Draw N measurements from rho in random local bases.
        rho: a qutip quantum state.
        N: int, number of measurements to draw.

        returns: settings, samples
            settings = (N, L, 2) array of angles used to define the pauli operator
            measured at each site, in each measurement.
            samples = (N, L) integer tensor of outcomes in each case.
            """

    L = get_L(rho)
    # draw a set of random angles
    angles = sample_random_angles((N,L))
    N_per_setting=1
    return get_measurements_from_settings(rho, angles, N_per_setting, how='angles')
