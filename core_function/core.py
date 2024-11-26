import math
import numpy as np
from collections.abc import Sequence
########################################
# single-qubit X rotation
########################################

def furx(x: np.ndarray, theta: float, q: int) -> np.ndarray:
    """
        Applies e^{-i theta X} on qubit indexed by q
    Args:
        x (np.ndarray): 
        theta (float): 
        q (int): 
    """
    n_states = len(x)
    n_group = n_states //2 
    """masking means extract pices of info from larger one withou affective actual information
    masking act of mask to the value by using bitwise like AND OR XOR
    ANDing: extract a subset of the bits in the value (can use as test to see if the user permission to do sth)
    ORing: set a subset of the bits in the value
    XORing: Toggle a subset of the bits in the value   
    """
    mask1 = (1 << q) - 1 
    mask2 = mask1 ^ ((n_states - 1) >> 1)

    wa = math.cos(theta)
    wb = 1j *math.sin(theta)

    for i in range(n_group):
        ia = (i & mask1) | ((i & mask2) << 1)
        ib = ia | (1 << (q -1 ))
        if ib < n_states:  # Ensure ib is within bounds
    
            x[ia], x[ib] = wa * x[ia] + wb * x[ib], wb * x[ia] + wa * x[ib]
 # shift the 1 to left by q bits like (001) will be (100) when q is 2
# the subtract will be subtract 1 from result: (100) will be (011)
# will be a binary number with the lowest q bits set to 1
# mask2 = mask1 ^ (( n_states - 1) >>1) # XOR operator
        else:
            print(" ib is bigger than n_state")
    return x



########################################
# single-qubit X rotation on all qubits
########################################
def furx_all(x:np.array, theta:float, np_qubits: int) -> np.ndarray:

    #applies e^{-i theta X} (furx function) on all qubits
    for i in range(np_qubits):
        furx(x, theta, i)


def get_complex_array(sv: np.ndarray):
    """create complex Numpy array from a numpy array or return the object as is if 
    its already a complex numpy array

    Args:
        sv (np.ndarray): statvector 
    """
    if not sv.dtype == 'complex':
        sv = sv.astype("complex")
    return sv


#improve the # single-qubit X rotation
def X_rotation_on_gate(sv: np.array, theta: float, q: int) -> np.ndarray:
    """a single qubit pualix_rotatio
        Rx(theta) = exp(-i * theta * X/2)
        where X is  the Pauli-X operator
        The operation will be in-place if the input is a numpy.ndarray of complex
    data type. Otherwise the input will be copied to a new numpy.ndarray.
    Args:
        sv (np.array): statevector on which the rotation is applied
        theta (float): rotation angel
        q (int): qubit index to apply the rotation

    Returns:
        np.ndarray: statvector after rotation
    """
    sv = get_complex_array(sv)
    furx(sv, 0.5 * theta, q)
    return sv

#finally apply qaoa curcate  
def apply_qaoa_furx(sv, gammas: Sequence[float], betas: Sequence[float], hc_diagonal: np.ndarray, n_qubits:int ):
    """
    apply a QAOA with the X mixer defined by
    U(beta) = sum_{j} exp(-i*beta*X_j/2)
    where X_j is the Pauli-X operator applied on the jth qubit.
    @param sv array NumPy array (dtype=complex) of length n containing the statevector
    @param gammas parameters for the phase separating layers
    @param betas parameters for the mixing layers
    @param hc_diag array of length n containing diagonal elements of the diagonal cost Hamiltonian
    @param n_qubits total number of qubits represented by the statevector
    """
    for gamma, beta in zip(gammas, betas):
        sv *= np.exp(-0.5 * gamma * hc_diagonal)
        furx_all(sv, beta, n_qubits)


