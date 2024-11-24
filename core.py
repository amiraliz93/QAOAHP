import math
import numpy as np

########################################
# single-qubit X rotation
########################################

def furx(x: np.ndarray, theta: float, q: int) -> np.ndarray:
    """
        Applies e^{-i theta X} on qubit indexed by q
    Args:
        x (np.ndarray): _description_
        theta (float): _description_
        q (int): _description_
    """
    n_states = len(x)
    n_group = n_states //2 
    """masking means extract pices of info from larger one withou affective actual information
    masking act of mask to the value by using bitwise like AND OR XOR
    ANDing: extract a subset of the bits in the value (can use as test to see if the user permission to do sth)
    ORing: set a subset of the bits in the value
    XORing: Toggle a subset of the bits in the value   
    """
    for qubits in range(0, q +1):
        mask1 = (1 << qubits) -1 
        mask2 = mask1 ^ (( n_states - 1) >>1)

        wa = math.cos(theta)
        wb = 1j *math.sin(theta)

        for i in range(n_group):
            ia = (i & mask1) | ((i & mask2) << 1)
            ib = ia | (1 << qubits)
            print(f" the pairs for q ={qubits} the pairs need to be update \t {ia} and {ib} \n")
            x[ia], x[ib] = wa * x[ia] + wb * x[ib], wb * x[ia] + wa * x[ib]
 # shift the 1 to left by q bits like (001) will be (100) when q is 2
# the subtract will be subtract 1 from result: (100) will be (011)
# will be a binary number with the lowest q bits set to 1
# mask2 = mask1 ^ (( n_states - 1) >>1) # XOR operator

    return x


x = [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111]
a = furx(x, 1.0, 2 )

def furx_all(x:np.array, theta:float, np_qubits: int) -> np.ndarray:

    #applies e^{-i theta X} (furx function) on all qubits
    for i in range(np_qubits):
        furx(x, theta, i)

def get_complex_array(sv: np.ndarray) -> np.adarray:
    """create complex Numpy array from a numpy array or return the object as is if 
    its already a complex numpy array

    Args:
        sv (np.ndarray): statvector 
    """
    if not sv.dtype == 'complex':
        sv = sv.astype("complex")
    return sv



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

