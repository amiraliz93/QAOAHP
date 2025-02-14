import math 
import numpy as np

def furx(x: np.ndarray, theta: float, q: int) -> np.ndarray:
    """
    Applies e^{-i theta X} on qubit indexed by q
    x: representing the state vector of the quantum system.
    theta: angle for operation x
    q: This is an integer index representing the qubit on which the X rotation gate is applied.
    """
    n_states = len(x) # length of state has 2^n
    n_groups = n_states // 2 

    mask1 = (1 << q) - 1 # determine which location of q bit
    mask2 = mask1 ^ ((n_states - 1) >> 1)

    wa = math.cos(theta)
    wb = -1j * math.sin(theta)

    for i in range(n_groups):
        ia = (i & mask1) | ((i & mask2) << 1)
        ib = ia | (1 << q)
        x[ia], x[ib] = wa * x[ia] + wb * x[ib], wb * x[ia] + wa * x[ib]

        
 # shift the 1 to left by q bits like (001) will be (100) when q is 2
# the subtract will be subtract 1 from result: (100) will be (011)
# will be a binary number with the lowest q bits set to 1
# mask2 = mask1 ^ (( n_states - 1) >>1) # XOR operator

def furx_all(x: np.ndarray, theta: float, n_qubits: int) -> np.ndarray:
    """
    Applies e^{-i theta X} on all qubits
    """
    for i in range(n_qubits):
        furx(x, theta, i)
    return x