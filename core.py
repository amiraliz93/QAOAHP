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
    
    mask1 = (1<< q) -1 
    mask2 = mask1 ^ (( n_states - 1) >>1)

x= [0,1,0,1,1]
q = 2
n_states = len(x)
n_group = n_states //2
mask1 = (1<< q) -1 # shift the 1 to left by q bits like (001) will be (100) when q is 2
# the subtract will be subtract 1 from result: (100) will be (011)
# will be a binary number with the lowest q bits set to 1
mask2 = mask1 ^ (( n_states - 1) >>1)
print(n_group) 