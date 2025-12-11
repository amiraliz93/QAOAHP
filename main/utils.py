import numpy as np
import pandas as pd
from itertools import product
import warnings



def reverse_array_index_bit_order(arr):
    arr = np.array(arr)
    n = int(np.log2(len(arr)))  # Calculate the value of N
    if n % 1:
        raise ValueError("Input vector has to have length 2**N where N is integer")

    index_arr = np.arange(len(arr))
    new_index_arr = np.zeros_like(index_arr)
    while n > 0:
        last_8 = np.unpackbits(index_arr.astype(np.uint8), axis=0, bitorder="little")
        repacked_first_8 = np.packbits(last_8).astype(np.int64)
        if n < 8:
            new_index_arr += repacked_first_8 >> (8 - n)
        else:
            new_index_arr += repacked_first_8 << (n - 8)
        index_arr = index_arr >> 8
        n -= 8
    return arr[new_index_arr]


def precompute_energies(obj_f, nbits: int, *args: object, **kwargs: object):
    """
    Precomputed a vector of objective function values
    that accelerates the energy computation in obj_from_statevector

    For LABS-specific, accelerated version see get_precomputed_labs_merit_factors in qaoa_objective_labs.py


    Parameters
    ----------
    obj_f : callable
        Objective function to precompute
    nbits : int
        Number of parameters obj_f takes
    num_processes : int
        Number of processes to use. Default: 1 (serial)
        if num_processes > 1, pathos.Pool is used
    *args, **kwargs : Objec
        Parameters to be passed directly to obj_f

    Returns
    -------
    energies : np.array
        vector of energies such that E = energies.dot(amplitudes)
        where amplitudes are absolute values squared of qiskit statevector

    """
    bit_strings = (((np.array(range(2**nbits))[:, None] & (1 << np.arange(nbits)))) > 0).astype(int)

    return np.array([obj_f(x, *args, **kwargs) for x in bit_strings])


def brute_force(obj_f, num_variables: int, minimize: bool = False, function_takes: str = "spins", *args: object, **kwargs: object):
    """Get the maximum of a function by complete enumeration
    Returns the maximum value and the extremizing bit string
    """
    if num_variables > 23:
        warnings.warn(
            f"Brute force with {num_variables} variables requires evaluating "
            f"{2**num_variables} configurations. This may take a very long time!"
        )    
    
    if minimize:
        best_value = float("inf")
        is_better = lambda x, y: x < y
    else:
        best_value = float("-inf")
        is_better = lambda x, y: x > y
    indices = np.arange(2**num_variables)
    bit_strings = ((indices[:, None] & (1 << np.arange(num_variables))) > 0).astype(int)
    best_bitstring = None
    
    # Evaluate objective for each configuration
    for bitstring in bit_strings:
        if function_takes == "spins":
            # Convert 0/1 to -1/+1
            config = 1 - 2 * bitstring
        elif function_takes == "bits":
            config = bitstring
        else:
            raise ValueError(f"function_takes must be 'spins' or 'bits', got '{function_takes}'")
        
        value = obj_f(config, *args, **kwargs)
        
        if is_better(value, best_value):
            best_value = value
            best_bitstring = bitstring
    
    return best_value, best_bitstring