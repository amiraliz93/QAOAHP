import numpy as np
import pandas as pd
from itertools import product
import warnings
import pulp

from .Base.maxcut import get_adjacency_matrix


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


def solve_maxcut_pulp(G):
    prob = pulp.LpProblem("MaxCut", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in G.nodes}
    y = {}

    for i, j in G.edges():
        y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", cat="Binary")

    for i, j in G.edges():
        prob += y[(i, j)] <= x[i]
        prob += y[(i, j)] <= x[j]
        prob += y[(i, j)] >= x[i] + x[j] - 1

    cut = []
    for i, j in G.edges():
        weight = G[i][j].get("weight", 1)
        cut.append(weight * (x[i] + x[j] - 2 * y[(i, j)]))
    prob += pulp.lpSum(cut), "TotalCutWeight"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    cut_value = pulp.value(prob.objective)
    solution = {i: int(pulp.value(x[i])) for i in G.nodes}
    return cut_value, solution


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


def invert_counts(counts):
    """Convert from lsb to msb ordering and vice versa, as fast as possible."""
    items = counts.items
    rev = slice(None, None, -1)
    return {k[rev]: v for k, v in items()}


def expected_maxcut_from_counts(counts, G):
    """
    Compute the expected MaxCut value from shot counts and adjacency matrix.

    Args:
        counts (dict): measurement counts from quantum circuit (bitstring → count)
        w (np.ndarray): adjacency matrix of the graph

    Vectorized MaxCut expectation:
      1) extract upper‑triangle edge list + weights,
      2) for each bitstring, build a bit‐mask via np.frombuffer,
      3) use one C‑speed dot( ) per shot.
    """
    w = get_adjacency_matrix(G)
    n = w.shape[0]

    iu, ju = np.triu_indices(n, k=1)
    w_triu = w[iu, ju]

    total_counts = sum(counts.values())
    total = 0.0

    for bitstr, cnt in counts.items():
        x = np.frombuffer(bitstr.encode("ascii"), dtype=np.uint8) & 1
        cut_val = (x[iu] ^ x[ju]).dot(w_triu)
        total += cut_val * cnt

    return total / total_counts
