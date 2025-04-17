import numpy as np
import pandas as pd
from itertools import product
# calculate maxcut solution with exact solution Mixed Integer Linear Program (MILP) 
import pulp


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

#Mixed Integer Linear Program
# maximize sigma (w_ij(x_i XOR x_j))
def solve_maxcut_pulp(G):
    prob = pulp.LpProblem("MaxCut", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in G.nodes}
    y = {}  # auxiliary variables for edge products

    for i, j in G.edges():
        y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", cat="Binary")

    # Add constraints for linearization: y_ij = x_i * x_j
    for i, j in G.edges():
        prob += y[(i, j)] <= x[i]
        prob += y[(i, j)] <= x[j]
        prob += y[(i, j)] >= x[i] + x[j] - 1

    # Objective: maximize cut edges
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
    if minimize:
        best_cost_brute = float("inf")
        compare = lambda x, y: x < y
    else:
        best_cost_brute = float("-inf")
        compare = lambda x, y: x > y
    bit_strings = (((np.array(range(2**num_variables))[:, None] & (1 << np.arange(num_variables)))) > 0).astype(int)
    for x in bit_strings:
        if function_takes == "spins":
            cost = obj_f(1 - 2 * np.array(x), *args, **kwargs)
        elif function_takes == "bits":
            cost = obj_f(np.array(x), *args, **kwargs)
        if compare(cost, best_cost_brute):
            best_cost_brute = cost
            xbest_brute = x
    return best_cost_brute, xbest_brute