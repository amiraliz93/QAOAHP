"""
Helper functions for the Maximum Cut (MaxCut) problem
"""
from .qaoa_simulator_base import TermsType
import numpy as np
import networkx as nx
from scipy import sparse

def maxcut_obj(x: np.ndarray, w: np.ndarray) -> float:
    """Compute the value of a cut.
    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix returned by get_adjacency_matrix
    Returns:
        float: value of the cut.
    """
    X = np.outer(x, (1 - x))
    return np.sum(w * X)  # type: ignore



def get_maxcut_terms(G: nx.Graph) -> TermsType:
    """Get terms corresponding to cost function value

    .. math::

        S = \\sum_{(i,j,w)\\in G} w*(1-s_i*s_j)/2

    Args:
        G: MaxCut problem graph
    Returns:
        terms to be used in the simulation
    """
    if nx.is_weighted(G):
        terms = [(-float(G[u][v]["weight"]) / 2, (int(u), int(v))) for u, v, *_ in G.edges()]
        total_w = sum([float(G[u][v]["weight"]) for u, v, *_ in G.edges()])

    else:
        terms = [(-(1 / 2), (int(e[0]), int(e[1]))) for e in G.edges()]
        total_w = int(G.number_of_edges())
    N = G.number_of_nodes()
    terms.append((+total_w / 2, tuple()))
    return terms



def get_adjacency_matrix(G: nx.Graph, nodelist=None, dtype=float) -> np.ndarray:
    """
    Return the (dense) adjacency matrix of G as a NumPy array.

    If G is weighted, edge weights are taken from the "weight" attribute;
    otherwise every edge contributes a 1.  Runs in C/SciPy so is far faster
    than iterating in Python.
    """
    # If you know your nodes are 0…n-1 you can omit nodelist entirely.
    if nodelist is None:
        nodelist = list(G.nodes())
    # to_numpy_array calls into SciPy and returns a dense ndarray
    return nx.to_numpy_array(G, nodelist=nodelist, dtype=dtype)