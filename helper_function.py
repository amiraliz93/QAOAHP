import numpy as np
import networkx as nx
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

def get_maxcut_terms(G: nx.Graph): #-> TermsType:
    """Get terms corresponding to cost function value

    .. math::

        S = \\sum_{(i,j,w)\\in G} w*(1-s_i*s_j)/2

    Args:
        G: MaxCut problem graph
    Returns:
        terms to be used in the simulation
    """