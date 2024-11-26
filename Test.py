import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from core_function.core import furx_all, furx


# Create a random graph
def create_random_graph(n_nodes, edge_probability, seed=None):
     """
    Creates a random graph with the specified number of nodes and edge probability.

    Args:
        n_nodes (int): Number of nodes in the graph.
        edge_probability (float): Probability of edge creation.

    Returns:
        G: A networkx graph object.
    """
     G = nx.erdos_renyi_graph(n_nodes, edge_probability, seed)
     return G

def generate_initial_statvectore(n_qubits):
     """
    Generates an initial uniform superposition state vector for n_qubits.

    Args:
        n_qubits (int): Number of qubits (nodes in the graph).

    Returns:
        np.ndarray: State vector in numpy array format.
    """
     n_states = 2 ** n_qubits
     initial_state_vectore = np.ones(n_states, dtype= "complex") / np.sqrt(n_states) 
     return initial_state_vectore


n_nodes = 3  # Number of nodes (qubits)
edge_probability = 0.6 

# graph 
G = create_random_graph(n_nodes, edge_probability, seed=100)
print(f"Random Graph:\nNodes: {G.nodes()}\nEdges: {G.edges()}")
#nx.draw(G, with_labels=True)
#plt.show()

# Generate the state vector
state_vector = generate_initial_statvectore(n_nodes)
print(f" he statvectore is: {state_vector} ")

theta = math.pi /4 
stat_vectore_update = furx(state_vector, theta, n_nodes)
print(stat_vectore_update)
