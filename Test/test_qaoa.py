import numpy as np
import networkx as nx
import numba.cuda
from QAOA_Objective_max_cut import get_qaoa_maxcut_objective

def test_qaoa_simulation():
    # Create a simple graph for MaxCut
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (2,3), (3,0)])  # 4-node ring graph

    # Set up QAOA parameters
    N = G.number_of_nodes()  # number of qubits
    p = 1  # number of QAOA layers

    # Get the QAOA objective function
    objective = get_qaoa_maxcut_objective(
        N=N,
        p=p,
        G=G,
        parameterization="theta",
        objective="expectation",
        simulator="auto"  # will automatically choose between CPU and GPU
    )

    # Test with some example parameters
    theta = np.array([0.5, 0.5])  # example parameters (gamma, beta)
    result = objective(theta)
    print(f"QAOA objective value: {result}")

    # Test with different parameters
    theta2 = np.array([0.3, 0.7])
    result2 = objective(theta2)
    print(f"QAOA objective value with different parameters: {result2}")

    # Optional: Test with GPU simulator explicitly
    if numba.cuda.is_available():
        print("\nTesting with GPU simulator:")
        objective_gpu = get_qaoa_maxcut_objective(
            N=N,
            p=p,
            G=G,
            parameterization="theta",
            objective="expectation",
            simulator="gpu"
        )
        result_gpu = objective_gpu(theta)
        print(f"GPU QAOA objective value: {result_gpu}")

if __name__ == "__main__":
    test_qaoa_simulation() 