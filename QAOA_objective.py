from __future__ import annotations
from calendar import c
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from functools import reduce
import numba.cuda

from Base import choose_simulator, qaoa_simulator_base

from parameter_utils import from_fourier_basis
import parameter_utils
from parameter_utils import QAOAParameterization

#from .qaoa_circuit_portfolio import measure_circuit

from Base.precomputation import precompute_vectorized_cpu_parallel


def _get_qiskit_objective(
     parameterized_circuit,
     precomputed_objectives=None,
    precomputed_optimal_bitstrings=None,
    objective: str = "expectation", 
    terms=None,
    parameterization: str | QAOAParameterization = "theta",
    mixer: str = "x",
    optimization_type="min"):
    N = parameterized_circuit.num_qubits
    if objective == "expectation":
        if  precomputed_objectives is None:
             if terms is None:
                 raise ValueError(f"precomputed_objectives or terms are required when using the {objective} objective")
             else:
                 precomputed_objectives = precompute_vectorized_cpu_parallel(terms, 0.0, N)
    
        def compute_objective_from_probabilities(probabilities):  
             if optimization_type == "max":
                 return -1 * precomputed_objectives.dot(probabilities)

    elif objective == "overlap":
        if precomputed_optimal_bitstrings is None:
            if precomputed_objectives is None:
                if terms is None:
                        raise ValueError(f"precomputed_objectives or terms are required when using the {objective} objective")
                else:
                      precomputed_objectives = precompute_vectorized_cpu_parallel(terms, 0.0, N)
            
            if optimization_type == "max":
                precomputed_objectives = -1 * np.asarray(precomputed_objectives)
                minval = precomputed_objectives.min()
                assert len(bitstring_loc) == 1
                bitstring_loc = bitstring_loc[0]
        else:
              # extract locations of the optimal_bitstrings in 2**N
              bitstring_loc = np.array([reduce(lambda a, b: 2 * a + b, x) for x in precomputed_optimal_bitstrings])
        
        def compute_objective_from_probabilities(probabilities):
             # compute overlap
            overlap = 0
            for i in range(len(bitstring_loc)):
                overlap += probabilities[bitstring_loc[i]]
            return 1 - overlap
    
    else:
        raise ValueError(f"Unknown objective passed to get_qaoa_objective: {objective}, allowed ['expectation', 'overlap']")
    
    if mixer == "x":
        backend = Aer.get_backend("aer_simulator_statevector")

        def g(gamma, beta):
            qc = parameterized_circuit.assign_parameters(list(np.hstack([beta, gamma])))
            sv = np.asarray(backend.run(qc).result().get_statevector())
            probs = np.abs(sv) ** 2
            return compute_objective_from_probabilities(probs)
        
    else:
        raise ValueError(f"Unknown mixer type passed to get_qaoa_objective: {mixer}, allowed ['x']")
    return g    




def get_qaoa_objective(
    N: int,
    precomputed_diagonal_hamiltonian=None, # not define
    precomputed_costs=None,
    terms=None, # we define this terms
    precomputed_optimal_bitstrings=None,
    parameterization: str | QAOAParameterization = "theta",
    objective: str = "expectation",
    parameterized_circuit=None,
    simulator: str = "auto", # we define this parameter
    mixer: str = "x",
    initial_state: np.ndarray | None = None,
    n_trotters: int = 1,
    optimization_type="min",
) -> typing.Callable:

    """Return QAOA objective to be minimized
 Parameters
    ----------
    N : int
        Number of qubits
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    parameterization : str
        If parameterization == 'theta', then f takes one parameter (gamma and beta concatenated)
        If parameterization == 'gamma beta', then f takes two parameters (gamma and beta)
        For below Fourier parameters, q=p
        If parameterization == 'freq', then f takes one parameter (fourier parameters u and v concatenated)
        If parameterization == 'u v', then f takes two parameters (fourier parameters u and v)
    objective : str
        If objective == 'expectation', then returns f(theta) = - < theta | C_{LABS} | theta > (minus for minimization)
        If objective == 'overlap', then returns f(theta) = 1 - Overlap |<theta|optimal_bitstring>|^2 (1-overlap for minimization)
    simulator : str
        If simulator == 'auto', implementation is chosen automatically
            (either the fastest CPU simulator or a GPU simulator if CUDA is available)
        If simulator == 'qiskit', implementation in qaoa_qiskit is used
    mixer : str
        If mixer == 'x', then uses the default Pauli X as the mixer
        If mixer == 'xy', then uses the ring-XY as the mixer
    initial_state : np.ndarray
        The initial state for QAOA, default is the uniform superposition state (corresponding to the X mixer)
    n_trotters : int
        Number of Trotter steps in each mixer layer for the xy mixer

    Returns
    -------
    f : callable
        Function returning the negative of expected value of QAOA with parameters theta
"""

# -- Qiskit edge case
    if simulator == "qiskit":

        if precomputed_costs is None:
            precomputed_costs = precomputed_diagonal_hamiltonian
            assert precomputed_costs is not None, 'the precomputed_costs still None {precomputed_costs}'
        g = _get_qiskit_objective(
                parameterized_circuit,
                precomputed_costs,
                precomputed_optimal_bitstrings,
                objective,
                terms,
                parameterization,
                mixer,
                optimization_type=optimization_type,
            )

        def fq(*args):
            gamma, beta = parameter_utils.convert_to_gamma_beta(*args, parameterization=parameterization)
            return g(gamma, beta)

        return fq
    # --------------
    if mixer == "x":
        simulator_cls = choose_simulator(name=simulator)
    
    else:
        raise ValueError(f"Unknown mixer type passed to get_qaoa_objective: {mixer}, allowed ['x', 'xy']")

    sim = simulator_cls(N, terms=terms, costs=precomputed_diagonal_hamiltonian)
    if precomputed_costs is None:
        precomputed_costs = sim.get_cost_diagonal()

    bitstring_loc = None

    if precomputed_optimal_bitstrings is not None and objective != "expectation":
        bitstring_loc = np.array([reduce(lambda a, b: 2 * a + b, x) for x in precomputed_optimal_bitstrings])

    # -- Final function
    # 
    def f(*args):
        gamma, beta = parameter_utils.convert_to_gamma_beta(*args, parameterization=parameterization)

        result = sim.simulate_qaoa(gamma, beta, initial_state, n_trotters=n_trotters)
        if objective == "expectation":
            return sim.get_expectation(result, costs=precomputed_costs, preserve_state=False, optimization_type=optimization_type)

        elif objective == "overlap":
            overlap = sim.get_overlap(result, costs=precomputed_costs, indices=bitstring_loc, preserve_state=False, optimization_type=optimization_type)
            return 1 - overlap
    
    return f


