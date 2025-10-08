from __future__ import annotations
from calendar import c
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from functools import reduce
import numba.cuda

from .Base import choose_simulator, qaoa_simulator_base
from .parameter_utils import from_fourier_basis, QAOAParameterization

#from .qaoa_circuit_portfolio import measure_circuit

from .Base.precomputation import precompute_vectorized_cpu_parallel


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
             else:
                 return precomputed_objectives.dot(probabilities)

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

def _create_qiskit_objective(
    N, precomputed_diagonal_hamiltonian, precomputed_costs,
    terms, precomputed_optimal_bitstrings, parameterization,
    objective, parameterized_circuit, optimization_type
):
    """Create a QAOA objective function using Qiskit as the simulator backend
    
    Args:
        N: Number of qubits
        precomputed_diagonal_hamiltonian: Precomputed diagonal Hamiltonian
        precomputed_costs: Precomputed cost array
        terms: Hamiltonian terms
        precomputed_optimal_bitstrings: Optimal bitstrings for overlap objective
        parameterization: Parameter format for QAOA angles
        objective: Type of objective function ("expectation" or "overlap")
        parameterized_circuit: Optional pre-built circuit
        optimization_type: Whether to minimize or maximize
        
    Returns:
        Callable objective function for optimization
    """
    # Set up simulator
    simulator = AerSimulator(method='statevector')
    
    # Get or create parameterized circuit
    if parameterized_circuit is None:
        from qaoa_circuit import get_parameterized_qaoa_circuit
        from create_ciecuit import get_parameterized_qaoa_circuit_from_terms
        
        # For MaxCut problems with terms
        if terms is not None:
            circuit_creator = lambda p: get_parameterized_qaoa_circuit_from_terms(
                N=N, 
                terms=terms, 
                p=p, 
                save_statevector=False
            )
        else:
            # Generic QAOA
            circuit_creator = lambda p: get_parameterized_qaoa_circuit(
                p=p,
                N=N,
                save_statevector=False
            )
    else:
        # Use provided circuit
        circuit_creator = lambda p: parameterized_circuit
        
    # Prepare cost array for calculations
    if precomputed_costs is None and precomputed_diagonal_hamiltonian is not None:
        costs = precomputed_diagonal_hamiltonian
    elif precomputed_costs is not None:
        costs = precomputed_costs
    elif terms is not None:
        costs = precompute_vectorized_cpu_parallel(terms, 0.0, N)
    else:
        raise ValueError("Either terms or precomputed costs must be provided")
    
    # Process optimal bitstring indices if needed
    optimal_indices = None
    if precomputed_optimal_bitstrings is not None and objective == "overlap":
        optimal_indices = _convert_bitstrings_to_indices(precomputed_optimal_bitstrings)
        
    def objective_function(*params):
        # Convert parameters to gamma/beta format
        gamma, beta = parameters_utils.convert_to_gamma_beta(
            *params, parameterization=parameterization
        )
        
        # Get parameterized circuit with appropriate depth
        p = len(gamma)
        qc = circuit_creator(p)
        
        # Bind the parameters to the circuit
        parameter_values = []
        for g, b in zip(gamma, beta):
            parameter_values.extend([g, b])
            
        # Transpile circuit for the simulator
        bound_qc = qc.bind_parameters(parameter_values)
        transpiled_qc = transpile(bound_qc, simulator)
        
        # Run the circuit and get statevector
        job = simulator.run(transpiled_qc)
        result = job.result()
        statevector = result.get_statevector()
        
        # Calculate objective value
        if objective == "expectation":
            # Calculate expectation value <ψ|H|ψ>
            probabilities = np.abs(statevector.data)**2
            expectation = costs.dot(probabilities)
            
            # Adjust sign based on optimization direction
            return -expectation if optimization_type == "max" else expectation
        
        elif objective == "overlap":
            # Calculate overlap with optimal states
            probabilities = np.abs(statevector.data)**2
            overlap = sum(probabilities[idx] for idx in optimal_indices)
            
            # Return 1-overlap for minimization
            return 1.0 - overlap
        
        else:
            raise ValueError(f"Unknown objective type: {objective}")
    
    return objective_function

def _get_simulator_implementation(simulator_name):
    """Get the appropriate simulator implementation class"""
    if simulator_name in ["fpga", "auto"]:
        return choose_simulator(name=simulator_name)
    else:
        raise ValueError(f"Unsupported simulator: {simulator_name}. Use 'fpga' or 'auto'.")


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
        """
        if precomputed_costs is None:
            precomputed_costs = precomputed_diagonal_hamiltonian
           # assert precomputed_costs is not None, f'the precomputed_costs still None {precomputed_costs}'
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

        #def fq(*args):
            gamma, beta = parameter_utils.convert_to_gamma_beta(*args, parameterization=parameterization)
            return g(gamma, beta)

        return fq
        """
        return _create_qiskit_objective(
            N, precomputed_diagonal_hamiltonian, precomputed_costs,
            terms, precomputed_optimal_bitstrings, parameterization,
            objective, parameterized_circuit, optimization_type
        )
    # --------------
    if mixer == "x": # for x mixer 
        simulator_cls = choose_simulator(name=simulator)
    
    else:
        raise ValueError(f"Unknown mixer type passed to get_qaoa_objective: {mixer}, allowed ['x', 'xy']")

    # Get appropriate simulator implementation
    simulator_implementation = _get_simulator_implementation(simulator)

    sim = simulator_implementation(N, terms=terms, costs=precomputed_diagonal_hamiltonian)
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
        else:
            raise ValueError(f"Unknown objective: {objective}")

    return f


