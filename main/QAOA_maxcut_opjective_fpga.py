from precomputation import precompute_vectorized_cpu_parallel
import numpy as np
from maxcut_utils import maxcut_obj, get_maxcut_terms, adjacency_matrix

from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from functools import reduce
import numba.cuda

# qiskit import 
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from .parameters_utils import QAOAParameterization 
from .Base import parameters_utils
from .Base import choose_simulator


def get_qaoa_objective(
    N: int,
    precomputed_diagonal_hamiltonian=None, 
    precomputed_costs=None,
    terms=None,
    precomputed_optimal_bitstrings=None,
    parameterization: str | QAOAParameterization = "theta",
    objective: str = "expectation",
    parameterized_circuit=None,
    simulator: str = "auto",
    initial_state: np.ndarray | None = None,
    n_trotters: int = 1,
    optimization_type="min",
):
    """Return QAOA objective to be minimized
    [documentation as before]
    """
    
    # Handle Qiskit special case
    if simulator == "qiskit":
        return _create_qiskit_objective(
            N, precomputed_diagonal_hamiltonian, precomputed_costs,
            terms, precomputed_optimal_bitstrings, parameterization,
            objective, parameterized_circuit, optimization_type
        )
    
    # Get appropriate simulator implementation
    simulator_implementation = _get_simulator_implementation(simulator)
    
    # Initialize simulator with problem parameters
    qaoa_sim = simulator_implementation(
        n_qubits = N,
        terms = terms,
        costs = precomputed_diagonal_hamiltonian
        )
    
    # Prepare cost array for calculations
    # its a array to feed to fpga as precomput costs
    energy_landscape = _prepare_energy_landscape(qaoa_sim, precomputed_costs)
    
    # Process optimal bitstring indices if needed
    optimal_indices = None
    if precomputed_optimal_bitstrings is not None and objective != "expectation":
        optimal_indices = _convert_bitstrings_to_indices(precomputed_optimal_bitstrings)
    
    # finally Create and return the objective function
    return _build_objective_function(
        qaoa_sim, energy_landscape, parameterization, objective,
        initial_state, n_trotters, optimization_type, optimal_indices
    )



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

def _prepare_energy_landscape(simulator, precomputed_costs):
    """Prepare the energy landscape array"""
    if precomputed_costs is not None:
        return precomputed_costs
    return simulator.get_cost_diagonal()

def _convert_bitstrings_to_indices(bitstrings):
    """Convert bitstring arrays to decimal indices"""
    return np.array([sum(bit << i for i, bit in enumerate(reversed(bitstring))) 
                    for bitstring in bitstrings])

def _build_objective_function(simulator, costs, parameterization, objective_type,
                             initial_state, trotter_steps, opt_type, optimal_indices):
    """Build the QAOA objective function"""
    
    def objective_function(*parameters):
        # Convert parameters to standard gamma/beta format
        angle_gamma, angle_beta = parameters_utils.convert_to_gamma_beta(
            *parameters, parameterization=parameterization
        )
        
        # Run simulation to get final quantum state
        final_state = simulator.simulate_qaoa(
            gamma_angles=angle_gamma, 
            beta_angles=angle_beta, 
            starting_state=initial_state, 
            n_trotters=trotter_steps
        )
        
        # Calculate appropriate metric based on objective type
        if objective_type == "expectation":
            # Calculate energy expectation value
            return _calculate_expectation(
                simulator, final_state, costs, opt_type
            )
        elif objective_type == "overlap":
            # Calculate overlap with target states
            return _calculate_overlap_objective(
                simulator, final_state, costs, optimal_indices, opt_type
            )
        else:
            raise ValueError(f"Unsupported objective type: {objective_type}")
    
    return objective_function

def _calculate_expectation(simulator, state, costs, optimization_type):
    """Calculate energy expectation value"""
    return simulator.get_expectation(
        state=state,
        costs=costs,
        preserve_state=False,
        optimization_type=optimization_type
    )

def _calculate_overlap_objective(simulator, state, costs, indices, optimization_type):
    """Calculate overlap-based objective"""
    overlap_value = simulator.get_overlap(
        state=state,
        costs=costs,
        indices=indices,
        preserve_state=False,
        optimization_type=optimization_type
    )
    # Return 1-overlap for minimization
    return 1.0 - overlap_value