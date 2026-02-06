"""
Generic QAOA circuit creation utilities.

This module provides tools to build QAOA circuits from Hamiltonian terms,
independent of the specific problem (MaxCut, LABS, TSP, etc.).

For problem-specific circuit creation (e.g., MaxCut), see the respective
problem module (e.g., maxcut.py).
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from typing import Sequence


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

def _append_z_prod_term(qc: QuantumCircuit, term: Sequence[int], gamma: float) -> None:
    """Apply a multi-qubit Pauli-Z product term to the circuit.
    
    Implements exp(-i * gamma * Z_i1 * Z_i2 * ... * Z_in) where
    the indices are specified in 'term'.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to modify
    term : Sequence[int]
        Qubit indices for Z operators
    gamma : float
        Rotation angle
        
    Notes
    -----
    - For 2-qubit terms: uses native RZZ gate
    - For 4-qubit terms: uses optimized decomposition
    - For other sizes: uses general CNOT ladder
    """
    n_qubits = len(term)
    
    if n_qubits == 0:
        # Constant term - no operation needed
        return
    elif n_qubits == 1:
        # Single qubit Z rotation
        qc.rz(2 * gamma, term[0])
    elif n_qubits == 2:
        # Two-qubit ZZ gate (native on some hardware)
        qc.rzz(2 * gamma, term[0], term[1])
    elif n_qubits == 4:
        # Optimized 4-qubit term (for LABS problem)
        # Assumes term is ordered
        qc.cx(term[0], term[1])
        qc.cx(term[3], term[2])
        qc.rzz(4 * gamma, term[1], term[2])
        qc.cx(term[3], term[2])
        qc.cx(term[0], term[1])
    else:
        # General n-qubit term using CNOT ladder
        target = term[-1]
        for control in term[:-1]:
            qc.cx(control, target)
        qc.rz(2 * gamma, target)
        for control in reversed(term[:-1]):
            qc.cx(control, target)


def _append_x_term(qc: QuantumCircuit, qubit: int, beta: float) -> None:
    """Apply an X rotation to a single qubit.
    
    Implements exp(-i * beta * X)
    
    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to modify
    qubit : int
        Qubit index
    beta : float
        Rotation angle
    """
    qc.rx(2 * beta, qubit)


def append_cost_operator_circuit(
    qc: QuantumCircuit, 
    terms: Sequence, 
    gamma: float
) -> None:
    """Apply the cost Hamiltonian operator to the circuit.
    
    Implements exp(-i * gamma * H_C) where H_C = sum of weighted Pauli-Z products.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to modify
    terms : Sequence
        Hamiltonian terms in one of two formats:
        1. Unweighted: [(q1, q2), (q1, q2, q3), ...]
        2. Weighted: [(coeff, (q1, q2)), (coeff, (q1, q2, q3)), ...]
    gamma : float
        Cost layer parameter
        
    Examples
    --------
    >>> qc = QuantumCircuit(3)
    >>> terms = [(0.5, (0, 1)), (0.3, (1, 2))]
    >>> append_cost_operator_circuit(qc, terms, gamma=0.1)
    """
    for term in terms:
        if len(term) == 2 and isinstance(term[1], tuple):
            # Weighted term: (coefficient, (qubits))
            coeff, qubits = term
            _append_z_prod_term(qc, qubits, gamma * coeff / 2)
        elif any(isinstance(i, tuple) for i in term):
            raise ValueError(f"Invalid term format: {term}")
        else:
            # Unweighted term: (qubits)
            _append_z_prod_term(qc, term, gamma)


def append_mixer_operator_circuit(qc: QuantumCircuit, beta: float) -> None:
    """Apply the standard X-mixer operator to all qubits.
    
    Implements exp(-i * beta * sum_i X_i)
    
    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to modify
    beta : float
        Mixer layer parameter
    """
    for qubit_idx in range(len(qc.qubits)):
        _append_x_term(qc, qubit_idx, beta)


# ============================================================================
# CIRCUIT CREATION - WITH SPECIFIC ANGLES
# ============================================================================

def get_qaoa_circuit_from_terms(
    N: int,
    terms: Sequence,
    gammas: Sequence[float],
    betas: Sequence[float],
    save_statevector: bool = True,
    qr: QuantumRegister | None = None,
    cr: ClassicalRegister | None =  None
) -> QuantumCircuit:
    """Create a QAOA circuit from Hamiltonian terms with specific angles.
    
    Parameters
    ----------
    N : int
        Number of qubits
    terms : Sequence
        Hamiltonian terms (see append_cost_operator_circuit for format)
    gammas : Sequence[float]
        Cost layer parameters (length p)
    betas : Sequence[float]
        Mixer layer parameters (length p)
    save_statevector : bool, default True
        Save final statevector for simulation
    qr : QuantumRegister, optional
        Custom quantum register
    cr : ClassicalRegister, optional
        Custom classical register for measurements
        
    Returns
    -------
    QuantumCircuit
        Complete QAOA circuit
        
    Examples
    --------
    >>> terms = [(1.0, (0, 1)), (0.5, (1, 2))]
    >>> gammas = [0.5, 0.3]
    >>> betas = [0.2, 0.4]
    >>> qc = get_qaoa_circuit_from_terms(3, terms, gammas, betas)
    """
    if len(gammas) != len(betas):
        raise ValueError(f"gamma and beta must have same length, got {len(gammas)} and {len(betas)}")
    
    p = len(gammas)
    
    # Create registers
    if qr is None:
        qr = QuantumRegister(N, 'q')
    elif qr.size < N:
        raise ValueError(f"Provided register has {qr.size} qubits, need at least {N}")
    
    qc = QuantumCircuit(qr, cr) if cr else QuantumCircuit(qr)
    
    # Initial state: uniform superposition
    qc.h(range(N))
    
    # Apply p QAOA layers
    for gamma, beta in zip(gammas, betas):
        append_cost_operator_circuit(qc, terms, gamma)
        append_mixer_operator_circuit(qc, beta)
    
    if save_statevector:
        qc.save_statevector()
    
    return qc


# ============================================================================
# CIRCUIT CREATION - PARAMETERIZED
# ============================================================================

def get_parameterized_qaoa_circuit_from_terms(
    N: int,
    terms: Sequence,
    p: int,
    save_statevector: bool = True,
    qr: QuantumRegister | None = None,
    cr: ClassicalRegister | None = None,
    return_parameter_vectors: bool = False,
):
    """Create a parameterized QAOA circuit from Hamiltonian terms.
    
    This creates a circuit with ParameterVector objects for gamma and beta,
    allowing efficient parameter binding for optimization.
    
    Parameters
    ----------
    N : int
        Number of qubits
    terms : Sequence
        Hamiltonian terms (see append_cost_operator_circuit for format)
    p : int
        Number of QAOA layers (will have 2*p parameters total)
    save_statevector : bool, default True
        Save final statevector for simulation
    qr : QuantumRegister, optional
        Custom quantum register
    cr : ClassicalRegister, optional
        Custom classical register for measurements
    return_parameter_vectors : bool, default False
        If True, return (circuit, betas, gammas)
        
    Returns
    -------
    QuantumCircuit or tuple
        Parameterized QAOA circuit
        If return_parameter_vectors=True, returns (qc, betas, gammas)
        
    Notes
    -----
    Parameters are ordered alphabetically: beta first, then gamma.
    To bind: qc.bind_parameters(np.hstack([beta_values, gamma_values]))
    
    Examples
    --------
    >>> terms = [(1.0, (0, 1)), (0.5, (1, 2))]
    >>> qc = get_parameterized_qaoa_circuit_from_terms(3, terms, p=2)
    >>> # Bind parameters
    >>> theta = np.array([0.1, 0.2, 0.3, 0.4])  # [beta0, beta1, gamma0, gamma1]
    >>> bound_qc = qc.bind_parameters(theta)
    """
    # Create registers
    if qr is None:
        qr = QuantumRegister(N, 'q')
    elif qr.size < N:
        raise ValueError(f"Provided register has {qr.size} qubits, need at least {N}")
    
    qc = QuantumCircuit(qr, cr) if cr else QuantumCircuit(qr)
    
    # Create parameter vectors
    betas = ParameterVector("beta", p)
    gammas = ParameterVector("gamma", p)
    
    # Initial state: uniform superposition
    qc.h(range(N))
    
    # Apply p QAOA layers with parameters
    for i in range(p):
        append_cost_operator_circuit(qc, terms, gammas[i])
        append_mixer_operator_circuit(qc, betas[i])
    
    if save_statevector:
        qc.save_statevector()
    
    if return_parameter_vectors:
        return qc, betas, gammas
    else:
        return qc



# generic quantum circuit 
def get_parameterized_qaoa_circuit(
    N: int,
    p: int,
    costs: np.ndarray | None = None,
    save_statevector: bool = True,
    qr: QuantumRegister | None = None,
    cr: ClassicalRegister | None = None,
    return_parameter_vectors: bool = False,
):
    """Create a parameterized QAOA circuit for a generic diagonal cost Hamiltonian.
    
    This version is for when you have a diagonal cost Hamiltonian represented
    as an array of energy values for each computational basis state, rather
    than explicit terms.
    
    Parameters
    ----------
    N : int
        Number of qubits
    p : int
        Number of QAOA layers (will have 2*p parameters total)
    costs : np.ndarray, optional
        Diagonal cost Hamiltonian as array of length 2^N
        If None, creates circuit without cost operator (mixer only)
    save_statevector : bool, default True
        Save final statevector for simulation
    qr : QuantumRegister, optional
        Custom quantum register
    cr : ClassicalRegister, optional
        Custom classical register for measurements
    return_parameter_vectors : bool, default False
        If True, return (circuit, betas, gammas)
        
    Returns
    -------
    QuantumCircuit or tuple
        Parameterized QAOA circuit
        If return_parameter_vectors=True, returns (qc, betas, gammas)
        
    Notes
    -----
    Parameters are ordered alphabetically: beta first, then gamma.
    To bind: qc.bind_parameters(np.hstack([beta_values, gamma_values]))
    
    For diagonal Hamiltonians, the cost operator is applied using
    phase rotation on the computational basis states.
    
    Examples
    --------
    >>> costs = np.array([0, 1, 1, 2])  # 2-qubit problem
    >>> qc = get_parameterized_qaoa_circuit(N=2, p=2, costs=costs)
    >>> theta = np.array([0.1, 0.2, 0.3, 0.4])  # [beta0, beta1, gamma0, gamma1]
    >>> bound_qc = qc.bind_parameters(theta)
    """
    # Create registers
    if qr is None:
        qr = QuantumRegister(N, 'q')
    elif qr.size < N:
        raise ValueError(f"Provided register has {qr.size} qubits, need at least {N}")
    
    qc = QuantumCircuit(qr, cr) if cr else QuantumCircuit(qr)
    
    # Create parameter vectors
    betas = ParameterVector("beta", p)
    gammas = ParameterVector("gamma", p)
    
    # Initial state: uniform superposition
    qc.h(range(N))
    
    # Apply p QAOA layers
    for i in range(p):
        # Cost operator (diagonal Hamiltonian)
        if costs is not None:
            _append_diagonal_cost_operator(qc, costs, gammas[i], N)
        
        # Mixer operator
        append_mixer_operator_circuit(qc, betas[i])
    
    if save_statevector:
        qc.save_state()
    
    if return_parameter_vectors:
        return qc, betas, gammas
    else:
        return qc
    

def _append_diagonal_cost_operator(
    qc: QuantumCircuit, 
    costs: np.ndarray, 
    gamma: float, 
    N: int
) -> None:
    """Apply diagonal cost Hamiltonian operator using multi-controlled phase gates.
    
    For a diagonal Hamiltonian H = diag(c_0, c_1, ..., c_{2^N-1}),
    applies exp(-i * gamma * H) by applying phase rotations to each basis state.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to modify
    costs : np.ndarray
        Diagonal cost values (length 2^N)
    gamma : float
        Cost layer parameter
    N : int
        Number of qubits
    """
    if len(costs) != 2**N:
        raise ValueError(f"costs array must have length 2^N = {2**N}, got {len(costs)}")
    
    # Apply phase rotation for each basis state
    for state_idx, cost_value in enumerate(costs):
        if abs(cost_value) > 1e-10:  # Skip near-zero costs
            # Convert state index to binary representation
            binary_state = format(state_idx, f'0{N}b')
            
            # Apply controlled phase rotation
            # For basis state |x⟩, apply phase e^(-i * gamma * cost_value)
            _apply_controlled_phase(qc, binary_state, -gamma * cost_value, N)


def _apply_controlled_phase(
    qc: QuantumCircuit, 
    binary_state: str, 
    phase: float, 
    N: int
) -> None:
    """Apply a phase to a specific computational basis state.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to modify
    binary_state : str
        Binary string representation of the state (e.g., "101")
    phase : float
        Phase to apply
    N : int
        Number of qubits
    """
    # Apply X gates to flip 0s to 1s
    for i, bit in enumerate(binary_state):
        if bit == '0':
            qc.x(i)
    
    # Apply multi-controlled Z rotation
    if N == 1:
        qc.p(phase, 0)
    else:
        # Multi-controlled phase gate
        qc.mcp(phase, list(range(N-1)), N-1)
    
    # Undo X gates
    for i, bit in enumerate(binary_state):
        if bit == '0':
            qc.x(i)