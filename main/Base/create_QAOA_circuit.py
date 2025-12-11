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
    qr: QuantumRegister = None,
    cr: ClassicalRegister = None
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
    qr: QuantumRegister = None,
    cr: ClassicalRegister = None,
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

