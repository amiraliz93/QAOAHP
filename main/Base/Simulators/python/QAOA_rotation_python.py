from collections.abc import Sequence
import numpy as np
from .core_rotation_python import furx_all


def apply_qaoa_furx(sv: np.ndarray, gammas: Sequence[float], betas: Sequence[float], hc_diag: np.ndarray, n_qubits: int) -> None:
    """
    apply a QAOA with the X mixer defined by
    U(beta) = sum_{j} exp(-i*beta*X_j/2)
    where X_j is the Pauli-X operator applied on the jth qubit.
    @param sv array NumPy array (dtype=complex) of length n containing the statevector
    @param gammas parameters for the phase separating layers
    @param betas parameters for the mixing layers
    @param hc_diag array of length n containing diagonal elements of the diagonal cost Hamiltonian
    @param n_qubits total number of qubits represented by the statevector
    """
    for gamma, beta in zip(gammas, betas):
        sv *= np.exp(-0.5j * gamma * hc_diag)
        furx_all(sv, beta, n_qubits)
        