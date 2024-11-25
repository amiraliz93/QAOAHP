
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from .base_simulator_class import QAOASimulationBase, CostsType, TermsType, ParamType
from .pre_computation_function import precompute_vectorized_cpu_parallel
from .core import apply_qaoa_furx

def binary_of_indice(arr):
    n = int(np.log2(len(arr))) # calculate the value of n

    # Create a binary representation of indices
    binary_indices = np.vectorize(lambda x: np.binary_repr(x, width=n))(np.arange(len(arr)))
    reversed_indices = np.array([int(bin_idx[::-1], 2) for bin_idx in binary_indices])
    new_arr = arr[reversed_indices]
    return new_arr


class QAOAFastSimulationPython(QAOASimulationBase):
    _hc_diag: np.ndarray

    def _diag_from_costs(self, costs: CostsType):
        return np.asarray(costs, dtype="float")
    
    def _diag_from_terms(self, terms: TermsType):
        a = precompute_vectorized_cpu_parallel(terms, 0.0, self.n_qubits)
        return a

    def get_cost_digonal(self):
        return self._hc_diag
    
    @property
    def defult_sv0(self):
        return np.full(self.n_states, 1.0 / np.sqrt(self.n_states), dtype="complex")
    
    def _apply_qaoa(self, sv: np.ndarray, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        raise NotImplementedError
    
    def simulate_qoao(self, gamma, betas, sv0, **kwargs):
           """
        simulator QAOA circuit using FUR
        @param gammas parameters for the phase separating layers
        @param betas parameters for the mixing layers
        @param sv0 (optional) initial statevector, default is uniform superposition state
        @return statevector or vector of probabilities
        """
           sv = sv0.astype("complex") if sv0 is not None else self.defult_sv0
           self._apply_qaoa(sv, list(gammas), list(betas), **kwargs)
           return sv

     # -- Outputs
    
    
    def get_statvector(self, result: np.ndarray, **kwargs) -> np.ndarray:
        return result

    def get_probabilities(self, result: np.ndarray, **kwargs) -> np.ndarray:
        return np.abs(result) ** 2
    
    def get_expectation(self, result: np.ndarray, costs: np.ndarray | None = None, optimization_type="min", **kwargs) -> float:
        if costs is None:
            costs = self._hc_diag
        if optimization_type == "max":
            return -1 * np.dot(costs, np.abs(result) ** 2)
        return np.dot(costs, np.abs(result) ** 2)
    
    def get_overlap(
        self, result: np.ndarray, costs: CostsType | None = None, indices: np.ndarray | Sequence[int] | None = None, optimization_type="min", **kwargs
    ) -> float:
        """
        Compute the overlap between the statevector and the ground state

        Parameters
        ----------
            result: statevector
            costs: (optional) diagonal of the cost Hamiltonian
            indices: (optional) indices of the ground state in the statevector
            preserve_state: (optional) if True, allocate a new array for probabilities
        """
        probs = self.get_probabilities(result, **kwargs)
        if indices is None:
            if costs is None:
                costs = self._hc_diag
            else:
                costs = self._diag_from_costs(costs)
            if optimization_type == "max":
                val = costs.max()
            else:
                val = costs.min()
            indices = (costs == val).nonzero()
        return probs[indices].sum()
    

class QAOAFURXSimulator(QAOAFastSimulatorPythonBase):
    def _apply_qaoa(self, sv: np.ndarray, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        apply_qaoa_furx(sv, gammas, betas, self._hc_diag, self.n_qubits)