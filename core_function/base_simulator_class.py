from __future__ import annotations
import typing
import numpy as np
from abc import ABC, abstractmethod
import sys


# Terms are a list of tuples (coeff, [qubit indices])
# hinting system
if sys.version_info(3, 10):
    from  collections.abc import Sequence
    TermsType = Sequence[tuple[float, Sequence[int]]]
    CostsType = Sequence[float] | np.ndarray
    ParamType = Sequence[float] | np.ndarray
else:
    from typing import Sequence
    sys.modules["collections.abc"].Sequence = typing.Sequence
    TermsType = Sequence[typing.Tuple[float, Sequence[int]]]
    CostsType = typing.Union[Sequence[float], np.ndarray]
    ParamType = typing.Union[Sequence[float], np.ndarray]
    from collections.abc import Sequence


class QAOASimulationBase(ABC):
    """Base class for simulator
    """

    def __init__(self, n_qubits: int, costs: CostsType | None= None, terms: TermsType | None = None) -> None:
        """either cost or terms must be uploaded

        Args:
            n_qubits (int): total number of qubits
            costs (CostsType | None, optional): array contain value of cost function
            at each computational basis state. Defaults to None.
            terms (TermsType | None, optional): list of weighted terms, where `term = (weight, [qubit indices]). Defaults to None.
        """

        self.n_qubits = n_qubits
        self.n_state = 2** n_qubits # state like (000, 001,...)
        if costs is None:
            if terms is None:
                raise ValueError("Either costs or terms must be provided")

            self._hc_diagonal = self._diag_from_terms(terms)  
        
        else:
            assert len(costs) == self.n_qubits
            self._hc_diagonal = self._diag_from_costs(costs)
        
        # this is abstract method means that must be implemented by any subclass
        # and contain abstract base class (ABS) wich can not be instantiated directly 
        
            # -- Internal methods
        @abstractmethod 
        def _diag_from_terms(self, terms: TermsType) -> typing.Any:
            """Precompute the diagonal of the cost Hamiltonian
            return implementation data.type For example,
        GPU simulator may return a GPU pointer. Consult the simulator
        implementation for details.

            Args:
                terms (TermsType): list of tuples (coef, [qubits indices])
            Returns:
                typing.Any: _description_
            """
            ...
        
        @abstractmethod
        def _diag_from_costs(self, costs: CostsType) -> typing.Any:
            """
            Adapt the costs array to the simulator-specific datatype

            Parameters
            ----------
                costs: A sequence or a numpy array of length 2**n_qubits
            """
            ... # elipsis; means that in this place sth intentionally left out or implementation provide later
        # public method

        @abstractmethod
        def get_cost_digonal(self) -> np.ndarray:
            """return the diagonal of the cost Hamiltonian
                np.ndarray: _description_
            """
            ...
        
        @abstractmethod
        def simulate_qoao(self, gamma: ParamType, betas: ParamType, sv0: np.ndarray, **kwargs) -> typing.Any:
            """simulator QAOA circuit
            parameters

            Args:
                gamma (ParamType): changing phase seprating
                betas (ParamType): for the mixing
                sv0 (np.ndarray): initial statvector
                default is uniform superposition state

            Returns:
                typing.Any: statvector depends on implementation
            """
            ...
        
        @abstractmethod
        def get_expectation(self, result, costs: typing.Any = None, optimisation_type = "min", **kwargs) -> float:
            """return the expectation value of the cost Hamiltonian

            Args:
                result (_type_): obtained from `sim.simulate_qaoa`
                costs (typing.Any, optional): _description_. Defaults to None.
                array containing values of the cost function at
                each computational basis state. Accepted types depend on the implementation
            """

            ...
        
        @abstractmethod
        def get_overlap(self, result, costs: CostsType | None = None, indices: np.ndarray | Sequence(int), optimisation_type = "min", **kwarg) -> float:
            """return overlap between the lowest energy state and statvector parameters

            Args:
                result (_type_): obtain from sim.simulate_qaoa
                costs (CostsType | None, optional): array containing values of the cost function at each computational basis state. accepted type depend on implemention. Defaults to None.
                optimisation_type (str, optional): _description_. Defaults to "min".

            Returns:
                float: _description_
            """
            ...
        @abstractmethod
        def get_statvector(self, result, **kwargs) -> np.ndarray:
            """return the statvetor as numpy array, which requires enough memory to store 2**n_qubits complex number

            Args:
                result  result: obtained from `sim.simulate_qaoa`

            Returns:
                np.ndarray: _description_
            """
            ...