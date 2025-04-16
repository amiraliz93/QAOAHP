###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################

# Utilities for parameter initialization

from __future__ import annotations
import numpy as np
from pathlib import Path
import pandas as pd
from importlib_resources import files
from enum import Enum
from typing import Callable
from functools import lru_cache, cached_property, cache
from datetime import datetime
from scipy.fft import dct, dst, idct, idst


def from_fourier_basis(u, v):
    """Convert u,v parameterizing QAOA in the Fourier basis
    to beta, gamma in standard parameterization
    following https://arxiv.org/abs/1812.01041
    Parameters
    ----------
    u : list-like
    v : list-like
        QAOA parameters in Fourier basis
    Returns
    -------
    beta, gamma : np.array
        QAOA parameters in standard parameterization
        (used e.g. by qaoa_qiskit.py)
    """
    assert len(u) == len(v)

    gamma = 0.5 * idst(u, type=4, norm="forward")  # difference of 1/2 due to normalization of idst
    beta = 0.5 * idct(v, type=4, norm="forward")  # difference of 1/2 due to normalization of idct

    return gamma, beta

def to_fourier_basis(gamma, beta):
    """Convert gamma,beta standard parameterizing QAOA to the Fourier basis
    of u, v in standard parameterization
    following https://arxiv.org/abs/1812.01041
    Parameters
    ----------
    gamma : list-like
    beta : list-like
        QAOA parameters in standard basis
    Returns
    -------
    u, v : np.array
        QAOA parameters in fourier parameterization
        (used e.g. by qaoa_qiskit.py)
    """
    assert len(gamma) == len(beta)
    u = 2 * dst(gamma, type=4, norm="forward")  # difference of 2 due to normalization of dst
    v = 2 * dct(beta, type=4, norm="forward")  # difference of 2 due to normalization of dct

    return u, v

class QAOAParameterization(Enum):
    """
    Enum class to specify the parameterization of the QAOA parameters
    """

    THETA = "theta"
    GAMMA_BETA = "gamma beta"
    FREQ = "freq"
    U_V = "u v"




def convert_to_gamma_beta(*args, parameterization: QAOAParameterization | str):
    """
    Convert QAOA parameters to gamma, beta parameterization
    """
    parameterization = QAOAParameterization(parameterization)
    if parameterization == QAOAParameterization.THETA:
        assert len(args) == 1, "theta parameterization requires a single argument"
        theta = args[0]
        p = int(len(theta) / 2)
        gamma = theta[:p]
        beta = theta[p:]
    elif parameterization == QAOAParameterization.FREQ:
        assert len(args) == 1, "freq parameterization requires two arguments"
        freq = args[0]
        p = int(len(freq) / 2)
        u = freq[:p]
        v = freq[p:]
        gamma, beta = from_fourier_basis(u, v)
    elif parameterization == QAOAParameterization.GAMMA_BETA:
        assert len(args) == 2, "gamma beta parameterization requires two arguments"
        gamma, beta = args
    elif parameterization == QAOAParameterization.U_V:
        assert len(args) == 2, "u v parameterization requires two arguments"
        u, v = args
        gamma, beta = from_fourier_basis(u, v)
    else:
        raise ValueError("Invalid parameterization")
    return gamma, beta
