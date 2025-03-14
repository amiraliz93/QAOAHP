import math
import numba.cuda
import numpy as np
from pathlib import Path
from functools import lru_cache
import warnings

import cupy as cp

#try:
#    import cupy as cp
#except ImportError:
 #  if numba.cuda.is_available():
    #    warnings.warn("Cupy import failed, which is required for X rotations on NVIDA GPUs", RuntimeWarning)



########################################
# single-qubit X rotation
########################################


@lru_cache
def get_furx_kernel(k_qubits: int, q_offset: int, state_mask: int):
    """
    Generate furx kernel for the specified sub-group size.
    """
    if k_qubits > 6:
        kernel_name = f"furx_kernel<{k_qubits},{q_offset},{state_mask}>"
    else:
        kernel_name = f"warp_furx_kernel<{k_qubits},{q_offset}>"

    code = open(Path(__file__).parent / "furx.cu").read()
    return cp.RawModule(code=code, name_expressions=[kernel_name], options=("-std=c++17",)).get_function(kernel_name)


def furx(sv: np.ndarray, a: float, b: float, k_qubits: int, q_offset: int, state_mask: int):
    """
    Apply in-place fast Rx gate exp(-1j * theta * X) on k consequtive qubits to statevector array x.

    sv: statevector
    a: cosine factor
    b: sine factor
    k_qubits: number of qubits to process concurrently
    q_offset: starting qubit number
    state_mask: mask for indexing
    """
    if k_qubits > 11:
        raise ValueError("k_qubits should be <= 11 because of shared memory constraints")

    seq_kernel = get_furx_kernel(k_qubits, q_offset, state_mask)

    if k_qubits > 6:
        threads = 1 << (k_qubits - 1)
    else:
        threads = min(32, len(sv))

    seq_kernel(((len(sv) // 2 + threads - 1) // threads,), (threads,), (sv, a, b))



def furx_all(sv: np.ndarray, theta: float, n_qubits: int):
    """
    Apply in-place fast uniform Rx gates exp(-1j * theta * X) to statevector array x.

    sv: statevector
    theta: rotation angle
    n_qubits: total number of qubits
    """
    n_states = len(sv)
    state_mask = (n_states - 1) >> 1

    a, b = math.cos(theta), - math.sin(theta)

    group_size = 10
    last_group_size = n_qubits % group_size

    cp_sv = cp.asarray(sv)

    for q_offset in range(0, n_qubits - last_group_size, group_size):
        furx(cp_sv, a, b, group_size, q_offset, state_mask)

    if last_group_size > 0:
        furx(cp_sv, a, b, last_group_size, n_qubits - last_group_size, state_mask)

        