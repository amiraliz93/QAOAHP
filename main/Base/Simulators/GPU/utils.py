###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# pyright: reportGeneralTypeIssues=false
import math
import numba.cuda

#This function computes the squared norm of a state vector by
#  replacing each element with its squared magnitude (absolute squared).
@numba.cuda.jit
def norm_squared_kernel(sv):
    n = len(sv)
    tid = numba.cuda.grid(1) # Get the thread index

    if tid < n:
        sv[tid] = abs(sv[tid]) ** 2 # Convert amplitudes to probabilities


def norm_squared(sv):
    """
    compute norm squared of a statevector
    i.e. convert amplitudes to probabilities
    Calls the kernel on the GPU with parallel execution over len(sv) threads.
    """
    norm_squared_kernel.forall(len(sv))(sv)



@numba.cuda.jit
def initialize_uniform_kernel(sv, scale):
    """This function initializes the state vector with
      equal amplitudes, creating a uniform superposition.
    """
    n = len(sv)
    tid = numba.cuda.grid(1)

    if tid < n:
        sv[tid] = scale / math.sqrt(n)


def initialize_uniform(sv, scale=1.0):
    """
    initialize a uniform superposition statevector on GPU
    """
    initialize_uniform_kernel.forall(len(sv))(sv, scale)


@numba.cuda.jit
def multiply_kernel(a, b): 
    n = len(a)
    tid = numba.cuda.grid(1)

    if tid < n:
        a[tid] = a[tid] * b[tid]


def multiply(a, b): # Launches the kernel on the GPU.
    multiply_kernel.forall(len(a))(a, b)


@numba.cuda.jit 
def copy_kernel(a, b):
    n = len(a)
    tid = numba.cuda.grid(1)

    if tid < n:
        a[tid] = b[tid]


def copy(a, b):
    copy_kernel.forall(len(a))(a, b)


@numba.cuda.reduce  #Performs a parallel reduction (sum of all elements in an array).
def sum_reduce(a, b):
    return a + b


@numba.cuda.reduce #Finds the maximum real part of all elements in an array.
def real_max_reduce(a, b):
    return max(a.real, b.real)