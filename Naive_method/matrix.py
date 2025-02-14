import numpy as np
X = np.array([[0, 1], [1, 0]])

Z = np.array([[1, 0], [0, -1]])

Y = -1j * Z @ X

def kron(*args: np.ndarray):
    if len(args) == 1:
        return args[0]
    return np.kron(args[0], kron(*args[1:]))

