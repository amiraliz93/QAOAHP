import numpy as np
import pandas as pd
from itertools import product



def reverse_array_index_bit_order(arr):
    arr = np.array(arr)
    n = int(np.log2(len(arr)))  # Calculate the value of N
    if n % 1:
        raise ValueError("Input vector has to have length 2**N where N is integer")

    index_arr = np.arange(len(arr))
    new_index_arr = np.zeros_like(index_arr)
    while n > 0:
        last_8 = np.unpackbits(index_arr.astype(np.uint8), axis=0, bitorder="little")
        repacked_first_8 = np.packbits(last_8).astype(np.int64)
        if n < 8:
            new_index_arr += repacked_first_8 >> (8 - n)
        else:
            new_index_arr += repacked_first_8 << (n - 8)
        index_arr = index_arr >> 8
        n -= 8
    return arr[new_index_arr]