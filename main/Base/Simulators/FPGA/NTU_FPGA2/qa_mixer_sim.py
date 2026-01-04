import numpy as np 

sv = []
N = 3 # 3 qbits
NS = 2**N # dimension of state vector

# initialize state vector 
for i in range(NS):
    sv.append(complex(0, 0))

sv[0] = complex(1, 0)

Np = 8 # number of p layers.
beta = 0.1
sinb = np.sin(beta)
cosb = np.cos(beta)
def swap_bits(i, a, b):
    # 1. Extract the values of the bits at position a and b
    bit_a = (i >> a) & 1
    bit_b = (i >> b) & 1

    # 2. If the bits are different, we need to swap them
    if bit_a != bit_b:
        # Create a bitmask with 1s at position a and b
        # Example: (1 << 2) | (1 << 5) results in 00100100
        mask = (1 << a) | (1 << b)
        
        # XOR the original integer with the mask
        # This flips the bits at those two positions
        i ^= mask
        
    return i
import random
lcq = list(range(N))
random.shuffle(lcq)
print(lcq)
for p in range(Np):
    # output the current state vector
    f = open(f"sim_mixer_{p}-th.txt", "w")
    for i in range(NS):
        f.write(f"{sv[i]}\n")
    f.close()

    # apply mixer operator
    for cq in lcq: #counter of qbit.
        for id2 in range(NS//2):
            sa = id2*2
            sb = id2*2 + 1

            # swap bits, so that a is an index only flipped cq-th bit of b. 
            # in other words, a is a neighbor index of b in terms of cq-th bit.
            a = swap_bits(sa, cq, 0)
            b = swap_bits(sb, cq, 0)

            # apply rotation
            
            tsa = cosb * sv[a] + 1j * sinb * sv[b]
            tsb = 1j*sinb * sv[a] + cosb * sv[b]
            sv[a] = tsa
            sv[b] = tsb

            # p'_a = cos p_a + i sin p_b
            # p'_b = i sin p_a + cos p_b
        


