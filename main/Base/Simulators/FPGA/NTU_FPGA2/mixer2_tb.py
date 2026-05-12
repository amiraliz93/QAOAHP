
import struct
import random
import math


def bfp64(b):
   return struct.unpack('<d', b)[0]
def ib8(i):
    return i.to_bytes(8, "little")
def ib1(i):
    return i.to_bytes(1, "little")

def fp64b(f, width=64, frac_bits=61):
    # Multiply by 2^61 to shift into fixed-point representation
    scaled = int(round(f * (1 << frac_bits)))
    # Mask and handle sign
    mask = (1 << width) - 1
    scaled &= mask
    if scaled >= (1 << (width - 1)):
        scaled -= (1 << width)
    return scaled.to_bytes(8, 'little', signed=True)    

def float_to_verilog_array(f: float, endianness='little') -> str:
    if endianness == 'little':
        packed_bytes = struct.pack('<d', f)
    elif endianness == 'big':
        packed_bytes = struct.pack('>d', f)
    else:
        raise ValueError("Endianness must be 'little' or 'big'")

    byte_values = list(packed_bytes)
    formatted_hex_values = [f"8'h{byte:02x}" for byte in byte_values]
    verilog_string = ", ".join(formatted_hex_values)
    return f"{verilog_string}"

N = 4 # number of elements
data = []
Hr = []
costF = []
solutionM = [N]
solutionC = [N]
seed = 123
random.seed(seed)
beta = random.uniform(-2*math.pi, 2*math.pi)
random.seed(seed)
gamma = random.uniform(-2*math.pi, 2*math.pi)

cosb, sinb = math.cos(beta), math.sin(beta)
lineend = ""
of = open("mixer2_tb_in.sv", "w")
print(f"cosb = 64'h{fp64b(cosb).hex()};", file=of, end=lineend)
print(f"sinb = 64'h{fp64b(sinb).hex()};", file=of, end=lineend)
for i in range(N):
    solutionM.append(1 + 1j)
    solutionC.append(1 + 1j)
    theta = random.uniform(-1, 1)
    r1 = math.cos(theta*2*math.pi) + 1j*math.sin(theta*2*math.pi)
    Hrt = random.uniform(-1, 1)
    # costFt = 1 + 1j*0
    costFt = math.cos(gamma*Hrt) + 1j*math.sin(gamma*Hrt)
    data.append(r1)
    Hr.append(Hrt)
    costF.append(costFt)
def q3_61_int(x, frac_bits=61):
    s = int(round(x * (1 << frac_bits)))
    # Optional range check for signed 64-bit
    if s < -(1 << 63) or s > (1 << 63) - 1:
        raise ValueError("Out of range for signed 64-bit Q3.61")
    return s

def q3_61_hex(x, frac_bits=61):
    s = q3_61_int(x, frac_bits)
    return f"{(s & ((1 << 64) - 1)):016x}"

def q3_61_dec(x, frac_bits=61):
    s = q3_61_int(x, frac_bits)
    return s / float(1 << frac_bits)

for id2 in range(N//2):
    sa = id2*2
    sb = id2*2 + 1
    a = sa
    b = sb
    # swap bits, so that a is an index only flipped cq-th bit of b. 
    # in other words, a is a neighbor index of b in terms of cq-th bit.
    # apply rotation
    tsa = cosb * data[a] - 1j * sinb * data[b]
    tsb = -1j*sinb * data[a] + cosb * data[b]
    solutionM[a] = tsa
    solutionM[b] = tsb

for i in range(N):
    r = data[i]
    costFt = costF[i]
    cf = costFt*r
    solutionC[i] = cf
print("data = {", file=of, end=lineend)
for i in range(N):
    v = data[i]
    end = ","
    if i == N-1:
        end = ""
    print(f"64'h{fp64b(v.real).hex()}, 64'h{fp64b(v.imag).hex()}{end}", file=of, end=lineend)
print("};", file=of, end=lineend)
print("costF = {", file=of, end=lineend)
for i in range(N):
    v = costF[i]
    end = ","
    if i == N-1:
        end = ""
    print(f"64'h{fp64b(v.real).hex()}, 64'h{fp64b(v.imag).hex()}{end}", file=of, end=lineend)
print("};", file=of, end=lineend)

print("solM = {", file=of, end=lineend)
for i in range(N):
    v = solutionM[i]
    end = ","
    if i == N-1:
        end = ""
    print(f"64'h{fp64b(v.real).hex()}, 64'h{fp64b(v.imag).hex()}{end}", file=of, end=lineend)
print("};", file=of, end=lineend)
    
print("solC = {", file=of, end=lineend)
for i in range(N):
    v = solutionC[i]
    end = ","
    if i == N-1:
        end = ""
    print(f"64'h{fp64b(v.real).hex()}, 64'h{fp64b(v.imag).hex()}{end}", file=of, end=lineend)
print("};", file=of, end=lineend)

a = 0.21023
b0 = 4.120319
b3 = 2.1
for i in range(12):
    dr = fp64b(b0)
    drR = dr[::-1]
    print(float_to_verilog_array(b0) + "," + f" // {b0}")
    # print("64'h"+ dr.hex() + ",")
    b0 = b0 + a
print(f"64'h{fp64b(b0).hex()}, // {b0}")
print(f"64'h{fp64b(-0.1).hex()}, // {-0.1}")
print("64'h"+ fp64b(10).hex() + ",")
print("64'h"+ fp64b(-0.1).hex() + ",")



