import serial
import struct
import time
from serial.tools import list_ports
import struct
import random
import math

def fp64b(f):
    return struct.pack('>d', f)
def bfp64(b):
    return struct.unpack('<d', b)[0]
def ib8(i):
    return i.to_bytes(8, "little")
def ib1(i):
    return i.to_bytes(1, "little")

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

N = 32 # number of elements
data = []
Hr = []
costF = []
solutionM = [N]
solutionC = [N]

beta = random.uniform(-2*math.pi, 2*math.pi)
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

for id2 in range(N//2):
    sa = id2*2
    sb = id2*2 + 1
    a = sa
    b = sb
    # swap bits, so that a is an index only flipped cq-th bit of b. 
    # in other words, a is a neighbor index of b in terms of cq-th bit.
    # apply rotation
    tsa = cosb * data[a] + 1j * sinb * data[b]
    tsb = 1j*sinb * data[a] + cosb * data[b]
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
print(float_to_verilog_array(10) + "," + f" // {10}")
print(float_to_verilog_array(-0.1) + "," + f" // {-0.1}")
print("64'h"+ fp64b(10).hex() + ",")
print("64'h"+ fp64b(-0.1).hex() + ",")
