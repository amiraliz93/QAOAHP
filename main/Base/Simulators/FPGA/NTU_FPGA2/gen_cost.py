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
solutionC = [N]

gamma = random.uniform(-math.pi, math.pi)

lineend = "\n"
of = open("gen_cost_in.sv", "w")
comment = ""
if lineend != "":
    comment = f" // {gamma}"
print(f"gamma = 64'h{fp64b(gamma).hex()};{comment}", file=of, end=lineend)
for i in range(N):
    Hrt = random.uniform(-1, 1)
    costFt = math.cos(gamma*Hrt) + 1j*math.sin(gamma*Hrt)
    Hr.append(Hrt)
    costF.append(costFt)

print("data = {", file=of, end=lineend)
for i in range(N):
    v = Hr[i]
    end = ","
    if i == N-1:
        end = ""
    comment = ""
    if lineend != "":
        comment = f" // {v}"
    print(f"64'h{fp64b(v.real).hex()}{end}{comment}", file=of, end=lineend)
print("};", file=of, end=lineend)

print("costF = {", file=of, end=lineend)
for i in range(N):
    v = costF[i]
    end = ","
    if i == N-1:
        end = ""
    if lineend != "":
        comment = f" // {v}"
    print(f"64'h{fp64b(v.real).hex()}, 64'h{fp64b(v.imag).hex()}{end}{comment}", file=of, end=lineend)
print("};", file=of, end=lineend)
