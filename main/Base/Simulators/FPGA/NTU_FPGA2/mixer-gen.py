import serial
import struct
import time
from serial.tools import list_ports
import struct

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
