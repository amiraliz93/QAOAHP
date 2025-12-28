import struct
import numpy as np
np.random.seed(77)
def get_binary_ieee754(fp_number):
    packed_bytes = struct.pack('>d', fp_number)
    packed_integer = struct.unpack('>Q', packed_bytes)[0]
    binary_string = format(packed_integer, '064b')
    return binary_string

def float_to_hex(f):
    """Converts a float to its raw IEEE 754 hexadecimal bit pattern."""
    packed_bytes = struct.pack('>d', f)
    hex_string = packed_bytes.hex()
    
    return hex_string
    
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
    return f"{{{verilog_string}}}"
# Example usage with your variable fp1
fp1 = 4.149
cosb = np.cos(np.pi/25)
sinb = np.sin(np.pi/25)
a = 2*4.149*4.149
fp2 = 89.123
fp3 = fp1*fp2

b1 = get_binary_ieee754(a)
hx = float_to_hex(a)
b2 = get_binary_ieee754(fp2)
b3 = get_binary_ieee754(fp3)

print(f"The 64-bit IEEE 754 binary representation:")
print(f"fp1 <= 64'b{b1}; // {fp1}")
print(f"cosb <= 64'h{float_to_hex(cosb)}; // {cosb}")
print(f"sinb <= 64'h{float_to_hex(sinb)}; // {sinb}")
print(f"fp2 <= 64'b{b2}; // {fp2}")
print(f"fp3 <= 64'b{b3}; // {fp3}")

print("test of mixer2.sv")
def generate_ab_math():
    """Generates two random floats a and b that satisfy a^2 + b^2 = 1."""
    # Generate a random angle theta between 0 and 2*pi radians
    theta = np.random.uniform(0, 2 * np.pi)
    
    # Calculate a and b using the cosine and sine of the angle
    a = np.cos(theta)
    b = np.sin(theta)
    return a, b
N = 12
vp = []
print("initial quantities")
for i in range(N):
    p_ar, p_ai = generate_ab_math()
    vp.append(p_ar)
    vp.append(p_ai)
    print(f"64'h{float_to_hex(p_ar)}, // {p_ar}")
    print(f"64'h{float_to_hex(p_ai)}, // {p_ai}")
for i in range(N//2):
    p_ar, p_ai = vp[i*4 + 0], vp[i*4 + 1]
    p_br, p_bi = vp[i*4 + 2], vp[i*4 + 3]
    pp_ar = cosb *p_ar - sinb *p_bi
    pp_ai = cosb *p_ai + sinb *p_br
    pp_br = - sinb* p_ai + cosb *p_br
    pp_bi = sinb *p_ar + cosb *p_bi
    vp[i*4 + 0], vp[i*4 + 1] = pp_ar, pp_ai
    vp[i*4 + 2], vp[i*4 + 3] = pp_br, pp_bi
print("result  quantities")
for i in range(N):
    p_ar = vp[i*2 + 0]
    p_ai = vp[i*2 + 1]
    vp.append((p_ar, p_ai))
    print(f"64'h{float_to_hex(p_ar)}, // {p_ar}")
    print(f"64'h{float_to_hex(p_ai)}, // {p_ai}")
