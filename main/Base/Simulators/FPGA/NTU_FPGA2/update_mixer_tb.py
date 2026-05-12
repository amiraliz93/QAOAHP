import random
import math


# ---------- Q3.61 signed 64-bit fixed-point helpers ----------

def q3_61_int(x, frac_bits=61):
    s = int(round(x * (1 << frac_bits)))
    if s < -(1 << 63) or s > (1 << 63) - 1:
        raise ValueError(f"Value {x} out of range for signed 64-bit Q3.61")
    return s

def q3_61_hex(x, frac_bits=61):
    s = q3_61_int(x, frac_bits)
    return f"{(s & ((1 << 64) - 1)):016x}"

def q3_61_dec(x, frac_bits=61):
    return q3_61_int(x, frac_bits) / float(1 << frac_bits)


# ---------- parameters ----------

N = 4  # number of elements
seed = 123

random.seed(seed)
beta  = random.uniform(-2*math.pi, 2*math.pi)
gamma = random.uniform(-2*math.pi, 2*math.pi)  # next draw, no reseed


# ---------- stimulus ----------

cosb, sinb = math.cos(beta), math.sin(beta)

data      = []
Hr        = []
costF     = []
solutionM = [0+0j] * N
solutionC = [0+0j] * N

for i in range(N):
    theta  = random.uniform(-1, 1)
    r1     = math.cos(theta*2*math.pi) + 1j*math.sin(theta*2*math.pi)
    Hrt    = random.uniform(-1, 1)
    costFt = math.cos(gamma*Hrt) + 1j*math.sin(gamma*Hrt)
    data.append(r1)
    Hr.append(Hrt)
    costF.append(costFt)


# ---------- golden reference: mixer layer ----------
# Pairs (a, b) = (2k, 2k+1) get the XX rotation.
# If N is odd, the last qubit is unpaired and passes through unchanged.

for id2 in range(N // 2):
    a = id2*2
    b = id2*2 + 1
    solutionM[a] = cosb     * data[a] + 1j*sinb * data[b]
    solutionM[b] = 1j*sinb  * data[a] + cosb    * data[b]

if N % 2 == 1:
    solutionM[N-1] = data[N-1]


# ---------- golden reference: cost layer ----------

for i in range(N):
    solutionC[i] = costF[i] * data[i]


# ---------- write SystemVerilog testbench input ----------

def fmt_complex_array(name, arr):
    lines = [f"{name} = {{"]
    for i, v in enumerate(arr):
        sep = "," if i < len(arr) - 1 else ""
        lines.append(f"    64'h{q3_61_hex(v.real)}, 64'h{q3_61_hex(v.imag)}{sep}")
    lines.append("};")
    return "\n".join(lines)

with open("mixer2_tb_in.sv", "w") as of:
    print(f"cosb = 64'h{q3_61_hex(cosb)};", file=of)
    print(f"sinb = 64'h{q3_61_hex(sinb)};", file=of)
    print(fmt_complex_array("data",  data),      file=of)
    print(fmt_complex_array("costF", costF),     file=of)
    print(fmt_complex_array("solM",  solutionM), file=of)
    print(fmt_complex_array("solC",  solutionC), file=of)

print("Wrote mixer2_tb_in.sv")
print(f"  beta  = {beta:+.6f} rad")
print(f"  gamma = {gamma:+.6f} rad")