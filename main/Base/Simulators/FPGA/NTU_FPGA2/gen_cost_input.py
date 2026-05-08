import struct
import random
import math

# ---------------- helpers ----------------
def fp64b(f):
    """float -> 8 bytes, big-endian (so .hex() prints in natural FP64 order)."""
    return struct.pack('>d', f)

# 64-bit signed fixed-point with 60 fractional bits.
Q_FRAC  = 60
Q_SCALE = 1 << Q_FRAC          # 2^60
Q_MIN   = -(1 << 63)           # most negative signed 64-bit
Q_MAX   =  (1 << 63) - 1       # most positive signed 64-bit
Q_LIMIT = 8.0                  # representable range is [-8, +8)

def to_q4_60_hex(x: float) -> str: 
    """Convert a real x in [-8, 8) to a 64-bit signed Q4.60 hex string (16 chars)."""
    if x >= Q_LIMIT or x < -Q_LIMIT:
        raise ValueError(f"value {x} outside Q4.60 range [-8, 8)")
    v = int(round(x * Q_SCALE))
    if v > Q_MAX: v = Q_MAX
    if v < Q_MIN: v = Q_MIN
    if v < 0:
        v += (1 << 64)         # two's complement
    return f"{v:016x}"


# ---------------- parameters ----------------
N = 2
H_RANGE  = 1.0                  # H in [-1, 1)
GAMMA_RANGE = math.pi         # gamma in [-pi, pi) so gamma*H stays in (-pi, pi) 
SEED = 10           

# ---------------- generate stimulus ----------------
gamma = random.uniform(-GAMMA_RANGE, GAMMA_RANGE)

H        = []
costF_re = []
costF_im = []

for i in range(N):
    h     = random.uniform(-H_RANGE, H_RANGE)
    theta = gamma * h
    H.append(h)
    costF_re.append(math.cos(theta))
    costF_im.append(math.sin(theta))


# ---------------- write the .sv stimulus file ----------------
with open("C:/new_fpga/QAOAHP/main/Base/Simulators/FPGA/NTU_FPGA2/gen_cost_tb_in.sv", "w") as of:
    # Module-scope localparam declarations (not procedural assignments)
    # gamma (Q4.60)
    print(f"localparam logic [63:0] gamma_data = 64'h{to_q4_60_hex(gamma)};   // gamma = {gamma:+.15g} rad (Q4.60)", file=of)
    
    # data[]: H values in Q4.60, each in [-1, 1]
    data_entries = [f"64'h{to_q4_60_hex(H[i])}" for i in range(N)]
    data_indecimal = [f"{H[i]:+.15g}" for i in range(N)]
    print(f"localparam logic [63:0] data_sample [{N-1}:0] = '{{{', '.join(data_entries)}}};", file=of)
    print(f"// H_Q4.60 = {', '.join(data_indecimal)}", file=of)

    # costF[]: interleaved cos/sin, Q4.60
    costf_entries = []
    for i in range(N):
        c, s = costF_re[i], costF_im[i]
        costf_entries.append(f"64'h{to_q4_60_hex(c)}")
        costf_entries.append(f"64'h{to_q4_60_hex(s)}")
    print(f"localparam logic [63:0] costF_sample [{2*N-1}:0] = '{{{', '.join(costf_entries)}}};", file=of)
    cost_real_decimal = [f"{costF_re[i]:+.15g}" for i in range(N)]
    cost_img_decimal = [f"{costF_im[i]:+.15g}" for i in range(N)]
    print(f"// (cos, sin) Q4.60 = {','.join((cost_real_decimal  + cost_img_decimal))}", file=of)


# ---------------- console summary ----------------
print(f"Generated 1 gamma value and {N} H values (all Q4.60).")
print(f"gamma = {gamma:+.15g} rad  (Q4.60: 64'h{to_q4_60_hex(gamma)})")
print(f"Expected {2*N} cos/sin entries (all Q4.60).")
print("Wrote: gen_cost_tb_in.sv")
print()
print("First 3 entries (sanity check):")
for i in range(N):
    theta = gamma * H[i]
    print(f"  H[{i}] = {H[i]:+.15g} -> theta = {theta:+.15g}, "
          f"cos = {math.cos(theta):+.15g}, sin = {math.sin(theta):+.15g}")
