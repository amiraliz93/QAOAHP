import struct
import random
import math

# ---------------- helpers ----------------
def fp64b(f):
    """float -> 8 bytes, big-endian (so .hex() prints in natural FP64 order)."""
    return struct.pack('>d', f)

# Q2.61: 1 sign bit, 1 integer bit, 61 fractional bits. Range [-2, 2).
Q_FRAC  = 61
Q_SCALE = 1 << Q_FRAC          # 2^61
Q_MIN   = -(1 << 63)           # most negative signed 64-bit
Q_MAX   =  (1 << 63) - 1       # most positive signed 64-bit
Q_LIMIT = 2.0                  # representable range is [-2, +2)

def to_q2_61_hex(x: float) -> str: 
    """Convert a real x in [-2, 2) to a 64-bit signed Q2.61 hex string (16 chars)."""
    if x >= Q_LIMIT or x < -Q_LIMIT:
        raise ValueError(f"value {x} outside Q2.61 range [-2, 2)")
    v = int(round(x * Q_SCALE))
    if v > Q_MAX: v = Q_MAX
    if v < Q_MIN: v = Q_MIN
    if v < 0:
        v += (1 << 64)         # two's complement
    return f"{v:016x}"


# ---------------- parameters ----------------
N           = 5
H_RANGE     = 1.0                  # H in [-1, 1)
GAMMA_RANGE = math.pi / 2.0        # gamma in [-pi/2, pi/2) so gamma*H stays in (-pi/2, pi/2)
SEED        = None                 # set to int for reproducibility, e.g. 42

if SEED is not None:
    random.seed(SEED)


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
with open("gen_cost_tb_in.sv", "w") as of:
    # gamma (Q2.61)
    print(f"gamma = 64'h{to_q2_61_hex(gamma)};   // gamma = {gamma:+.15g} rad (Q2.61)", file=of)

    # data[]: 32 H values, Q2.61
    data_entries = [f"64'h{to_q2_61_hex(H[i])}" for i in range(N)]
    data_line = "data = {" + ", ".join(data_entries) + "};"
    print(data_line, file=of)

    # costF[]: 64 entries, interleaved cos/sin, Q2.61
    costf_entries = []
    for i in range(N):
        c, s = costF_re[i], costF_im[i]
        costf_entries.append(f"64'h{to_q2_61_hex(c)}")
        costf_entries.append(f"64'h{to_q2_61_hex(s)}")
    costf_line = "costF = {" + ", ".join(costf_entries) + "};"
    print(costf_line, file=of)


# ---------------- console summary ----------------
print(f"gamma = {gamma:+.15g} rad  (Q2.61: 64'h{to_q2_61_hex(gamma)})")
print(f"Generated {N} H values, {2*N} expected cos/sin entries (all Q2.61).")
print("Wrote: gen_cost_tb_in.sv")
print()
print("First 3 entries (sanity check):")
for i in range(3):
    theta = gamma * H[i]
    print(f"  H[{i}] = {H[i]:+.15g} -> theta = {theta:+.15g}, "
          f"cos = {math.cos(theta):+.15g}, sin = {math.sin(theta):+.15g}")
