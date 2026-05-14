import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

filename = "statistics.txt"

# Constants (theory)
M = 0
K = 2**M
F = 320  # MHz

# -------- Storage --------
data = defaultdict(lambda: {
    "python": [],
    "fpga": [],
    "trans": [],
    "ratio": [],
    "np": [],
})

# -------- Parse file --------
with open(filename) as f:
    block = {}

    for line in f:
        line = line.strip()

        if line.startswith("Python time"):
            block["python"] = float(line.split(":")[1].split()[0])

        elif line.startswith("PFGA time"):
            block["fpga"] = float(line.split(":")[1].split()[0])

        elif line.startswith("Transmission"):
            block["trans"] = float(line.split(":")[1])

        elif line.startswith("python/PFGA"):
            block["ratio"] = float(line.split(":")[1])

        elif line.startswith("NQ"):
            block["nq"] = int(line.split()[1])

        elif line.startswith("Np"):
            block["np"] = int(line.split(":")[1])

        # End of block
        if line.startswith("MAE"):
            nq = block["nq"]

            data[nq]["python"].append(block["python"])
            data[nq]["fpga"].append(block["fpga"])
            data[nq]["trans"].append(block["trans"])
            data[nq]["ratio"].append(block["ratio"])
            data[nq]["np"].append(block["np"])

            block = {}

# -------- Compute stats --------
nq_values = sorted(data.keys())

py_mean, py_std = [], []
fpga_mean, fpga_std = [], []
trans_mean, trans_std = [], []
ratio_mean, ratio_std = [], []
S_values = []
Tt_values = []

for nq in nq_values:
    py = np.array(data[nq]["python"])
    fpga = np.array(data[nq]["fpga"])
    trans = np.array(data[nq]["trans"])
    ratio = np.array(data[nq]["ratio"])
    np_vals = np.array(data[nq]["np"])

    py_mean.append(py.mean())
    py_std.append(py.std())

    fpga_mean.append(fpga.mean())
    fpga_std.append(fpga.std())

    trans_mean.append(trans.mean())
    trans_std.append(trans.std())

    ratio_mean.append(ratio.mean())
    ratio_std.append(ratio.std())

    # --- Theoretical S ---
    Np_val = np_vals.mean()
    D = 2**nq
    Dp = D / K

    Tcost = Dp / F * 1e-6
    TMix = Dp / F * nq * 1e-6

    S = (Tcost + TMix) * Np_val
    Tt = (D*3*5*8 + 64*D*2 + D*64 + Np_val*3*64)/115200*10/8
    S_values.append(S)
    Tt_values.append(Tt)

# Convert to numpy
nq_values = np.array(nq_values)

py_mean = np.array(py_mean)
py_std = np.array(py_std)
fpga_mean = np.array(fpga_mean)
fpga_std = np.array(fpga_std)
trans_mean = np.array(trans_mean)
trans_std = np.array(trans_std)
ratio_mean = np.array(ratio_mean)
ratio_std = np.array(ratio_std)
S_values = np.array(S_values)
Tt_values = np.array(Tt_values)

# -------- Graph 1: Times + Transmission --------
plt.figure()

plt.plot(nq_values, py_mean, 'o-', label='Python')
plt.plot(nq_values, fpga_mean, 's-', label='FPGA')
plt.plot(nq_values, S_values, '^-', label='Theory. FPGA')
plt.plot(nq_values, Tt_values, '^-', label='Theory. Transfer')
plt.plot(nq_values, trans_mean, 'd-', label='Transfer')

# Std bands (only meaningful for measured data)
plt.fill_between(nq_values, py_mean - py_std, py_mean + py_std, alpha=0.2)
plt.fill_between(nq_values, fpga_mean - fpga_std, fpga_mean + fpga_std, alpha=0.2)
plt.fill_between(nq_values, trans_mean - trans_std, trans_mean + trans_std, alpha=0.2)

plt.yscale("log")

plt.xlabel("NQ")
plt.ylabel("Value (log scale)")
plt.title("Python, FPGA, Theoretical Time, and Transmission")

plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.savefig("time_with_theory.svg")
plt.close()

# -------- Graph 2: Ratio --------
plt.figure()

plt.plot(nq_values, ratio_mean, 'o-', label='python/FPGA')

plt.fill_between(
    nq_values,
    ratio_mean - ratio_std,
    ratio_mean + ratio_std,
    alpha=0.3
)

plt.xlabel("NQ")
plt.ylabel("python / FPGA")
plt.title("Python/FPGA Ratio vs NQ")

plt.grid(True)
plt.legend()

plt.savefig("ratio.svg")
plt.close()

print("Generated:")
print(" - time_with_theory.svg")
print(" - ratio.svg")