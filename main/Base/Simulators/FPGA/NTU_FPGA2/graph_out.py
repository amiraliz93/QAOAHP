import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.ticker as ticker
import sys

from pathlib import Path

print(sys.argv)
if len(sys.argv) < 2:
    print("need argument of file name")
    quit()

filename = sys.argv[1]
basename = Path(filename).stem


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
    "TTh": [],
    "np": [],
})

# -------- Parse file --------
with open(filename) as f:
    block = {"TTh": 1}

    for line in f:
        line = line.strip()

        if line.startswith("Python time"):
            block["python"] = float(line.split(":")[1].split()[0])

        elif line.startswith("PFGA time"):
            block["fpga"] = float(line.split(":")[1].split()[0])

        elif line.startswith("Transmission"):
            block["trans"] = float(line.split(":")[1].split()[0])

        elif line.startswith("python/PFGA"):
            block["ratio"] = float(line.split(":")[1].split()[0])

        elif line.startswith("Theory. FPGA"):
            block["TTh"] = float(line.split(":")[1].split()[0])

        elif line.startswith("NQ"):
            block["nq"] = int(line.split(":")[1].split()[0])

        elif line.startswith("Np"):
            block["np"] = int(line.split(":")[1].split()[0])

        # End of block
        if line.startswith("MAE"):
            nq = block["nq"]

            data[nq]["python"].append(block["python"])
            data[nq]["fpga"].append(block["fpga"])
            data[nq]["trans"].append(block["trans"])
            data[nq]["ratio"].append(block["ratio"])
            data[nq]["TTh"].append(block["TTh"])
            data[nq]["np"].append(block["np"])

            block = {"TTh": 1}

# -------- Compute stats --------
nq_values = sorted(data.keys())

py_mean, py_std = [], []
fpga_mean, fpga_std = [], []
trans_mean, trans_std = [], []
ratio_mean, ratio_std = [], []
TTh_mean, TTh_std = [], []
S_values = []
Tt_values = []

for nq in nq_values:
    py = np.array(data[nq]["python"])
    fpga = np.array(data[nq]["fpga"])
    TTh = np.array(data[nq]["TTh"])
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
    LatencyAddGen = 7
        
    F = 320*1e6
    NQ = nq
    Np = Np_val # number of p layers.
    LP_BRAM_A = 2
    LP_BRAM_D = 1
    LP_GEN_COST = 2
    LP_MIXER_IN = 1
    LP_MIXER_OUT = 1
    L_BRAM_R = LP_BRAM_A + LP_BRAM_D + 2
    L_BRAM_W = LP_BRAM_A + LP_BRAM_D + 2

    Lc = 223 + 1 + L_BRAM_R + 1 + LP_GEN_COST + LP_GEN_COST# cost gen latency, output H latency, memory and register latency, address output latency.
    Lm = 52 + 1 + L_BRAM_R + 1 + L_BRAM_W + 1 + LP_MIXER_IN + LP_MIXER_OUT  # mixer latency 52, output latency to mixer + memory read, output of address + write latency + write address latency.
    LInit = 24

    NS = 2**NQ # number of layers

    LPipe = NS
    tl = Lm + NS//2 + NS%2 
    if tl >= NS:
        LPipe = tl
        print(f"Lm + NS//2 + NS%2  = {tl} >= NS = {NS}. LPipe become {LPipe}.")
    else:
        print(f"LPipe = NS = {NS}")

    DVTc = Lc // LPipe + 1; # make sure, pipe*DVTc-Tc-2 > LInit.

    tGenCost = DVTc*LPipe - Lc
    if tGenCost < LInit:
        tGenCost += LPipe

        print(f"tGenCost = {tGenCost} greater than LInit = {LInit}.")
    tbGenCost = LPipe*(NQ+1)-Lc
    t_Mixer = LPipe
    if tbGenCost < LInit:
        t_Mixer = LPipe + LInit - tbGenCost
        print(f"tbGenCost = {tbGenCost} < {LInit} = LInit. set tbGenCost={LInit}, t_Mixer ={t_Mixer}")
        # need to prepare additional time for gen cost
        tbGenCost = LInit

    SClocks = LatencyAddGen + tGenCost + Lc + (t_Mixer + LPipe*NQ)*Np + LPipe
    if "gtf" in sys.argv:
        TTh = SClocks/F
        TTh_mean.append(TTh)
        TTh_std.append(TTh)
    else:
        TTh_mean.append(TTh.mean())

    
    S = (Tcost + TMix) * Np_val
    Tt = (D*3*5*8 + 64*D*2 + D*64 + Np_val*3*64)/115200*10/8
    Tt_values.append(Tt)

# Convert to numpy
nq_values = np.array(nq_values)

py_mean = np.array(py_mean)
py_std = np.array(py_std)
fpga_mean = np.array(fpga_mean)
fpga_std = np.array(fpga_std)
trans_mean = np.array(trans_mean)
trans_std = np.array(trans_std)
TTh_mean = np.array(TTh_mean)
ratio_mean = np.array(ratio_mean)
ratio_std = np.array(ratio_std)
Tt_values = np.array(Tt_values)

# -------- Graph 1: Times + Transmission --------
plt.figure()

if "p" in sys.argv:
    plt.plot(nq_values, py_mean, 'o-', label='Python')
if "f" in sys.argv:
    plt.plot(nq_values, fpga_mean, 's-', label='FPGA')
if "tf" in sys.argv or "gtf" in sys.argv:
    plt.plot(nq_values, TTh_mean, '^-', label='Theory. FPGA')
if "tt" in sys.argv:
    plt.plot(nq_values, Tt_values, '^-', label='Theory. Transfer')
if "t" in sys.argv:
    plt.plot(nq_values, trans_mean, 'd-', label='Transfer')

# Std bands (only meaningful for measured data)
plt.fill_between(nq_values, py_mean - py_std, py_mean + py_std, alpha=0.2)
#plt.fill_between(nq_values, fpga_mean - fpga_std, fpga_mean + fpga_std, alpha=0.2)
plt.fill_between(nq_values, trans_mean - trans_std, trans_mean + trans_std, alpha=0.2)

plt.yscale("log")

ax = plt.gca()

# Major ticks: 10^n
ax.yaxis.set_major_locator(
    ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
)

# Minor ticks: 2*10^n, ..., 9*10^n
ax.yaxis.set_minor_locator(
    ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
)

# Format major labels as 10^n
ax.yaxis.set_major_formatter(
    ticker.LogFormatterMathtext(base=10.0)
)

# Hide minor tick labels
ax.yaxis.set_minor_formatter(
    ticker.NullFormatter()
)

# Grid for both major and minor ticks
plt.grid(True, which="major", linestyle="-", linewidth=0.7)
plt.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.6)

plt.xlabel("NQ")
plt.ylabel("Value (log scale)")
plt.title("Python, FPGA, Theoretical Time, and Transmission")

plt.legend()

path = "time_" + basename + ".svg"
plt.savefig(path)
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
path = "ratio_" + basename + ".svg"

plt.savefig(path)
plt.close()

print("Generated:")
print(" - time_with_theory.svg")
print(" - ratio.svg")