# stats.py
import subprocess
import time

# arguments
NQ = 12
# for NQ in [4, 8, 10, 12, 14, 16]:
for NQ in [8, 10]:
    for NP in range(1, 33):
        args = ["python3", "-m", "main.Base.Simulators.FPGA.fpga_sim_test", "--sim", "cpp", "--NQ", str(NQ), "--port", "COM3", "--Np", str(NP)]
        print("running...", args)
        for k in range(1):
            subprocess.run(args, capture_output=False, text=True)
            time.sleep(2)