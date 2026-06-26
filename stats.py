# stats.py
import subprocess
import time

# arguments
for NQ in range(2, 17):
    args = ["python3", "-m", "main.Base.Simulators.FPGA.fpga_sim_test", "--sim", "cpp", "--NQ", str(NQ), "--port", "COM3", "--Np", str(1)]
    print("running...", args)
    for k in range(4):
        subprocess.run(args, capture_output=False, text=True)
        time.sleep(2)