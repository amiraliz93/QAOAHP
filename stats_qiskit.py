# stats.py
import subprocess
import time

# arguments
for NQ in range(2, 17):
    args = ["python3", "-m", "main.Test.q2test", "--NQ", str(NQ), "--Np", str(1), "--niter", str(8)]
    print("running...", args)
    subprocess.run(args, capture_output=False, text=True)

    