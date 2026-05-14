# stats.py
import subprocess
import time

# arguments
for NQ in range(2, 16):    
    args = ["python", "w.py", str(NQ), "15"]
    print("running...", args)
    for k in range(8):
        subprocess.run(args, capture_output=False, text=True)
        time.sleep(2)
    

