# Run inside the Colab kernel (colab exec -s <session> -f keepalive_full.py --timeout 0) so the
# runtime is not treated as idle while the 3-donor generation runs as detached shell processes.
import os
import time

FLAG = "/content/stack-repro/logs/FULL_DONE"
t0 = time.time()
while time.time() - t0 < 4 * 3600:
    done = os.path.exists(FLAG)
    print(f"alive {int((time.time() - t0) / 60)} min, done={done}", flush=True)
    if done:
        break
    time.sleep(60)
