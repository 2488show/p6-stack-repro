# Run inside the Colab kernel (colab exec -f keepalive.py --timeout 0) so the runtime is not
# treated as idle while the real work runs as detached shell processes.
import os
import time

t0 = time.time()
while time.time() - t0 < 3 * 3600:
    done = os.path.exists("/content/stack-repro/logs/run_all.log") and "ALL_DONE" in open(
        "/content/stack-repro/logs/run_all.log").read()
    print(f"alive {int((time.time() - t0) / 60)} min, all_done={done}", flush=True)
    if done:
        break
    time.sleep(60)
