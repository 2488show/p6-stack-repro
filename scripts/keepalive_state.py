# Run inside the Colab kernel (colab exec -s p6state -f scripts/keepalive_state.py --timeout 0) so the
# runtime is not reclaimed as idle while the State jobs run as detached shell processes.
# Stops after 10 h or when /content/p6state/STOP_KEEPALIVE exists.
import os
import time

t0 = time.time()
while time.time() - t0 < 10 * 3600:
    if os.path.exists("/content/p6state/STOP_KEEPALIVE"):
        break
    print(f"alive {int((time.time() - t0) / 60)} min", flush=True)
    time.sleep(60)
