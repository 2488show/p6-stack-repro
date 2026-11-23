"""Copy a Lightning checkpoint without optimizer / loop state (1.17 GB -> about a third), so the best
checkpoint can be pulled off the Colab runtime while training is still running. Enough for `state tx infer`.
Usage: python scripts/state_slim_ckpt.py <src.ckpt> <dst.ckpt>
"""
import os
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
for k in ["optimizer_states", "lr_schedulers", "loops", "callbacks"]:
    ck.pop(k, None)
torch.save(ck, dst + ".tmp")
os.replace(dst + ".tmp", dst)
print(f"slim {src} -> {dst}: step {ck.get('global_step')}, epoch {ck.get('epoch')}, {os.path.getsize(dst) / 1e6:.0f} MB")
