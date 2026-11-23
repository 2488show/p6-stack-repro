"""Wrapper around the `state` CLI (arc-state 0.11.1) that can load our own Lightning checkpoints.

torch >= 2.6 defaults torch.load to weights_only=True, which rejects self-trained Lightning checkpoints
(they contain numpy scalars). Lightning passes weights_only explicitly, so it has to be overridden, not
defaulted. Only used for checkpoints this project trained itself. Nothing else is changed.
Usage: python scripts/state_cli.py tx train ... | tx infer ...
"""
import os
import sys

import numpy as np
import torch

try:
    from numpy._core.multiarray import scalar
except ImportError:  # older numpy
    from numpy.core.multiarray import scalar
torch.serialization.add_safe_globals([scalar, np.dtype, np.ndarray])
_orig_load = torch.load


def _load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)


torch.load = _load

# Optional: STATE_ACCEL=mps lets arc-state train on Apple-silicon GPUs (its entry point picks "gpu" if CUDA
# is available and "cpu" otherwise). Only the Trainer's accelerator argument changes.
ACCEL = os.environ.get("STATE_ACCEL")
if ACCEL:
    import lightning.pytorch as pl
    _init = pl.Trainer.__init__

    def _patched(self, *a, **k):
        if k.get("accelerator") == "cpu":
            k["accelerator"] = ACCEL
        return _init(self, *a, **k)

    pl.Trainer.__init__ = _patched

if __name__ == "__main__":
    from state.__main__ import main
    sys.exit(main())
