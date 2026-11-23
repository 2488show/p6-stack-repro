"""Wrapper around the `state` CLI (arc-state 0.11.1) that can load our own Lightning checkpoints.

torch >= 2.6 defaults torch.load to weights_only=True, which rejects self-trained Lightning checkpoints
(they contain numpy scalars). Lightning passes weights_only explicitly, so it has to be overridden, not
defaulted. Only used for checkpoints this project trained itself. Nothing else is changed.
Usage: python scripts/state_cli.py tx train ... | tx infer ...
"""
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

if __name__ == "__main__":
    from state.__main__ import main
    sys.exit(main())
