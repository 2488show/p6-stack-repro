#!/usr/bin/env python
"""Check that the State runtime's evaluation environment reproduces the original evaluation.

The State results were scored in a new venv (.venv-eval, pinned to env_freeze.txt; scripts/20). The
no-downsample tcell_copy evaluation uses no random numbers, so re-running it there must give exactly the
numbers of the original run (results/eval/T5_gen0_allcells_seed0_per_pert.csv).
Run:  python scripts/03_evaluate.py --tag CHECK_tcell_allcells --seed 0 --no-downsample --methods tcell_copy
      python scripts/27_check_eval_env.py <original csv> <new csv>
"""
import sys

import numpy as np
import pandas as pd

old = pd.read_csv(sys.argv[1])
old = old[old["method"] == "tcell_copy"]
new = pd.read_csv(sys.argv[2])
m = new.merge(old, on=["cell_type", "perturbation"], suffixes=("_new", "_old"))
cols = ["pearson_delta", "pr_auc", "overlap_at_N", "de_spearman_lfc_sig", "de_spearman_sig",
        "de_nsig_counts_real", "de_nsig_counts_pred"]
print(f"units compared: {len(m)} (both cell types x 14 compounds)")
for c in cols:
    print(f"{c:22s} max |new - old| = {float(np.nanmax(np.abs(m[c + '_new'] - m[c + '_old']))):.3g}")
