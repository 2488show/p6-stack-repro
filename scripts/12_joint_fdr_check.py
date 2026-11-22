#!/usr/bin/env python
"""Why did the hand-computed DE count differ for one unit? pdex (used by cell-eval) applies
Benjamini-Hochberg once over all (compound, gene) tests of an evaluation run, not per compound.
This script recomputes the real DE counts both ways for the myeloid cells and compares them
with the pipeline's counts.
"""
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import false_discovery_control, mannwhitneyu

CTRL, PERT, CT = "Dimethyl Sulfoxide", "sm_name", "broad_cell_class"
full = ad.read_h5ad("data/openproblems_donor1.h5ad")
model_genes = {str(g).upper() for g in pickle.load(open("model/basecount_1000per_15000max.pkl", "rb"))}
full = full[:, [g for g in full.var_names if g.upper() in model_genes]].copy()
full.X = full.X.astype(np.float32)
sc.pp.normalize_total(full, target_sum=1e4)
sc.pp.log1p(full)
ref = pd.read_csv("results/eval/T5_gen0_allcells_seed0_per_pert.csv")

for ct in ["myeloid leukocyte", "lymphocyte of b lineage"]:
    sub = full[full.obs[CT] == ct].copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=2000, flavor="seurat")
    hv = sub.var["highly_variable"].to_numpy()
    xb = sub[(sub.obs[PERT] == CTRL).to_numpy()].X[:, hv].toarray()
    drugs = sorted(d for d in sub.obs[PERT].astype(str).unique() if d != CTRL)
    P = {}
    for d in drugs:
        xa = sub[(sub.obs[PERT] == d).to_numpy()].X[:, hv].toarray()
        P[d] = np.array([mannwhitneyu(xa[:, j], xb[:, j], alternative="two-sided").pvalue
                         if (xa[:, j].any() or xb[:, j].any()) else 1.0 for j in range(xa.shape[1])])
    joint = false_discovery_control(np.concatenate([P[d] for d in drugs]), method="bh").reshape(len(drugs), -1)
    no_pc = [d for d in drugs if d not in ("Belinostat", "Dabrafenib")]
    joint12 = false_discovery_control(np.concatenate([P[d] for d in no_pc]), method="bh").reshape(len(no_pc), -1)
    print(f"\n{ct}: real DE genes at FDR<0.05")
    print(f"{'compound':26s} per-compound BH   joint BH (14)   joint BH (12, no pos. controls)   pipeline")
    for i, d in enumerate(drugs):
        per = int((false_discovery_control(P[d], method='bh') < 0.05).sum())
        j14 = int((joint[i] < 0.05).sum())
        j12 = int((joint12[no_pc.index(d)] < 0.05).sum()) if d in no_pc else -1
        pipe = int(ref[(ref.cell_type == ct) & (ref.perturbation == d) & (ref.method == "stack")]["de_nsig_counts_real"].iloc[0])
        print(f"{d:26s} {per:8d} {j14:14d} {j12:22d} {pipe:22d}")
