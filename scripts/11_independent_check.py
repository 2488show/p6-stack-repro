#!/usr/bin/env python
"""Independent re-computation of a few reported numbers, without cell-eval and without the
helper functions of 03_evaluate.py. Uses the all-cells variant because it involves no random
downsampling, so the numbers must agree with results/eval/T5_gen0_allcells_seed0_per_pert.csv.

Checks, for two (cell type, compound) units:
  - Pearson Delta of Stack and of tcell_copy (numpy only)
  - number of real DE genes (scipy Mann-Whitney U + statsmodels BH) vs pdex's count
"""
import pickle
import re

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu, pearsonr
from statsmodels.stats.multitest import multipletests

CTRL, PERT, CT = "Dimethyl Sulfoxide", "sm_name", "broad_cell_class"
UNITS = [("lymphocyte of b lineage", "Porcn Inhibitor III"), ("myeloid leukocyte", "Penfluridol")]

full = ad.read_h5ad("data/openproblems_donor1.h5ad")
model_genes = [str(g).upper() for g in pickle.load(open("model/basecount_1000per_15000max.pkl", "rb"))]
shared = [g for g in full.var_names if g.upper() in set(model_genes)]
full = full[:, shared].copy()
full.X = full.X.astype(np.float32)
sc.pp.normalize_total(full, target_sum=1e4)
sc.pp.log1p(full)

ref = pd.read_csv("results/eval/T5_gen0_allcells_seed0_per_pert.csv")


def load_pred(drug: str) -> ad.AnnData:
    p = ad.read_h5ad("out/tutorial_seed0/" + re.sub(r"[^0-9A-Za-z_.-]+", "_", drug) + ".h5ad")
    p.var_names = [str(g).upper() for g in p.var_names]
    p = p[:, [g.upper() for g in shared]].copy()
    p.X = p.X.astype(np.float32)
    sc.pp.normalize_total(p, target_sum=1e4)
    sc.pp.log1p(p)
    return p


pred_ctrl_all = load_pred(CTRL)
for ct, drug in UNITS:
    sub = full[full.obs[CT] == ct].copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=2000, flavor="seurat")
    hv = sub.var["highly_variable"].to_numpy()

    def m(a, mask):
        return np.asarray(a[mask].X[:, hv].mean(axis=0)).ravel()

    real_d = m(sub, (sub.obs[PERT] == drug).to_numpy())
    real_c = m(sub, (sub.obs[PERT] == CTRL).to_numpy())
    p = load_pred(drug)
    pd_stack = pearsonr(real_d - real_c,
                        m(p, (p.obs[CT] == ct).to_numpy()) - m(pred_ctrl_all, (pred_ctrl_all.obs[CT] == ct).to_numpy()))[0]
    t = full[full.obs[CT] == "t cell"]
    pd_tcell = pearsonr(real_d - real_c,
                        m(t, (t.obs[PERT] == drug).to_numpy()) - m(t, (t.obs[PERT] == CTRL).to_numpy()))[0]

    xa = sub[(sub.obs[PERT] == drug).to_numpy()].X[:, hv].toarray()
    xb = sub[(sub.obs[PERT] == CTRL).to_numpy()].X[:, hv].toarray()
    pvals = np.array([mannwhitneyu(xa[:, j], xb[:, j], alternative="two-sided").pvalue
                      if (xa[:, j].any() or xb[:, j].any()) else 1.0 for j in range(xa.shape[1])])
    nsig = int((multipletests(pvals, method="fdr_bh")[1] < 0.05).sum())

    def refval(method, col):
        r = ref[(ref.cell_type == ct) & (ref.perturbation == drug) & (ref.method == method)]
        return float(r[col].iloc[0])

    print(f"{ct} / {drug}")
    print(f"  Pearson Delta stack      by hand {pd_stack:.6f}   pipeline {refval('stack', 'pearson_delta'):.6f}")
    print(f"  Pearson Delta tcell_copy by hand {pd_tcell:.6f}   pipeline {refval('tcell_copy', 'pearson_delta'):.6f}")
    print(f"  real DE genes (FDR<0.05) by hand {nsig}   pipeline {int(refval('stack', 'de_nsig_counts_real'))}")
