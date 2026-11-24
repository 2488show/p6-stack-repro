#!/usr/bin/env python
"""Independent re-computation of Pearson Delta for the 3-donor run (numpy/scanpy only, no cell-eval,
no helpers from 43_evaluate_full.py). Uses the all-cells runs (no random downsampling), so the
numbers must agree with results/eval_full/F_<design>_allcells_seed0_per_pert.csv.
"""
import pickle
import re

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr

CTRL, PERT, CT = "Dimethyl Sulfoxide", "sm_name", "broad_cell_class"
PROMPT = {"T": "t cell", "B": "lymphocyte of b lineage", "M": "myeloid leukocyte"}
UNITS = [("d2_T", "myeloid leukocyte", "Palbociclib"), ("d1_B", "t cell", "Penfluridol"),
         ("d3_M", "lymphocyte of b lineage", "Crizotinib")]
model = {str(g).upper() for g in pickle.load(open("model/basecount_1000per_15000max.pkl", "rb"))}


def lognorm(a):
    a = a.copy()
    a.X = a.X.astype(np.float32)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


for design, ct, drug in UNITS:
    donor, p = design.split("_")
    full = ad.read_h5ad(f"data/full/{donor}.h5ad")
    full = full[:, [g for g in full.var_names if g.upper() in model]].copy()
    full = lognorm(full)
    sub = full[full.obs[CT] == ct].copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=2000, flavor="seurat")
    hv = sub.var["highly_variable"].to_numpy()
    genes = [g.upper() for g in sub.var_names[hv]]

    def m(a, mask):
        return np.asarray(a[mask].X[:, hv].mean(axis=0)).ravel()

    real = m(sub, (sub.obs[PERT] == drug).to_numpy()) - m(sub, (sub.obs[PERT] == CTRL).to_numpy())
    pr = full[full.obs[CT] == PROMPT[p]]
    copy = m(pr, (pr.obs[PERT] == drug).to_numpy()) - m(pr, (pr.obs[PERT] == CTRL).to_numpy())

    def stack_mean(d):
        q = ad.read_h5ad(f"out/full/{design}/compact/" + re.sub(r"[^0-9A-Za-z_.-]+", "_", d) + ".h5ad")
        q = q[(q.obs[CT] == ct).to_numpy()].copy()
        q.var_names = [str(g).upper() for g in q.var_names]
        q = q[:, [g.upper() for g in full.var_names]].copy()   # same universe as the data
        q = lognorm(q)
        pos = {g: i for i, g in enumerate(q.var_names)}
        return np.asarray(q.X[:, [pos[g] for g in genes]].mean(axis=0)).ravel()

    st = stack_mean(drug) - stack_mean(CTRL)
    ref = pd.read_csv(f"results/eval_full/F_{design}_allcells_seed0_per_pert.csv")

    def refval(method):
        r = ref[(ref.cell_type == ct) & (ref.perturbation == drug) & (ref.method == method)]
        return float(r["pearson_delta"].iloc[0])

    print(f"{design} / {ct} / {drug}")
    print(f"  stack       by hand {pearsonr(real, st)[0]:.6f}   pipeline {refval('stack'):.6f}")
    print(f"  prompt_copy by hand {pearsonr(real, copy)[0]:.6f}   pipeline {refval('prompt_copy'):.6f}")
