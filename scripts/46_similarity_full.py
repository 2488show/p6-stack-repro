#!/usr/bin/env python
"""Does Stack's gain over copying the prompt depend on how different the target is from the prompt?

For every (donor, prompt class, target class): Pearson correlation between the DMSO pseudobulks of the
prompt and the target class (log-normalised, on the target's 2,000 HVGs, the genes that are scored),
next to the design's mean Pearson Delta of Stack minus prompt_copy (results/summary_full/by_design_target.csv).
"""
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr

SH = {"t cell": "T/NK", "lymphocyte of b lineage": "B", "myeloid leukocyte": "Myeloid"}
rows = []
for donor in ["d1", "d2", "d3"]:
    a = ad.read_h5ad(f"data/full/{donor}.h5ad")
    a = a[a.obs["sm_name"].astype(str) == "Dimethyl Sulfoxide"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    bc = a.obs["broad_cell_class"].astype(str).to_numpy()
    for tgt in SH:
        t = a[bc == tgt].copy()
        sc.pp.highly_variable_genes(t, n_top_genes=2000, flavor="seurat")
        g = t.var["highly_variable"].to_numpy()
        pb_t = np.asarray(t.X[:, g].mean(0)).ravel()
        for p in SH:
            if p == tgt:
                continue
            pb_p = np.asarray(a[bc == p].X[:, g].mean(0)).ravel()
            rows.append({"design": f"{donor}_{SH[p][0]}", "donor": donor, "prompt": SH[p], "target": SH[tgt],
                         "dmso_similarity": pearsonr(pb_p, pb_t)[0]})
sim = pd.DataFrame(rows)
bd = pd.read_csv("results/summary_full/by_design_target.csv")
m = bd.merge(sim, on=["design", "prompt", "target"])
m["stack_minus_copy"] = m["stack"] - m["prompt_copy"]
cols = ["design", "prompt", "target", "dmso_similarity", "stack", "prompt_copy", "stack_minus_copy", "stack_wins_vs_copy", "n_units"]
m = m[cols].sort_values("dmso_similarity")
pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
print(m.to_string(index=False))
r, p = spearmanr(m["dmso_similarity"], m["stack_minus_copy"])
r2, p2 = spearmanr(m["dmso_similarity"], m["prompt_copy"])
r3, p3 = spearmanr(m["dmso_similarity"], m["stack"])
print(f"\nSpearman (n={len(m)}): similarity vs Stack-minus-copy rho={r:.2f} p={p:.3f}; "
      f"vs prompt_copy rho={r2:.2f} p={p2:.3f}; vs stack rho={r3:.2f} p={p3:.3f}")
m.to_csv("results/summary_full/similarity_vs_gain.csv", index=False)
