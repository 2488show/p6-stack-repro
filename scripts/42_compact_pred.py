"""Shrink stack-generation outputs so they can be pulled off the Colab runtime.

Keeps only the genes shared with the dataset (the evaluation universe; library size is computed over
it in the evaluation anyway), stores X as sparse float32 with gzip, and keeps the obs columns the
evaluation needs. Usage: 42_compact_pred.py <gen_dir> <donor_h5ad> <genelist.pkl> <out_dir>
"""
import os
import pickle
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

gen_dir, data, genelist, out_dir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
model_genes = [str(g).upper() for g in pickle.load(open(genelist, "rb"))]
data_genes = {g.upper() for g in ad.read_h5ad(data, backed="r").var_names}
cols = np.array([i for i, g in enumerate(model_genes) if g in data_genes])
total_in = total_out = 0
for f in sorted(os.listdir(gen_dir)):
    if not f.endswith(".h5ad"):
        continue
    p = ad.read_h5ad(os.path.join(gen_dir, f))
    assert [str(g).upper() for g in p.var_names[:50]] == model_genes[:50]
    X = p.X[:, cols]
    X = sp.csr_matrix(X.astype(np.float32))
    keep = [c for c in ["broad_cell_class", "fine_cell_type", "sm_name", "gen_logit"] if c in p.obs]
    q = ad.AnnData(X=X, obs=p.obs[keep].copy())
    q.var_names = [model_genes[i] for i in cols]
    q.write_h5ad(os.path.join(out_dir, f), compression="gzip")
    total_in += os.path.getsize(os.path.join(gen_dir, f))
    total_out += os.path.getsize(os.path.join(out_dir, f))
    print(f, p.shape, "->", q.shape, f"nnz frac {X.nnz / max(1, X.shape[0] * X.shape[1]):.3f}", flush=True)
print(f"compact: {total_in / 1e6:.0f} MB -> {total_out / 1e6:.0f} MB")
