"""Inspect the OpenProblems donor-1 example data before running anything."""
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

a = ad.read_h5ad("data/openproblems_donor1.h5ad")
print(a)
print("raw is None:", a.raw is None)
X = a.X
print("X type:", type(X), X.dtype, "sparse:", sp.issparse(X))
sub = X[:2000]
sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
print("X min/max:", sub.min(), sub.max(), "integer-valued:", bool(np.allclose(sub, np.round(sub))))
print("row sums (first 5):", sub.sum(1)[:5])
print("var head:", list(a.var_names[:8]), "n_vars", a.n_vars)
print("var columns:", list(a.var.columns))
print("obs columns:", list(a.obs.columns))
for c in a.obs.columns:
    n = a.obs[c].nunique()
    if n <= 40:
        print(f"\n[{c}] n={n}")
        print(a.obs[c].value_counts().to_string())
    else:
        print(f"\n[{c}] n={n} (e.g. {list(a.obs[c].astype(str).unique()[:5])})")

ct = "broad_cell_class"
tab = pd.crosstab(a.obs["sm_name"], a.obs[ct])
print("\nN drugs:", tab.shape[0])
print("cells per drug x cell class: quantiles")
print(tab.describe().T.to_string())
print("\ndrugs with >=1 cell per class:")
print((tab > 0).sum().to_string())
print("\ndrugs with >=50 cells per class:")
print((tab >= 50).sum().to_string())
print("\nDMSO row:")
print(tab.loc["Dimethyl Sulfoxide"].to_string())
tab.to_csv("results/cells_per_drug_celltype.csv")

for f in ["data/openproblems_context.h5ad", "data/openproblems_test.h5ad"]:
    b = ad.read_h5ad(f, backed="r")
    print("\n", f, b.shape)
    print(b.obs[ct].value_counts().to_string())
    print("n sm_name:", b.obs["sm_name"].nunique())
