"""Build the 3-donor Setting 1 inputs from the full OpenProblems file.

For every donor: the same 14 compounds as the Stack example (12 + positive controls Belinostat,
Dabrafenib) plus DMSO, taken from the plates on which that donor received those compounds; raw
counts; obs = the example's columns + fine_cell_type (the original 6 OpenProblems cell types) +
broad_cell_class (mapped exactly as in the example). Check: donor 1 must reproduce the example file.

Designs: every donor x every broad class as prompt (3 x 3 = 9). Context = the prompt class under all
compounds + DMSO; test = DMSO cells of the other two classes, capped at --query-cap per class
(seeded) to keep generation time bounded on a T4.
Usage: 40_build_full.py <full.h5ad> <example_donor1.h5ad> <outdir> [--query-cap 1000]
"""
import argparse
import json
import os

import anndata as ad
import h5py
from anndata.io import read_elem
import numpy as np
import pandas as pd
import scipy.sparse as sp

BROAD = {"T cells CD4+": "t cell", "T cells CD8+": "t cell", "T regulatory cells": "t cell",
         "NK cells": "t cell", "B cells": "lymphocyte of b lineage", "Myeloid cells": "myeloid leukocyte"}
SHORT = {"t cell": "T", "lymphocyte of b lineage": "B", "myeloid leukocyte": "M"}
CTRL = "Dimethyl Sulfoxide"

ap = argparse.ArgumentParser()
ap.add_argument("full")
ap.add_argument("example")
ap.add_argument("outdir")
ap.add_argument("--query-cap", type=int, default=1000)
ap.add_argument("--cell-type-col", default="cell_type")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

ex = ad.read_h5ad(args.example)
compounds = sorted(ex.obs["sm_name"].astype(str).unique())
assert CTRL in compounds and len(compounds) == 15
# Read obs/var only, then pull the needed rows of the raw-count matrix straight from HDF5:
# anndata's backed mode would load layers["counts"] (~7 GB) into RAM.
H5 = h5py.File(args.full, "r")
o = read_elem(H5["obs"])
var_names = np.asarray(read_elem(H5["var"]).index.astype(str))
CNT = H5["layers"]["counts"] if "layers" in H5 and "counts" in H5["layers"] else H5["X"]
print("full:", o.shape, "genes", len(var_names), "counts from", CNT.name)


def load_rows(rows: np.ndarray) -> ad.AnnData:
    """Rows (sorted) of the CSR count matrix, read in contiguous runs."""
    indptr = CNT["indptr"][:]
    runs = np.split(rows, np.where(np.diff(rows) != 1)[0] + 1)
    data, ind, ptr = [], [], [0]
    for r in runs:
        a, b = indptr[r[0]], indptr[r[-1] + 1]
        data.append(CNT["data"][a:b].astype(np.float32))
        ind.append(CNT["indices"][a:b])
        ptr.extend((indptr[r[0] + 1:r[-1] + 2] - a + ptr[-1]).tolist())
    X = sp.csr_matrix((np.concatenate(data), np.concatenate(ind), np.array(ptr)), shape=(len(rows), len(var_names)))
    out = ad.AnnData(X=X, obs=o.iloc[rows].copy())
    out.var_names = list(var_names)
    print(f"  read {len(rows)} rows in {len(runs)} runs")
    return out


for c in o.columns:
    if o[c].nunique() < 40:
        print(c, o[c].astype(str).value_counts().to_dict())
ctcol = args.cell_type_col
print("var head", list(var_names[:5]))

info = {"compounds": compounds, "query_cap": args.query_cap, "donors": {}}
designs = []
donors = sorted(o["donor_id"].astype(str).unique())
for donor in donors:
    dmask = (o["donor_id"].astype(str) == donor).to_numpy()
    drug12 = [c for c in compounds if c not in (CTRL, "Belinostat", "Dabrafenib")]
    plates = sorted(o.loc[dmask & o["sm_name"].astype(str).isin(drug12).to_numpy(), "plate_name"].astype(str).unique())
    m = dmask & o["plate_name"].astype(str).isin(plates).to_numpy() & o["sm_name"].astype(str).isin(compounds).to_numpy()
    sub = load_rows(np.where(m)[0])
    X = sp.csr_matrix(sub.X)
    xs = X[:2000].toarray()
    assert np.allclose(xs, np.round(xs)), "X is not raw counts"
    sub.X = X.astype(np.float32)
    sub.obs["fine_cell_type"] = sub.obs[ctcol].astype(str)
    sub.obs["broad_cell_class"] = sub.obs["fine_cell_type"].map(BROAD)
    assert sub.obs["broad_cell_class"].notna().all(), sub.obs["fine_cell_type"].unique()
    for c in sub.obs.columns:
        if sub.obs[c].dtype == object:
            sub.obs[c] = sub.obs[c].astype(str)
    sub.var_names = [str(g) for g in sub.var_names]
    tag = donor.replace("donor_", "d").replace("Donor ", "d")
    f = os.path.join(args.outdir, f"{tag}.h5ad")
    sub.write_h5ad(f)
    tab = pd.crosstab(sub.obs["sm_name"], sub.obs["broad_cell_class"])
    tab.to_csv(os.path.join(args.outdir, f"{tag}_cells_per_compound.csv"))
    info["donors"][donor] = {"plates": plates, "n_cells": int(sub.n_obs), "n_genes": int(sub.n_vars),
                             "cells_per_class": sub.obs["broad_cell_class"].value_counts().to_dict()}
    print(donor, plates, sub.shape)
    print(tab.to_string())

    if tag == "d1":
        a = tab.sort_index()
        b = pd.crosstab(ex.obs["sm_name"].astype(str), ex.obs["broad_cell_class"].astype(str)).sort_index()
        same_tab = a.equals(b.loc[a.index, a.columns]) if set(a.index) == set(b.index) else False
        same_cells = set(sub.obs_names) == set(ex.obs_names)
        same_genes = list(sub.var_names) == list(ex.var_names)
        same_x = False
        if same_cells and same_genes:
            xa = sp.csr_matrix(sub[ex.obs_names[:1000]].X).toarray()
            xb = sp.csr_matrix(ex.X[:1000]).toarray()
            same_x = bool(np.array_equal(xa, xb))
        info["donor1_check"] = {"same_table": bool(same_tab), "same_cells": bool(same_cells),
                                "same_genes": bool(same_genes), "same_counts_first1000": same_x}
        diff = (a - b.reindex(index=a.index, columns=a.columns)) if set(a.index) == set(b.index) else None
        info["donor1_check"]["n_cells_full_vs_example"] = [int(sub.n_obs), int(ex.n_obs)]
        if diff is not None:
            info["donor1_check"]["max_abs_cell_count_diff"] = int(diff.abs().to_numpy().max())
            print("cells: this build minus example\n" + diff.to_string())
        print("DONOR1 CHECK", info["donor1_check"])

    rng = np.random.default_rng(0)
    for p in ["t cell", "lymphocyte of b lineage", "myeloid leukocyte"]:
        name = f"{tag}_{SHORT[p]}"
        bc = sub.obs["broad_cell_class"].to_numpy()
        ctx = sub[bc == p].copy()
        idx = []
        for q in sorted(set(bc) - {p}):
            cand = np.where((bc == q) & (sub.obs["sm_name"].to_numpy() == CTRL))[0]
            if len(cand) > args.query_cap:
                cand = np.sort(rng.choice(cand, args.query_cap, replace=False))
            idx.append(cand)
        test = sub[np.concatenate(idx)].copy()
        ctx.write_h5ad(os.path.join(args.outdir, f"{name}_ctx.h5ad"))
        test.write_h5ad(os.path.join(args.outdir, f"{name}_test.h5ad"))
        open(os.path.join(args.outdir, f"{name}.donorfile"), "w").write(f + "\n")
        info.setdefault("designs", {})[name] = {"prompt": p, "ctx_cells": int(ctx.n_obs),
                                                "test_cells": test.obs["broad_cell_class"].value_counts().to_dict()}
        designs.append(name)

order = ["d2_T", "d3_T", "d1_B", "d1_M", "d2_B", "d2_M", "d3_B", "d3_M", "d1_T"]
order = [d for d in order if d in designs] + [d for d in designs if d not in order]
open(os.path.join(args.outdir, "designs.txt"), "w").write("\n".join(order) + "\n")
json.dump(info, open(os.path.join(args.outdir, "build_info.json"), "w"), indent=1, default=str)
print("designs", order)
