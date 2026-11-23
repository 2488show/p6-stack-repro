#!/usr/bin/env python
"""Convert `state tx infer` output into one AnnData per compound for scripts/03_evaluate.py.

`state tx infer` reports "Saved count predictions to adata.X", but the values are in the training space:
log1p of counts normalised to a fixed library size (target_sum in data/state/build_info.json). They are
mapped back with expm1. No rounding: 03_evaluate.py re-normalises every cell to 1e4 over the shared genes
and log1p-transforms it, so the scale does not matter. Optional clipping to the training range (--clip)
is off by default; the number of predicted values above the training maximum is recorded either way.
Reads the prediction file in backed mode, one compound at a time (the T4 runtime has 12 GB RAM).
"""
from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

CT_LONG = {"t_nk": "t cell", "b_lineage": "lymphocyte of b lineage", "myeloid": "myeloid leukocyte"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--build-info", default=os.path.join(os.environ.get("P6_ROOT", "/content/p6state"), "data/state/build_info.json"))
    ap.add_argument("--clip", type=float, default=None, help="clip log1p predictions to this maximum")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    info = json.load(open(args.build_info))
    tmax = info["train_log1p_max"]

    p = ad.read_h5ad(args.pred, backed="r")
    pert = p.obs["pert"].astype(str).to_numpy()
    ct = p.obs["ct"].astype(str).map(CT_LONG).to_numpy()
    # plain object-dtype indices (the infer output, written by anndata 0.13, uses nullable strings)
    var = pd.DataFrame(index=pd.Index([str(g) for g in p.var_names], dtype=object))
    names = np.array([str(n) for n in p.obs_names], dtype=object)
    stats = {"shape": [int(p.n_obs), int(p.n_vars)], "train_log1p_max": tmax, "target_sum": info["target_sum"],
             "clip": args.clip, "min": np.inf, "max": -np.inf, "n_values_above_train_max": 0,
             "n_cells_with_value_above_train_max": 0, "n_zero": 0, "per_file": {}}
    sums = []
    for lab in sorted(set(pert)):
        rows = np.where(pert == lab)[0]
        X = p.X[rows[0]:rows[-1] + 1] if np.all(np.diff(rows) == 1) else p.X[rows]
        X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
        stats["min"] = min(stats["min"], float(X.min()))
        stats["max"] = max(stats["max"], float(X.max()))
        above = X > tmax
        stats["n_values_above_train_max"] += int(above.sum())
        stats["n_cells_with_value_above_train_max"] += int(above.any(1).sum())
        stats["n_zero"] += int((X <= 0).sum())
        del above
        if args.clip is not None:
            np.clip(X, None, args.clip, out=X)
        np.clip(X, 0, None, out=X)
        np.expm1(X, out=X)
        sums.append(X.sum(1))
        a = ad.AnnData(X=sp.csr_matrix(X), var=var,
                       obs=pd.DataFrame({"broad_cell_class": pd.Categorical(ct[rows])}, index=pd.Index(names[rows], dtype=object)))
        a.write_h5ad(os.path.join(args.outdir, f"{lab}.h5ad"))
        stats["per_file"][lab] = {k: int(v) for k, v in pd.Series(ct[rows]).value_counts().items()}
        del X, a
    s = np.concatenate(sums)
    stats["cell_sum_expm1_median"] = float(np.median(s))
    stats["cell_sum_expm1_q01_q99"] = [float(np.quantile(s, 0.01)), float(np.quantile(s, 0.99))]
    stats["frac_zero"] = stats.pop("n_zero") / (stats["shape"][0] * stats["shape"][1])
    json.dump(stats, open(os.path.join(args.outdir, "conversion_stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_file"}, indent=1))
    print("files:", len(stats["per_file"]), "example:", next(iter(stats["per_file"].items())))


if __name__ == "__main__":
    main()
