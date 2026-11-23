#!/usr/bin/env python
"""Build the training and inference inputs for the State baseline (Setting 1 hold-out).

Hold-out rule (same split as the Stack evaluation):
- the 12 main compounds x {B lineage, myeloid}: removed from the file entirely (these are the 24 test units)
- the 2 positive controls (Belinostat, Dabrafenib) x {B lineage, myeloid}: kept in the file but assigned to
  cell-load's val/test split, so they are never trained on. Their loss is the validation signal used to pick
  the checkpoint; they are not part of the 24-unit main table.
- everything else is training data: T/NK cells under all 14 compounds + DMSO, and the DMSO B-lineage and
  myeloid cells (the query cells, which State needs as the basal/control population of those cell types).

So at test time every (compound, cell type) combination is unseen, but every compound has been seen (in T/NK
cells) and every cell type has been seen (under DMSO). State encodes compounds as one-hot vectors, so it can
only be asked about compounds it saw in training; this design satisfies that.

Gene space: the 11,018 genes shared by the dataset and Stack's gene list, i.e. exactly the universe the
evaluation uses. Per cell: normalize_total to the median library size of the training cells, log1p
(the convention of `state tx preprocess_train`); obsm["X_hvg"] = 2,000 HVGs (seurat flavour) computed on the
training cells only.

Outputs (under data/state/):
  train/pbmc.h5ad      training/validation file read by cell-load (cells grouped by cell type, then compound)
  pbmc.toml            cell-load split config
  infer_input.h5ad     query DMSO cells, plus one copy of them labelled with each of the 14 compounds
  build_info.json      cell counts, target_sum, HVG list size

Run with .venv-eval (anndata 0.11.4): it writes obs/_index and var/_index as plain string datasets, which
cell-load reads with h5py; anndata 0.13 writes them as nullable-string groups that cell-load cannot read.
"""
from __future__ import annotations

import os
import json
import pickle
import re
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

ROOT = Path(os.environ.get("P6_ROOT", "/content/p6state"))
OUT = ROOT / "data/state"
CTRL_RAW = "Dimethyl Sulfoxide"
PCS = ["Belinostat", "Dabrafenib"]
CT_SHORT = {"t cell": "t_nk", "lymphocyte of b lineage": "b_lineage", "myeloid leukocyte": "myeloid"}
HELD = ["b_lineage", "myeloid"]


def sanitize(v: str) -> str:
    """Same rule as scripts/03_evaluate.py (and stack-generation) uses for file names."""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(v)) or "split"


def add_gene_name(path: Path, names) -> None:
    """cell-load reads gene names with h5py; anndata 0.13 writes var/_index as a group it cannot read."""
    with h5py.File(path, "r+") as f:
        if "gene_name" not in f["var"]:
            f["var"].create_dataset("gene_name", data=np.array(list(names), dtype=object),
                                    dtype=h5py.string_dtype(encoding="utf-8"))


def main() -> None:
    (OUT / "train").mkdir(parents=True, exist_ok=True)
    full = ad.read_h5ad(ROOT / "data/openproblems_donor1.h5ad")
    model_genes = {str(g).upper() for g in pickle.load(open(ROOT / "model/basecount_1000per_15000max.pkl", "rb"))}
    keep = np.array([str(g).upper() in model_genes for g in full.var_names])
    full = full[:, keep].copy()
    print("genes kept (shared with Stack list):", full.n_vars)

    o = full.obs
    o["ct"] = o["broad_cell_class"].astype(str).map(CT_SHORT)
    o["pert"] = o["sm_name"].astype(str).map(sanitize)
    o["plate"] = o["plate_name"].astype(str)
    ctrl = sanitize(CTRL_RAW)
    held_ct = o["ct"].isin(HELD).to_numpy()
    is_ctrl = (o["pert"] == ctrl).to_numpy()
    is_pc = o["sm_name"].astype(str).isin(PCS).to_numpy()
    test_mask = held_ct & ~is_ctrl & ~is_pc          # 12 compounds x 2 held-out types: removed
    val_mask = held_ct & is_pc                         # positive controls x 2 held-out types: val split
    train_mask = ~test_mask & ~val_mask
    assert not (train_mask & held_ct & ~is_ctrl).any(), "a held-out treated cell would be trained on"

    X = sp.csr_matrix(full.X, dtype=np.float32)
    lib = np.asarray(X.sum(1)).ravel()
    target_sum = float(np.median(lib[train_mask]))
    print(f"target_sum (median library of training cells, shared genes) = {target_sum:.1f}")

    def lognorm(rows: np.ndarray) -> sp.csr_matrix:
        sub = X[rows]
        l = lib[rows].copy()
        l[l == 0] = 1
        m = sp.diags((target_sum / l).astype(np.float32)) @ sub
        m = sp.csr_matrix(m, dtype=np.float32)
        m.data = np.log1p(m.data)
        return m

    keep_rows = np.where(train_mask | val_mask)[0]
    a = ad.AnnData(X=lognorm(keep_rows), obs=o.iloc[keep_rows][["ct", "pert", "plate", "sm_name", "broad_cell_class"]].copy(),
                   var=pd.DataFrame(index=full.var_names.astype(str)))
    a.obs["split_role"] = np.where(val_mask[keep_rows], "val_pc", "train")

    # HVGs on training cells only
    tr = a[a.obs["split_role"] == "train"].copy()
    sc.pp.highly_variable_genes(tr, n_top_genes=2000, flavor="seurat")
    hv = tr.var["highly_variable"].to_numpy()
    a.var["highly_variable"] = hv
    a.obsm["X_hvg"] = np.asarray(a.X[:, hv].todense(), dtype=np.float32)

    # cell-load wants cells grouped contiguously by (cell type, perturbation)
    order = np.lexsort((a.obs["pert"].to_numpy(), a.obs["ct"].to_numpy()))
    a = a[order].copy()
    for c in ["ct", "pert", "plate", "split_role"]:
        a.obs[c] = pd.Categorical(a.obs[c].astype(str))
    p_train = OUT / "train/pbmc.h5ad"
    a.write_h5ad(p_train)
    add_gene_name(p_train, a.var_names)

    pcs = ", ".join(f'"{sanitize(p)}"' for p in PCS)
    (OUT / "pbmc.toml").write_text(f'''[datasets]
pbmc = "{OUT / 'train'}/"

[training]
pbmc = "train"

[zeroshot]

[fewshot]

[fewshot."pbmc.b_lineage"]
val = [{pcs}]
test = [{pcs}]

[fewshot."pbmc.myeloid"]
val = [{pcs}]
test = [{pcs}]
''')

    # inference input: query DMSO cells of the held-out types, once as controls and once per compound
    q = a[(a.obs["ct"].isin(HELD)) & (a.obs["pert"] == ctrl)].copy()
    drugs = sorted(d for d in a.obs["pert"].unique() if d != ctrl)
    parts = []
    for lab in [ctrl] + drugs:
        b = q.copy()
        b.obs["pert"] = lab
        b.obs_names = [f"{lab}|{n}" for n in q.obs_names]
        parts.append(b)
    inf = ad.concat(parts, axis=0, merge="same")
    inf.obs["pert"] = pd.Categorical(inf.obs["pert"].astype(str))
    inf.obs["ct"] = pd.Categorical(inf.obs["ct"].astype(str))
    inf.obsm["X_hvg"] = np.asarray(inf.obsm["X_hvg"], dtype=np.float32)
    p_inf = OUT / "infer_input.h5ad"
    inf.write_h5ad(p_inf)
    add_gene_name(p_inf, inf.var_names)

    info = {
        "n_genes": int(a.n_vars), "n_hvg": int(hv.sum()), "target_sum": target_sum,
        "train_log1p_max": float(a.X[(a.obs["split_role"] == "train").to_numpy()].max()),
        "cells_removed_test": int(test_mask.sum()), "cells_val_pc": int(val_mask.sum()),
        "cells_train": int(train_mask.sum()),
        "train_cells_by_ct_pert": a.obs[a.obs["split_role"] == "train"].groupby(["ct", "pert"], observed=True).size()
            .reset_index().astype(str).values.tolist(),
        "infer_rows": int(inf.n_obs), "query_cells": int(q.n_obs), "compounds": drugs,
    }
    (OUT / "build_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "train_cells_by_ct_pert"}, indent=1))


if __name__ == "__main__":
    main()
