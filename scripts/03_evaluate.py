#!/usr/bin/env python
"""Evaluate Stack in-context predictions against real held-out cells, next to simple baselines.

Protocol (paper Methods, "Setting 1"; deviations are listed in the README):
- gene universe: genes present in both the dataset and Stack's 15,012-gene list
- every matrix is library-size normalised to 1e4 over that universe, then log1p
- metrics on the top-2,000 HVGs of the held-out cell type (real target + query cells)
- per (cell type, compound) each predicted group is downsampled to the number of real cells
- cell-eval v0.6.6, Wilcoxon + Benjamini-Hochberg, profile="full"

Methods compared
- stack         Stack generation; control = Stack's own DMSO generation ("synthetic control")
- tcell_copy    the prompt T/NK cells themselves; control = DMSO T/NK cells
- perturb_mean  query cells + (mean T/NK compound - mean T/NK DMSO); control = query cells
- no_effect     query cells unchanged; control = query cells
- real_half     (only with --split-real) an independent half of the real cells = noise ceiling
- state         State transition model trained on this split (scripts/21-24, --state-dir);
                control = State's own prediction for DMSO (its "synthetic control", same rule as stack)
- state_realctrl  the same State predictions, control = real query cells (sensitivity only)

With --split-real every method is scored against a random half of the real cells, and the
other half is scored as a prediction. That shows how high the metrics can go at this sample size.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import re

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import scipy.sparse as sp

CTRL = "Dimethyl Sulfoxide"
PERT = "sm_name"
CT = "broad_cell_class"
PROMPT_CT = "t cell"
TARGET_SUM = 1e4


def sanitize(v: str) -> str:
    """Same rule stack-generation uses to name its output files."""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(v)) or "split"


def lognorm_cols(X, cols: np.ndarray) -> np.ndarray:
    """log1p(counts / library * 1e4) for a subset of columns; library = sum over all columns."""
    X = sp.csr_matrix(X)
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    lib[lib == 0] = 1.0
    sub = X[:, cols].toarray().astype(np.float64)
    return np.log1p(sub / lib[:, None] * TARGET_SUM).astype(np.float32)


def make_adata(groups: dict[str, np.ndarray], ctrl: np.ndarray, genes: np.ndarray) -> ad.AnnData:
    mats, labels = [ctrl], [CTRL] * ctrl.shape[0]
    for name, m in groups.items():
        mats.append(m)
        labels += [name] * m.shape[0]
    a = ad.AnnData(X=np.vstack(mats).astype(np.float32))
    a.var_names = list(genes)
    a.obs[PERT] = pd.Categorical(labels)
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    return a


def jaccard_from_de(de_real: pl.DataFrame, de_pred: pl.DataFrame, fdr: float = 0.05) -> dict[str, float]:
    out = {}
    r = de_real.filter(pl.col("fdr") < fdr).to_pandas()
    p = de_pred.filter(pl.col("fdr") < fdr).to_pandas()
    targets = set(de_real["target"].unique().to_list())
    for t in targets:
        a = set(r.loc[r["target"] == t, "feature"])
        b = set(p.loc[p["target"] == t, "feature"])
        out[t] = len(a & b) / len(a | b) if (a | b) else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/openproblems_donor1.h5ad")
    ap.add_argument("--genelist", default="model/basecount_1000per_15000max.pkl")
    ap.add_argument("--pred-dir", default=None, help="directory written by stack-generation")
    ap.add_argument("--state-dir", default=None, help="per-compound State predictions (scripts/24_state_to_eval.py)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="results/eval")
    ap.add_argument("--seed", type=int, default=0, help="seed for downsampling / splitting")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--methods", nargs="+", default=["stack", "tcell_copy", "perturb_mean", "no_effect"])
    ap.add_argument("--logit-max", type=float, default=None, help="drop generated cells with gen_logit above this")
    ap.add_argument("--no-downsample", action="store_true",
                    help="sensitivity check: keep every predicted cell instead of matching the real cell number")
    ap.add_argument("--split-real", action="store_true",
                    help="score against a random half of the real cells; add the other half as 'real_half'")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    from cell_eval import MetricsEvaluator

    rng = np.random.default_rng(args.seed)
    methods = list(args.methods)
    if args.split_real and "real_half" not in methods:
        methods.append("real_half")

    # ---- real data, restricted to genes Stack models --------------------------------------
    full = ad.read_h5ad(args.data)
    model_genes = [str(g).upper() for g in pickle.load(open(args.genelist, "rb"))]
    model_set = set(model_genes)
    data_upper = np.array([g.upper() for g in full.var_names])
    assert len(set(data_upper)) == len(data_upper), "duplicate gene symbols after upper-casing"
    keep = np.array([g in model_set for g in data_upper])
    full = full[:, keep].copy()
    genes_upper = data_upper[keep]
    print(f"gene universe: {keep.sum()} shared genes of {len(data_upper)} (dataset) / {len(model_genes)} (model)")

    obs = full.obs
    drugs = sorted(d for d in obs[PERT].astype(str).unique() if d != CTRL)
    held_out = sorted(c for c in obs[CT].astype(str).unique() if c != PROMPT_CT)

    # ---- Stack predictions: one file per compound, columns = model gene list --------------
    pos_in_model = {g: i for i, g in enumerate(model_genes)}
    pred_cols = np.array([pos_in_model[g] for g in genes_upper])
    stack_pred: dict[str, ad.AnnData] = {}
    if "stack" in methods:
        for d in drugs + [CTRL]:
            p = ad.read_h5ad(os.path.join(args.pred_dir, sanitize(d) + ".h5ad"))
            assert p.n_vars == len(model_genes)
            assert [str(g).upper() for g in p.var_names[:50]] == model_genes[:50]
            p = p[:, pred_cols].copy()
            if args.logit_max is not None and "gen_logit" in p.obs:
                p = p[p.obs["gen_logit"].to_numpy() <= args.logit_max].copy()
            stack_pred[d] = p

    # ---- State predictions: one file per compound, columns = dataset genes (shared universe) ----
    state_pred: dict[str, ad.AnnData] = {}
    if any(m.startswith("state") for m in methods):
        assert args.state_dir, "--state-dir is required for the state methods"
        for d in drugs + [CTRL]:
            p = ad.read_h5ad(os.path.join(args.state_dir, sanitize(d) + ".h5ad"))
            pos = {str(g).upper(): i for i, g in enumerate(p.var_names)}
            assert all(g in pos for g in genes_upper), "State output misses genes of the evaluation universe"
            state_pred[d] = p[:, [pos[g] for g in genes_upper]].copy()

    rows = []
    for ct in held_out:
        ct_mask = (obs[CT].astype(str) == ct).to_numpy()
        # HVGs from the real held-out cells of this type (target + query), log-normalised
        a = full[ct_mask].copy()
        a.X = a.X.astype(np.float32)
        sc.pp.normalize_total(a, target_sum=TARGET_SUM)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=args.n_hvg, flavor="seurat")
        hvg_cols = np.where(a.var["highly_variable"].to_numpy())[0]
        hvg_names = genes_upper[hvg_cols]
        del a

        def real_block(cell_type: str, drug: str) -> np.ndarray:
            m = ((obs[CT].astype(str) == cell_type) & (obs[PERT].astype(str) == drug)).to_numpy()
            return lognorm_cols(full.X[m], hvg_cols)

        def take(m: np.ndarray, n: int) -> np.ndarray:
            if args.no_downsample:
                return m
            n = min(n, m.shape[0])
            return m[rng.choice(m.shape[0], n, replace=False)]

        query_all = real_block(ct, CTRL)                   # real DMSO cells of the held-out type
        real_all = {d: real_block(ct, d) for d in drugs}
        t_ctrl = real_block(PROMPT_CT, CTRL)
        t_groups = {d: real_block(PROMPT_CT, d) for d in drugs}
        t_delta = {d: t_groups[d].mean(0) - t_ctrl.mean(0) for d in drugs}

        if args.split_real:
            def halves(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                p = rng.permutation(m.shape[0])
                h = m.shape[0] // 2
                return m[p[:h]], m[p[h:]]
            query_a, query = halves(query_all)             # a -> prediction side, b -> "real" side
            pairs = {d: halves(real_all[d]) for d in drugs}
            real_a = {d: pairs[d][0] for d in drugs}
            real_groups = {d: pairs[d][1] for d in drugs}
            base_query = query_a
        else:
            query, real_groups, base_query = query_all, real_all, query_all
            query_a, real_a = None, None

        n_real = {d: real_groups[d].shape[0] for d in drugs}
        real = make_adata(real_groups, query, hvg_names)
        real_ctrl_mean = query.mean(0)

        de_real_path = None
        for method in methods:
            if method == "stack":
                def sp_block(d):
                    p = stack_pred[d]
                    return lognorm_cols(p.X[(p.obs[CT].astype(str) == ct).to_numpy()], hvg_cols)
                pred_ctrl = sp_block(CTRL)
                groups = {d: take(sp_block(d), n_real[d]) for d in drugs}
            elif method in ("state", "state_realctrl"):
                def st_block(d):
                    p = state_pred[d]
                    return lognorm_cols(p.X[(p.obs[CT].astype(str) == ct).to_numpy()], hvg_cols)
                pred_ctrl = st_block(CTRL) if method == "state" else base_query
                groups = {d: take(st_block(d), n_real[d]) for d in drugs}
            elif method == "tcell_copy":
                pred_ctrl = take(t_ctrl, query.shape[0])
                groups = {d: take(t_groups[d], n_real[d]) for d in drugs}
            elif method == "perturb_mean":
                pred_ctrl = base_query
                groups = {d: np.clip(take(base_query, n_real[d]) + t_delta[d], 0, None) for d in drugs}
            elif method == "no_effect":
                pred_ctrl = base_query
                groups = {d: take(base_query, n_real[d]) for d in drugs}
            elif method == "real_half":
                pred_ctrl = query_a
                groups = real_a
            else:
                raise ValueError(method)

            pred = make_adata(groups, pred_ctrl, hvg_names)
            out = os.path.join(args.outdir, args.tag, f"seed{args.seed}", sanitize(ct), method)
            ev = MetricsEvaluator(
                adata_pred=pred, adata_real=real.copy(), de_real=de_real_path,
                control_pert=CTRL, pert_col=PERT, num_threads=2, outdir=out,
            )
            res, _ = ev.compute(profile="full", write_csv=True)
            if de_real_path is None:
                de_real_path = os.path.join(out, "real_de.csv")
            de_r = pl.read_csv(de_real_path, schema_overrides={"target": pl.Utf8, "feature": pl.Utf8})
            de_p = pl.read_csv(os.path.join(out, "pred_de.csv"), schema_overrides={"target": pl.Utf8, "feature": pl.Utf8})
            jac = jaccard_from_de(de_r, de_p)

            # effect-size bookkeeping on pseudobulk deltas (does the method shrink the effect?)
            pc = pred_ctrl.mean(0)
            stats = {}
            for d in drugs:
                dr = real_groups[d].mean(0) - real_ctrl_mean
                dp = groups[d].mean(0) - pc
                stats[d] = (
                    float(np.linalg.norm(dr)), float(np.linalg.norm(dp)),
                    float(dp @ dr / (dr @ dr)) if dr @ dr > 0 else float("nan"),
                )

            df = res.to_pandas()
            df["jaccard"] = df["perturbation"].map(jac)
            df["delta_norm_real"] = df["perturbation"].map({d: s[0] for d, s in stats.items()})
            df["delta_norm_pred"] = df["perturbation"].map({d: s[1] for d, s in stats.items()})
            df["delta_slope"] = df["perturbation"].map({d: s[2] for d, s in stats.items()})
            df["n_real_cells"] = df["perturbation"].map(n_real)
            df["n_pred_cells"] = df["perturbation"].map({d: g.shape[0] for d, g in groups.items()})
            df.insert(0, "method", method)
            df.insert(0, "cell_type", ct)
            df.insert(0, "seed", args.seed)
            df.insert(0, "tag", args.tag)
            rows.append(df)
            print(f"done {ct} / {method}: {df.shape}")

    allres = pd.concat(rows, ignore_index=True)
    dst = os.path.join(args.outdir, f"{args.tag}_seed{args.seed}_per_pert.csv")
    allres.to_csv(dst, index=False)
    print("wrote", dst, allres.shape)
    print("columns:", list(allres.columns))


if __name__ == "__main__":
    main()
