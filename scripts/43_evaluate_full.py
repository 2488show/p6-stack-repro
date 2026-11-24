#!/usr/bin/env python
"""Evaluate one 3-donor design (donor x prompt cell class) with the same protocol as 03_evaluate.py.

Design = one donor, one broad cell class as prompt (all compounds + DMSO), the other two broad
classes held out. Query = DMSO cells of the held-out classes (capped at build time for generation;
the real control used for scoring is every DMSO cell of the held-out class).

Methods
- stack         Stack generation (T=5); control = Stack's own DMSO generation ("synthetic control")
- prompt_copy   the prompt cells under compound d (all fine types); control = DMSO prompt cells
                (= tcell_copy in 03_evaluate.py when the prompt is T/NK)
- nearest_ct    paper baseline "nearest cell type in prompt": every query cell is matched to the prompt
                fine cell type whose DMSO pseudobulk correlates best with it (Pearson on HVGs); the
                prediction samples cells of the matched types under d in those proportions;
                control = DMSO cells of the matched types in the same proportions
- perturb_mean  query + (mean prompt d - mean prompt DMSO); control = query
- no_effect     query unchanged; control = query
- real_half     (with --split-real) independent half of the real cells = noise ceiling

Per unit every method predicts min(real cells, prompt cells under d) cells (= the real number
whenever the prompt is T/NK, as in 03_evaluate.py). Units with fewer than --min-real real cells are skipped (listed in the log and in skipped_units.csv).
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
FINE = "fine_cell_type"
TARGET_SUM = 1e4


def sanitize(v: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(v)) or "split"


def lognorm_cols(X, cols: np.ndarray) -> np.ndarray:
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
    for t in set(de_real["target"].unique().to_list()):
        a = set(r.loc[r["target"] == t, "feature"])
        b = set(p.loc[p["target"] == t, "feature"])
        out[t] = len(a & b) / len(a | b) if (a | b) else float("nan")
    return out


def rowwise_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson between every row of A (cells) and every row of B (pseudobulks)."""
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return A @ B.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="one donor, written by 40_build_full.py")
    ap.add_argument("--prompt-ct", required=True)
    ap.add_argument("--genelist", default="model/basecount_1000per_15000max.pkl")
    ap.add_argument("--pred-dir", default=None, help="compact Stack predictions (42_compact_pred.py)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="results/eval_full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--min-real", type=int, default=10)
    ap.add_argument("--methods", nargs="+",
                    default=["stack", "prompt_copy", "nearest_ct", "perturb_mean", "no_effect"])
    ap.add_argument("--no-downsample", action="store_true")
    ap.add_argument("--split-real", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    from cell_eval import MetricsEvaluator

    rng = np.random.default_rng(args.seed)
    methods = list(args.methods)
    if args.split_real and "real_half" not in methods:
        methods.append("real_half")

    full = ad.read_h5ad(args.data)
    model_set = {str(g).upper() for g in pickle.load(open(args.genelist, "rb"))}
    data_upper = np.array([g.upper() for g in full.var_names])
    assert len(set(data_upper)) == len(data_upper), "duplicate gene symbols after upper-casing"
    keep = np.array([g in model_set for g in data_upper])
    full = full[:, keep].copy()
    genes_upper = data_upper[keep]
    print(f"gene universe: {keep.sum()} shared genes of {len(data_upper)} (dataset)")

    obs = full.obs
    ct_s = obs[CT].astype(str).to_numpy()
    pert_s = obs[PERT].astype(str).to_numpy()
    fine_s = obs[FINE].astype(str).to_numpy()
    drugs_all = sorted(d for d in np.unique(pert_s) if d != CTRL)
    held_out = sorted(c for c in np.unique(ct_s) if c != args.prompt_ct)
    prompt_fine = sorted(np.unique(fine_s[ct_s == args.prompt_ct]))
    print("prompt", args.prompt_ct, "fine types", prompt_fine, "held out", held_out)

    stack_pred: dict[str, ad.AnnData] = {}
    if "stack" in methods:
        for d in drugs_all + [CTRL]:
            p = ad.read_h5ad(os.path.join(args.pred_dir, sanitize(d) + ".h5ad"))
            pos = {str(g).upper(): i for i, g in enumerate(p.var_names)}
            stack_pred[d] = p[:, [pos[g] for g in genes_upper]].copy()

    rows, skipped = [], []
    for ct in held_out:
        ct_mask = ct_s == ct
        n_real_all = {d: int((ct_mask & (pert_s == d)).sum()) for d in drugs_all}
        drugs = [d for d in drugs_all if n_real_all[d] >= args.min_real]
        for d in drugs_all:
            if d not in drugs:
                skipped.append({"cell_type": ct, "perturbation": d, "n_real": n_real_all[d]})
        print(ct, "units kept", len(drugs), "skipped", [d for d in drugs_all if d not in drugs])

        a = full[ct_mask & np.isin(pert_s, drugs + [CTRL])].copy()
        a.X = a.X.astype(np.float32)
        sc.pp.normalize_total(a, target_sum=TARGET_SUM)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=args.n_hvg, flavor="seurat")
        hvg_cols = np.where(a.var["highly_variable"].to_numpy())[0]
        hvg_names = genes_upper[hvg_cols]
        del a

        def block(mask: np.ndarray) -> np.ndarray:
            return lognorm_cols(full.X[mask], hvg_cols)

        def take(m: np.ndarray, n: int) -> np.ndarray:
            if args.no_downsample:
                return m
            n = min(n, m.shape[0])
            return m[rng.choice(m.shape[0], n, replace=False)]

        p_ct = ct_s == args.prompt_ct
        query_all = block(ct_mask & (pert_s == CTRL))
        real_all = {d: block(ct_mask & (pert_s == d)) for d in drugs}
        p_ctrl = block(p_ct & (pert_s == CTRL))
        p_groups = {d: block(p_ct & (pert_s == d)) for d in drugs}
        p_delta = {d: p_groups[d].mean(0) - p_ctrl.mean(0) for d in drugs}
        # fine prompt types for nearest_ct
        f_ctrl = {f: block(p_ct & (fine_s == f) & (pert_s == CTRL)) for f in prompt_fine}
        f_ctrl = {f: m for f, m in f_ctrl.items() if m.shape[0] >= 5}
        f_names = list(f_ctrl)
        f_pb = np.vstack([f_ctrl[f].mean(0) for f in f_names])

        if args.split_real:
            def halves(m):
                perm = rng.permutation(m.shape[0])
                h = m.shape[0] // 2
                return m[perm[:h]], m[perm[h:]]
            query_a, query = halves(query_all)
            pairs = {d: halves(real_all[d]) for d in drugs}
            real_a = {d: pairs[d][0] for d in drugs}
            real_groups = {d: pairs[d][1] for d in drugs}
            base_query = query_a
        else:
            query, real_groups, base_query = query_all, real_all, query_all
            real_a = None

        n_real = {d: real_groups[d].shape[0] for d in drugs}
        # every method predicts the same number of cells per unit: the real cell number, or fewer
        # when the prompt has fewer cells under d (B-lineage / myeloid prompts for the T/NK target)
        n_use = {d: min(n_real[d], p_groups[d].shape[0]) for d in drugs}
        real = make_adata(real_groups, query, hvg_names)
        real_ctrl_mean = query.mean(0)

        # nearest prompt fine type per query cell
        assign = np.argmax(rowwise_pearson(base_query.copy(), f_pb.copy()), axis=1)
        frac = np.bincount(assign, minlength=len(f_names)) / len(assign)
        nearest_info = {f: round(float(x), 3) for f, x in zip(f_names, frac)}
        print(ct, "nearest_ct assignment", nearest_info)

        def mix(pools: dict[str, np.ndarray], n: int) -> np.ndarray:
            """n cells drawn from the fine-type pools in the query-assignment proportions."""
            if args.no_downsample:
                n = sum(p.shape[0] for p in pools.values())
            counts = rng.multinomial(n, frac)
            out = []
            for f, k in zip(f_names, counts):
                pool = pools[f]
                if k == 0 or pool.shape[0] == 0:
                    continue
                out.append(pool[rng.choice(pool.shape[0], k, replace=k > pool.shape[0])])
            return np.vstack(out)

        de_real_path = None
        for method in methods:
            if method == "stack":
                def sp_block(d):
                    p = stack_pred[d]
                    return lognorm_cols(p.X[(p.obs[CT].astype(str) == ct).to_numpy()], hvg_cols)
                pred_ctrl = sp_block(CTRL)
                groups = {d: take(sp_block(d), n_use[d]) for d in drugs}
            elif method == "prompt_copy":
                pred_ctrl = take(p_ctrl, query.shape[0])
                groups = {d: take(p_groups[d], n_use[d]) for d in drugs}
            elif method == "nearest_ct":
                pred_ctrl = mix(f_ctrl, query.shape[0])
                groups = {}
                for d in drugs:
                    pools = {f: block(p_ct & (fine_s == f) & (pert_s == d)) for f in f_names}
                    groups[d] = mix(pools, n_use[d])
            elif method == "perturb_mean":
                pred_ctrl = base_query
                groups = {d: np.clip(take(base_query, n_use[d]) + p_delta[d], 0, None) for d in drugs}
            elif method == "no_effect":
                pred_ctrl = base_query
                groups = {d: take(base_query, n_use[d]) for d in drugs}
            elif method == "real_half":
                pred_ctrl = query_a
                groups = real_a
            else:
                raise ValueError(method)

            pred = make_adata(groups, pred_ctrl, hvg_names)
            out = os.path.join(args.outdir, args.tag, f"seed{args.seed}", sanitize(ct), method)
            ev = MetricsEvaluator(
                adata_pred=pred, adata_real=real.copy(), de_real=de_real_path,
                control_pert=CTRL, pert_col=PERT, num_threads=args.threads, outdir=out,
            )
            res, _ = ev.compute(profile="full", write_csv=True)
            if de_real_path is None:
                de_real_path = os.path.join(out, "real_de.csv")
            de_r = pl.read_csv(de_real_path, schema_overrides={"target": pl.Utf8, "feature": pl.Utf8})
            de_p = pl.read_csv(os.path.join(out, "pred_de.csv"), schema_overrides={"target": pl.Utf8, "feature": pl.Utf8})
            jac = jaccard_from_de(de_r, de_p)

            pc = pred_ctrl.mean(0)
            stats = {}
            for d in drugs:
                dr = real_groups[d].mean(0) - real_ctrl_mean
                dp = groups[d].mean(0) - pc
                stats[d] = (float(np.linalg.norm(dr)), float(np.linalg.norm(dp)),
                            float(dp @ dr / (dr @ dr)) if dr @ dr > 0 else float("nan"))

            df = res.to_pandas()
            df["jaccard"] = df["perturbation"].map(jac)
            df["delta_norm_real"] = df["perturbation"].map({d: s[0] for d, s in stats.items()})
            df["delta_norm_pred"] = df["perturbation"].map({d: s[1] for d, s in stats.items()})
            df["delta_slope"] = df["perturbation"].map({d: s[2] for d, s in stats.items()})
            df["n_real_cells"] = df["perturbation"].map(n_real)
            df["n_pred_cells"] = df["perturbation"].map({d: g.shape[0] for d, g in groups.items()})
            df["n_query_cells"] = query.shape[0]
            df["nearest_assign"] = str(nearest_info)
            df.insert(0, "method", method)
            df.insert(0, "cell_type", ct)
            df.insert(0, "prompt_ct", args.prompt_ct)
            df.insert(0, "seed", args.seed)
            df.insert(0, "tag", args.tag)
            rows.append(df)
            print(f"done {ct} / {method}: {df.shape}", flush=True)

    allres = pd.concat(rows, ignore_index=True)
    os.makedirs(args.outdir, exist_ok=True)
    dst = os.path.join(args.outdir, f"{args.tag}_seed{args.seed}_per_pert.csv")
    allres.to_csv(dst, index=False)
    pd.DataFrame(skipped, columns=["cell_type", "perturbation", "n_real"]).to_csv(
        os.path.join(args.outdir, f"{args.tag}_seed{args.seed}_skipped.csv"), index=False)
    print("wrote", dst, allres.shape)


if __name__ == "__main__":
    main()
