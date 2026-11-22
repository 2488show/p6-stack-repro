#!/usr/bin/env python
"""Summarise per-perturbation metrics: method means, paired win counts, seed spread.

Reads every results/eval/*_per_pert.csv, averages over downsampling seeds first, then
reports one row per (cell type, compound) unit so the unit of analysis is the condition.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

# cell-eval name -> name used in the Stack paper (higher is better for all of these)
PAPER_METRICS = {
    "pearson_delta": "Pearson Delta",
    "de_spearman_lfc_sig": "DE Spearman LFC",
    "pr_auc": "PR AUC",
    "overlap_at_N": "DE overlap acc",
    "de_spearman_sig": "Spearman effect size",
    "de_direction_match": "DE direction match",
    "precision_at_N": "Precision@N",
    "jaccard": "Jaccard",
}
EXTRA = ["delta_slope", "delta_norm_real", "delta_norm_pred", "de_nsig_counts_real", "de_nsig_counts_pred"]
POSITIVE_CONTROLS = ["Belinostat", "Dabrafenib"]


def load(pattern: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pattern))], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/eval/*_per_pert.csv")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/summary")
    ap.add_argument("--compare-tags", nargs="*", default=None,
                    help="also print Stack's means for these tags (generation runs), seed 0 only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:.3f}".format)

    everything = load(args.glob)
    df = everything[everything["tag"] == args.tag]
    metrics = [m for m in PAPER_METRICS if m in df.columns]
    extra = [m for m in EXTRA if m in df.columns]
    seeds = sorted(df["seed"].unique())
    print(f"tag={args.tag} seeds={[int(s) for s in seeds]} rows={len(df)}")

    # mean over downsampling seeds -> one value per (cell type, compound, method)
    unit = df.groupby(["cell_type", "perturbation", "method"], as_index=False)[metrics + extra].mean()
    unit.to_csv(os.path.join(args.out, f"{args.tag}_unit_level.csv"), index=False)

    def table(sub: pd.DataFrame, label: str) -> pd.DataFrame:
        mean = sub.groupby("method")[metrics].mean().rename(columns=PAPER_METRICS)
        mean.insert(0, "subset", label)
        mean["n_units"] = sub.groupby("method")["pearson_delta"].size()
        return mean

    no_pc = unit[~unit["perturbation"].isin(POSITIVE_CONTROLS)]
    subsets = {
        "12 compounds": no_pc,
        "14 (with positive controls)": unit,
        "2 positive controls": unit[unit["perturbation"].isin(POSITIVE_CONTROLS)],
    }
    for ct in sorted(unit["cell_type"].unique()):
        subsets[f"12 compounds, {ct}"] = no_pc[no_pc["cell_type"] == ct]
    tabs = pd.concat([table(s, k) for k, s in subsets.items()])
    tabs.to_csv(os.path.join(args.out, f"{args.tag}_method_means.csv"))
    print(tabs.to_string())

    # paired comparison on the 12 compounds: on how many units does Stack beat each baseline?
    per_unit = [m for m in metrics if m != "de_spearman_sig"]  # that one is per cell type, not per unit
    wide = no_pc.pivot_table(index=["cell_type", "perturbation"], columns="method", values=per_unit)
    rows = []
    for m in per_unit:
        if "stack" not in wide[m]:
            continue
        for b in [c for c in wide[m].columns if c != "stack"]:
            pair = wide[m][["stack", b]].dropna()
            d = pair["stack"] - pair[b]
            rows.append({
                "metric": PAPER_METRICS[m], "baseline": b, "n_pairs": len(pair),
                "stack_wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                "mean_diff": d.mean(), "median_diff": d.median(),
            })
    wins = pd.DataFrame(rows)
    wins.to_csv(os.path.join(args.out, f"{args.tag}_paired_wins.csv"), index=False)
    print("\nPaired comparison, 12 compounds (unit = cell type x compound):")
    print(wins.to_string(index=False))

    # seed-to-seed spread of the method means (downsampling noise), 12 compounds
    if len(seeds) > 1:
        d12 = df[~df["perturbation"].isin(POSITIVE_CONTROLS)]
        per_seed = d12.groupby(["seed", "method"])[metrics].mean().groupby("method").agg(["mean", "std"])
        per_seed = per_seed.rename(columns=PAPER_METRICS)
        per_seed.to_csv(os.path.join(args.out, f"{args.tag}_seed_spread.csv"))
        print("\nMethod means across downsampling seeds, 12 compounds (mean, std):")
        print(per_seed.to_string())

    if extra:
        print("\nEffect size and DE counts, 12 compounds (median over units):")
        print(no_pc.groupby("method")[extra].median().to_string())
        print("\nEffect size and DE counts, 2 positive controls (median over units):")
        print(unit[unit["perturbation"].isin(POSITIVE_CONTROLS)].groupby("method")[extra].median().to_string())

    if args.compare_tags:
        s = everything[(everything["method"] == "stack") & (everything["seed"] == 0)
                       & everything["tag"].isin(args.compare_tags)
                       & ~everything["perturbation"].isin(POSITIVE_CONTROLS)]
        comp = s.groupby("tag")[metrics].mean().rename(columns=PAPER_METRICS)
        comp.to_csv(os.path.join(args.out, "stack_generation_runs.csv"))
        print("\nStack across generation runs, 12 compounds, downsampling seed 0:")
        print(comp.to_string())


if __name__ == "__main__":
    main()
