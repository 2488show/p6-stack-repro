#!/usr/bin/env python
"""Final tables: headline comparison, generation-seed spread, one-step vs five-step, noise ceiling.

All numbers are for the 12 compounds (positive controls Belinostat and Dabrafenib excluded),
unit = (held-out cell type, compound), 24 units.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

M = {
    "pearson_delta": "Pearson Delta",
    "de_spearman_lfc_sig": "DE Spearman LFC",
    "pr_auc": "PR AUC",
    "overlap_at_N": "DE overlap acc",
    "de_spearman_sig": "Spearman effect size",
}
PC = ["Belinostat", "Dabrafenib"]
OUT = "results/summary"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:.3f}".format)

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/eval/*_per_pert.csv"))], ignore_index=True)
df = df[~df["perturbation"].isin(PC)]
metrics = list(M)


def unit_means(tag: str) -> pd.DataFrame:
    d = df[df["tag"] == tag]
    return d.groupby(["cell_type", "perturbation", "method"], as_index=False)[metrics].mean()


# 1) headline: T5_gen0, mean over downsampling seeds, +- sd of the seed-level means
d0 = df[df["tag"] == "T5_gen0"]
seed_means = d0.groupby(["seed", "method"])[metrics].mean()
head = seed_means.groupby("method").agg(["mean", "std"]).rename(columns=M)
print(f"1) Headline (generation run 0, downsampling seeds {sorted(d0['seed'].unique())}), 12 compounds")
print(head.to_string())
head.to_csv(f"{OUT}/headline_mean_sd.csv")

# 2) paired wins on unit means
u = unit_means("T5_gen0")
per_unit = [m for m in metrics if m != "de_spearman_sig"]
w = u.pivot_table(index=["cell_type", "perturbation"], columns="method", values=per_unit)
rows = []
for m in per_unit:
    for b in [c for c in w[m].columns if c != "stack"]:
        p = w[m][["stack", b]].dropna()
        diff = p["stack"] - p[b]
        rows.append({"metric": M[m], "vs": b, "n": len(p), "stack_wins": int((diff > 0).sum()),
                     "mean_diff": diff.mean()})
wins = pd.DataFrame(rows)
print("\n2) Stack vs each baseline, paired over units (seed-averaged)")
print(wins.to_string(index=False))
wins.to_csv(f"{OUT}/paired_wins.csv", index=False)

# 3) generation-seed spread (Stack only, downsampling seed 0)
g = df[(df["method"] == "stack") & (df["seed"] == 0) & df["tag"].isin(["T5_gen0", "T5_gen1", "T5_gen2"])]
gs = g.groupby("tag")[metrics].mean().rename(columns=M)
print("\n3) Stack across three generation seeds (downsampling seed 0)")
print(gs.to_string())
print("mean:", gs.mean().round(3).to_dict())
print("sd:  ", gs.std().round(3).to_dict())
gs.to_csv(f"{OUT}/stack_generation_seeds.csv")

# 4) one-step vs five-step (same generation seed, same downsampling seed), paired over units
t = df[(df["method"] == "stack") & (df["seed"] == 0) & df["tag"].isin(["T5_gen0", "T1_gen0"])]
tw = t.pivot_table(index=["cell_type", "perturbation"], columns="tag", values=per_unit)
rows = []
for m in per_unit:
    p = tw[m][["T5_gen0", "T1_gen0"]].dropna()
    diff = p["T5_gen0"] - p["T1_gen0"]
    rows.append({"metric": M[m], "n": len(p), "T5_mean": p["T5_gen0"].mean(), "T1_mean": p["T1_gen0"].mean(),
                 "T5_wins": int((diff > 0).sum()), "mean_diff": diff.mean()})
steps = pd.DataFrame(rows)
print("\n4) Five-step vs one-step generation (Stack)")
print(steps.to_string(index=False))
steps.to_csv(f"{OUT}/steps_T5_vs_T1.csv", index=False)

# 5) noise ceiling: everything scored against half of the real cells
s = df[df["tag"] == "T5_gen0_split"]
if len(s):
    sm = s.groupby(["seed", "method"])[metrics].mean().groupby("method").agg(["mean", "std"]).rename(columns=M)
    print(f"\n5) Split-half ceiling (split seeds {sorted(s['seed'].unique())}): real_half = independent half of real cells")
    print(sm.to_string())
    sm.to_csv(f"{OUT}/split_half_ceiling.csv")
    extra = [c for c in ["delta_slope", "de_nsig_counts_real", "de_nsig_counts_pred", "n_real_cells"] if c in s.columns]
    print(s.groupby("method")[extra].median().to_string())

# 6) effect-size shrinkage
if "delta_slope" in d0.columns:
    e = d0.dropna(subset=["delta_slope"])
    print("\n6) Slope of predicted on real pseudobulk delta, and DE-gene counts (median over units)")
    print(e.groupby("method")[["delta_slope", "delta_norm_pred", "delta_norm_real",
                               "de_nsig_counts_pred", "de_nsig_counts_real"]].median().to_string())
