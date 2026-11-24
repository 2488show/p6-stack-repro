#!/usr/bin/env python
"""Tables for the 3-donor run (results/eval_full/F_*_per_pert.csv -> results/summary_full/).

Unit = (design, held-out cell class, compound); design = donor x prompt class. 12 compounds
(positive controls excluded). Main numbers = mean over downsampling seeds 0-2 per unit.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

M = {"pearson_delta": "Pearson Delta", "pr_auc": "PR AUC", "overlap_at_N": "DE overlap",
     "de_spearman_lfc_sig": "DE Spearman LFC", "de_spearman_sig": "Spearman effect size"}
PC = ["Belinostat", "Dabrafenib"]
OUT = "results/summary_full"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format)
SH = {"t cell": "T/NK", "lymphocyte of b lineage": "B", "myeloid leukocyte": "Myeloid"}

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/eval_full/F_*_per_pert.csv"))], ignore_index=True)
df = df[~df["perturbation"].isin(PC)].copy()
df["design"] = df["tag"].str.replace("^F_", "", regex=True).str.replace("_(split|allcells)$", "", regex=True)
df["donor"] = df["design"].str.split("_").str[0]
df["prompt"] = df["prompt_ct"].map(SH)
df["target"] = df["cell_type"].map(SH)
metrics = list(M)
keys = ["design", "donor", "prompt", "target", "perturbation", "method"]


def units(kind: str) -> pd.DataFrame:
    if kind == "main":
        d = df[~df["tag"].str.endswith(("_split", "_allcells"))]
    else:
        d = df[df["tag"].str.endswith("_" + kind)]
    return d.groupby(keys, as_index=False)[metrics + ["n_real_cells", "delta_slope"]].mean()


def wins(u: pd.DataFrame, a: str, b: str, m: str) -> tuple[int, int, float, float]:
    w = u.pivot_table(index=["design", "target", "perturbation"], columns="method", values=m)[[a, b]].dropna()
    diff = w[a] - w[b]
    p = wilcoxon(diff).pvalue if len(diff) > 5 and (diff != 0).any() else float("nan")
    return int((diff > 0).sum()), len(diff), float(diff.mean()), float(p)


lines = []
def say(s=""):
    print(s)
    lines.append(str(s))


u = units("main")
say(f"designs: {sorted(u['design'].unique())}")
say(f"units per method: {u.groupby('method').size().to_dict()}")
say("\n1) All designs pooled: mean over units (12 compounds, positive controls excluded)")
t1 = u.groupby("method")[metrics].mean().rename(columns=M).sort_values("Pearson Delta")
say(t1.to_string()); t1.to_csv(f"{OUT}/pooled_means.csv")

say("\n2) Paired wins, Stack vs each baseline (unit = design x target x compound; Wilcoxon p)")
rows = []
for m in ["pearson_delta", "pr_auc", "overlap_at_N", "de_spearman_lfc_sig"]:
    for b in ["prompt_copy", "nearest_ct", "perturb_mean", "no_effect"]:
        k, n, md, p = wins(u, "stack", b, m)
        rows.append({"metric": M[m], "vs": b, "stack_wins": k, "n": n, "mean_diff": md, "wilcoxon_p": p})
t2 = pd.DataFrame(rows); say(t2.to_string(index=False)); t2.to_csv(f"{OUT}/paired_wins.csv", index=False)

say("\n3) Pearson Delta by design and target (and Stack wins vs prompt_copy)")
pd_ = u.pivot_table(index=["design", "prompt", "target"], columns="method", values="pearson_delta")
w = u.pivot_table(index=["design", "prompt", "target", "perturbation"], columns="method", values="pearson_delta")
pd_["stack_wins_vs_copy"] = (w["stack"] > w["prompt_copy"]).groupby(level=[0, 1, 2]).sum()
pd_["n_units"] = w["stack"].groupby(level=[0, 1, 2]).count()
say(pd_.to_string()); pd_.to_csv(f"{OUT}/by_design_target.csv")

say("\n4) By prompt -> target pair (pooled over donors), Pearson Delta")
t4 = u.pivot_table(index=["prompt", "target"], columns="method", values="pearson_delta", aggfunc="mean")
w4 = u.pivot_table(index=["prompt", "target", "donor", "perturbation"], columns="method", values="pearson_delta")
t4["stack_wins_vs_copy"] = (w4["stack"] > w4["prompt_copy"]).groupby(level=[0, 1]).sum()
t4["n"] = w4["stack"].groupby(level=[0, 1]).count()
say(t4.to_string()); t4.to_csv(f"{OUT}/by_prompt_target.csv")

say("\n5) By donor, Pearson Delta")
t5 = u.pivot_table(index="donor", columns="method", values="pearson_delta")
say(t5.to_string()); t5.to_csv(f"{OUT}/by_donor.csv")

s = units("split")
if len(s):
    say("\n6) Noise ceiling (scored against half of the real cells, seed 0), pooled")
    t6 = s.groupby("method")[metrics].mean().rename(columns=M).sort_values("Pearson Delta")
    say(t6.to_string()); t6.to_csv(f"{OUT}/noise_ceiling.csv")
a = units("allcells")
if len(a):
    say("\n7) Sensitivity: every predicted cell kept (no downsampling), pooled")
    t7 = a.groupby("method")[metrics].mean().rename(columns=M)
    say(t7.to_string()); t7.to_csv(f"{OUT}/allcells_means.csv")
    rows = []
    for m in ["pearson_delta", "pr_auc", "overlap_at_N", "de_spearman_lfc_sig"]:
        for b in ["prompt_copy", "perturb_mean"]:
            k, n, md, p = wins(a, "stack", b, m)
            rows.append({"metric": M[m], "vs": b, "stack_wins": k, "n": n, "mean_diff": md, "wilcoxon_p": p})
    t7b = pd.DataFrame(rows); say(t7b.to_string(index=False)); t7b.to_csv(f"{OUT}/allcells_paired_wins.csv", index=False)

say("\n8) Effect-size shrinkage: median delta_slope (pred delta on real delta)")
say(u.groupby("method")["delta_slope"].median().to_string())

sk = pd.concat([pd.read_csv(f).assign(file=os.path.basename(f)) for f in glob.glob("results/eval_full/F_*_seed0_skipped.csv")
                if not f.endswith(("_split_seed0_skipped.csv", "_allcells_seed0_skipped.csv"))], ignore_index=True) \
    if glob.glob("results/eval_full/F_*_seed0_skipped.csv") else pd.DataFrame()
say(f"\n9) units skipped for < 10 real cells: {len(sk)}")
if len(sk):
    say(sk.to_string(index=False))
open(f"{OUT}/report.txt", "w").write("\n".join(lines) + "\n")
