#!/usr/bin/env python
"""Tables for the State baseline, next to the existing Stack / baseline results.

Inputs: results/eval/*_per_pert.csv
  T5_gen0            existing main run (stack, tcell_copy, perturb_mean, no_effect), downsampling seeds 0-2
  T5_gen0_allcells   existing no-downsample run (stack, tcell_copy, perturb_mean), seed 0
  STATE_<run>        State (best checkpoint of training run <run>), downsampling seeds 0-2
  STATE_<run>_allcells, STATE_<run>_realctrl   sensitivity runs, seed 0
12 compounds (positive controls excluded; for State they were the validation set), unit = (cell type,
compound), 24 units. Paired wins use unit means over the downsampling seeds. The downsampling draws of
State and of the existing methods are independent (separate evaluation runs); with 3 seeds averaged this
only adds noise of the size shown in the sd column.
Usage: python scripts/25_state_report.py <run> [<run2> ...]   (first run = main State row)
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

M = {
    "pearson_delta": "Pearson Delta",
    "pr_auc": "PR AUC",
    "overlap_at_N": "DE overlap acc",
    "de_spearman_lfc_sig": "DE Spearman LFC",
    "de_spearman_sig": "Spearman effect size",
}
PER_UNIT = ["pearson_delta", "pr_auc", "overlap_at_N", "de_spearman_lfc_sig"]
PC = ["Belinostat", "Dabrafenib"]
OUT = "results/summary"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220, "display.max_columns", 40, "display.float_format", "{:.3f}".format)

runs = sys.argv[1:] or ["s42"]
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/eval/*_per_pert.csv"))], ignore_index=True)
df = df[~df["perturbation"].isin(PC)]
metrics = list(M)


def rows_for(tag: str, method: str, label: str) -> pd.DataFrame:
    d = df[(df["tag"] == tag) & (df["method"] == method)].copy()
    d["label"] = label
    return d


def headline(parts: list[pd.DataFrame]) -> pd.DataFrame:
    d = pd.concat(parts)
    sm = d.groupby(["label", "seed"])[metrics].mean()
    h = sm.groupby("label").agg(["mean", "std"])
    h[("n_seeds", "")] = sm.groupby("label").size()
    return h.rename(columns=M)


def wins(parts: list[pd.DataFrame], ref: str, others: list[str]) -> pd.DataFrame:
    d = pd.concat(parts)
    u = d.groupby(["cell_type", "perturbation", "label"], as_index=False)[PER_UNIT].mean()
    rows = []
    for m in PER_UNIT:
        w = u.pivot_table(index=["cell_type", "perturbation"], columns="label", values=m)
        for o in others:
            p = w[[ref, o]].dropna()
            diff = p[ref] - p[o]
            rows.append({"metric": M[m], "ref": ref, "vs": o, "n_units": len(p), "ref_wins": int((diff > 0).sum()),
                         "mean_diff": diff.mean()})
            for ct in sorted(p.index.get_level_values(0).unique()):
                pc = p.xs(ct, level=0)
                dc = pc[ref] - pc[o]
                rows.append({"metric": M[m], "ref": ref, "vs": o, "cell_type": ct, "n_units": len(pc),
                             "ref_wins": int((dc > 0).sum()), "mean_diff": dc.mean(),
                             "ref_mean": pc[ref].mean(), "vs_mean": pc[o].mean()})
    return pd.DataFrame(rows)


main = runs[0]
base = [rows_for("T5_gen0", m, m) for m in ["no_effect", "perturb_mean", "tcell_copy", "stack"]]
state_rows = [rows_for(f"STATE_{r}", "state", f"state_{r}") for r in runs]
state_rows = [s for s in state_rows if len(s)]
if not state_rows or f"STATE_{main}" not in set(df["tag"]):
    sys.exit(f"no State results for run '{main}' in results/eval (expected tag STATE_{main})")

print("1) Main table: 12 compounds, 24 units, downsampled to the real cell number; mean over downsampling seeds (sd)")
h = headline(base + state_rows)
print(h.to_string())
h.to_csv(f"{OUT}/state_headline_mean_sd.csv")

print(f"\n2) Paired wins of State ({main}) over 24 units (unit means over downsampling seeds), overall and by cell type")
w = wins(base + state_rows, f"state_{main}", ["tcell_copy", "stack", "perturb_mean", "no_effect"])
print(w.to_string(index=False))
w.to_csv(f"{OUT}/state_paired_wins.csv", index=False)

alls = [rows_for("T5_gen0_allcells", m, m) for m in ["perturb_mean", "tcell_copy", "stack"]]
sa = rows_for(f"STATE_{main}_allcells", "state", f"state_{main}")
if len(sa):
    print("\n3) Sensitivity: keep every predicted cell (no downsampling), seed 0")
    d = pd.concat(alls + [sa])
    t = d.groupby("label")[metrics + ["n_pred_cells"]].mean().rename(columns=M)
    print(t.to_string())
    t.to_csv(f"{OUT}/state_allcells.csv")
    wa = wins(alls + [sa], f"state_{main}", ["tcell_copy", "stack", "perturb_mean"])
    wa = wa[wa["cell_type"].isna()] if "cell_type" in wa else wa
    print(wa.to_string(index=False))
    wa.to_csv(f"{OUT}/state_allcells_paired_wins.csv", index=False)

rc = rows_for(f"STATE_{main}_realctrl", "state_realctrl", "state_realctrl")
if len(rc):
    print("\n4) Sensitivity: State predictions with the real query cells as DE control (seed 0) vs State's own control")
    st0 = rows_for(f"STATE_{main}", "state", "state_ownctrl")
    st0 = st0[st0["seed"] == 0]
    t = pd.concat([st0, rc]).groupby("label")[metrics].mean().rename(columns=M)
    print(t.to_string())
    t.to_csv(f"{OUT}/state_control_choice.csv")

extra = ["delta_slope", "delta_norm_pred", "delta_norm_real", "de_nsig_counts_pred", "de_nsig_counts_real"]
d = pd.concat(base + state_rows)
if all(c in d.columns for c in extra):
    print("\n5) Effect size: slope of predicted on real pseudobulk delta, delta norms, DE-gene counts (median over units x seeds)")
    e = d.groupby("label")[extra].median()
    print(e.to_string())
    e.to_csv(f"{OUT}/state_effect_size.csv")
