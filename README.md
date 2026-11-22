# p6-stack-repro: reproducing Stack's zero-shot in-context perturbation prediction

Independent reproduction of one claim from *Stack: In-Context Learning of Single-Cell Biology*
(Dong et al., bioRxiv 2026, doi:10.64898/2026.01.09.698608): given drug-treated cells of one cell
type as a prompt, the released model predicts the drug response of other cell types without any
fine-tuning, and does so better than simple baselines (paper Fig. 3E, "Setting 1").

Nothing is trained here. The released checkpoint is run as-is and its predictions are scored
against real held-out cells, next to zero-parameter baselines and a split-half noise ceiling.

## What is run

| Item | Value |
|---|---|
| Model | `arcinstitute/Stack-Large-Aligned` (217M parameters), sha256 `f93cf6f4...6ad36` |
| Code | `ArcInstitute/stack` @ `cacc2e4b` (arc-stack 0.1.3), torch 2.5.1+cu121 |
| Data | `arcinstitute/Stack-DrugPBMC-Example`: OpenProblems PBMC donor 1, 32,530 cells, 14 compounds + DMSO |
| Prompt | T/NK cells of donor 1 under each compound (318-6,319 cells per condition) |
| Query | 2,848 DMSO-treated B-lineage (1,320) and myeloid (1,528) cells of the same donor |
| Target | the real B-lineage and myeloid cells under each compound (13-128 cells; positive controls 862-1,873) |
| Generation | `stack-generation`, mask-diffusion mode, T=5, prompt ratio 0.25, context ratio 0.2 to 0.4 (tutorial settings) |
| Evaluation | cell-eval 0.6.6, Wilcoxon + BH, top-2,000 HVGs per held-out cell type, log-normalised to 1e4 |
| Hardware | Google Colab T4 (16 GB), 2 vCPU, 12 GB RAM; one generation run takes 9.6 min |

Only 11,018 of the model's 15,012 genes exist in this dataset; the model zero-pads the rest, and all
metrics are computed inside the shared gene set.

## Methods compared

| Name | Prediction for compound d in held-out cell type c | Control used for DE |
|---|---|---|
| `stack` | Stack generation with T/NK cells under d as prompt | Stack generation with DMSO T/NK cells as prompt ("synthetic control", as in the paper) |
| `tcell_copy` | the prompt cells themselves (T/NK under d) | DMSO T/NK cells |
| `perturb_mean` | query cells + (mean T/NK under d - mean T/NK DMSO) | query cells |
| `no_effect` | query cells, unchanged | query cells |
| `real_half` | an independent half of the real target cells (ceiling runs only) | half of the real DMSO cells |

Each predicted group is downsampled to the number of real cells of that (cell type, compound).
The unit of analysis is one (held-out cell type, compound) pair: 2 x 12 = 24 units. The two
positive controls (Belinostat, Dabrafenib) are reported separately because the paper's panel shows 12 drugs.

## Reproduce

```bash
# on a Colab T4 runtime (colab new -s stack --gpu T4), from /content/stack-repro
bash scripts/00_setup_colab.sh     # venv, pinned installs, model + data from Hugging Face
bash scripts/10_run_all.sh         # 4 generation runs + all evaluations + report
```

`scripts/keepalive.py` and `scripts/sync_results.sh` exist because the first runtime was reclaimed
after about 75 minutes with all results on it: the kernel has to stay busy, and results are pulled
off the runtime after every evaluation.

## Results

All numbers: 12 compounds, 24 units, generation seed 0 unless stated. Files in `results/summary/`.

### 1. Headline (mean over 3 downsampling seeds; sd of the seed-level means in brackets)

| Method | Pearson Delta | PR AUC | DE overlap acc | DE Spearman LFC | Spearman effect size |
|---|---|---|---|---|---|
| no_effect | -0.011 (0.023) | 0.094 (0.000) | 0.000 (0.000) | -0.180 (0.039) | -0.156 (0.079) |
| perturb_mean | 0.332 (0.017) | 0.106 (0.001) | 0.109 (0.001) | 0.229 (0.011) | 0.274 (0.014) |
| tcell_copy | 0.364 (0.008) | 0.195 (0.004) | 0.118 (0.004) | 0.069 (0.030) | 0.799 (0.045) |
| **stack** | 0.343 (0.011) | 0.183 (0.006) | 0.086 (0.005) | 0.065 (0.043) | 0.726 (0.076) |

Stack across three generation seeds (downsampling seed 0): Pearson Delta 0.356 / 0.338 / 0.343.

Paired over the 24 units, Stack beats `no_effect` on 24/24 (Pearson Delta, PR AUC); `tcell_copy`
on 12/24 (Pearson Delta), 11/24 (PR AUC), 6/24 (DE overlap); `perturb_mean` on 14/24, 21/24, 10/24.

By cell type (Pearson Delta, seed-averaged): B lineage, stack 0.380 vs tcell_copy 0.468 (3/12 wins);
myeloid, stack 0.306 vs tcell_copy 0.259 (9/12 wins; 9, 8 and 6 of 12 across the three generation seeds).

### 2. Five-step vs one-step generation (paper: multi-step has "modest advantages")

| Metric | T=5 | T=1 | units where T=5 is higher |
|---|---|---|---|
| Pearson Delta | 0.356 | 0.298 | 21/24 |
| PR AUC | 0.188 | 0.164 | 20/24 |
| DE overlap acc | 0.092 | 0.069 | 17/24 |
| DE Spearman LFC | 0.113 | 0.034 | 20/24 |

### 3. Split-half noise ceiling (everything scored against half of the real cells, about 34 per unit; 3 split seeds)

| Method | Pearson Delta | PR AUC | DE overlap acc |
|---|---|---|---|
| real_half (independent half of the real cells) | 0.614 | 0.296 | 0.290 |
| tcell_copy | 0.294 | 0.127 | 0.088 |
| perturb_mean | 0.269 | 0.067 | 0.052 |
| stack | 0.266 | 0.111 | 0.064 |

### 4. Sensitivity: keep every predicted cell instead of matching the real cell number

| Method | predicted cells / unit | Pearson Delta | PR AUC | DE overlap acc | DE Spearman LFC |
|---|---|---|---|---|---|
| stack | 1,424 | 0.468 | 0.257 | 0.234 | 0.433 |
| tcell_copy | 390 | 0.449 | 0.267 | 0.193 | 0.220 |
| perturb_mean | 1,424 | 0.444 | 0.125 | 0.138 | 0.489 |

Stack vs `tcell_copy`, paired: Pearson Delta 13/24, PR AUC 11/24, DE overlap 15/24. The ranking on
DE-set metrics depends on how many predicted cells are scored; the tie on Pearson Delta and PR AUC does not.

### 5. Paper Fig. 3E for comparison (bar heights read by eye, about +-0.02; `results/paper_fig3e_readoff.csv`)

| | Pearson Delta | PR AUC | DE overlap acc | DE Spearman LFC | Spearman effect size |
|---|---|---|---|---|---|
| Stack | 0.57 | 0.39 | 0.38 | 0.50 | 0.67 |
| Closest cell type in prompt | 0.52 | 0.36 | 0.17 | 0.44 | 0.54 |
| PerturbMean | 0.50 | 0.11 | 0.12 | 0.49 | 0.59 |
| Original query | 0.00 | 0.08 | 0.02 | 0.00 | n/a |

The paper's benchmark is not the same slice: 3 donors and randomly sampled prompt cell types among 6
cell types, versus one donor and a fixed T/NK prompt here.

### 6. Checks

- Same seed, two independent Colab runtimes: identical metrics (compared to 3 decimals).
- `scripts/11_independent_check.py` (output in `results/checks/`): Pearson Delta recomputed with numpy matches the pipeline to 6 decimals (2 units x 2 methods).
- `scripts/12_joint_fdr_check.py`: real DE-gene counts match the pipeline exactly for 28/28 units, but only when
  Benjamini-Hochberg is applied once over all compounds of the run, which is what pdex does. Per-compound BH gives
  different counts (myeloid, Palbociclib: 33 per compound, 114 jointly over 14 compounds, 85 jointly over 12).
- Classifier-logit filter used for Perturb Sapiens (logit <= 2.5) keeps at least 99.9% of generated cells; no material change.

## Limits

- One donor, one prompt design, 12 compounds; ground-truth groups have 13-128 cells.
- State and scVI baselines of the paper were not run; `tcell_copy` and `perturb_mean` are my implementations of
  the paper's one-line descriptions of "closest cell type in prompt" and "PerturbMean".
- The paper's cell-count matching rule is ambiguous to me; section 4 shows the result depends on it.
- Headline uses one generation seed (three for Stack's spread); the T=5 vs T=1 comparison uses one seed.
- Only "Setting 1" inference is covered: no pre-training, no post-training, no embedding benchmarks, no Perturb Sapiens.
- The effect of joint BH on the metric values themselves is not quantified.

## Licence note

Stack code is CC BY-NC-SA 4.0; the weights are under the Arc Research Institute Stack Model Non-Commercial License.
The example data is CC-BY-4.0 (OpenProblems single-cell perturbations, Luecken et al. 2025).
