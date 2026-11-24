# p6-stack-repro: reproducing Stack's zero-shot in-context perturbation prediction

Independent reproduction of one claim from *Stack: In-Context Learning of Single-Cell Biology*
(Dong et al., bioRxiv 2026, doi:10.64898/2026.01.09.698608): given drug-treated cells of one cell
type as a prompt, the released model predicts the drug response of other cell types without any
fine-tuning, and does so better than simple baselines (paper Fig. 3E, "Setting 1").

**Summary.** On the example donor (sections 1-7) Stack ties with copying the prompt cells' response. On all 3
donors x 3 prompt designs (section 8, 216 units) it is slightly below that zero-parameter baseline with matched
cell numbers (Pearson Delta 0.416 vs 0.445), beats it only for distant prompt/target pairs (T/NK -> myeloid 26/36),
and leads on Pearson Delta, PR AUC and the DE metrics only when scored with more predicted cells than the baseline. A State model trained on
the donor scores lower still (section 7).

Stack is not trained here. The released checkpoint is run as-is and its predictions are scored
against real held-out cells, next to zero-parameter baselines and a split-half noise ceiling.

## What is run

| Item | Value |
|---|---|
| Model | `arcinstitute/Stack-Large-Aligned` (217M parameters), sha256 `f93cf6f4...6ad36` |
| Code | `ArcInstitute/stack` @ `cacc2e4b` (arc-stack 0.1.3), torch 2.5.1+cu121 |
| Data | `arcinstitute/Stack-DrugPBMC-Example`: OpenProblems PBMC "donor_1" (= OpenProblems Donor 3, section 8), 32,530 cells, 14 compounds + DMSO |
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

### 7. State baseline (trained on this donor, single seed)

The paper compares Stack with State, a model trained on perturbation data. This section adds Arc's State
transition model (`arc-state` 0.11.1, `model=state`, trained from scratch here, no pretrained weights) as a
trained baseline in the same evaluation. The first training run was lost with its Colab runtime and Colab then
refused a new T4; the run reported here was trained on the Mac mini's Apple-silicon GPU (MPS) with the same
split, settings and scripts (see "Run on the Mac" below).

**Split (Setting 1, `scripts/21_build_state_input.py`, `results/state/build_info.json`).**
Removed from the training file: the 24 test units (12 compounds x B-lineage/myeloid, 1,634 cells).
Validation only: the positive controls (Belinostat, Dabrafenib) x B-lineage/myeloid (5,014 cells), never trained
on; their loss picks the checkpoint, and they are not part of the main table. Training: T/NK cells under all 14
compounds and DMSO, plus the DMSO B-lineage and myeloid cells, which are the query cells (25,882 cells).
Genes: the 11,018 genes shared with Stack's gene list (the evaluation universe); per cell normalised to the median
library size of the training cells and log1p; input embedding = 2,000 HVGs of the training cells.

**Can State represent this setting?** State encodes the compound as a one-hot vector and does not get the cell
type as an input token (its batch and cell-type encoders are off by default); the cell type enters only through
the set of DMSO cells it transforms. Every test unit is an unseen (compound, cell type) combination of a seen
compound (in T/NK cells) and a seen cell type (under DMSO), so State can be asked about all 24 units. It is not
zero-shot: unlike Stack it was trained on this donor, including all T/NK x compound cells and all DMSO cells.
A compound never seen in training could not be predicted at all (one-hot).

**Training (`scripts/22_state_train.sh`).** arc-state defaults for `model=state` (hidden 768, 8 bidirectional
llama layers, energy loss, lr 1e-4), except: cell-set length 256 (the default 512 is larger than most training
groups; T/NK x compound has 318-533 cells for the 12 main compounds), batch 8 sets, 2 data workers (2 vCPUs),
cell-load's consecutive (batched) reader, validation every 100 steps, `best.ckpt` = lowest validation loss.
With the per-cell reader a 60-step benchmark took 9 min and the GPU sat idle; with the batched reader 3 min.

**Prediction and scoring (`scripts/23`, `24`, `26`).** `state tx infer` gets the 2,848 query cells once as DMSO
and once per compound, so every compound gets 1,320 B-lineage and 1,528 myeloid predicted cells, the same pool
Stack is downsampled from. Its output is log1p of normalised expression (not counts, despite the log message);
it is mapped back with `expm1`, without rounding, because `03_evaluate.py` re-normalises every cell. The control
for DE is State's own prediction for DMSO, the same rule as Stack's synthetic control (`--methods state`);
`state_realctrl` uses the real query cells instead (sensitivity). Everything else is the unchanged evaluation:
downsampling seeds 0-2, no-downsample sensitivity, paired wins vs `tcell_copy` and Stack (`scripts/25_state_report.py`).

**Results (2,500 steps, training seed 42, best checkpoint; `results/summary/state_*.csv`).**

| Method (12 compounds, 24 units, 3 downsampling seeds) | Pearson Delta | PR AUC | DE overlap | DE Spearman LFC | Spearman effect size |
| --- | --- | --- | --- | --- | --- |
| no_effect | -0.011 | 0.094 | 0.000 | -0.180 | -0.156 |
| perturb_mean | 0.332 | 0.106 | 0.109 | 0.229 | 0.274 |
| **State (trained on this donor)** | **0.237** (sd 0.020) | 0.162 | 0.087 | 0.027 | 0.741 |
| Stack (zero-shot) | 0.343 | 0.183 | 0.086 | 0.065 | 0.726 |
| tcell_copy | 0.364 | 0.195 | 0.118 | 0.069 | 0.799 |

- Paired over the 24 units (Pearson Delta): State beats no_effect in 23/24, but tcell_copy in 0/24 and Stack in
  1/24; by cell type, State is 0.341 on B-lineage (tcell_copy 0.468, Stack 0.380) and 0.132 on myeloid cells
  (tcell_copy 0.259, Stack 0.306). On PR AUC State beats perturb_mean in 22/24, tcell_copy in 7/24, Stack in 9/24.
- No-downsample sensitivity (seed 0, every predicted cell): State 0.336 Pearson Delta vs Stack 0.468 and
  tcell_copy 0.449; 0/24 wins vs tcell_copy, 1/24 vs Stack.
- DE control: with the real query cells instead of State's own DMSO prediction (seed 0), Pearson Delta 0.196
  instead of 0.217 and Spearman effect size 0.28 instead of 0.75; the own-control rule (as for Stack) is the main row.
- Predicted effects are small: median slope of predicted on real pseudobulk delta 0.08 (Stack 0.22, tcell_copy
  0.25); median DE genes per unit 19.5 (Stack 20.5, real 128).
- Validation loss (positive controls) was still falling at the last step (best checkpoint = step 2,499,
  `results/state/metrics_s42_mac.csv`), unlike the lost Colab run where it flattened; State may be under-trained
  here. One training seed only.

So on this slice a model trained on the same donor's T/NK x compound data does **worse** than both the zero-shot
Stack and the zero-parameter "copy the T cells" baseline. This is one donor, one seed and 2,500 steps; it is not
a test of State in its intended setting (many cell lines, large training sets).

**Run on the Mac (2026-11-23).** Same split (the regenerated `build_info.json` is identical to the Colab one),
same hyper-parameters, except 4 data workers. Scripts take `P6_ROOT` (working folder, default `/content/p6state`),
`STATE_ACCEL=mps` (`scripts/state_cli.py` switches Lightning's accelerator from cpu to Apple's GPU; nothing else
changes) and `NW` (data workers). Training took 35 min (about 1.2 steps/s); prediction and all five evaluations
about 3 min. Package versions: `results/state/env_freeze_*_mac.txt` (arc-state 0.11.1, cell-load 0.10.4,
anndata 0.13.4, torch 2.15.0; evaluation venv as on Colab).

**History: the lost Colab run.**
- Two 60-step benchmarks (per-cell vs batched reader) and a full test of infer -> conversion -> `03_evaluate.py`
  on the 60-step model: all steps work. (That model has essentially no signal; its numbers are not reported.)
- A 4,000-step training run was last seen at about step 2,650 after about 70 min; about 15 min later the runtime
  disappeared (after about 1 h 50 min of T4 time), with the checkpoints and logs on it. In the live log the
  validation loss fell steeply for the first ~500 steps and flattened after ~1,300, with its lowest value (of the
  steps seen) near step 2,300, while the training loss kept falling (overfitting). Those files were lost, so
  no loss values are given here, and the benchmark timings above come from the same lost logs.
- From then on every request for a T4 runtime was refused (HTTP 503 "Service Unavailable"; every 5 min for
  75 min with `scripts/retry_gpu.sh`, log in `results/state/retry_gpu.log`, and once more 15 min later), which
  is what the free tier does after its GPU quota is used up. The account has 0 compute units.
- Saved from a CPU runtime: `results/state/build_info.json` (cell counts of the split above), `pbmc.toml`
  (cell-load split), `env_freeze_state.txt` / `env_freeze_eval.txt` (package versions).
- On a CPU runtime: the evaluation environment of this section reproduces the original no-downsample
  `tcell_copy` evaluation exactly (all 28 units, all metrics, `results/checks/state_eval_env_check.txt`).

**Reproduce** (Mac with Apple silicon; on Colab drop the first line's `STATE_ACCEL`/`NW` and use the defaults):

```bash
export P6_ROOT=$PWD STATE_ACCEL=mps PYTORCH_ENABLE_MPS_FALLBACK=1 NW=4   # use a path without spaces
bash scripts/20_setup_state_colab.sh                    # venvs + data (needs uv; set HF_HOME/UV_CACHE_DIR if ~/.cache is not writable)
.venv-eval/bin/python scripts/21_build_state_input.py   # split, toml, inference input
bash scripts/28_state_run.sh s42 2500 42 full           # train, infer, evaluate (needs about 5 GB free disk for the prediction file)
.venv-eval/bin/python scripts/25_state_report.py s42    # tables in results/summary/state_*.csv
```

### 8. Three donors, every prompt design (2026-11-24)

Sections 1-7 use one donor with a fixed T/NK prompt. The paper samples prompt cell types and uses all 3 donors,
so this section repeats the evaluation on the full OpenProblems data with every broad cell class as the prompt.

**Data (`scripts/40_build_full.py`).** `sc_counts.h5ad` from the OpenProblems NeurIPS 2023 bucket (raw counts in
`layers/counts`, 301,785 cells). For each donor: the same 14 compounds + DMSO as the example, from the 2 plates on
which that donor received them (31,361 / 34,043 / 32,530 cells); `cell_type` mapped to the example's three broad
classes. **The example's "donor_1" is OpenProblems "Donor 3"**: same 32,530 cells, identical cell numbers and
UMI totals per compound x cell type (the example drops 10 genes; `results/checks/donor_mapping_check.txt`).
For donor 3 the example file is used, so donor 3 / T/NK prompt is the slice of sections 1-7.

**Designs.** 3 donors x 3 prompt classes (T/NK, B-lineage, myeloid) = 9 designs; the other two classes are held
out (2 of 3 classes, the paper's hold-out ratio 0.75 rounded). Prompt = the prompt class under all compounds +
DMSO; query = DMSO cells of the held-out classes, capped at 1,000 per class (seeded) to bound T4 time. Generation
as before (T=5, generation seed 0); 7.0-7.6 min per design on a T4 (`scripts/41_run_full.sh`). 9 designs x 2
targets x 12 compounds = **216 units**; every unit has at least 13 real cells (none skipped).

**Baselines (`scripts/43_evaluate_full.py`).** `prompt_copy` = `tcell_copy` with any prompt class.
New: `nearest_ct`, the paper's "closest cell type in prompt": every query cell is matched to the prompt's fine
cell type (CD4+ T, CD8+ T, Treg, NK for the T/NK class) whose DMSO pseudobulk correlates best with it, and the
prediction samples cells of the matched types under d in those proportions (for B and myeloid prompts there is
one fine type, so it equals `prompt_copy`). Cell numbers: every method predicts min(real cells, prompt cells
under d) per unit; this equals the real number whenever the prompt is T/NK (as in sections 1-7), and is smaller
for B/myeloid prompts with a T/NK target (those prompts have 13-164 cells per compound, the T/NK target 181-533).
Scoring is unchanged: downsampling seeds 0-2, split-half ceiling, no-downsample sensitivity.
Compact predictions were pulled to the Mac after each design and scored there (`scripts/42`, `44`);
tables from `scripts/45_report_full.py` and `46_similarity_full.py`, in `results/summary_full/`.

**Pooled over 216 units (12 compounds, mean over downsampling seeds):**

| Method | Pearson Delta | PR AUC | DE overlap | DE Spearman LFC | Spearman effect size |
|---|---|---|---|---|---|
| no_effect | 0.007 | 0.091 | 0.001 | -0.164 | -0.098 |
| perturb_mean | 0.361 | 0.102 | 0.086 | 0.198 | 0.204 |
| **stack** | **0.416** | 0.210 | 0.102 | 0.159 | 0.454 |
| nearest_ct | 0.426 | 0.221 | 0.189 | 0.197 | 0.644 |
| prompt_copy | 0.445 | 0.246 | 0.178 | 0.192 | 0.705 |
| real_half (ceiling run, about half the real cells) | 0.712 | 0.405 | 0.388 | 0.280 | 0.914 |

Paired over the 216 units, Stack wins against `prompt_copy` on Pearson Delta in 95/216 (mean difference -0.029),
PR AUC 69/216, DE overlap 37/216; against `nearest_ct` 99/216, 80/216, 29/216; against `perturb_mean` 132/216,
201/216, 105/216; against `no_effect` 213/216. (Wilcoxon p-values are in `paired_wins.csv`; units share compounds
and donors, so they are not independent and the p-values are descriptive only.)

**By prompt -> target (Pearson Delta, 3 donors pooled, 36 units each):**

| Prompt -> target | Stack | prompt_copy | Stack wins | DMSO similarity of prompt and target |
|---|---|---|---|---|
| T/NK -> myeloid | **0.369** | 0.299 | **26/36** | 0.52-0.59 |
| myeloid -> T/NK | **0.490** | 0.461 | 21/36 | 0.75-0.76 |
| B -> myeloid | 0.289 | 0.325 | 17/36 | 0.54-0.60 |
| myeloid -> B | 0.355 | 0.393 | 13/36 | 0.74-0.78 |
| T/NK -> B | 0.488 | 0.534 | 14/36 | 0.84-0.90 |
| B -> T/NK | 0.502 | 0.656 | 4/36 | 0.85-0.89 |

By donor (Pearson Delta, Stack vs prompt_copy): donor 1 0.432 vs 0.465, donor 2 0.466 vs 0.485, donor 3 0.348 vs
0.384. The only designs where Stack wins both targets on average are donor 2 / T/NK prompt (B 0.541 vs 0.529,
myeloid 0.434 vs 0.360). Across the 18 design x target cells, Stack's margin over `prompt_copy` falls as the prompt
and target look more alike under DMSO (Spearman rho = -0.75, n = 18; similarity = Pearson of the DMSO pseudobulks
on the target's HVGs, `similarity_vs_gain.csv`): copying is hard to beat when the target resembles the prompt.
Similarity is not the whole story: B -> myeloid is as dissimilar as T/NK -> myeloid, yet Stack loses there; the B
prompt has only 41-164 cells per compound.

**Sensitivity: every predicted cell kept** (Stack 1,000 generated cells per class, `prompt_copy` all prompt cells
under d): Stack 0.529 / 0.332 / 0.287 / 0.496 (Pearson Delta / PR AUC / DE overlap / DE Spearman LFC) vs
`prompt_copy` 0.475 / 0.278 / 0.212 / 0.243; Stack wins 147/216, 148/216, 154/216, 188/216. With matched cell
numbers Stack loses; when it may contribute more cells than the baseline it wins on all four (Spearman effect size
stays lower: 0.39 vs 0.68). These numbers are
also the closest to the paper's Fig. 3E (0.57 / 0.39 / 0.38 / 0.50). I do not know which rule the paper used.

**Effect size.** Median slope of predicted on real pseudobulk delta: Stack 0.36, `prompt_copy` 0.43, `nearest_ct`
0.44, `perturb_mean` 0.28.

**Checks.**
- Donor 3 / T/NK reproduces sections 1-7 with the new code and a different query sample (seed 0: Stack 0.340,
  copy 0.360, vs 0.343 / 0.364 above).
- `scripts/47_check_full.py`: Pearson Delta recomputed with numpy/scanpy for 3 units x 2 methods in 3 designs
  matches the pipeline to 6 decimals (`results/checks/independent_check_full.txt`).
- The donor files built on Colab (used for generation) and on the Mac (used for scoring) have identical
  compound x cell type tables.

**Reproduce.**

```bash
# Colab T4 (colab new -s p6full --gpu T4), /content/stack-repro, after scripts/00_setup_colab.sh
wget -O /content/full/sc_counts.h5ad https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/sc_counts.h5ad
python scripts/40_build_full.py /content/full/sc_counts.h5ad data/openproblems_donor1.h5ad data/full
python scripts/40_build_full.py data/openproblems_donor1.h5ad data/openproblems_donor1.h5ad data/full_ex
#   donor 3 = the example file: copy data/full_ex/d1_{T,B,M}_{ctx,test}.h5ad to data/full/d3_*, point d3_*.donorfile at data/full_ex/d1.h5ad
bash scripts/41_run_full.sh                      # 9 generations, one .tgz of compact predictions per design
# Mac: same two builds (d3.h5ad = data/full_ex/d1.h5ad), then pull + score while Colab generates
scripts/44_pull_eval_full.sh p6full
.venv-eval/bin/python scripts/45_report_full.py && .venv-eval/bin/python scripts/46_similarity_full.py
.venv-eval/bin/python scripts/47_check_full.py
```

**Verdict.** On 3 donors and all prompt designs, with matched cell numbers, zero-shot Stack does not beat copying
the prompt cells' response (or the closest prompt cell type). It does beat the mean-shift baseline and clearly beats
"no change". Its advantage is limited to prompt/target pairs that differ more (T/NK -> myeloid: 26/36). The
paper-like lead appears only when Stack is scored with more predicted cells than the baseline.

## Limits

- 3 donors, 9 prompt designs, 12 compounds (sections 1-7: one donor, T/NK prompt); ground-truth groups have 13-533 cells.
- Section 8: one generation seed per design; query capped at 1,000 DMSO cells per class; `nearest_ct` is my
  implementation of the paper's description; units are not independent.
- State baseline: one training seed, 2,500 steps, validation loss still falling (section 7); scVI was not run. `tcell_copy` and
  `perturb_mean` are my implementations of the paper's one-line descriptions of "closest cell type in prompt"
  and "PerturbMean".
- The paper's cell-count matching rule is ambiguous to me; section 4 shows the result depends on it.
- Headline uses one generation seed (three for Stack's spread); the T=5 vs T=1 comparison uses one seed.
- Only "Setting 1" inference is covered: no pre-training, no post-training, no embedding benchmarks, no Perturb Sapiens.
- The effect of joint BH on the metric values themselves is not quantified.

## Licence note

Stack code is CC BY-NC-SA 4.0; the weights are under the Arc Research Institute Stack Model Non-Commercial License.
The example data is CC-BY-4.0 (OpenProblems single-cell perturbations, Luecken et al. 2025).
