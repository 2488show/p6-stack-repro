#!/usr/bin/env bash
# Predict B-lineage and myeloid cells under each compound with a trained State model, then convert the
# output to the per-compound files that scripts/03_evaluate.py reads (--state-dir).
# Usage: 23_state_infer.sh <run_name> <best|last|final> <infer_seed> <out_tag>
# `state tx infer` groups rows by cell type and compound and, for every row, feeds a control (DMSO) cell
# of the same cell type, sampled with replacement, together with the compound's one-hot vector.
# The input has the 2,848 query cells once as DMSO and once per compound, so every compound gets 1,320
# B-lineage and 1,528 myeloid predicted cells; the DMSO rows give State's own predicted control.
set -euo pipefail
NAME="${1:?run}"; CK="${2:-best}"; SEED="${3:-42}"; TAG="${4:?out tag}"
R=/content/p6state
cd "$R"
export MPLBACKEND=Agg
RUN=$R/runs/$NAME/pbmc
OUT=$R/out/$TAG
mkdir -p "$OUT"
.venv-state/bin/python scripts/state_cli.py tx infer \
  --model-dir "$RUN" --checkpoint "$RUN/checkpoints/$CK.ckpt" \
  --adata "$R/data/state/infer_input.h5ad" --embed-key X_hvg \
  --pert-col pert --celltype-col ct --batch-col plate --control-pert Dimethyl_Sulfoxide \
  --seed "$SEED" --output "$OUT/pred.h5ad"
.venv-eval/bin/python scripts/24_state_to_eval.py --pred "$OUT/pred.h5ad" --outdir "$OUT/per_compound"
echo "INFER_DONE $NAME $CK seed=$SEED -> $OUT"
