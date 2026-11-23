#!/usr/bin/env bash
# Train State's transition model (arc-state 0.11.1, model=state) from scratch on the Setting 1 split.
# Usage: 22_state_train.sh <run_name> <max_steps> <train_seed> [val_freq]
# Re-entrant: if runs/<run_name>/pbmc/checkpoints/last.ckpt exists, state resumes from it.
# Choices (see README, "State baseline"):
#   cell_set_len 256 (default 512 is larger than most training groups: T/NK x compound has 318-533 cells
#   for the 12 main compounds), batch_size 8 sets, everything else = arc-state defaults of model=state
#   (hidden 768, 8 llama layers, energy loss, lr 1e-4). num_workers 2: the T4 runtime has 2 vCPUs, 12 GB RAM.
#   val_loss = positive controls (Belinostat, Dabrafenib) in B-lineage and myeloid cells, never trained on;
#   best.ckpt = lowest val_loss, last.ckpt = most recent.
#   use_consecutive_loading=true (cell-load's batched reader): sets are read as contiguous blocks of the file
#   (grouped by cell type, then compound) instead of cell by cell. In a 60-step benchmark this took about
#   3 min instead of 9 min; the per-cell reader on 2 vCPUs left the GPU idle (README section 7).
set -uo pipefail
NAME="${1:?run name}"; STEPS="${2:?max steps}"; SEED="${3:-42}"; VF="${4:-100}"
R=/content/p6state
cd "$R"
export MPLBACKEND=Agg
mkdir -p logs runs
echo "START $NAME steps=$STEPS seed=$SEED val_freq=$VF $(date -u +%FT%TZ)"
.venv-state/bin/python scripts/state_cli.py tx train \
  data.kwargs.toml_config_path=$R/data/state/pbmc.toml \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.pert_col=pert \
  data.kwargs.cell_type_key=ct \
  data.kwargs.batch_col=plate \
  data.kwargs.control_pert=Dimethyl_Sulfoxide \
  data.kwargs.output_space=all \
  data.kwargs.num_workers=2 \
  data.kwargs.use_consecutive_loading=true \
  model=state \
  model.kwargs.cell_set_len=256 \
  training.batch_size=8 \
  training.max_steps="$STEPS" \
  training.val_freq="$VF" \
  training.ckpt_every_n_steps="$VF" \
  training.train_seed="$SEED" \
  use_wandb=false \
  output_dir=$R/runs/$NAME \
  name=pbmc ${EXTRA:-}
echo "TRAIN_EXIT=$? $NAME $(date -u +%FT%TZ)"
