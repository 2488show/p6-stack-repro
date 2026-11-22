#!/usr/bin/env bash
# Sensitivity check: do the conclusions depend on matching the number of predicted cells
# to the number of real cells? Here every predicted cell is kept (maximum DE power for predictions).
set -uo pipefail
cd /content/stack-repro && source .venv/bin/activate && export MPLBACKEND=Agg
while ! grep -q FINAL_DONE logs/final_chain.log; do sleep 10; done
python scripts/03_evaluate.py --pred-dir out/tutorial_seed0 --tag T5_gen0_allcells --seed 0 --no-downsample \
  --methods stack tcell_copy perturb_mean > logs/eval_T5_gen0_allcells.log 2>&1
echo "SENS_DONE $(date)"
