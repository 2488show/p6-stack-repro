#!/usr/bin/env bash
# Noise-ceiling runs: score every method against a random half of the real cells and
# score the other half as if it were a prediction ("real_half"). Three split seeds.
set -uo pipefail
cd /content/stack-repro && source .venv/bin/activate && export MPLBACKEND=Agg
for S in 0 1 2; do
  python scripts/03_evaluate.py --pred-dir out/tutorial_seed0 --tag T5_gen0_split --seed "$S" --split-real \
    > "logs/eval_T5_gen0_split_seed$S.log" 2>&1
  echo "SPLIT_seed${S}_DONE $(date)"
done
echo "CEILING_DONE $(date)"
