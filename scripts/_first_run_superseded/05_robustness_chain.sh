#!/usr/bin/env bash
# Robustness runs, executed one after another on the Colab runtime:
#   1) one-step generation (paper: multi-step "modest advantage" over one-step)
#   2) two more generation seeds for T=5 (generation is stochastic)
#   3) two more downsampling seeds for the first T=5 run (all methods)
set -uo pipefail
cd /content/stack-repro && source .venv/bin/activate && export MPLBACKEND=Agg

while ! grep -q EVAL_DONE logs/eval_T5_gen0_seed0.log; do sleep 5; done

bash scripts/02_run_generation.sh 0 1 out/onestep_seed0 16 > logs/gen_onestep_seed0.log 2>&1
python scripts/03_evaluate.py --pred-dir out/onestep_seed0 --tag T1_gen0 --seed 0 --methods stack > logs/eval_T1_gen0_seed0.log 2>&1
echo "STAGE1_DONE $(date)"

for G in 1 2; do
  bash scripts/02_run_generation.sh "$G" 5 "out/tutorial_seed$G" 16 > "logs/gen_seed$G.log" 2>&1
  python scripts/03_evaluate.py --pred-dir "out/tutorial_seed$G" --tag "T5_gen$G" --seed 0 --methods stack > "logs/eval_T5_gen${G}_seed0.log" 2>&1
  echo "STAGE2_gen${G}_DONE $(date)"
done

for S in 1 2; do
  python scripts/03_evaluate.py --pred-dir out/tutorial_seed0 --tag T5_gen0 --seed "$S" > "logs/eval_T5_gen0_seed$S.log" 2>&1
  echo "STAGE3_seed${S}_DONE $(date)"
done
echo "CHAIN_DONE $(date)"
