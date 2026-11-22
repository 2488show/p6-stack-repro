#!/usr/bin/env bash
# Whole experiment on one Colab T4 runtime, two lanes:
#   GPU lane: four generation runs (T=5 seeds 0,1,2 and one-step seed 0)
#   CPU lane: evaluations, started as soon as the predictions they need exist
# After every evaluation the small result files are packed into snapshot.tgz so they can be
# pulled off the runtime while it is still alive (a first runtime was reclaimed mid-run).
set -uo pipefail
cd /content/stack-repro && source .venv/bin/activate && export MPLBACKEND=Agg
mkdir -p logs results/eval results/summary out

snap() {
  tar czf snapshot.tmp.tgz results/eval/*_per_pert.csv results/summary logs/*.log env_freeze.txt 2>/dev/null
  mv snapshot.tmp.tgz snapshot.tgz
}
ev() {  # ev <logname> <args...>
  local log="$1"; shift
  python scripts/03_evaluate.py "$@" > "logs/$log.log" 2>&1
  echo "EVAL $log exit=$? $(date -u +%H:%M:%S)"
  snap
}
wait_for() { while [ ! -f "$1" ]; do sleep 5; done; }

gpu_lane() {
  bash scripts/02_run_generation.sh 0 5 out/tutorial_seed0 16 > logs/gen_seed0.log 2>&1;        touch out/.gen0; echo "GEN gen0 $(date -u +%H:%M:%S)"
  bash scripts/02_run_generation.sh 0 1 out/onestep_seed0 16  > logs/gen_onestep_seed0.log 2>&1; touch out/.t1;   echo "GEN t1 $(date -u +%H:%M:%S)"
  bash scripts/02_run_generation.sh 1 5 out/tutorial_seed1 16 > logs/gen_seed1.log 2>&1;        touch out/.gen1; echo "GEN gen1 $(date -u +%H:%M:%S)"
  bash scripts/02_run_generation.sh 2 5 out/tutorial_seed2 16 > logs/gen_seed2.log 2>&1;        touch out/.gen2; echo "GEN gen2 $(date -u +%H:%M:%S)"
}

cpu_lane() {
  wait_for out/.gen0
  ev eval_T5_gen0_seed0       --pred-dir out/tutorial_seed0 --tag T5_gen0 --seed 0
  ev eval_T5_gen0_split_seed0 --pred-dir out/tutorial_seed0 --tag T5_gen0_split --seed 0 --split-real
  wait_for out/.t1
  ev eval_T1_gen0_seed0       --pred-dir out/onestep_seed0 --tag T1_gen0 --seed 0 --methods stack
  ev eval_T5_gen0_seed1       --pred-dir out/tutorial_seed0 --tag T5_gen0 --seed 1
  wait_for out/.gen1
  ev eval_T5_gen1_seed0       --pred-dir out/tutorial_seed1 --tag T5_gen1 --seed 0 --methods stack
  ev eval_T5_gen0_split_seed1 --pred-dir out/tutorial_seed0 --tag T5_gen0_split --seed 1 --split-real
  wait_for out/.gen2
  ev eval_T5_gen2_seed0       --pred-dir out/tutorial_seed2 --tag T5_gen2 --seed 0 --methods stack
  ev eval_T5_gen0_seed2       --pred-dir out/tutorial_seed0 --tag T5_gen0 --seed 2
  ev eval_T5_gen0_split_seed2 --pred-dir out/tutorial_seed0 --tag T5_gen0_split --seed 2 --split-real
  ev eval_T5_gen0_allcells    --pred-dir out/tutorial_seed0 --tag T5_gen0_allcells --seed 0 --no-downsample --methods stack tcell_copy perturb_mean
  ev eval_T5_gen0_logit       --pred-dir out/tutorial_seed0 --tag T5_gen0_logit2.5 --seed 0 --methods stack --logit-max 2.5
  python scripts/07_report.py > results/summary/report.txt 2>&1
  python - > results/summary/gen_logit.txt 2>&1 <<'EOF'
import anndata as ad, glob
for f in sorted(glob.glob("out/tutorial_seed0/*.h5ad")):
    l = ad.read_h5ad(f, backed="r").obs["gen_logit"].to_numpy()
    print(f.split("/")[-1], "n", len(l), "frac_logit<0", round(float((l < 0).mean()), 3), "frac_logit<=2.5", round(float((l <= 2.5).mean()), 3))
EOF
  snap
  echo "ALL_DONE $(date -u +%H:%M:%S)"
}

echo "START $(date -u +%H:%M:%S)"
gpu_lane &
cpu_lane &
wait
