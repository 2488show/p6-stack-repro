#!/bin/zsh
# Mac side of the 3-donor run: pull each finished design's compact Stack predictions off the Colab
# runtime as soon as it exists, then score it locally (up to 3 evaluations at a time):
#   main  : downsampling seeds 0, 1, 2       (tag F_<design>)
#   split : noise ceiling, seed 0            (tag F_<design>_split)
#   all   : no downsampling, seed 0          (tag F_<design>_allcells)
# Needs data/full/<donor>.h5ad built locally by 40_build_full.py (d3 = the Stack example file). Usage: 44_pull_eval_full.sh <session>
S="${1:-p6full}"
cd "$HOME/p6r" || exit 1
export MPLBACKEND=Agg
PY=.venv-eval/bin/python
mkdir -p out/full logs/full results/eval_full
typeset -A PCT
PCT=(T "t cell" B "lymphocyte of b lineage" M "myeloid leukocyte")
DESIGNS=(d2_T d3_T d1_B d1_M d2_B d2_M d3_B d3_M d1_T)

ev() {  # ev <log> <args...>
  local log=$1; shift
  $PY scripts/43_evaluate_full.py "$@" > "logs/full/$log.log" 2>&1
  echo "EVAL $log exit=$? $(date +%H:%M:%S)"
}
eval_design() {
  local D=$1 P=${PCT[${1##*_}]} DON=data/full/${1%%_*}.h5ad PD=out/full/$1/compact
  for s in 0 1 2; do ev "F_${D}_s$s" --data $DON --prompt-ct "$P" --pred-dir $PD --tag F_$D --seed $s; done
  ev "F_${D}_split" --data $DON --prompt-ct "$P" --pred-dir $PD --tag F_${D}_split --seed 0 --split-real
  ev "F_${D}_all" --data $DON --prompt-ct "$P" --pred-dir $PD --tag F_${D}_allcells --seed 0 --no-downsample \
    --methods stack prompt_copy perturb_mean
  touch out/full/$D.evaluated
}

END=$(( $(date +%s) + 6 * 3600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  left=0
  for D in $DESIGNS; do
    [ -f out/full/$D.evaluated ] && continue
    left=1
    [ -f out/full/$D.evalstarted ] && continue
    if [ ! -f out/full/$D.tgz ]; then
      if scripts/colab_ssh.sh $S "test -f /content/stack-repro/out/full/$D.done" 2>/dev/null; then
        scripts/colab_ssh.sh $S "cat /content/stack-repro/out/full/$D.tgz" > out/full/$D.part 2>/dev/null \
          && tar tzf out/full/$D.part > /dev/null 2>&1 && mv out/full/$D.part out/full/$D.tgz \
          && echo "PULLED $D $(du -h out/full/$D.tgz | cut -f1) $(date +%H:%M:%S)"
        scripts/colab_ssh.sh $S "cat /content/stack-repro/logs/gen_$D.log; cat /content/stack-repro/out/full/$D.done" \
          > logs/full/gen_$D.log 2>/dev/null
      fi
    fi
    if [ -f out/full/$D.tgz ] && [ -f data/full/${D%%_*}.h5ad ]; then
      while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 10; done
      mkdir -p out/full/$D && tar xzf out/full/$D.tgz -C out/full/$D
      touch out/full/$D.evalstarted
      echo "EVAL_START $D $(date +%H:%M:%S)"
      eval_design $D &
    fi
  done
  [ $left -eq 0 ] && break
  sleep 60
done
wait
echo "ALL_EVAL_DONE $(date +%H:%M:%S)"
