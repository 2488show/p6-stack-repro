#!/usr/bin/env bash
# After training: predict with the best checkpoint (lowest validation loss) and score it with
# scripts/03_evaluate.py exactly like the other methods.
# Usage: 26_state_eval_all.sh <run_name> [full|main]
#   main: downsampled evaluation, downsampling seeds 0, 1, 2           (tag STATE_<run>)
#   full: main + no-downsample sensitivity (STATE_<run>_allcells) + real-control sensitivity (STATE_<run>_realctrl)
# After every evaluation the small result files are packed into snapshot_state.tgz for pulling.
set -uo pipefail
RUN="${1:?run}"; MODE="${2:-full}"
R="${P6_ROOT:-/content/p6state}"
cd "$R"
export MPLBACKEND=Agg
mkdir -p results/eval results/summary logs

snap() {
  tar czf snapshot_state.tmp.tgz results/eval/STATE_*_per_pert.csv results/eval/CHECK_*_per_pert.csv \
    results/summary logs data/state/build_info.json data/state/pbmc.toml env_freeze_*.txt \
    runs/*/pbmc/version_0/metrics.csv runs/*/pbmc/config.yaml out/*/per_compound/conversion_stats.json 2>/dev/null
  mv snapshot_state.tmp.tgz snapshot_state.tgz
}
ev() {  # ev <logname> <args...>
  local log="$1"; shift
  .venv-eval/bin/python scripts/03_evaluate.py "$@" > "logs/$log.log" 2>&1
  echo "EVAL $log exit=$? $(date -u +%H:%M:%S)"
  snap
}

echo "EVAL_ALL_START $RUN $MODE $(date -u +%FT%TZ)"
bash scripts/23_state_infer.sh "$RUN" best 42 "state_$RUN" > "logs/infer_$RUN.log" 2>&1
echo "INFER exit=$? $(date -u +%H:%M:%S)"
SD="out/state_$RUN/per_compound"
for s in 0 1 2; do
  ev "eval_STATE_${RUN}_seed$s" --state-dir "$SD" --tag "STATE_$RUN" --seed "$s" --methods state
done
if [ "$MODE" = "full" ]; then
  ev "eval_STATE_${RUN}_allcells" --state-dir "$SD" --tag "STATE_${RUN}_allcells" --seed 0 --no-downsample --methods state
  ev "eval_STATE_${RUN}_realctrl" --state-dir "$SD" --tag "STATE_${RUN}_realctrl" --seed 0 --methods state_realctrl
fi
snap
echo "EVAL_ALL_DONE $RUN $(date -u +%FT%TZ)"
