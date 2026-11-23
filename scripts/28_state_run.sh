#!/usr/bin/env bash
# One State run on the Colab runtime, start to finish: train, keep a slim copy of the best checkpoint
# up to date (so it can be pulled off the runtime at any time), then predict and evaluate.
# Usage: 28_state_run.sh <run_name> <max_steps> <train_seed> [full|main]
# The first runtime was lost after about 1 h 50 min of T4 time with everything on it (README), hence the
# slim-checkpoint loop and the Mac-side puller (scripts/29_pull_state.sh).
set -uo pipefail
RUN="${1:?run}"; STEPS="${2:?steps}"; SEED="${3:-42}"; MODE="${4:-full}"
R=/content/p6state
cd "$R"
CK=runs/$RUN/pbmc/checkpoints
slim_loop() {
  while true; do
    if [ -f "$CK/best.ckpt" ] && { [ ! -f "$CK/best_slim.ckpt" ] || [ "$CK/best.ckpt" -nt "$CK/best_slim.ckpt" ]; }; then
      sleep 30
      .venv-state/bin/python scripts/state_slim_ckpt.py "$CK/best.ckpt" "$CK/best_slim.ckpt" >> logs/slim_$RUN.log 2>&1
    fi
    [ -f "logs/.train_done_$RUN" ] && break
    sleep 120
  done
}
slim_loop &
bash scripts/22_state_train.sh "$RUN" "$STEPS" "$SEED" 100 > "logs/train_$RUN.log" 2>&1
grep TRAIN_EXIT "logs/train_$RUN.log"
.venv-state/bin/python scripts/state_slim_ckpt.py "$CK/best.ckpt" "$CK/best_slim.ckpt" >> logs/slim_$RUN.log 2>&1
touch "logs/.train_done_$RUN"
wait
bash scripts/26_state_eval_all.sh "$RUN" "$MODE"
echo "RUN_DONE $RUN $(date -u +%FT%TZ)"
