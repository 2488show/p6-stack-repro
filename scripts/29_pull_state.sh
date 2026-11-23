#!/bin/zsh
# Mac side: every 3 minutes, pull the State run's small outputs (logs, metrics, results) and the slim best
# checkpoint off the Colab runtime into results/remote/state/ (git-ignored), so nothing is lost if the
# runtime disappears. Stops when RUN_DONE <last run> is in the log, or after <hours>.
# Usage: scripts/29_pull_state.sh <session> <last_run_name> [hours]
S=$1; LAST=$2; H=${3:-6}
cd "$HOME/Final Project/p6-stack-repro" || exit 1
D=results/remote/state; mkdir -p $D
RS=(rsync -a --timeout=600 -e "ssh -o ControlPath=/tmp/cm-$S -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR")
END=$(( $(date +%s) + H * 3600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  zsh scripts/colab_ssh.sh $S true 2>/dev/null
  $RS --include='*/' --include='logs/***' --include='results/***' --include='data/state/build_info.json' \
      --include='data/state/pbmc.toml' --include='env_freeze_*.txt' --include='runs/*/pbmc/version_0/metrics.csv' \
      --include='runs/*/pbmc/*.yaml' --include='runs/*/pbmc/*.pt' --include='runs/*/pbmc/*.pkl' --include='runs/*/pbmc/*.torch' \
      --include='runs/*/pbmc/checkpoints/best_slim.ckpt' --include='out/*/per_compound/conversion_stats.json' \
      --exclude='*' --prune-empty-dirs root@colab-$S:/content/p6state/ $D/ 2>&1 | tail -2
  echo "$(date -u +%FT%TZ) pulled: $(ls $D/results/eval 2>/dev/null | wc -l | tr -d ' ') eval csv; slim: $(ls -la $D/runs/*/pbmc/checkpoints/best_slim.ckpt 2>/dev/null | wc -l | tr -d ' ')"
  grep -q "RUN_DONE $LAST" $D/logs/pipeline.log 2>/dev/null && { echo "RUN_DONE seen"; break; }
  sleep 180
done
