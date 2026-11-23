#!/bin/zsh
# Ask Colab for a T4 runtime every 5 minutes until one is granted or the time limit is reached.
# Usage: scripts/retry_gpu.sh <session> <minutes>
S=$1; MIN=${2:-60}
END=$(( $(date +%s) + MIN * 60 ))
while [ "$(date +%s)" -lt "$END" ]; do
  if "$HOME/.local/bin/colab" new -s "$S" --gpu T4 2>&1 | grep -q "Session READY"; then
    echo "$(date -u +%FT%TZ) GPU_READY $S"; exit 0
  fi
  echo "$(date -u +%FT%TZ) no T4 (Service Unavailable); retry in 5 min"
  sleep 300
done
echo "$(date -u +%FT%TZ) GIVE_UP after $MIN min"; exit 1
