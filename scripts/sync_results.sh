#!/usr/bin/env bash
# Local side: pull the small result snapshot off the Colab runtime every 90 s and unpack it,
# so results survive if the runtime is reclaimed. Stops when ALL_DONE is seen or after 3 h.
cd "$HOME/Final Project/p6-stack-repro" || exit 1
C="$HOME/.local/bin/colab"
mkdir -p results/remote
END=$(( $(date +%s) + 10800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  if "$C" download -s stack /content/stack-repro/snapshot.tgz results/remote/snapshot.new.tgz > /dev/null 2>&1; then
    if tar tzf results/remote/snapshot.new.tgz > /dev/null 2>&1; then
      mv results/remote/snapshot.new.tgz results/remote/snapshot.tgz
      tar xzf results/remote/snapshot.tgz -C results/remote
      echo "$(date +%H:%M:%S) synced $(ls results/remote/results/eval 2>/dev/null | wc -l | tr -d ' ') csv"
    fi
  fi
  "$C" download -s stack /content/stack-repro/logs/run_all.log results/remote/run_all.log > /dev/null 2>&1
  if grep -q ALL_DONE results/remote/run_all.log 2>/dev/null; then
    sleep 20
    "$C" download -s stack /content/stack-repro/snapshot.tgz results/remote/snapshot.tgz > /dev/null 2>&1
    tar xzf results/remote/snapshot.tgz -C results/remote
    echo "$(date +%H:%M:%S) ALL_DONE final sync"
    break
  fi
  sleep 90
done
