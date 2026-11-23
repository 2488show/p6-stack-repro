#!/bin/zsh
# Reach a Colab runtime through one shared ssh master connection (`colab ssh` allows only one
# connection at a time; with ControlMaster, commands and rsync can share it).
# Usage: scripts/colab_ssh.sh <session> [ssh args / command...]
S=$1; shift
exec ssh -o "ProxyCommand=$HOME/.local/bin/colab ssh --proxy-mode -s $S -i $HOME/.ssh/id_ed25519.pub" \
  -o ControlMaster=auto -o "ControlPath=/tmp/cm-$S" -o ControlPersist=12h \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
  root@colab-$S "$@"
