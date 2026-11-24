#!/usr/bin/env bash
# Colab GPU lane for the 3-donor run: one Stack generation per design (donor x prompt class), with the
# official tutorial parameters (T=5, prompt-ratio 0.25, context-ratio 0.4, seed 0). After each design
# the predictions are shrunk (42_compact_pred.py) and packed into out/full/<design>.tgz, which the
# Mac pulls (44_pull_full.sh) and scores locally, so a reclaimed runtime loses at most one design.
# Usage: 41_run_full.sh [designs file]   (default data/full/designs.txt, one design per line)
set -uo pipefail
cd /content/stack-repro
source .venv/bin/activate
export MPLBACKEND=Agg
DATA="${DATA:-data/full}"
LIST="${1:-$DATA/designs.txt}"
mkdir -p out/full logs
echo "FULL_START $(date -u +%FT%TZ)"
while read -r D; do
  [ -z "$D" ] && continue
  [ -f "out/full/$D.done" ] && { echo "skip $D"; continue; }
  DONOR_FILE=$(cat "$DATA/$D.donorfile")
  S=$(date +%s)
  stack-generation --checkpoint model/bc_large_aligned.ckpt \
    --base-adata "$DATA/${D}_ctx.h5ad" --test-adata "$DATA/${D}_test.h5ad" \
    --genelist model/basecount_1000per_15000max.pkl --split-column sm_name \
    --batch-size 16 --num-steps 5 --num-workers 2 --random-seed 0 \
    --output-dir "out/full/$D/gen" --prompt-ratio 0.25 --context-ratio 0.4 > "logs/gen_$D.log" 2>&1
  RC=$?
  G=$(( $(date +%s) - S ))
  python scripts/42_compact_pred.py "out/full/$D/gen" "$DONOR_FILE" model/basecount_1000per_15000max.pkl \
    "out/full/$D/compact" > "logs/compact_$D.log" 2>&1
  tar czf "out/full/$D.tmp.tgz" -C "out/full/$D" compact && mv "out/full/$D.tmp.tgz" "out/full/$D.tgz"
  rm -rf "out/full/$D/gen"
  echo "$G" > "out/full/$D.done"
  echo "GEN $D exit=$RC gen_seconds=$G $(date -u +%H:%M:%S)"
done < "$LIST"
[ "$DATA" = data/full ] && touch logs/FULL_DONE
echo "FULL_DONE $(date -u +%FT%TZ)"
