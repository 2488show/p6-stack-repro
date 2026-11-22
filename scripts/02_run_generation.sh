#!/usr/bin/env bash
# Official tutorial setting (paper "Setting 1"): OpenProblems PBMC donor 1.
# Prompt  = T/NK cells of donor 1 under each compound (split by sm_name, DMSO included).
# Query   = DMSO-treated B and myeloid cells of donor 1.
# Output  = one predicted AnnData per compound (+ DMSO = "synthetic control").
# Usage: 02_run_generation.sh <seed> <num_steps> <outdir> [batch_size]
set -euo pipefail
SEED="${1:-0}"; STEPS="${2:-5}"; OUT="${3:-out/tutorial_seed0}"; BS="${4:-16}"
cd /content/stack-repro
source .venv/bin/activate
export MPLBACKEND=Agg  # the Colab kernel exports an inline backend the venv does not have

if [ ! -f data/ctx_rebuilt.h5ad ]; then
python - <<'EOF'
# Rebuild context/test exactly as in notebooks/tutorial-predict.ipynb (cell 6).
import scanpy as sc
adata = sc.read_h5ad("data/openproblems_donor1.h5ad")
ctx = adata[adata.obs["broad_cell_class"].isin(["t cell"])]
test = adata[(~adata.obs["broad_cell_class"].isin(["t cell"])) & (adata.obs["sm_name"] == "Dimethyl Sulfoxide")]
ctx.write("data/ctx_rebuilt.h5ad")
test.write("data/test_rebuilt.h5ad")
print("context", ctx.shape, "test", test.shape)
EOF
fi

START=$(date +%s)
stack-generation --checkpoint model/bc_large_aligned.ckpt \
  --base-adata data/ctx_rebuilt.h5ad --test-adata data/test_rebuilt.h5ad \
  --genelist model/basecount_1000per_15000max.pkl --split-column sm_name \
  --batch-size "$BS" --num-steps "$STEPS" --num-workers 2 --random-seed "$SEED" \
  --output-dir "$OUT" --prompt-ratio 0.25 --context-ratio 0.4
END=$(date +%s)
echo "GENERATION_DONE seed=$SEED steps=$STEPS out=$OUT seconds=$((END-START))"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
ls -la "$OUT"
