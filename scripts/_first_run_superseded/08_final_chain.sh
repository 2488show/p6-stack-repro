#!/usr/bin/env bash
# After the other chains: (a) re-run the first evaluation and diff it against the saved copy
# (determinism check), (b) sensitivity to the classifier-logit filter used for Perturb Sapiens.
set -uo pipefail
cd /content/stack-repro && source .venv/bin/activate && export MPLBACKEND=Agg
while ! grep -q CHAIN_DONE logs/chain.log; do sleep 10; done
while ! grep -q CEILING_DONE logs/ceiling_chain.log; do sleep 10; done

python scripts/03_evaluate.py --pred-dir out/tutorial_seed0 --tag T5_gen0 --seed 0 > logs/eval_T5_gen0_seed0_rerun.log 2>&1
python - <<'EOF'
import numpy as np, pandas as pd
a = pd.read_csv("results/T5_gen0_seed0_per_pert.v1.bak")
b = pd.read_csv("results/eval/T5_gen0_seed0_per_pert.csv")
num = [c for c in a.columns if a[c].dtype.kind in "fi"]
key = ["cell_type", "method", "perturbation"]
m = a.merge(b, on=key, suffixes=("_a", "_b"))
worst = max(float(np.nanmax(np.abs(m[c + "_a"] - m[c + "_b"]))) for c in num if c not in ("seed",))
print("DETERMINISM rows", len(a), len(b), len(m), "max abs diff over", len(num), "numeric columns:", worst)
EOF

python scripts/03_evaluate.py --pred-dir out/tutorial_seed0 --tag T5_gen0_logit2.5 --seed 0 --methods stack --logit-max 2.5 \
  > logs/eval_T5_gen0_logit.log 2>&1
python - <<'EOF'
import anndata as ad, glob, numpy as np
for f in sorted(glob.glob("out/tutorial_seed0/*.h5ad"))[:4]:
    l = ad.read_h5ad(f, backed="r").obs["gen_logit"].to_numpy()
    print("LOGIT", f.split("/")[-1], "n", len(l), "frac<0", round(float((l < 0).mean()), 3), "frac<=2.5", round(float((l <= 2.5).mean()), 3))
EOF
echo "FINAL_DONE $(date)"
