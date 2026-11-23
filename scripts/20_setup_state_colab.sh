#!/usr/bin/env bash
# State baseline: environment + data on a Colab T4 runtime (colab new -s p6state --gpu T4).
# Two venvs:
#   .venv-eval   the evaluation stack of scripts/03_evaluate.py, pinned to the versions in env_freeze.txt
#                (cell-eval 0.6.6 etc.; Stack itself and its checkpoint are not needed here)
#   .venv-state  arc-state 0.11.1 (State transition model, trained from scratch; no pretrained weights)
# Data: the same Hugging Face dataset as scripts/00_setup_colab.sh, plus Stack's gene list
# (only the 925 kB .pkl, which 03_evaluate.py uses to define the shared gene universe).
set -euo pipefail
ROOT=/content/p6state
mkdir -p "$ROOT" && cd "$ROOT"

if [ ! -x .venv-eval/bin/python ]; then
  uv venv --python 3.10 .venv-eval
  uv pip install --python .venv-eval/bin/python \
    "cell-eval==0.6.6" "anndata==0.11.4" "scanpy==1.11.5" "polars==2.0.0" "pdex==0.1.28" \
    "numpy==2.2.6" "pandas==2.3.3" "scipy==1.15.3" "scikit-learn==1.7.2" "statsmodels==0.15.0" \
    "numba==0.68.1" "h5py==3.16.0" "adpbulk==0.1.4" "huggingface-hub==2.4.0"
fi
if [ ! -x .venv-state/bin/state ]; then
  uv venv --python 3.12 .venv-state
  uv pip install --python .venv-state/bin/python "arc-state==0.11.1" "cell-load==0.10.4" "anndata==0.13.4"
fi

.venv-state/bin/python - <<'EOF'
import torch, importlib.metadata as m
print("state venv: torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| arc-state", m.version("arc-state"), "| cell-load", m.version("cell-load"), "| anndata", m.version("anndata"))
EOF
.venv-eval/bin/python - <<'EOF'
import importlib.metadata as m
print("eval venv:", {p: m.version(p) for p in ["cell-eval", "anndata", "scanpy", "polars", "pdex"]})
EOF

.venv-eval/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("arcinstitute/Stack-Large-Aligned", repo_type="model", local_dir="model",
                  allow_patterns=["basecount_1000per_15000max.pkl"])
snapshot_download("arcinstitute/Stack-DrugPBMC-Example", repo_type="dataset", local_dir="data",
                  allow_patterns=["openproblems_donor1.h5ad"])
EOF
ls -la model data
sha256sum model/basecount_1000per_15000max.pkl data/openproblems_donor1.h5ad
uv pip freeze --python .venv-state/bin/python > env_freeze_state.txt
uv pip freeze --python .venv-eval/bin/python > env_freeze_eval.txt
echo "SETUP_DONE"
