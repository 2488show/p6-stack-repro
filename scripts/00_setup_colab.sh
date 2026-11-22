#!/usr/bin/env bash
# Environment + data setup on the Colab T4 runtime.
# Pins: Stack repo commit, torch build, cell-eval version used in the paper (v0.6.6).
set -euo pipefail

STACK_COMMIT="cacc2e4b09435c3e536d46237d10b50f222dd144"
ROOT=/content/stack-repro
mkdir -p "$ROOT" && cd "$ROOT"

if [ ! -d .venv ]; then
  uv venv --python 3.10 .venv
fi
source .venv/bin/activate

uv pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
uv pip install \
  "git+https://github.com/ArcInstitute/stack.git@${STACK_COMMIT}" \
  "cell-eval==0.6.6" scanpy huggingface_hub

python - <<'EOF'
import torch, stack, importlib.metadata as md
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
for p in ["arc-stack", "cell-eval", "anndata", "scanpy", "scvi-tools", "pytorch-lightning"]:
    print(p, md.version(p))
EOF

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("arcinstitute/Stack-Large-Aligned", repo_type="model", local_dir="model")
snapshot_download("arcinstitute/Stack-DrugPBMC-Example", repo_type="dataset", local_dir="data")
EOF

ls -la model data
sha256sum model/*.ckpt
uv pip freeze > env_freeze.txt
echo "SETUP_DONE"
