#!/usr/bin/env bash
# Run as the VM's training user, on its persistent local disk.
set -euo pipefail
TRAINING_COMMIT=${1:?Pass the pushed training commit SHA}
INFERENCE_COMMIT=6e38839e46a61d2583dc6c72dd6c672a19cd6a94
WORKSPACE="$HOME/mmai"
sudo apt-get update
sudo apt-get install -y git curl rsync build-essential python3-dev ninja-build
curl --fail --show-error --location https://astral.sh/uv/install.sh --output /tmp/install-uv.sh
sh /tmp/install-uv.sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WORKSPACE/data/no_phi" "$WORKSPACE/models" "$WORKSPACE/cache"
cd "$WORKSPACE"
if [[ ! -d matchminer-ai-training-kehl/.git ]]; then
  git clone --branch v23 https://github.com/kenlkehl/matchminer-ai-training-kehl.git
fi
git -C matchminer-ai-training-kehl fetch origin v23
git -C matchminer-ai-training-kehl checkout --detach "$TRAINING_COMMIT"
if [[ ! -d matchminer-ai-inference-kehl/.git ]]; then
  git clone --branch v23 https://github.com/kenlkehl/matchminer-ai-inference-kehl.git
fi
git -C matchminer-ai-inference-kehl checkout --detach "$INFERENCE_COMMIT"
uv venv --python 3.13 "$HOME/thisenv"
uv pip install --python "$HOME/thisenv/bin/python" \
  -r matchminer-ai-training-kehl/pyproject.toml -e ./matchminer-ai-inference-kehl pytest
# FlashInfer JIT requires the compiler and PyTorch's runtime headers to share
# their CUDA major/minor version; an unconstrained nvcc can resolve newer.
MMAI_CUDA_RANGE=$("$HOME/thisenv/bin/python" -c 'import torch; major, minor = map(int, torch.version.cuda.split(".")[:2]); print(f">={major}.{minor},<{major}.{minor+1}")')
uv pip install --python "$HOME/thisenv/bin/python" \
  "nvidia-cuda-nvcc$MMAI_CUDA_RANGE" "nvidia-nvvm$MMAI_CUDA_RANGE" "nvidia-cuda-crt$MMAI_CUDA_RANGE"
# The CUDA pip wheels use lib/ and a versioned libcudart name. FlashInfer's
# toolkit linker expects the conventional lib64/ and libcudart.so names.
"$HOME/thisenv/bin/python" - <<'PY'
from pathlib import Path
import sysconfig
import torch
major = torch.version.cuda.split('.')[0]
root = Path(sysconfig.get_paths()['purelib']) / 'nvidia' / f'cu{major}'
if not (root / 'lib64').exists():
    (root / 'lib64').symlink_to('lib', target_is_directory=True)
library = root / 'lib' / 'libcudart.so'
if not library.exists():
    library.symlink_to(f'libcudart.so.{major}')
PY
uv pip freeze --python "$HOME/thisenv/bin/python" > "$WORKSPACE/environment-freeze.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$HOME/thisenv/bin/python" -c 'import torch, transformers, sentence_transformers, vllm; print("GPUs:", torch.cuda.device_count(), "Torch:", torch.__version__, "Transformers:", transformers.__version__, "vLLM:", vllm.__version__)'
