#!/usr/bin/env bash
# Run as the VM's training user, on its persistent local disk.
set -euo pipefail
TRAINING_COMMIT=${1:?Pass the pushed training commit SHA}
INFERENCE_COMMIT=6e38839e46a61d2583dc6c72dd6c672a19cd6a94
WORKSPACE="$HOME/mmai"
sudo apt-get update
sudo apt-get install -y git curl rsync build-essential python3-dev
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
uv pip freeze --python "$HOME/thisenv/bin/python" > "$WORKSPACE/environment-freeze.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$HOME/thisenv/bin/python" -c 'import torch, transformers, sentence_transformers, vllm; print("GPUs:", torch.cuda.device_count(), "Torch:", torch.__version__, "Transformers:", transformers.__version__, "vLLM:", vllm.__version__)'
