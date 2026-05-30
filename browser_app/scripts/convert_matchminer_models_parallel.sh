#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-artifacts/models}"
CACHE_DIR="${CACHE_DIR:-../../models/hf_cache}"
GPUS="${GPUS:-5,6,7}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
MODELS=(TrialSpace TrialChecker BoilerplateChecker)

for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
  echo "[start] $model on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  env -u VIRTUAL_ENV PYTHONNOUSERSITE=1 \
  uv run --no-project \
    --with torch \
    --with transformers \
    --with huggingface_hub \
    --with onnx \
    --with onnxruntime \
    --with safetensors \
    python scripts/convert_matchminer_models.py \
      --only "$model" \
      --output-dir "$OUT_DIR" \
      --cache-dir "$CACHE_DIR" \
      "$@" &
done

wait
echo "[done] parallel conversion complete"
