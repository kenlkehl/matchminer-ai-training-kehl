#!/usr/bin/env bash
# Start the GCP vLLM pool and label synthetic clinical oncologist notes through it.
#
# Required/typical environment:
#   GCP_INSTANCES_FILE=/path/to/instances.txt
#   MODEL=nvidia/Gemma-4-31B-IT-NVFP4
#
# Optional environment:
#   PYTHON_ENV=/home/kenneth_kehl/thisenv
#   SERVERS_FILE=/tmp/mmai_synthetic_clinical_servers.json
#   INCLUDE_SELF=0,1,2,3,4,5,6,7   # use orchestrator GPUs too; default empty
#   STOP_INSTANCES_ON_EXIT=1        # stop worker VMs after labeling; default 1
#   GCP_MAX_MODEL_LEN=50000
#   GCP_MAX_NUM_SEQS=900
#   GCP_GPU_MEMORY_UTILIZATION=0.90
#   GCP_GPUS_PER_SERVER=1
#   MAX_CONCURRENT_PER_SERVER=50
#   RESULTS_PER_SHARD=200
#   LABEL_REQUEST_TIMEOUT=600
#   MAX_TOKENS=10000
#   RESUME=1
#
# Extra CLI args passed to label_oncology_notes.py can be appended after --.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$(cd "$REPO_ROOT/.." && pwd)/data/no_phi}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GCP_INSTANCES_FILE="${GCP_INSTANCES_FILE:-$REPO_ROOT/instances.txt}"
PYTHON_ENV="${PYTHON_ENV:-/home/kenneth_kehl/thisenv}"
SERVERS_FILE="${SERVERS_FILE:-/tmp/mmai_synthetic_clinical_servers.json}"
MODEL="${MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}"
REASONING_PARSER="${REASONING_PARSER:-auto}"
INCLUDE_SELF="${INCLUDE_SELF:-}"
STOP_INSTANCES_ON_EXIT="${STOP_INSTANCES_ON_EXIT:-1}"

INPUT_PARQUET="${INPUT_PARQUET:-$DATA_DIR/synthetic_clinical.parquet}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$DATA_DIR/synthetic_clinical_vllm_labeled.jsonl}"

GCP_MAX_MODEL_LEN="${GCP_MAX_MODEL_LEN:-50000}"
GCP_MAX_NUM_SEQS="${GCP_MAX_NUM_SEQS:-900}"
GCP_GPU_MEMORY_UTILIZATION="${GCP_GPU_MEMORY_UTILIZATION:-0.90}"
GCP_GPUS_PER_SERVER="${GCP_GPUS_PER_SERVER:-1}"
GCP_BASE_PORT="${GCP_BASE_PORT:-8000}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
MIN_SERVERS="${MIN_SERVERS:-1}"

MAX_CONCURRENT_PER_SERVER="${MAX_CONCURRENT_PER_SERVER:-50}"
RESULTS_PER_SHARD="${RESULTS_PER_SHARD:-200}"
LABEL_REQUEST_TIMEOUT="${LABEL_REQUEST_TIMEOUT:-600}"
MAX_TOKENS="${MAX_TOKENS:-10000}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RESUME="${RESUME:-1}"

ORCH_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  if [[ -n "$ORCH_PID" ]]; then
    kill -TERM "$ORCH_PID" 2>/dev/null || true
    wait "$ORCH_PID" 2>/dev/null || true
    ORCH_PID=""
  fi

  if [[ -n "$INCLUDE_SELF" ]]; then
    pkill -TERM -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
    sleep 3
    pkill -KILL -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  fi

  if [[ "$STOP_INSTANCES_ON_EXIT" == "1" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/gcp_vllm_orchestrator.py" stop-instances \
      --instances-file "$GCP_INSTANCES_FILE" || true
  fi

  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ ! -f "$GCP_INSTANCES_FILE" ]]; then
  echo "GCP instances file not found: $GCP_INSTANCES_FILE" >&2
  exit 1
fi

if [[ ! -f "$INPUT_PARQUET" ]]; then
  echo "Input parquet not found; preparing synthetic extractor inputs..." >&2
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_synthetic_data.py"
fi

serve_cmd=(
  "$PYTHON_BIN" "$REPO_ROOT/gcp_vllm_orchestrator.py" serve
  --instances-file "$GCP_INSTANCES_FILE"
  --python-env "$PYTHON_ENV"
  --servers-file "$SERVERS_FILE"
  --model "$MODEL"
  --reasoning-parser "$REASONING_PARSER"
  --gpus-per-server "$GCP_GPUS_PER_SERVER"
  --base-port "$GCP_BASE_PORT"
  --max-model-len "$GCP_MAX_MODEL_LEN"
  --max-num-seqs "$GCP_MAX_NUM_SEQS"
  --gpu-memory-utilization "$GCP_GPU_MEMORY_UTILIZATION"
  --download-dir "${DOWNLOAD_DIR:-~/models}"
)

if [[ -n "$INCLUDE_SELF" ]]; then
  serve_cmd+=(--include-self "$INCLUDE_SELF")
fi

if [[ -n "${GCP_EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_vllm_args=($GCP_EXTRA_VLLM_ARGS)
  serve_cmd+=(--extra-vllm-args "${extra_vllm_args[@]}")
fi

"${serve_cmd[@]}" &
ORCH_PID=$!

"$PYTHON_BIN" "$REPO_ROOT/gcp_vllm_orchestrator.py" wait-for-ready \
  --servers-file "$SERVERS_FILE" \
  --timeout "$WAIT_TIMEOUT" \
  --min-servers "$MIN_SERVERS"

label_cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/label_oncology_notes.py"
  --input "$INPUT_PARQUET"
  --output "$OUTPUT_JSONL"
  --model "$MODEL"
  --server_urls_file "$SERVERS_FILE"
  --max_concurrent_per_server "$MAX_CONCURRENT_PER_SERVER"
  --results_per_shard "$RESULTS_PER_SHARD"
  --request_timeout "$LABEL_REQUEST_TIMEOUT"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
)

if [[ "$RESUME" == "1" ]]; then
  label_cmd+=(--resume)
fi

if [[ -n "${REMOTE_SHARD_DIR:-}" ]]; then
  label_cmd+=(--remote-shard-dir "$REMOTE_SHARD_DIR")
fi

if [[ "${NO_JSON_MODE:-0}" == "1" ]]; then
  label_cmd+=(--no-json-mode)
fi

if [[ "${INCLUDE_INPUT:-0}" == "1" ]]; then
  label_cmd+=(--include-input)
fi

"${label_cmd[@]}" "$@"
