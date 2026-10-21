#!/usr/bin/env bash
# Shared resumable data-production and training pipeline.
# Default: train from existing summaries. --start-at selects an upstream stage;
# --stop-after limits the run. --wait-for-generation observes an existing job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" run_pipeline.py "$@"
