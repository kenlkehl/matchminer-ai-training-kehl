#!/usr/bin/env bash
# Compatibility entrypoint for the same resumable data-production and training pipeline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" run_pipeline.py "$@"
