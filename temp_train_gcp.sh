#!/usr/bin/env bash
# Both environments run the same resumable two-model pipeline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" train_from_summaries.py "$@"
