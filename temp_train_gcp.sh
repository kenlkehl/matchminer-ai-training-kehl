#!/usr/bin/env bash
# Both environments run the same resumable two-model pipeline.
# Accepts the same --trial-spaces, --synthetic-notes, and --patient-summaries
# input overrides as train_all.sh and train_all_gcp.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" train_from_summaries.py "$@"
