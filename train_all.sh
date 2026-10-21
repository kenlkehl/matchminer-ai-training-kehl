#!/usr/bin/env bash
# Both environments run the same resumable two-model pipeline.
# Input overrides are forwarded unchanged: --trial-spaces (the compact 0b
# lineitems, including completed_ exports), --synthetic-notes, --patient-summaries.
# Defaults use the canonical filenames under --data-dir. This runner starts
# after synthetic-note generation and patient summarization have completed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" train_from_summaries.py "$@"
