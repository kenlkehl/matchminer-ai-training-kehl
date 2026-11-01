#!/usr/bin/env bash
# Shared resumable data-production and training pipeline (run_pipeline.py). Stages:
#   parse-trials, extract-cohorts (0b), sample-trials (0c),
#   synthetic-protocols (synthetic_protocols/generate_batch.py),
#   protocol-cohorts (synthetic_protocols/extract_cohorts.py),
#   registry-cohorts (0b --trials-jsonl on the protocol seeds),
#   generation-cohorts (synthetic_protocols/patient_cohorts.py),
#   retrieval-cohorts (synthetic_protocols/retrieval_cohorts.py: protocol + registry versions),
#   generate-synthetic, prepare-notes, summarize-patients, prepare-training, train.
# Patients come from protocol cohorts; retrieval, labeling and training use both versions.
# Default: train from existing summaries. --start-at selects an upstream stage;
# --stop-after limits the run. --wait-for-generation observes an existing job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" run_pipeline.py "$@"
