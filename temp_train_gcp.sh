#!/usr/bin/env bash
# Shared resumable data-production and training pipeline (run_pipeline.py). Stages:
#   parse-trials (0a_parse_ctgov_json.py), screen-trials (0b_screen_trials.py: registry rule, DFCI holdout, cancer-treatment screen),
#   sample-trials (0c_sample_trials.py: seeded draw of screened trials),
#   synthetic-protocols (synthetic_protocols/generate_batch.py),
#   protocol-cohorts (synthetic_protocols/extract_cohorts.py),
#   registry-cohorts (0d_extract_registry_cohorts.py --trials-jsonl on the protocol seeds),
#   generation-cohorts (synthetic_protocols/patient_cohorts.py),
#   retrieval-cohorts (synthetic_protocols/retrieval_cohorts.py: protocol + registry versions),
#   generate-synthetic, prepare-notes, summarize-patients, prepare-training, train.
# Patients come from protocol cohorts; retrieval, labeling and training use both versions.
# Default: train from existing summaries. --start-at selects an upstream stage;
# --stop-after limits the run. --wait-for-generation observes an existing job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" run_pipeline.py "$@"
