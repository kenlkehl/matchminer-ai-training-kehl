# matchminer-ai-training

Data production and model training for MatchMiner-AI. This repository turns
ClinicalTrials.gov records into trial cohorts, generates synthetic patients
(event histories and clinical notes) for those cohorts, summarizes the patients,
labels patient–cohort pairs with an LLM teacher, and trains the models that the
inference package (`matchminer-ai-inference-kehl`, Python package `matchminer_ai`)
uses at runtime.

Training data is synthetic or public. The only code that touches real patient
data is the evaluation code in `real_eval_code/`.

Runtime behavior that the inference package owns is imported, not copied: the
cohort extraction prompt and parser, patient summarization prompts, the ColBERT
encoder and scorer, sampling profiles, the endpoint pool, synthetic protocol
generation, and full-protocol cohort extraction.

## Setup

Use a virtual environment on local disk (not on the network mount) and install
this repository plus the sibling inference package:

```bash
python -m venv ~/thisenv
~/thisenv/bin/pip install -e ../matchminer-ai-inference-kehl -e .
```

Both repositories' commits are recorded in every training run's identity
(`pipeline_manifest.json`), along with hashes of the production sources they
use. Data-production stages put `../matchminer-ai-inference-kehl/src` on
`PYTHONPATH` and hash the package sources into their stage fingerprints.

LLM stages never start or stop the shared vLLM fleet. They use an existing
server registry (`--server-urls-file` / `--teacher-servers-file`, a JSON file
re-read while running) or fixed URLs; the integrated training runner can also
start one vLLM teacher per local GPU.

## Current workflow (October 2026)

The process now being followed, with the runs that produced the current
artifacts (paths under `../data/no_phi`; `AR` is
`archives/trialspace_reset_20261026T190832Z`). Steps 1–4 and 7–8 have run on
registry-derived cohorts; steps 5–6 are the synthetic-protocol track, and
generating patients from its cohorts (step 7) is not wired up yet.

1. **ClinicalTrials.gov export.** The JSON export
   `cancer_OR_tumor_OR_oncology_OR_lymphoma_OR_leukemia_OR_carcinoma_OR_sarcoma_OR_melanoma_OR_myelodysplastic_OR_myeloproliferative_OR_myeloma_OR_malignancy_ctgov_10-15-26.json`
   (the search query is the file name; downloaded 2026-10-15) is parsed without
   filtering by `parse-trials` (`0a_parse_ctgov_json.py`) into
   `ctgov_trials.csv`: 148,056 trials.
2. **Treatment screening and registry cohorts, before sampling.** Trials first
   posted in 2020 or later, interventional, with every phase in early phase 1
   to phase 4 (30,167 trials) go to the inference package's combined cohort
   prompt (`trial.cohorts.user.txt`, schema `trial-treatment-cohorts-v3`). One
   answer decides whether the trial is a cancer treatment trial and, if so, its
   cohorts; there is no separate screening pass. This ran on all 30,167 trials
   on 2026-10-24, 01:37–03:03 UTC, with Gemma 4 31B (NVFP4 on
   `sn4622130540:8000`, FP8 on `172.24.216.113:8001`; thinking on, 50,000
   completion tokens): 23,818 treatment trials and 6,349 others, with no errors
   (`AR/ctgov_treatment_cohorts_since2020_gemma4_31b_20261023/`). `extract-cohorts`
   (`0b_extract_trial_cohorts.py`) is the pipeline form of this step.
3. **Sample 5,000 trials.** `sample-trials` (`0c_sample_trial_spaces.py`) keeps
   screened treatment trials with complete cohort sets, excludes DFCI trials
   open in 2025–2026 (`dfci_trials_summary_10-18-26.csv`), leaving 22,734, and
   draws 5,000 with `random_state=42` (2026-10-24).
4. **Re-extract the sample with named cohorts.** The same 5,000 were extracted
   again on 2026-10-26 with Gemma 4 31B NVFP4 (cohort names, representation
   `trial-treatment-cohorts-v4`). 19 were rejected and not replaced, leaving
   **4,981 trials and 6,680 cohorts** in `sample_trial_cohort_lineitems.csv`
   (`cohort_reextract_named_sample5000_gemma4_31b_20261026/`).
5. **Synthetic protocols.** For each of the 4,981 trials, a synthetic full
   protocol is generated from its registry record, fetched again from the
   ClinicalTrials.gov API on 2026-10-28 (so the text can differ slightly from
   the 10-15 export), by Nemotron 3.5 Super VL (NVFP4 and FP8) with prompt v11:
   `synthetic_protocols/generate_batch.py`, into
   `synthetic_protocols_from_registry/batch_v11_sample5000_20261029/`.
6. **Extract cohorts from the synthetic protocols.**
   `synthetic_protocols/extract_cohorts.py` (Nemotron, schema
   `trial-protocol-cohorts-v1` with structured outputs) writes
   `cohort_extraction/cohorts` and `trials` tables that keep each cohort's
   synthetic protocol and its original registry record and eligibility
   criteria. The protocol prompt decides again whether each trial is a cancer
   treatment trial (`extraction_status` `not_cancer_treatment`). See
   [`synthetic_protocols/README.md`](synthetic_protocols/README.md).
7. **Patient seeds, event histories, notes and summaries.** `generate-synthetic`
   (`generate_synthetic_data.py`) writes five positive and five near-miss
   patient seeds per cohort from the cohort's eligibility criteria alone, then
   event histories, then notes; then `prepare-notes`, `summarize-patients`
   (serial summaries) and `prepare-training`. The default model is the teacher,
   `nvidia/Gemma-4-31B-IT-NVFP4`. The last completed run used the registry
   cohorts from step 4: 66,800 patients, 1,661,021 events and 1,137,527 notes
   (`synthetic_namedcohorts_5000_gemma4_31b_20261026/`, 2026-10-26 to 10-27).
   Before it can use the step-6 cohorts, an adapter has to add a contiguous
   `cohort_index` and a `split`, and handle single-cohort rows, whose
   `cohort_criteria_source` is `protocol`, without substituting the registry
   criteria.
8. **Retrieve, label, train.** The current round is the experimental
   EmbeddingGemma 2 track (`trialspace_comparison_namedcohorts_5000_20261026/`;
   [below](#experimental-embeddinggemma-2-comparison-track)): retrieval in both
   directions over all patients and cohorts, gated 0–6 labels from the Gemma 4
   31B NVFP4 teacher (2,403,278 pairs), and EmbeddingGemma 2 training, stopped
   by the user at checkpoint 11,600 on 2026-10-28. The integrated runner,
   `train_from_summaries.py` (ColBERT plus OncoReasoning), is described under
   [Training](#training-train_from_summariespy).

## Pipeline

`train_all.sh`, `train_all_gcp.sh` and `temp_train_gcp.sh` are identical
wrappers: `exec "${PYTHON:-python}" run_pipeline.py "$@"`. `run_pipeline.py` runs
a contiguous range of stages (`--start-at` to `--stop-after`, inclusive). The
default range is `train` only.

| Stage | Command | Main outputs (defaults under `../data/no_phi`) |
| --- | --- | --- |
| `parse-trials` | `0a_parse_ctgov_json.py` (requires `--ctgov-json`, a CT.gov API v2 export) | `ctgov_trials.csv` |
| `extract-cohorts` | `0b_extract_trial_cohorts.py` | `trial_cohort_lineitems.csv`, `trials_with_cohorts.csv`, `trial_cohorts.json` |
| `sample-trials` | `0c_sample_trial_spaces.py` | `synthetic_generation/sample_trial_cohort_lineitems.csv` + `.sampling.json` |
| `generate-synthetic` | `generate_synthetic_data.py` | `synthetic_generation/all_synthetic_events.parquet`, `all_synthetic_notes.parquet`, `manifest.json` |
| `prepare-notes` | `prepare_synthetic_training.py notes` | `all_synthetic_notes.parquet`, `synthetic_patient_metadata.parquet` |
| `summarize-patients` | `6_summarize_patients.py` | `patient_summaries.parquet`, `patient_serial_summaries.parquet` |
| `prepare-training` | `prepare_synthetic_training.py summaries` | `patient_summaries_with_spaces.parquet` |
| `train` | `train_from_summaries.py` | models under `../models/colbert_pipeline_v1` |

Inspect a plan, then remove `--dry-run` to execute:

```bash
./train_all.sh --start-at parse-trials --stop-after prepare-training \
  --ctgov-json ../data/no_phi/ctgov_export.json \
  --data-dir ../data/no_phi/new_dataset \
  --run-dir ../data/no_phi/new_training_run \
  --models-dir ../models/new_training_run \
  --server-urls-file ../data/no_phi/live_pool/servers.json \
  --generation-model gemma4-31b \
  --generation-tokenizer RedHatAI/Gemma-4-31B-IT-FP8-Dynamic \
  --dry-run
```

**Resume and safety.** All data-production paths must be under `data/no_phi`.
Each stage's fingerprint covers its command, input size/mtime stamps and source
hashes. A stage with a matching `completed/<stage>.json` is skipped; a changed
fingerprint is an error (use a fresh `--pipeline-dir`). A lock in the pipeline
directory prevents concurrent runs, and logs go to `<pipeline-dir>/logs/`.
`--wait-for-generation` attaches to a generation job that is already running
(it polls `status.json` and never starts or stops it), and is implied when
`generation.lock` is held.

The default LLM for data production and labeling is
`nvidia/Gemma-4-31B-IT-NVFP4` (`oncoreasoning_training/teacher.py`), overridden
with `--generation-model` / `--generation-tokenizer` (upstream) and
`--teacher-model` / `--teacher-served-model` (training; an endpoint's model root
must match the teacher model).

## Data stages

- **0a — parse.** Flattens the CT.gov JSON to one row per trial: NCT ID, study
  type, phases, registry status and dates (with original precision and
  ACTUAL/ESTIMATED types, `ctgov_metadata.py`), title, brief summary,
  eligibility criteria and `trial_text`. No filtering and no LLM.
- **0b — cohorts.** Selects interventional trials first posted in `--start-year`
  (2020) or later with early phase 1 through phase 4, assigns 80/10/10 splits
  with seed 42 (or keeps an existing `split`), and calls
  `matchminer_ai.trials.summarize_trials` with the package's combined
  treatment-screening and cohort prompt. Each trial gets one decision
  (`cancer_treatment_trial`, `multiple_cohorts`, named cohorts with
  cohort-specific and common criteria). Remote only; per-trial gzip checkpoints
  and a run fingerprint make it resumable, and failed trials leave `.partial`
  outputs and an error.
- **0c — sample.** Keeps cancer-treatment trials, drops trials with invalid or
  incomplete cohorts, excludes DFCI trials open during 2025–2026
  (`dfci_trials_summary_10-18-26.csv`), and samples 5,000 whole trials with all
  their cohorts (seed 42). `--extraction-dir` samples a completed extraction
  export without new LLM calls; `repair_single_cohort_criteria.py` fixes exports
  made before the single-cohort correction.
- **Synthetic patients** (`generate_synthetic_data.py`, prompts in
  `synthetic_event_prompts.py`, note prompt and masking in
  `2_make_synthetic_notes_sharded.py`). For every cohort: five positive seeds
  (meet all criteria) and five near-miss negative seeds; each seed becomes a
  validated event chronology (demographics, diagnoses, treatments, adverse
  events, notes, imaging, pathology and NGS reports); each report event becomes a
  note written with the full history and the target event marked. Validation
  failures are fed back and retried (up to 16 attempts). Runs against a remote
  pool with adaptive concurrency; `--stop-after-events` stops before notes.
- **Prepare notes** (`prepare_synthetic_training.py notes`). Assigns explicitly
  fabricated calendar dates from event ages, ending at `--synthetic-date-anchor`
  (2025-01-01), and writes patient metadata. All synthetic splits are pooled for
  training (`synthetic_pool.py`); the original split is kept as `source_split`.
- **Summarize** (`6_summarize_patients.py`, with `summary_pool.py` and
  `summary_workspace.py`). Serial, chronological summarization with the inference
  package's prompts: each patient's dated notes are concatenated, split into
  overlapping token chunks (the pipeline uses 50,000 tokens with 500 overlap), and
  each chunk updates the running summary. Invalid chunks are rewound and redone.
  There is no note tagging or retrieval step.
- **Prepare training** (`prepare_synthetic_training.py summaries`). Joins final
  summaries to patient and cohort IDs, class, source trial and split.

CSV inputs share `trial_cohort_format.read_clinical_csv`, which keeps literal text
such as `None` or `NA`; only empty cells are missing.

## Training (`train_from_summaries.py`)

The integrated runner trains a ColBERT retriever and the OncoReasoning adapters.
Its stages, in order:

1. `prepare-trialspace`
2. Round 1: mine with the base model, label, train.
3. Round 2: mine with the round-1 model, label, train.
4. `mine-3` with the round-2 model (candidates for downstream training).
5. `catalog`, `validate-catalog`: the patient-free GoodOption drug evidence
   catalog (`train_good_option_checker.py catalog`).
6. `prepare-exclusions`: teacher extraction of trial exclusion rules
   (`oncoreasoning_training/prepare_boilerplate.py`).
7. `prepare-oncoreasoning`, `distill-oncoreasoning`, `build-oncoreasoning`,
   `train-oncoreasoning`.

**Retriever.** Base `lightonai/GTE-ModernColBERT-v1`, with the encoder,
retrieval instruction and MaxSim scorer from the inference package. The default
loss (`--trialspace-loss legacy_ranking`) is CoSENT over MaxSim scores in both
directions plus a contrastive term comparing each positive only with its own
query's known zero-score pairs (scale 20, learning rate 2e-5, 3 epochs, 1%
warmup, 8,192-token context). Models are saved to
`trialspace_round{1,2}/` under `--models-dir`.

**Labels.** The teacher answers five yes/no questions per pair: an eligibility
gate (`potentially_eligible`), then, only if the gate passes, cancer type,
disease burden, prior treatment and biomarker. The gated score is 0–6: a
matched biomarker counts two points and every other yes one
(`colbert_experiments/regression.py`, `contracts.py`).

**Mining.** Each patient searches the cohorts of 500 sampled trials; candidates
are the top 20 cohorts per patient and top 40 patients per cohort, plus TF-IDF,
random and source-trial pairs (`trialspace_training.py mine`).

**OncoReasoning.** Student base `google/gemma-4-E4B-it` with two independent
LoRA adapters on one frozen base (rank 64, alpha 128, all linear layers):
`summarization` and `clinical_qa`. The QA adapter covers trial checking,
exclusion (boilerplate) checking and GoodOption checking. Outputs go to
`oncoreasoning/` under `--models-dir` with `oncoreasoning_contract.json`. There
are no separate TrialChecker or BoilerplateChecker classifier trainers here.

```bash
~/thisenv/bin/python train_from_summaries.py --dry-run
~/thisenv/bin/python train_from_summaries.py \
  --patient-summaries ../data/no_phi/patient_summaries_with_spaces.parquet \
  --trial-cohorts ../data/no_phi/synthetic_generation/sample_trial_cohort_lineitems.csv \
  --teacher-servers-file ../data/no_phi/live_pool/servers.json \
  --teacher-served-model gemma4-31b --gpus 2,3
```

Defaults: `--run-dir ../data/no_phi/colbert_pipeline_v1`,
`--models-dir ../models/colbert_pipeline_v1`, `--inference-repo
../matchminer-ai-inference-kehl`. Teachers come from `--teacher-servers-file`,
repeated `--teacher-url`, or one vLLM server started per GPU in `--gpus`
(default 0–7). Completed stages are marked in `<run-dir>/completed/` against
the run identity. Any change to settings, inputs, sources or either repository's
commit makes a resume fail unless a scoped upgrade is registered
(`scripts/register_*_upgrade.py`).

### Labeling and retrieval storage

- **Labeling** (`trialspace_training.py label --gated-regression`,
  `label_workspace.py`) runs on local SSD by default
  (`/tmp/mmai-labels/<hash>`, or `MMAI_LABEL_SCRATCH_DIR`; 16 GiB reserve;
  `MMAI_LOCAL_LABEL_STORAGE=0` disables). Concurrency per server starts at 96 and
  can reach 768. Status is mirrored to the output directory every 10 seconds. On
  completion, `scores.jsonl`, `labels.jsonl`, the response database (prompts,
  answers and reasoning) and manifests are published with `storage_complete.json`;
  failed runs keep their scratch for resume.
- **Retrieval** (`trialspace_training.py mine`, `retrieval_workspace.py`) stages
  work in `/tmp/mmai-retrieval` (`--retrieval-scratch-dir`; network filesystems
  are rejected), publishes candidates, rankings and `complete.json` to
  `--output`, then removes the scratch. Unchanged encodings can be reused across
  merged corpora (`retrieval_cache_import.py`); rankings never are.

Models and data are published only by size-checked copies to shared `no_phi`
storage. There is no Hugging Face upload code in this repository.

## Synthetic trial protocols

`synthetic_protocols/` builds a corpus of synthetic full protocol documents
seeded by registry records, with realistic, often messily specified eligibility
covering one or more cohorts. Generation runs through
`matchminer_ai.trials.generate_synthetic_protocols`; see
[`synthetic_protocols/README.md`](synthetic_protocols/README.md). Training is
planned on these synthetic protocols, not on real ones. Real protocol PDFs posted
on ClinicalTrials.gov (`scripts/download_ctgov_protocols.py`; check licensing
before redistributing) are for comparison and evaluation only.
`synthetic_protocols/extract_cohorts.py` extracts each synthetic protocol's
cohorts and eligibility criteria with
`matchminer_ai.trials.extract_protocol_cohorts`, keeping every cohort's
provenance: the synthetic protocol and its generation, and the original
ClinicalTrials.gov record and eligibility criteria it was seeded from. Neither
the synthetic corpus nor its extracted cohorts are used by the pipeline above
yet.

## Experimental: EmbeddingGemma 2 comparison track

A separate workflow compares retrievers on the same labels:
`scripts/trialspace_comparison_workflow.py` drives `trialspace_comparison.py`
(`dense-mine`), shared labeling, and `finetune_embedder.py`, which trains
`google/embeddinggemma-2` with known-positive-masked MNRL plus CoSENT on the 0–6
scores. Related modules: `trialspace_embedding_losses.py`,
`trialspace_prompt_profiles.py`, `dense_training_preparation.py`,
`embeddinggemma_round_two.py`, `training_workspace.py` (local SSD training with
copy-back), `summary_overlap.py`, `retrieval_publication.py`,
`adopted_pipeline.py` and `scripts/benchmark_embeddinggemma_batches.py`. Run
state and decisions for the current comparison are recorded in `AGENTS.md` and
`colbert_experiments/README.md`.

## Repository map

| Path | Purpose | Status |
| --- | --- | --- |
| `run_pipeline.py`, `train_all*.sh` | Stage orchestrator and wrappers | Current |
| `0a_*`, `0b_*`, `0c_*`, `ctgov_metadata.py`, `trial_cohort_format.py`, `trial_space_format.py`, `repair_single_cohort_criteria.py` | Trial parsing, cohort extraction, sampling | Current |
| `generate_synthetic_data.py`, `synthetic_event_prompts.py`, `2_make_synthetic_notes_sharded.py`, `prepare_synthetic_training.py`, `synthetic_pool.py` | Synthetic patients and notes | Current (the note script's standalone modes are legacy) |
| `6_summarize_patients.py`, `summary_pool.py`, `summary_workspace.py` | Serial summarization | Current |
| `train_from_summaries.py`, `trialspace_training.py`, `colbert_experiments/`, `retrieval_workspace.py`, `label_workspace.py`, `retrieval_cache_import.py` | Retriever mining, labeling and training | Current |
| `oncoreasoning_training/` | Exclusion extraction, distillation, OncoReasoning adapters | Current (its README is provisional) |
| `train_good_option_checker.py` | GoodOption catalog (`catalog`, `validate-catalog`) | Current; its `label`/`train` checker modes are deprecated |
| `remote_vllm_pool.py`, `adaptive_controller.py`, `vllm_reasoning_utils.py` | Server pool, adaptive concurrency, reasoning parsers | Current |
| `synthetic_protocols/` | Synthetic protocol corpus and its cohort extraction | Current |
| `scripts/` | GCP fleet (`gcp_note_fleet.py`, `gcp_note_worker.py`, `bootstrap_gcp.sh`), remote TrialSpace jobs, `teacher_proxy.py`, `post_notes_workflow.py`, `label_to_trialspace.py`, `check_patient_summaries.py`, `download_ctgov_protocols.py`, comparison workflow, one-off `register_*_upgrade.py` resume upgrades | Mixed; see each docstring |
| `claude_oracle/` | Claude Code plugin with judge skills for evaluations | Current |
| `real_eval_code/` | Retrospective evaluations on real (PHI) data | Current; some scripts use the legacy extractor |
| `legacy_trial_space_extraction.py`, `trial_hierarchy*.py`, `trial_prompts/`, `trial_space_export.py`, `trial_treatment_screening.py` | Pre-cohort trial-space extraction, superseded by `0b` | Legacy |
| `llm_check_trials.py`, `14_check_boilerplate.py`, `make_top_matches.py`, `train_good_option_cosplit.py` | Earlier checker labeling and retrieval, superseded by gated labeling and OncoReasoning | Legacy |
| `gcp_vllm_orchestrator.py`, `pdfify_synthetic_notes.py`, `synthetic_eval_code/` | Earlier fleet management, synthetic-note PDFs, a one-patient demo | Legacy |
| `pull_spacify_and_embed_ctgov` | Pulls current ClinicalTrials.gov cancer trials, extracts legacy trial spaces and embeds them for `real_eval_code/real_time_eval/` | Legacy (uses the legacy extractor) |

## Tests

```bash
~/thisenv/bin/python -m pytest -q tests/
```

`tests/` holds only critical contracts: trial selection, cohort extraction and
sampling, synthetic seed and event validation, note preparation, pipeline
planning and resume, CSV literals, the remote pool, embedder losses and the
labeling-prompt contract. Operational scaffolding (supervisors, fleets, receipts)
is not tested here by policy. Embedder tests skip in environments older than the
`pyproject.toml` pins (sentence-transformers 6.1, transformers 5.19), such as
`/data1/ken/envs/gptoss3` on camus.

## Data and clinical safety

- Synthetic data, registry text and public protocols are the only inputs to
  training. Keep real patient data in `real_eval_code/` workflows, never in
  fixtures, logs or committed files.
- Synthetic calendar dates are fabricated and must not be treated as real.
- Scores and model outputs are research ranking signals, not eligibility
  determinations.
