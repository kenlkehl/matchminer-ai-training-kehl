# matchminer-ai-training

Data production and model training for MatchMiner-AI. This repository turns
ClinicalTrials.gov records into trial cohorts, generates synthetic patients
(event histories and clinical notes) for those cohorts, summarizes the patients,
labels patient–cohort pairs with an LLM teacher, and trains the models that the
inference package (`matchminer-ai-inference-kehl`, Python package `matchminer_ai`)
uses at runtime: the TrialSpace retriever (dense EmbeddingGemma 2), TrialChecker,
BoilerplateChecker and the OncoReasoning adapters. TrialSpace is the only
retriever: every ColBERT retrieval path was removed from both repositories on
2026-11-04 (user instruction; see [History](#history-the-october-2026-retriever-comparison)).

Training data is synthetic or public. The only code that touches real patient
data is the evaluation code in `real_eval_code/`.

Runtime behavior that the inference package owns is imported, not copied: the
cohort extraction prompts and parser, the merge and boilerplate steps, patient
summarization prompts and the split of a summary into its key and boilerplate
sections, the TrialSpace prompts (`matchminer_ai.embedding.trialspace_contract`),
the match-label prompt and 0–6 score (`matchminer_ai.matching.match_labels`), the
boilerplate teacher prompt, both checker input templates, sampling profiles, the
endpoint pool, synthetic protocol generation, and full-protocol cohort extraction.

**Key criteria** (2026-11-04, user instruction) are a cohort's criteria that are
not boilerplate. Retrieval, match labels and TrialChecker read the cohort's key
criteria and the patient summary's key section (its cancer history);
BoilerplateChecker reads the cohort's boilerplate criteria and the summary's
boilerplate section (`Other relevant history:`).

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

## Current workflow (October–November 2026)

The process now being followed, with the runs that produced the current
artifacts (paths under `../data/no_phi`; `AR` is
`archives/trialspace_reset_20261026T190832Z`). Steps 1–4 and 8 have run on
registry-derived cohorts; steps 5–6 are the synthetic-protocol track, and
step 7 is now generating patients from its cohorts.

1. **ClinicalTrials.gov export.** The JSON export
   `cancer_OR_tumor_OR_oncology_OR_lymphoma_OR_leukemia_OR_carcinoma_OR_sarcoma_OR_melanoma_OR_myelodysplastic_OR_myeloproliferative_OR_myeloma_OR_malignancy_ctgov_10-15-26.json`
   (the search query is the file name; downloaded 2026-10-15) is parsed without
   filtering by `parse-trials` (`0a_parse_ctgov_json.py`) into
   `ctgov_trials.csv`: 148,056 trials.

**Sampling from 2026-11-04 (user instruction; production `run_pipeline.py`).**
The 4,981-trial sample below came out of an older screen and a later
re-extraction that dropped 19 trials without replacement. Training now draws
its sample in this order, so the denominator is exactly 5,000:

- the export above (148,056 trials) →
- the registry rule: first posted in 2020 or later, interventional, every
  phase early phase 1 to phase 4 (30,167) →
- the DFCI holdout: trials open at DFCI in 2025–2026
  (`dfci_trials_summary_10-18-26.csv`; 1,162 of them) →
- the cohort workflow's cancer-treatment screen (`trial.cohorts.screen.user.txt`)
  on each listing (title, brief summary, eligibility criteria). This is
  `screen-trials` (`0b_screen_trials.py`): every answer and its reasoning goes
  to `<run>/upstream/screen/screen.jsonl`, and the counts and removed IDs of
  every step go to `screened_trials.csv.screening.json` →
- `sample-trials` (`0c_sample_trial_spaces.py --screened-trials`): a draw of
  5,000 accepted trials with `random_state=42`
  (`registry_sample_trials.csv` + `.sampling.json`).

Each sampled trial then gets a synthetic protocol from its registry record.
The first run on this path, `training_run_screened_sample5000_20261104/`
(2026-11-04), screened 29,005 listings (21,789 cancer treatment trials, 7,216
not, 0 errors) and drew 5,000 with seed 42; its synthetic protocols are
generating. The steps below describe the October runs.

2. **Treatment screening and registry cohorts, before sampling.** Trials first
   posted in 2020 or later, interventional, with every phase in early phase 1
   to phase 4 (30,167 trials) go to the inference package's combined cohort
   prompt (`trial.cohorts.user.txt`, schema `trial-treatment-cohorts-v3`). One
   answer decides whether the trial is a cancer treatment trial and, if so, its
   cohorts; there is no separate screening pass. This ran on all 30,167 trials
   on 2026-10-24, 01:37–03:03 UTC, with Gemma 4 31B (NVFP4 on
   `sn4622130540:8000`, FP8 on `172.24.216.113:8001`; thinking on, 50,000
   completion tokens): 23,818 treatment trials and 6,349 others, with no errors
   (`AR/ctgov_treatment_cohorts_since2020_gemma4_31b_20261023/`). The pipeline
   now runs `screen-trials` instead (above).
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
   `trial-protocol-cohorts-v2` with structured outputs) writes `cohorts` and
   `trials` tables that keep each cohort's synthetic protocol and its original
   registry record and eligibility criteria. The protocol prompt decides again
   whether each trial is a cancer treatment trial (`extraction_status`
   `not_cancer_treatment`). The current run, `cohort_extraction_v2/`
   (2026-10-29, 17:22–19:35 UTC), gave 7,721 cohorts from 4,848 trials; 132
   were judged not cancer treatment and 1 failed. It replaces the schema-v1
   run in `cohort_extraction/`, whose criteria lost every line break and
   double quote. See [`synthetic_protocols/README.md`](synthetic_protocols/README.md).
   **Registry versions (new default, 2026-10-31).** The same trials are also
   extracted from the registry records their protocols were written from, with
   the registry prompt brought into line with the protocol prompt (schema
   `trial-registry-cohorts-v1`), by `0b_extract_trial_cohorts.py --trials-jsonl`;
   `synthetic_protocols/retrieval_cohorts.py` combines both versions into the
   retrieval cohort table (see [`synthetic_protocols/README.md`](synthetic_protocols/README.md)).
   **Three-step extraction (default since 2026-11-03, user instruction).**
   Protocols and registry listings now share one inference workflow and prompt
   set (schema `trial-cohorts-stepwise-v1`): a cancer-treatment screen, the
   cohort names, and one request per cohort for its complete criteria (a
   listing is its title, brief summary and eligibility criteria as one
   document). Rows use representation `trial-treatment-cohorts-v5`, without the
   specific/common split. The runs above used the earlier single-request
   prompts; rerunning `protocol-cohorts` and `registry-cohorts` re-extracts
   every trial.
7. **Patient seeds, event histories, notes and summaries.** `generate-synthetic`
   (`generate_synthetic_data.py`) writes five positive and five near-miss
   patient seeds per cohort from the cohort's eligibility criteria alone (the
   merged protocol cohort's criteria, boilerplate included), then
   event histories, then notes; then `prepare-notes`, `summarize-patients`
   (serial summaries) and `prepare-training`. The default model is the teacher,
   `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`. The last completed run used the registry
   cohorts from step 4: 66,800 patients, 1,661,021 events and 1,137,527 notes
   (`synthetic_namedcohorts_5000_gemma4_31b_20261026/`, 2026-10-26 to 10-27).
   For the step-6 cohorts, `synthetic_protocols/patient_cohorts.py` builds the
   generation input (contiguous `cohort_index`, source `split`, protocol
   criteria kept for single cohorts), and generation runs with Nemotron 3.5
   Super VL in `synthetic_protocolcohorts_v11_cohortsv2_nemotron35_20261029/`
   (started 2026-10-29; the schema-v1 run
   `synthetic_protocolcohorts_v11_nemotron35_20261029/` was stopped).
8. **Retrieve, label, train.** The last round run was the October EmbeddingGemma
   2 comparison track (`trialspace_comparison_namedcohorts_5000_20261026/`;
   [History](#history-the-october-2026-retriever-comparison)): retrieval in both
   directions over all patients and cohorts on complete cohort criteria, the
   then-current gated 0–6 labels from the Gemma 4 31B NVFP4 teacher (2,403,278
   pairs; that label design was replaced on 2026-11-04), and EmbeddingGemma 2
   training, stopped by the user at checkpoint 11,600 on 2026-10-28. The
   integrated runner, `train_from_summaries.py` (TrialSpace, TrialChecker,
   BoilerplateChecker and OncoReasoning), is described under
   [Training](#training-train_from_summariespy).

## Pipeline

`train_all.sh`, `train_all_gcp.sh` and `temp_train_gcp.sh` are identical
wrappers: `exec "${PYTHON:-python}" run_pipeline.py "$@"`. `run_pipeline.py` runs
a contiguous range of stages (`--start-at` to `--stop-after`, inclusive). The
default range is `train` only.

| Stage | Command | Main outputs (defaults under `../data/no_phi`) |
| --- | --- | --- |
| `parse-trials` | `0a_parse_ctgov_json.py` (requires `--ctgov-json`, a CT.gov API v2 export) | `ctgov_trials.csv` |
| `screen-trials` | `0b_screen_trials.py` | `screened_trials.csv` + `.screening.json`: every trial meeting the registry rule, with its DFCI holdout flag and screen decision; the counts and removed IDs of each step. Every screen answer and its reasoning goes to `upstream/screen/screen.jsonl`. |
| `sample-trials` | `0c_sample_trial_spaces.py --screened-trials` | `registry_sample_trials.csv` + `.sampling.json`: the seeded draw of `--sample-trials` (5,000) accepted trials, which seed the protocols and give the source splits |
| `synthetic-protocols` | `synthetic_protocols/generate_batch.py` | `synthetic_protocols/` (`seeds.jsonl`, `protocols/<NCT>/protocol.md`, `protocols.csv`) |
| `protocol-cohorts` | `synthetic_protocols/extract_cohorts.py` | `synthetic_protocols/cohort_extraction/` or `--protocol-cohort-dir` (`cohorts.parquet` merged, `cohorts_unmerged.parquet`, `cohorts_deboilerplated.parquet` merged with key and boilerplate criteria, `trials.parquet`) |
| `registry-cohorts` | `0b_extract_trial_cohorts.py --trials-jsonl` (the protocol seeds) | `registry_cohorts/` (`trial_cohort_lineitems.csv` merged, `trial_cohort_lineitems_unmerged.csv`, `trial_cohort_lineitems_deboilerplated.csv` merged with key and boilerplate criteria, `trials_with_cohorts.csv`, `trial_cohorts.json`) |
| `generation-cohorts` | `synthetic_protocols/patient_cohorts.py` | `synthetic_generation/sample_trial_cohort_lineitems.csv` (merged protocol cohorts with all their criteria, for patient generation) and `sample_trial_cohort_lineitems_deboilerplated.csv` (the same rows, for retrieval: `cohort_eligibility_criteria` keeps all the criteria, `cohort_key_criteria` holds the key criteria and `cohort_boilerplate_criteria` the boilerplate), each + `.provenance.json` |
| `retrieval-cohorts` | `synthetic_protocols/retrieval_cohorts.py` | `retrieval_cohorts.csv` (both versions of every trial) + `.provenance.json` |
| `generate-synthetic` | `generate_synthetic_data.py` | `synthetic_generation/all_synthetic_events.parquet`, `all_synthetic_notes.parquet`, `manifest.json` |
| `prepare-notes` | `prepare_synthetic_training.py notes` | `all_synthetic_notes.parquet`, `synthetic_patient_metadata.parquet` |
| `summarize-patients` | `6_summarize_patients.py` | `patient_summaries.parquet`, `patient_serial_summaries.parquet` |
| `prepare-training` | `prepare_synthetic_training.py summaries` | `patient_summaries_with_spaces.parquet` (key section in `patient_summary`, boilerplate section in `patient_boilerplate_text`) |
| `train` | `train_from_summaries.py` (corpus from `retrieval_cohorts.csv`) | TrialSpace rounds, TrialChecker, BoilerplateChecker and OncoReasoning under `../models/embeddinggemma2_pipeline_v1` |

Patients are generated only from the protocol cohorts (`generation-cohorts`),
with all their criteria; retrieval, labeling and training use
`retrieval_cohorts.csv`, which holds each trial's protocol and registry versions
(see Mining below) with their key and boilerplate criteria. The three LLM
extraction/generation stages after sampling share `--server-urls-file`; a trial
that fails every attempt in one of them keeps whichever version it has.

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
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` (`oncoreasoning_training/teacher.py`), overridden
with `--generation-model` / `--generation-tokenizer` (upstream) and
`--teacher-model` / `--teacher-served-model` (training; an endpoint's model root
must match the teacher model or be an approved quantization of it, listed in
`EQUIVALENT_TEACHER_ROOTS` in `trialspace/labeling.py`: the Gemma 4 31B
builds, and Nemotron 3.5 Super VL NVFP4 and FP8, whose tokenizers and chat
templates are identical).

## Data stages

- **0a — parse.** Flattens the CT.gov JSON to one row per trial: NCT ID, study
  type, phases, registry status and dates (with original precision and
  ACTUAL/ESTIMATED types, `ctgov_metadata.py`), title, brief summary,
  eligibility criteria and `trial_text`. No filtering and no LLM.
- **0b — screen** (`0b_screen_trials.py`). The registry rule, the DFCI holdout,
  then the cohort workflow's cancer-treatment screen on every remaining listing,
  with every answer and its reasoning saved (see Sampling above).
- **0b — cohorts.** Selects interventional trials first posted in `--start-year`
  (2020) or later with early phase 1 through phase 4, assigns 80/10/10 splits
  with seed 42 (or keeps an existing `split`), or with `--trials-jsonl` extracts
  every given record, and calls `matchminer_ai.trials.summarize_trials` with the
  package's three-step workflow (screen, cohort names, one request per cohort
  for its criteria, copied verbatim). For training, equivalent cohorts are then
  merged and each merged cohort's boilerplate separated
  (`--output-deboilerplated-cohorts`: key criteria in `cohort_key_criteria`,
  boilerplate in `cohort_boilerplate_criteria`, `cohort_eligibility_criteria`
  unchanged). Remote only; per-trial gzip checkpoints and a run fingerprint make
  it resumable, and failed trials leave `.partial` outputs and an error.
- **0c — sample.** Production (`--screened-trials`) draws 5,000 whole trials
  (seed 42) from those `0b_screen_trials.py` accepted, before any cohort
  extraction. The older modes keep cancer-treatment trials from an extracted
  cohort table, drop trials with invalid or incomplete cohorts, exclude DFCI
  trials open during 2025–2026 (`dfci_trials_summary_10-18-26.csv`), and sample
  5,000 whole trials with all their cohorts (seed 42); `--extraction-dir`
  samples a completed extraction export without new LLM calls, and
  `repair_single_cohort_criteria.py` fixes exports made before the single-cohort
  correction.
- **Synthetic patients** (`generate_synthetic_data.py`, prompts in
  `synthetic_event_prompts.py`, note prompt and masking in
  `2_make_synthetic_notes_sharded.py`). For every cohort: five positive seeds
  (meet all criteria) and five near-miss negative seeds; each seed becomes a
  validated event chronology (demographics, diagnoses, treatments, adverse
  events, notes, imaging, pathology and NGS reports); each report event becomes a
  note written with the full history and the target event marked. Validation
  failures are fed back and retried (up to 16 attempts). A patient whose history
  fails validation on every attempt gets a new seed, drawn with the unchanged
  seed prompt and distinct from the cohort's other seeds, and is expanded again
  (`--max-seed-replacements`, default 3); replacements are recorded in
  `patient_seed_replacements/` and as `seed_replacement` in patient provenance,
  and seed batches are never rewritten. Transport failures keep their seed. Runs
  against a remote pool with adaptive concurrency; `--stop-after-events` stops
  before notes.
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
  summaries to patient and cohort IDs, class, source trial and split, and keeps
  each summary's two sections apart (2026-11-04): `patient_summary` is the key
  section (the cancer history, before `Other relevant history:`) and
  `patient_boilerplate_text` the boilerplate section.

CSV inputs share `trial_cohort_format.read_clinical_csv`, which keeps literal text
such as `None` or `NA`; only empty cells are missing.

## Training (`train_from_summaries.py`)

The integrated runner trains the TrialSpace retriever (EmbeddingGemma 2),
TrialChecker, BoilerplateChecker and the OncoReasoning adapters. Its stages, in
order:

1. `prepare-trialspace` (`trialspace_training.py prepare`): the corpus every
   round shares (below).
2. Round 1: mine with the base model (`mine-1`), label (`label-trialspace-1`,
   `--task match`), project the labels into a pair table
   (`project-trialspace-1`), fit (`train-trialspace-1`).
3. Round 2: the same with the round-1 model; labeling imports round-1
   judgments of pairs mined again (`--reuse-labels`).
4. Round 3: mine with the round-2 model and label (`mine-3`,
   `label-trialspace-3`, importing round-2 judgments). Its candidates feed the
   stages below; there is no third fit.
5. `label-boilerplate`: BoilerplateChecker labels for every unique combination
   of patient boilerplate section and cohort boilerplate criteria among the
   rounds 1-3 candidate pairs.
6. `train-trial-checker`, `train-boilerplate-checker` (`train_checkers.py
   trial` / `boilerplate`).
7. `catalog`, `validate-catalog`: the patient-free GoodOption drug evidence
   catalog (`train_good_option_checker.py catalog`).
8. `prepare-oncoreasoning`, `distill-oncoreasoning`, `build-oncoreasoning`,
   `train-oncoreasoning`: OncoReasoning asks the five match labels on key texts
   and each cohort boilerplate criterion line on the patient's boilerplate
   section (`oncoreasoning_training/README.md`); the earlier teacher extraction
   of exclusions (`prepare-exclusions`) is gone.

**Corpus.** `trialspace_training.py prepare` (`trialspace/data.py`) gives each
patient `text` = the summary's key section and `boilerplate` = its boilerplate
section (a complete summary is split at `Other relevant history:`), and each
cohort `text` = `cohort_key_criteria` and `boilerplate` =
`cohort_boilerplate_criteria`. A cohort table without `cohort_key_criteria`
falls back to `cohort_eligibility_criteria` (all criteria), so pass the
de-boilerplated `retrieval_cohorts.csv`, not the patient-generation table.

**Retriever.** TrialSpace is the only retriever. Base `google/embeddinggemma-2`
at revision `914f7f89`, mined by exhaustive cosine similarity over the key texts
only, with 8,192-token inputs (`trialspace_comparison.py dense-mine`; its other
commands are `project-dense-labels` and `evaluate-dense`), and fit with
`finetune_embedder.py --prompt-profile trialspace` (the patient and cohort
prompts are saved under the inference names `patient_summary` and
`this_space`): CoSENT over every labeled pair plus MNRL over positives with
known positives masked, 96 pairs per GPU (gradient-cache chunks of 32), BF16,
learning rate 2e-5 with 1% linear warmup and decay over 1 epoch
(`--trialspace-epochs`), 4,096-token training cap (key texts only). Round 2
starts from the round-1 model. The fit runs one rank per GPU in `--gpus`
(`torch.distributed.run`). Each rank's ranking losses see only its own 96
pairs, and gradients are averaged, so an optimizer step covers 96 times the
GPU count. Rank zero stages inputs to local SSD and publishes the model.
Models are saved to `trialspace_round{1,2}/` under
`--models-dir`. Dense stages run with `--embedding-python` (an environment
with EmbeddingGemma 2 support).

**Match labels** (`trialspace_training.py label --task match`; 2026-11-04, user
instruction). One teacher request per pair answers all five match labels,
`reasonableness`, `cancer_type`, `burden`, `treatment_history` and `biomarker`,
each yes/no with an `evidence_status` and a `reason`, using the inference
package's `match_quality.labels.system.txt` (`matchminer_ai.matching.match_labels`)
on the patient's key section and the cohort's key criteria. The score is 0–6:
0 unless reasonable, else 1 plus one each for cancer type, burden and treatment
history and two for biomarker (`trialspace/regression.py`, `contracts.py`).
`scores.jsonl` rows keep all five answers (never null) and `question_keys`
`{"match": key}`. The earlier gated design (`potentially_eligible` first,
specificity questions only after a yes, one question per request, prompts in
`colbert_experiments/prompts/`) is gone; its label directories do not match the
current contract.

**Boilerplate labels** (`label-boilerplate`: `trialspace_training.py label
--task boilerplate --combine-candidates <round 1 and 2 candidates>`; user,
2026-11-04). Every unique combination of patient boilerplate section and cohort
boilerplate criteria among the rounds 1-3 candidate pairs (pairs whose cohort
has boilerplate criteria; one representative pair each, counts in
`combinations.json`) is asked the inference
package's `llm_exclusion_criteria.user.txt` (Yes!/No!: does the patient clearly
fail any boilerplate criterion) on the patient's boilerplate section and the
cohort's boilerplate criteria. Pairs whose cohort has no boilerplate are
skipped. Answers go to `labels.jsonl` as `excluded`, with their reasoning.

**Checkers** (`train_checkers.py`, one rank per GPU in `--gpus`). Base
`answerdotai/ModernBERT-large` (`--checker-model`), inputs truncated at 8,192
tokens (`--checker-max-length`), 2 epochs (`--checker-epochs`), 8 pairs per GPU
(`--checker-batch-size`), learning rate 2e-5 (`--checker-learning-rate`) with 1%
warmup and linear decay, BF16, every synthetic pair (no holdout). Each model
writes `checker_contract.json` (task, labels, template and its SHA-256, length,
label sources, counts, truncated examples).

- **TrialChecker** (`trial`, `trial_checker/` under `--models-dir`): one model
  with five sigmoid heads (`multi_label_classification`, mean BCE; `id2label` =
  the five match labels), trained on every pair in all three rounds'
  `scores.jsonl` (each pair once; a later round wins a conflict). Input: the
  inference package's `match_quality_checker_template.txt` with the cohort's key
  criteria, then the patient's key section.
- **BoilerplateChecker** (`boilerplate`, `boilerplate_checker/`): a separate
  two-class model (`NEGATIVE`/`POSITIVE` = excluded) on the boilerplate labels.
  Input: `exclusion_criteria_checker_template.txt` with the patient's
  boilerplate section and the cohort's boilerplate criteria.

**Mining.** EmbeddingGemma 2 (`trialspace_comparison.py dense-mine`) produces
this candidate set by default, with no flags needed:

- Each patient searches only the cohorts of 500 randomly sampled trials
  (`--patient-trial-sample`; 0 searches all trials). Each patient gets its own
  sample, drawn reproducibly from `--seed` (42), the patient ID and the
  mining round: every round draws a fresh sample (`--trial-sample-round`,
  which the runner sets to 2 and 3; round 1 keeps the original draws).
- A trial with both a synthetic-protocol version and a registry
  (ClinicalTrials.gov) version of its cohorts (`synthetic_protocols/retrieval_cohorts.py`)
  is represented by one of them: in each patient's search, a version chosen at
  random per patient and trial; as a cohort query, a version chosen at random
  per trial (`trialspace/retrieval.py`). Trials with one version use it.
  Each cohort's version is `source` in the prepared corpus and `cohort_source`
  in the exported pair tables.
- Per patient: the top 20 cohorts by the retriever, plus 5 TF-IDF and 5 random
  cohorts, all from that patient's sampled trials.
- Per cohort: the top 40 patients over the whole corpus, plus 5 TF-IDF and 5
  random patients.
- Every patient's pairs with the cohorts of its source trial, in both versions.

The TF-IDF and random extras (`--lexical-k`, `--random-k`, default 5 each)
supply teacher-labeled pairs the retriever would not rank highly. Keep them
unless an experiment says otherwise. Each round's `mining/` also holds
per-direction tables (`patient_to_cohort.parquet`, `cohort_to_patient.parquet`)
with the key texts as `patient_summary` / `this_space` (also
`cohort_key_criteria`) and the boilerplate texts as `patient_boilerplate_text` /
`trial_boilerplate_text`; round 3's feed the catalog, exclusion and OncoReasoning
stages.

**OncoReasoning.** Student base `google/gemma-4-E4B-it` with two independent
LoRA adapters on one frozen base (rank 64, alpha 128, all linear layers):
`summarization` and `clinical_qa`. The QA adapter covers trial checking,
exclusion (boilerplate) checking and GoodOption checking. Outputs go to
`oncoreasoning/` under `--models-dir` with `oncoreasoning_contract.json`. These
QA families are separate from the ModernBERT TrialChecker and BoilerplateChecker
above.

```bash
~/thisenv/bin/python train_from_summaries.py --dry-run
~/thisenv/bin/python train_from_summaries.py \
  --patient-summaries ../data/no_phi/patient_summaries_with_spaces.parquet \
  --trial-cohorts ../data/no_phi/retrieval_cohorts.csv \
  --teacher-servers-file ../data/no_phi/live_pool/servers.json \
  --teacher-served-model nemotron-3.5-super-vl-120b \
  --embedding-python /path/to/embeddinggemma2-env/bin/python --gpus 2,3
```

Defaults: `--run-dir ../data/no_phi/embeddinggemma2_pipeline_v1`,
`--models-dir ../models/embeddinggemma2_pipeline_v1`, `--inference-repo
../matchminer-ai-inference-kehl`. Teachers come from `--teacher-servers-file`,
repeated `--teacher-url`, or one vLLM server started per GPU in `--gpus`
(default 0–7). Completed stages are marked in `<run-dir>/completed/` against
the run identity. Any change to settings, inputs, sources or either repository's
commit makes a resume fail unless a scoped upgrade is registered
(`scripts/register_*_upgrade.py`). Checker settings: `--checker-model`,
`--checker-max-length`, `--checker-epochs`, `--checker-batch-size`,
`--checker-learning-rate`.

### Labeling and retrieval storage

- **Labeling** (`trialspace_training.py label --task match|boilerplate`,
  `label_workspace.py`) runs on local SSD by default
  (`/tmp/mmai-labels/<hash>`, or `MMAI_LABEL_SCRATCH_DIR`; 16 GiB reserve;
  `MMAI_LOCAL_LABEL_STORAGE=0` disables). Concurrency per server starts at 96 and
  can reach 768. Status is mirrored to the output directory every 10 seconds. On
  completion, `scores.jsonl` (match labels only), `labels.jsonl`, the response
  database (prompts, answers and reasoning) and manifests are published with
  `storage_complete.json`; failed runs keep their scratch for resume.
- **Retrieval** (`trialspace_comparison.py dense-mine`, `retrieval_workspace.py`)
  stages work in `/tmp/mmai-retrieval` (`--retrieval-scratch-dir`; network
  filesystems are rejected), publishes candidates, rankings, encodings and
  `complete.json` to `--output`, then removes the scratch. Unchanged dense
  encodings can be reused across merged corpora (`retrieval_cache_import.py`);
  rankings never are.

Models and data are published only by size-checked copies to shared `no_phi`
storage. There is no Hugging Face upload code in this repository.

## Synthetic trial protocols

`synthetic_protocols/` builds a corpus of synthetic full protocol documents
seeded by registry records, with realistic, often messily specified eligibility
covering one or more cohorts. Generation runs through
`matchminer_ai.trials.generate_synthetic_protocols`; see
[`synthetic_protocols/README.md`](synthetic_protocols/README.md). Training uses
these synthetic protocols, not real ones. Real protocol PDFs posted
on ClinicalTrials.gov (`scripts/download_ctgov_protocols.py`; check licensing
before redistributing) are for comparison and evaluation only.
`synthetic_protocols/extract_cohorts.py` extracts each synthetic protocol's
cohorts and eligibility criteria with
`matchminer_ai.trials.extract_protocol_cohorts`, keeping every cohort's
provenance: the synthetic protocol and its generation, and the original
ClinicalTrials.gov record and eligibility criteria it was seeded from. In the
pipeline above, these cohorts seed the synthetic patients (`generation-cohorts`)
and, beside the registry versions, form the retrieval cohort table
(`retrieval-cohorts`).

## History: the October 2026 retriever comparison

From 2026-10-26 to 10-28 a separate workflow compared EmbeddingGemma 2 with
ModernColBERT (`lightonai/GTE-ModernColBERT-v1`, MaxSim scoring) on shared
gated labels; its run state is recorded in `AGENTS.md`. On 2026-11-04 (user
instruction) every ColBERT retrieval path was removed from both repositories.
The package `colbert_experiments/` became `trialspace/` without its ColBERT-only
modules, and `retrieval_publication.py`, `scripts/trialspace_comparison_workflow.py`,
`embeddinggemma_round_two.py`, `dense_training_preparation.py`,
`adopted_pipeline.py` and the package's `export_completed_scores.py` and
`import_score_journal.py` were deleted. `train_from_summaries.py` has no
`--trialspace-retriever`; `trialspace_training.py` has only `prepare` and
`label`; `trialspace_comparison.py` has only `dense-mine`,
`project-dense-labels` and `evaluate-dense`. The prompt profile
`colbert_matching` is now `trialspace` (same strings). The run logs are in
`docs/history/`: `COMPARISON_GB300_20261026.md`,
`HANDOFF_GB300_TRAINING_20261026.md`, `GB300_MEMORY_RESULTS_20261026.md` / `.json`
and `colbert_experiments_README.md`; paths and commands in them refer to the
code of that time. TrialSpace support modules that remain:
`trialspace_embedding_losses.py`, `trialspace_prompt_profiles.py`,
`training_workspace.py` (local SSD training with copy-back), `summary_overlap.py`
and `scripts/benchmark_embeddinggemma_batches.py`.

## Repository map

| Path | Purpose | Status |
| --- | --- | --- |
| `run_pipeline.py`, `train_all*.sh` | Stage orchestrator and wrappers | Current |
| `0a_*`, `0b_*`, `0c_*`, `ctgov_metadata.py`, `trial_cohort_format.py`, `trial_space_format.py`, `repair_single_cohort_criteria.py` | Trial parsing, cohort extraction, sampling | Current |
| `generate_synthetic_data.py`, `synthetic_event_prompts.py`, `2_make_synthetic_notes_sharded.py`, `prepare_synthetic_training.py`, `synthetic_pool.py` | Synthetic patients and notes | Current (the note script's standalone modes are legacy) |
| `6_summarize_patients.py`, `summary_pool.py`, `summary_workspace.py` | Serial summarization | Current |
| `train_from_summaries.py`, `trialspace_training.py`, `trialspace/`, `trialspace_comparison.py`, `finetune_embedder.py`, `trialspace_embedding_losses.py`, `trialspace_prompt_profiles.py`, `retrieval_workspace.py`, `label_workspace.py`, `retrieval_cache_import.py`, `training_workspace.py` | TrialSpace corpus, mining, match and boilerplate labeling, fitting | Current |
| `train_checkers.py` | TrialChecker (five heads) and BoilerplateChecker (two classes) | Current |
| `oncoreasoning_training/` | Exclusion extraction, distillation, OncoReasoning adapters | Current (its README is provisional) |
| `train_good_option_checker.py` | GoodOption catalog (`catalog`, `validate-catalog`) | Current; its `label`/`train` checker modes are deprecated |
| `remote_vllm_pool.py`, `adaptive_controller.py`, `vllm_reasoning_utils.py` | Server pool, adaptive concurrency, reasoning parsers | Current |
| `synthetic_protocols/` | Synthetic protocol corpus and its cohort extraction | Current |
| `scripts/` | GCP fleet (`gcp_note_fleet.py`, `gcp_note_worker.py`, `bootstrap_gcp.sh`), remote TrialSpace jobs, `teacher_proxy.py`, `post_notes_workflow.py`, `label_to_trialspace.py`, `check_patient_summaries.py`, `download_ctgov_protocols.py`, `cohort_dashboard.py` (cohort review: as extracted, merged, or key and boilerplate criteria separated), one-off `register_*_upgrade.py` resume upgrades | Mixed; see each docstring |
| `docs/history/` | October 2026 retriever comparison logs (ColBERT removed 2026-11-04) | Historical |
| `claude_oracle/` | Claude Code plugin with judge skills for evaluations | Current |
| `real_eval_code/` | Retrospective evaluations on real (PHI) data | Current; some scripts use the legacy extractor |
| `legacy_trial_space_extraction.py`, `trial_hierarchy*.py`, `trial_prompts/`, `trial_space_export.py`, `trial_treatment_screening.py` | Pre-cohort trial-space extraction, superseded by `0b` | Legacy |
| `14_check_boilerplate.py`, `make_top_matches.py`, `train_good_option_cosplit.py` | Earlier checker labeling and retrieval, superseded by the match and boilerplate labels, `train_checkers.py` and OncoReasoning (the single-score LLM labeler `llm_check_trials.py` was removed on 2026-11-04) | Legacy |
| `gcp_vllm_orchestrator.py`, `pdfify_synthetic_notes.py`, `synthetic_eval_code/` | Earlier fleet management, synthetic-note PDFs, a one-patient demo | Legacy |
| `pull_spacify_and_embed_ctgov` | Pulls current ClinicalTrials.gov cancer trials, extracts legacy trial spaces and embeds them for `real_eval_code/real_time_eval/` | Legacy (uses the legacy extractor) |

## Tests

```bash
~/thisenv/bin/python -m pytest -q tests/
```

`tests/` holds only critical contracts: trial selection, cohort extraction and
sampling, synthetic seed and event validation, note preparation, pipeline
planning and resume, CSV literals, the remote pool, embedder losses, the
teacher label contracts (five match answers in one request, boilerplate) and the
checker training examples. Operational scaffolding (supervisors, fleets, receipts)
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
