# Synthetic trial protocols

ClinicalTrials.gov eligibility text is much less granular than the eligibility
sections of full protocols. This workflow generates a corpus of synthetic full
protocol documents, one per registry seed, whose eligibility reads like real
protocols: often messily specified, with one or more protocol-defined cohorts,
parts or arms. The corpus is training material for cohort and criteria
extraction and matching; it is **not** a set of real trial criteria.

## Components

| Piece | Location |
| --- | --- |
| Generation API | `matchminer_ai.trials.generate_synthetic_protocols` (inference package) |
| Prompt (v11) | `matchminer_ai/prompts/trial.synthetic_protocol.system.txt`, `trial.synthetic_protocol.user.txt` |
| Eligibility layouts | `matchminer_ai/data/synthetic_protocol_layouts.json` |
| Config | `synthetic_protocol_generation` section of the inference preset |
| Batch driver | `synthetic_protocols/generate_batch.py` (this repository) |
| Cohort extraction API | `matchminer_ai.trials.extract_protocol_cohorts` (inference package) |
| Cohort extraction driver | `synthetic_protocols/extract_cohorts.py` (this repository) |
| Registry-version cohorts | `0b_extract_trial_cohorts.py --trials-jsonl` on the batch's `seeds.jsonl` (this repository) |
| Retrieval cohort table (both versions) | `synthetic_protocols/retrieval_cohorts.py` (this repository) |
| Prompt history and findings | [`PROMPT_HISTORY.md`](PROMPT_HISTORY.md) |
| Real protocol PDFs (comparison data) | `scripts/download_ctgov_protocols.py` |

Requires the inference package at commit `acb37a0` or later (overlapping
chunks); cohort extraction needs `cbd7aff` or later (saved, resumable decisions
and Markdown input).

## What the prompt does

The registry record (official title, brief summary, detailed description and
eligibility criteria) is a **seed, not a specification**: the protocol may refine
or differ from it, as a real protocol does from its registry entry. The prompt
asks for a complete sponsor-style protocol in Markdown, written prospectively
(no enrollment status or results), with cohorts defined in the study design,
eligibility detail scaled to the trial's kind and sponsor, sponsor texture
(introductory sentence, NOTE lines, cross-references to sections that exist),
label-derived criteria for every study agent, and no statement that the document
is synthetic. Eligibility details from the record stay in the study population
section and synopsis.

Each trial's eligibility layout is chosen by a stable hash of its NCT ID with
weights from a survey of 59 real multi-cohort protocols:

| Weight | Layout |
| --- | --- |
| 20 (~65%) | One inclusion and one exclusion list; differences between parts/cohorts inside criteria (disease-keyed sub-items, part-specific thresholds, NOTEs) |
| 9 (~29%) | One list each; only criteria limited to some parts/cohorts carry a short qualifier |
| 1 (~3%) | A "disease-specific inclusion criteria" subsection, then other criteria |
| 1 (~3%) | Core criteria plus additional criteria in later part/module sections |

Prompt headings never sort criteria into "common" and "cohort-specific" sets,
which would mirror the extraction schema. See `PROMPT_HISTORY.md` for the
v1–v11 iterations and the checks behind them.

## Running a batch

```bash
cd matchminer-ai-training-kehl
B=../data/no_phi/synthetic_protocols_from_registry/batch_v11_sample5000_20261029
python -I synthetic_protocols/generate_batch.py \
  --nct-ids-from ../data/no_phi/sample_trial_cohort_lineitems.csv \
  --output-dir "$B" --chunk-size 500 \
  --server http://sn4622130540:8000=256 \
  --server http://172.24.216.113:8001=32 --server http://172.24.216.113:8002=32 \
  --server http://172.24.216.113:8003=32 --server http://localhost:8002=128
```

- **Seeds:** `--nct-ids-from` fetches registry records for the CSV's `nct_id`
  column once into `<output-dir>/seeds.jsonl` (with `retrieved_at`);
  `--seeds FILE` uses an existing JSON-lines file instead. `--limit N` runs the
  first N seeds for a smoke test.
- **Servers:** on the first run `--server URL=CAP` entries are written to
  `<output-dir>/servers.json`, each with the model ID the server reports, so
  servers serving different builds (for example NVFP4 and FP8) can be mixed. The
  file is re-read about every second: remove or add an entry, or change its
  `upstream_concurrency` cap, without restarting. Later runs reuse the file and
  ignore `--server`.
- **Concurrency:** the inference package's endpoint pool runs adaptive
  concurrency per server (starting at 32, growing in saturated successful
  windows up to the cap, halving under vLLM load pressure read from `/metrics`).
  Two 500-trial chunks stay in flight so fast servers are not left idle while a
  slow server finishes a chunk.
- **Sampling:** the runtime model name (`--model-name`, default Nemotron 3.5
  Super VL NVFP4) selects the package's sampling profile (temperature 1.0,
  top-p 0.95, thinking on); `--max-tokens` defaults to 100,000.
- **Stopping and resuming:** stop with `kill <pid>`; rerun the same command. A
  trial is skipped when its `meta.json` records `finish_reason` `stop` and the
  current prompt hash, so a prompt or layout change regenerates everything.
  Requests still in flight when a run stops are lost and repeated.

## Outputs

```text
<output-dir>/
  seeds.jsonl            registry seeds and their retrieval time
  servers.json           live endpoint registry
  protocols/<NCT>/
    protocol.md          the synthetic protocol
    reasoning.txt        the model's reasoning trace
    meta.json            prompt hash, layout index, attempt, finish reason,
                         serving endpoint and model, seed fields, completion time
  protocols.csv          one row per generated trial (written at the end)
  run_metadata.json      errors, skipped count, model metadata
  trial_attempts.json    every attempt with its finish reason and endpoint
```

Empty or truncated outputs (Nemotron occasionally returns reasoning with no
answer) are retried up to `max_attempts` (3); trials that still fail are listed
in `run_metadata.json`.

## Extracting cohorts and eligibility criteria

```bash
python -I synthetic_protocols/extract_cohorts.py --batch-dir "$B" \
  --server http://sn4622130540:8000=256 \
  --server http://172.24.216.113:8001=32 --server http://172.24.216.113:8002=32 \
  --server http://172.24.216.113:8003=32 --server http://localhost:8002=128
```

The driver takes every protocol in the batch whose `meta.json` records a
completed generation, checks that its seed matches `seeds.jsonl`, and sends the
whole Markdown document to `extract_protocol_cohorts` with the full-protocol
prompt and JSON-schema structured outputs (default model Nemotron 3.5 Super VL
NVFP4, 50,000 completion tokens, chunks of 500 with two in flight). It refuses a
batch that mixes generation prompt versions. Servers, adaptive concurrency,
stopping and resuming work as for generation: each accepted decision is saved
when its chunk returns and reused on rerun when the extraction prompt, schema and
protocol bytes are unchanged. Output goes to `<batch-dir>/cohort_extraction/`
(or `--output-dir`):

```text
cohort_extraction/
  servers.json           live endpoint registry
  trials/<NCT>/
    extraction.json      raw answer, parsed decision, attempts with endpoint and
                         model, prompt hash, schema, protocol SHA-256
    reasoning.json       each step's reasoning
    merge.json           a multi-cohort trial's merge grouping, with the answer and
                         reasoning (reused on rerun)
    deboilerplate.json   each merged cohort's criteria with the boilerplate removed,
                         with the answers and reasoning (reused per unchanged cohort)
  cohorts_unmerged.parquet, .csv
                         one row per extracted cohort (criteria verbatim, boilerplate
                         included), with provenance
  cohorts.parquet, .csv  one row per cohort after merging those with essentially the
                         same criteria (merged_from: the extracted cohort numbers),
                         boilerplate included, with provenance
  cohorts_deboilerplated.parquet, .csv
                         the merged cohorts with their boilerplate criteria removed
                         (consent, compliance, samples/tissue/biopsy, measurable or
                         evaluable disease, contraception/pregnancy/lactation, vaccines,
                         staff and catch-all exclusions): the cohorts that seed patients
  trials.parquet, .csv   one row per protocol: provenance and extraction status
  run_metadata.json      git commits of both repositories, settings, servers,
                         status counts, failures, seeds without a protocol
  trial_attempts.json    every attempt, including failed trials
```

Each row in `cohorts` and `trials` carries its provenance:

| Columns | Source |
| --- | --- |
| `protocol_origin` (`synthetic`), `protocol_path`, `protocol_sha256` | The synthetic protocol |
| `synthetic_prompt_sha256`, `synthetic_layout_index`, `synthetic_attempt`, `synthetic_completed_at`, `synthetic_endpoint`, `synthetic_model` | Its generation (`meta.json`) |
| `registry_title`, `registry_brief_summary`, `registry_detailed_description`, `registry_eligibility_criteria`, `registry_last_update_post_date`, `registry_retrieved_at` | The ClinicalTrials.gov seed record, including its original eligibility criteria |
| `extraction_status`, `cohort_count`, `extraction_error`, `extraction_attempts`, `extraction_endpoint`, `extraction_model`, `extraction_prompt_sha256`, `extraction_schema_version`, `extraction_decision_path` | The extraction |

`cohorts` also has the inference package's cohort columns (`COHORT_COLUMNS`,
`cohort_criteria_source` = `protocol`). `extraction_status` is `cohorts`,
`not_cancer_treatment` or `failed`; only `cohorts` trials have cohort rows.

## Patient-generation input

```bash
python -I synthetic_protocols/patient_cohorts.py --batch-dir "$B" \
  --extraction-dir "$B/cohort_extraction_v2" \
  --output ../data/no_phi/<generation-run>/sample_trial_cohort_lineitems.csv
```

`patient_cohorts.py` turns an extraction's `cohorts_deboilerplated.parquet` (`--cohorts-file`;
`--extraction-dir`, default `<batch-dir>/cohort_extraction/`) into the cohort
table that `generate_synthetic_data.py` (`run_pipeline.py --sampled-cohorts`)
reads: accepted cohorts in seed order with a contiguous `cohort_index`, each
trial's source `split` copied from the registry sample it was seeded from
(`--splits-from`, default `../data/no_phi/sample_trial_cohort_lineitems.csv`),
and every provenance column. Single-cohort rows keep `cohort_criteria_source`
`protocol`, so the generator uses the protocol's criteria; the table has no
`eligibility_criteria` column, and the registry text stays in
`registry_eligibility_criteria`. The script checks that the generator's own
adapter reproduces the extracted criteria exactly, and writes
`<output>.provenance.json` (source stamps, counts, failures, prompt hashes).

## In the training pipeline

`run_pipeline.py` (and the identical `train_all.sh`, `train_all_gcp.sh` and
`temp_train_gcp.sh` wrappers) runs this track as stages after `sample-trials`:
`synthetic-protocols` (`generate_batch.py --nct-ids-from` the registry sample),
`protocol-cohorts` (`extract_cohorts.py`), `registry-cohorts`
(`0b_extract_trial_cohorts.py --trials-jsonl` on the batch's `seeds.jsonl`),
`generation-cohorts` (`patient_cohorts.py`, the patient-generation input) and
`retrieval-cohorts` (`retrieval_cohorts.py`, the `train` stage's cohort table).
There `generate_batch.py` and `extract_cohorts.py` use the pipeline's
`--server-urls-file` instead of writing their own `servers.json`, and
`--allow-failures` lets a stage finish when a few trials fail every attempt;
those trials keep whichever version they have.

## Registry versions and the retrieval cohort table

Users may be allowed to send some trials' full protocols to an LLM and only the
ClinicalTrials.gov text of others, so TrialSpace and the downstream models are
trained on both representations. Every sampled trial is therefore also
extracted from the registry record its protocol was written from. Since
2026-11-03 both extractions run the inference package's three-step workflow with
the same prompts (`trial.cohorts.screen.user.txt`, `trial.cohorts.list.user.txt`,
`trial.cohorts.criteria.user.txt`; schema `trial-cohorts-stepwise-v1`): a
cancer-treatment screen, the cohort names, and one request per cohort for its
complete criteria, with the listing's title, brief summary and eligibility
criteria as one document. Since 2026-11-04 (user instruction) the criteria are
copied verbatim with boilerplate included, and training gives registry listings the
same two follow-on steps as protocols: the merge, then a separate request per merged
cohort that removes its boilerplate criteria. Retrieval and labeling read both
versions with boilerplate removed (`--output-deboilerplated-cohorts` beside the
generation table, which comes from `cohorts_deboilerplated.parquet`). Earlier runs used
the single-request registry prompt (`trial.cohorts.user.txt`, schema
`trial-registry-cohorts-v1`):

```bash
python 0b_extract_trial_cohorts.py --trials-jsonl "$B/seeds.jsonl" \
  --work-dir <run>/registry_cohorts/work --output-trials <run>/registry_cohorts/trials_with_cohorts.csv \
  --output-cohorts <run>/registry_cohorts/trial_cohort_lineitems.csv --output-json <run>/registry_cohorts/trial_cohorts.json \
  --output-unmerged-cohorts <run>/registry_cohorts/trial_cohort_lineitems_unmerged.csv \
  --output-deboilerplated-cohorts <run>/registry_cohorts/trial_cohort_lineitems_deboilerplated.csv \
  --model <served alias> --tokenizer-model <model or tokenizer path> --server-urls-file <servers.json> \
  --max-attempts 5 --allow-failures
python -I synthetic_protocols/retrieval_cohorts.py \
  --protocol-cohorts <generation-run>/sample_trial_cohort_lineitems.csv \
  --registry-cohorts <run>/registry_cohorts/trial_cohort_lineitems_deboilerplated.csv \
  --registry-trials <run>/registry_cohorts/trials_with_cohorts.csv --output <run>/retrieval_cohorts.csv
```

`--trials-jsonl` extracts every record, without `0b`'s date and phase selection
or split assignment; `--allow-failures` exports even when a trial fails every
attempt (that trial keeps only its protocol version). `retrieval_cohorts.py`
keeps the protocol rows unchanged and adds the registry rows with IDs
`<NCT>-registry-cohort-<n>`, `cohort_criteria_source` `registry`, their registry
fields as `registry_*`, and the trial's split from `--splits-from`; a trial
accepted in only one version keeps that one. This table, not the
patient-generation table, is the `--cohorts` input to `trialspace_training.py
prepare`. Patients are still generated from the protocol cohorts only.

Mining then represents each trial by one version (main README, Mining): each
patient searches one version of each of its sampled trials, chosen at random
per patient and trial, and each trial queries for patients with one version,
chosen at random per trial. A patient's pairs with its own source trial
include both versions.

## Caveats

- These are fabricated documents. Label any dataset built from them as
  synthetic; the documents themselves deliberately carry no disclaimer.
- Seeds sent to the generation endpoints are public registry text. Do not add
  patient data to seeds.
- A batch can mix model builds (NVFP4 and FP8); each protocol's `meta.json`
  records the endpoint and model that produced it.
- Single-cohort seeds sometimes gain dose-escalation or expansion parts, which
  shifts the cohort-count distribution relative to the seeds.
- Training is planned on this synthetic corpus, not on real protocols; real
  protocol PDFs are for comparison and evaluation only. The extracted cohorts
  reach patient generation through `patient_cohorts.py` and
  `run_pipeline.py --sampled-cohorts`, and retrieval together with the
  registry versions through `retrieval_cohorts.py`.
- Extracted criteria come from the synthetic protocol, not the registry; the
  `registry_*` columns are the seed, which the protocol may refine or differ from.

## Batches

| Directory under `data/no_phi/synthetic_protocols_from_registry/` | Prompt | Status |
| --- | --- | --- |
| `batch_v5_sample5000_20261028/` | v5 | Stopped at 3,861 of 4,981; eligibility structure mirrored the extraction schema (superseded) |
| `batch_v11_sample5000_20261029/` | v11 via the inference package | Current: 4,981 protocols (2026-10-29). `cohort_extraction_v2/` (schema `trial-protocol-cohorts-v2`, `--output-dir`) is current: 7,721 cohorts from 4,848 trials, 132 judged not cancer treatment, 1 failed (duplicate cohort criteria). `cohort_extraction/` (schema v1: 7,716 cohorts from 4,850 trials) is superseded: its `minLength` strings kept vLLM structured outputs from writing any JSON escape, so cohort-specific and single-cohort criteria have no line breaks or double quotes |
