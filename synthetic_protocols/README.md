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
| Prompt history and findings | [`PROMPT_HISTORY.md`](PROMPT_HISTORY.md) |
| Real protocol PDFs (comparison data) | `scripts/download_ctgov_protocols.py` |

Requires the inference package at commit `acb37a0` or later (overlapping chunks).

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
  protocol PDFs are for comparison and evaluation only. Nothing in
  `train_all.sh` or `run_pipeline.py` consumes the corpus yet.

## Batches

| Directory under `data/no_phi/synthetic_protocols_from_registry/` | Prompt | Status |
| --- | --- | --- |
| `batch_v5_sample5000_20261028/` | v5 | Stopped at 3,861 of 4,981; eligibility structure mirrored the extraction schema (superseded) |
| `batch_v11_sample5000_20261029/` | v11 via the inference package | Current |
