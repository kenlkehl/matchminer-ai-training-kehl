# matchminer-ai-training

The active pipeline trains text-only GemmaEmbedding 2 TrialSpace,
then two OncoReasoning LoRA adapters on the same frozen Gemma 4 E4B text model:
one for patient summarization and one for individual TrialChecker,
BoilerplateChecker, and GoodOption questions.
The separate ModernBERT TrialChecker and BoilerplateChecker trainers have been
removed. GoodOption catalog construction remains part of training.

Use a venv on local disk (`~/thisenv` on the development machine), never inside
this network-mounted checkout. On the training VM, install the project and the
sibling inference repository into a uv venv, then run:

```bash
~/thisenv/bin/python train_from_summaries.py --dry-run
PYTHON=~/thisenv/bin/python bash train_all_gcp.sh
```

All three shell entrypoints use this same runner. It expects eight GPUs and a
sibling `matchminer-ai-inference-kehl` checkout pinned to
`fc025bf3407e2f13d7cb52ef65ad0720036faee7`. Review the pin and prompt contracts
before adopting changes in that dependency. The default teacher for every
label, catalog stage, exclusion extraction, and distillation request is
`nvidia/Gemma-4-31B-IT-NVFP4`. Eight text-only vLLM workers serve it on localhost;
they are stopped before student training. A separate vLLM executable can be
selected with `--vllm` when serving dependencies need their own environment.
TrialSpace labeling allows up to 96 concurrent requests per teacher GPU
(768 total), and each vLLM worker permits 96 active sequences. The labeler's
adaptive limiter ramps up after successful requests and backs off on errors.
Cataloging, exclusion extraction and distillation share a live endpoint pool.

To run the pipeline against an existing teacher, supply its OpenAI-compatible
endpoint instead of starting local vLLM workers:

```bash
~/thisenv/bin/python train_from_summaries.py \
  --teacher-url http://sn4622130540:8000/v1 \
  --teacher-served-model gemma4-31b --gpus 0,1 \
  --label-concurrency-per-server 288
```

Repeat `--teacher-url` for multiple servers. The runner verifies the served model
and context capacity, writes the shared endpoint registry, and leaves external
server processes running between stages. Mining and student fine-tuning still
use the selected GPUs on the machine running the pipeline. If the teacher has a
smaller context, set `--teacher-max-context` to its supported limit.
`--label-concurrency-per-server` persists the TrialSpace labeling ceiling across
stage transitions and restarts; the default remains 96 per endpoint. The shared
teacher above has been measured at 288. Catalog and long-context distillation use
the separate `--teacher-concurrency` setting (default `0`, automatic). Automatic
scheduling can fill the sum of the registry's `upstream_concurrency` caps, up to
1,536 pending workers; entries without a cap default to 96. An explicit positive
value limits total concurrent chains. The current primary/H200/Camus/cloud caps
are 288/64/32/96 per server. Registry additions and removals take effect during
each stage. Singleton catalog calls share the pool instead of repeatedly using
the first server. Transient connection, rate-limit and server failures retry
on healthy workers for up to 24 hours, allowing Spot workers to return. Endpoint
addresses are execution metadata and do not invalidate saved model responses.
Requests allow 7,200 seconds; `scripts/teacher_proxy.py` allows 7,260 seconds and
cancels upstream work when a caller disconnects. The run identity records the exact
trial-scoring rubric digest; changing that rubric requires discarding old labels
and resetting dependent stages rather than resuming mixed-prompt outputs.
The scoring instructions precede both summaries to provide a reusable cache
prefix. Required biomarkers follow the rubric's tested/standard-of-care inference
rules and its exception for novel screening biomarkers. Specificity requires
support for the complete applicable requirement. Binary TrialChecker questions
reuse the relevant current reasonableness clauses and specificity guidance;
their instructions and answer options also precede patient and trial context.
Prior-treatment specificity explicitly requires an on-therapy response when the
trial calls for it, matching the aggregate scoring rubric.
For a served alias, the server must report `--teacher-model` as its underlying
root; tokenizers still load from that HF model. OncoReasoning accumulation scales
with GPU count (32 steps on two GPUs, 8 on eight) to retain a batch of 64 examples
per optimizer step when the GPU count divides 64.

An existing local run can send TrialSpace fitting to the eight-GPU GCP VM through
an explicitly scoped `.mmai_cache/remote_trialspace.json` execution configuration.
The adapter leaves teacher traces in the local label artifacts, transfers the
unchanged columns used by the embedder, and waits for a durable cloud job.
Cloud teachers stop while fitting owns those GPUs and return afterward. The
Spot job resumes the latest complete checkpoint, saves every 500 steps, and
stores checkpoints on persistent disk. The local stage completes only after the
trained SentenceTransformer artifact has been copied back and checked. The
ordinary trainer retains its 2,500-step checkpoint interval; `--save-steps`
can override it.

For the active run, `scripts/register_downstream_upgrade.py` records the reviewed
teacher transport/prompt update while preserving the original completed-stage
fingerprints and prior round 2 correction. Register it with the local coordinator
stopped, then restart that coordinator. The durable cloud fit keeps running on
its recorded revision; its execution pins must remain unchanged until it finishes.

The run starts from `../data/no_phi/patient_summaries_with_spaces.parquet` and
`trial_space_lineitems.csv`, mines candidates with the base embedding model,
labels them, and trains TrialSpace round 1 from the base model. Round 2 mines
with that trained model, labels those candidates, and continues from the round 1
weights using only round 2 labels. Each fit has a fresh optimizer and learning-rate
schedule; checkpoints resume only interruptions within that fit. A third mining
round uses the trained round 2 model to supply OncoReasoning candidates.
For each patient, mining samples 500 **distinct NCT IDs**, expands all their
spaces, and keeps 20 spaces. For each trial space it samples up to 20,000
patients and keeps 40. Sampling is within original train/validation co-splits;
test rows are excluded and TrialSpace trains only on the train split.

The pipeline then builds and validates fresh drug and class evidence, extracts
individual exclusions, and distills summarization from `all_synthetic_notes.parquet`
plus all three QA families. Catalog web research receives public trial/drug
information only. Patient-bearing teacher requests use the configured local
workers or explicitly supplied teacher endpoints.
OncoReasoning trains the adapters sequentially with FSDP2 across eight GPUs.
Each starts from the original base and trains only on its own task category;
their data volumes do not compete in a mixed training objective. Defaults are
rank 64, alpha 128, all text-model linear layers, dropout 0.05, learning rate
5e-5 with 3% warmup and cosine decay, gradient clipping at 1.0, and one epoch.
Variable 10K–50K-token summary chunks and final answers without teacher
reasoning are unchanged. Outputs are `oncoreasoning/summarization` and
`oncoreasoning/clinical_qa`, each with its own checkpoints and validation metrics.
See the [OncoReasoning instructions](oncoreasoning_training/README.md) for the
answer-first contract and configurable student model.

Outputs default to `../data/no_phi/gemma_pipeline_v2` and
`../models/gemma_pipeline_v2`. `status.json`, per-stage logs, manifests, and
completion markers live in the run directory. Rerun the identical command to
resume after interruption: completed stages are checked, teacher outputs resume,
and student trainers restore their latest checkpoints. Source hashes, code
revisions, and settings must match. Keep these directories on persistent disk
on Spot VMs. A new run or changed inputs require fresh output directories.
For an existing run with unchanged inputs, `--skip-input-validation` reuses its
recorded source identities without reading or rehashing the input files. It does
not override prompt/settings compatibility or permit reuse of stale labels.

To correct a run stopped at round 2 training from the older cumulative-label
pipeline, stop its local runner and cloud job, archive the incorrect round 2
job directory and checkpoints on both machines, deploy this commit, and run
`~/thisenv/bin/python scripts/register_trialspace_upgrade.py --run-dir <run-dir>`.
The scoped receipt preserves completed mining, labels, and round 1 artifacts
while allowing only the round 2 weight/input change. Resume the original runner
with `--skip-input-validation`. Cloud execution must use a new operation ID and
the deployed commit; it must not reuse the old job or optimizer checkpoints.

For an already-running pre-adapter pipeline, pull this update without restarting
the service, then run:

```bash
~/thisenv/bin/python scripts/register_oncoreasoning_upgrade.py \
  --run-dir ../data/no_phi/gemma_pipeline_v2
```

The pending training entrypoint
automatically trains both adapters when reached. The explicit upgrade receipt
also permits restart after a Spot interruption while preserving the original
manifest and completion fingerprints. Registration rejects changes outside the
adapter update and changes to the earlier stage semantics; only the two teacher
batching limits may change. Registration must happen before OncoReasoning
training begins. Applying new batching limits requires draining and restarting
the teacher/labeling processes; saved label shards are reused.

## TrialSpace embedding model

TrialSpace fine-tunes `google/embeddinggemma-2` with only its text encoder
(`vision_config=None`, `audio_config=None`). Google documents these settings and
the input prefixes in its [Hugging Face model card](https://huggingface.co/google/embeddinggemma-2#best-practices).
Use `sentence-transformers>=6.1` and `transformers>=5.19` (`uv sync --group training`).

Patient summaries and trial spaces both receive
`task: sentence similarity | query: `, Google's symmetric similarity prefix,
because either side can be the retrieval query. The trainer adds the prefix once
to raw text before tokenization. The 2,500-token limit includes that prefix and
special tokens. Google's mean pooling (including prompt tokens), normalization,
and full 768-dimensional output are retained. Training uses float32 weights
with bfloat16 mixed precision on supported CUDA devices; float16 is disabled.

The saved SentenceTransformer sets `SentenceSimilarity` as its default prompt
and saves the same prefix under `query` and `document`. Encode raw summaries
with `model.encode(texts)` or `model.encode(texts, prompt_name="query")`.
`prompt="query"` would prepend the literal word, so it must not be used to
select a named prompt. Evaluation and mining scripts honor the saved prefix.

The integrated runner saves the final TrialSpace artifact as
`../models/gemma_pipeline_v2/trialspace_round2`. Regenerate patient and trial
embeddings for the new model; older vectors and indexes are incompatible.

## GoodOption evidence catalog

GoodOption is an evidence-counting signal for whether a trial's experimental
drug options have support relevant to a synthetic patient's cancer. It is not
an eligibility result, response probability, or treatment recommendation.

The trained four-logit GoodOptionChecker is deprecated. It read one patient and
one drug summary and never saw drug-class evidence, and no trained model was
published. GoodOption scoring now uses the inference package's LLM rubric
(`score_good_options_with_llm`), whose prompt packs drug and class evidence to
the teacher's context. The active pipeline builds that patient-free catalog,
then distills each rubric component into OncoReasoning. The `label`,
`train`, and `all` subcommands still work, with a `FutureWarning`, for
reproducing earlier checkers.

The v3 catalog workflow has a hard patient-free research boundary:

1. Collect unique NCT IDs from the configured candidate tables or an explicit
   ID list.
2. Fetch each ClinicalTrials.gov record and LLM-screen every `DRUG` or
   `BIOLOGICAL` entry for a concrete named agent with direct anticancer treatment
   intent. Audit and drop supportive/procedural medicines, diagnostic tracers,
   schedule/cohort labels, and unnamed standard-of-care placeholders; classify
   retained agents' trial roles, normalize them to NCIt where possible, and
   deduplicate them across trials.
3. Research each unique drug through authoritative oncology sources and a
   pluggable general-web provider. Fetch bounded document passages rather than
   relying only on result snippets; general-web queries explicitly include
   `cancer treatment` to reduce unrelated name collisions.
4. Synthesize structured facts and clean GoodOption/Help Me Choose summaries.
   Retain passage-supported early and immature evidence with its limitations,
   and materialize matched NCIt definitions as citable ledger passages.
5. Validate the complete catalog before any patient-bearing LLM request.
6. (Deprecated) Label every scoreable patient-drug pair on four independent
   binary criteria.
7. (Deprecated) Train a four-logit ModernBERT checker and aggregate all
   drug-by-criterion probabilities when producing a patient-trial score.

Genuine named anticancer control and background drugs remain available for
catalog coverage but are omitted from GoodOption prompts and denominators.
Supportive/procedural drugs are excluded before research. A retained agent whose
role remains uncertain is marked and scored. A completed search with no evidence
is a valid result; an unresolved technical research failure blocks affected
labels.

The unchanged criteria are:

1. human clinical benefit from the same drug or a regimen containing it in the
   patient's disease and relevant histology;
2. a directly targeted biomarker present in at least 20% of the full relevant
   disease population, or authoritatively described as common/frequent/highly
   expressed there;
3. explicit documentation of that target in this patient's own tumor; and
4. human benefit from therapeutically targeting that same patient biomarker.

### Commands

The distillation scripts use the inference package's public stage APIs:
`matchminer_ai.trials` owns catalog construction/loading/validation, and
`matchminer_ai.matching` owns LLM labeling, classifier-input formatting, and
rubric versions. Both the regular and co-split trainers use those shared
contracts. Prompt text is maintained in the inference package's `prompts/`
folder, not copied into training scripts.

For this unpublished workspace refactor, select the sibling checkout when
running either script (or install that checkout into your environment):

```bash
export PYTHONPATH="$(pwd)/../matchminer-ai-inference-kehl/src${PYTHONPATH:+:$PYTHONPATH}"
```

The API/prompt relocation preserves rendered teacher prompts, classifier
inputs, catalog compatibility IDs, and resumable checkpoints. Existing
compatible catalogs and label shards can be reused with the same CLI commands.

Deprecated: to reproduce an earlier checker, run the complete workflow in
sequence with one command. Each stage runs in an isolated process, validation
must succeed before labeling starts, and training is launched through
Accelerate:

```bash
python train_good_option_checker.py all \
  --model nvidia/Gemma-4-31B-IT-NVFP4 \
  --tensor-parallel-size 8 \
  --num-processes 8
```

All stage-specific options are accepted by `all`, including explicit NCT input,
research limits, remote teacher servers, label-shard settings, and training
checkpoint resumption. The individual commands remain available for inspecting
or operating stages separately.

Catalog construction checkpoints public-only state after every completed trial
fetch, LLM intervention screen, drug research operation, and drug synthesis.
Rerun the same `all` command after an interruption; compatible checkpoints in
`../data/no_phi/good_option_catalog_v3_checkpoints/` are reused automatically.
If the final catalog was already published, `all` validates it, verifies that
its trial IDs match the current inputs, and continues with labeling. Technically
blocked retrievals are retried when resuming an unpublished catalog.

Checkpoint manifests fingerprint the NCT list, sources and research settings,
ontology and prompt/schema versions, and teacher configuration. A changed run
fails rather than mixing incompatible evidence. To intentionally research from
scratch, use both `--overwrite` and `--reset-catalog-checkpoints` with the same
catalog paths. A catalog prompt/schema change also requires fresh label-shard,
aggregate-label, model-output, and model-checkpoint paths; label shards from a
different catalog compatibility ID are deliberately rejected.

Build and validate all patient-free evidence (the supported workflow, and what
step 17 of the orchestrators runs):

```bash
python train_good_option_checker.py catalog \
  --model nvidia/Gemma-4-31B-IT-NVFP4 \
  --tensor-parallel-size 8

python train_good_option_checker.py validate-catalog
```

Existing OpenAI-compatible servers can be supplied to both teacher stages:

```bash
python train_good_option_checker.py catalog \
  --server-urls http://host-a:8000/v1,http://host-b:8000/v1 \
  --model nvidia/Gemma-4-31B-IT-NVFP4
```

Deprecated: after validation, label and train a checker:

```bash
python train_good_option_checker.py label \
  --server-urls http://host-a:8000/v1,http://host-b:8000/v1 \
  --model nvidia/Gemma-4-31B-IT-NVFP4

accelerate launch --num_processes 8 train_good_option_checker.py train \
  --patient-validation-fraction 0.20 \
  --drug-validation-fraction 0.20
```

GoodOption labeling requests up to 100,000 output tokens. The GCP orchestration
serves a 131,072-token model context so that budget remains available after the
patient-and-drug prompt. Custom configs with a smaller active output budget are
rejected. For code-validation failures, only failed rows are retried: the next
turn includes the prior answer, exact parser error, and finish reason. The
default is three reasoning-enabled attempts followed by one final attempt with
thinking disabled. Rows that still have `parse_failed` status are intentionally
requeued when labeling resumes; later shards replace their failed predecessors
in the aggregate.

Use `--split-strategy none` to train on all valid synthetic rows when evaluation
will be performed on a separately governed real dataset.

For an interim run using completed label shards and the original synthetic
summary/trial co-splits, use `train_good_option_cosplit.py prepare --run-dir PATH`,
then launch `train_good_option_cosplit.py train --run-dir PATH` through torchrun.
Preparation freezes the available shards, reconstructs splits from
`patient_summaries_with_spaces.parquet` and `trial_space_lineitems.csv`, and keeps
only train/train for fitting and val/val for evaluation. Mixed and test splits
are excluded; the mining tables' constant `train` field is not used. Missing or
conflicting source assignments fail validation. Canonical drugs may overlap
between these original patient/trial co-splits.
Conflicting duplicate patient/drug targets are excluded in their entirety from
each partition and recorded in `conflict_audit.json`; the ordinary trainer's
default remains to reject conflicts. `finalize --run-dir PATH` repeats this
preparation from the frozen snapshot without reading newly generated shards.

This runner uses three fixed epochs and evaluates Val/val only at the end,
without checkpoint selection on that set. It saves the snapshot, split manifest,
model, per-criterion metrics, and validation probabilities under the run directory.
Its two-GPU defaults use an effective batch size of 64 (8 examples per GPU with
4 accumulation steps), BF16, and the locally cached ModernBERT-large base model.
ModernBERT's mean binary loss is explicitly normalized across accumulation steps.

The catalog command also accepts repeated `--nct-id` values or
`--nct-ids-file`. Text files contain one NCT ID per line; CSV/Parquet files use
`nct_id` or `trial_id`. Custom candidate inputs outside `../data/no_phi` require
`--confirm-inputs-are-non-phi` during `label` because their patient summaries
reach the configured teacher endpoint. The `catalog` command reads only the
`nct_id` column and does not accept or load patient context.

### Artifacts

- `../data/no_phi/good_option_catalog_v3/`: versioned Parquet catalog,
  intervention-screen audit, source ledger, retry audit, clean drug summaries,
  trial-drug index, and manifest.
- `../data/no_phi/good_option_catalog_v3_checkpoints/`: resumable, atomic,
  patient-free registry, intervention-screening, drug-evidence, and synthesis
  JSON.
- `../data/no_phi/good_option_four_point_label_shards_v2/`: resumable
  patient-trial labeling shards retaining the patient summary, exact clean drug
  information supplied to the teacher, and separate teacher final-response and
  reasoning/finish-reason columns. `llm_invocation_status` distinguishes a
  teacher call from an intentional skip such as `no_scoreable_drug`, so skipped
  rows do not look like unexplained blank generations. Resuming labeling
  backfills reconstructible drug-input and invocation-status columns in
  compatible earlier shards without relabeling them.
- `../data/no_phi/good_option_four_point_labels_v2.parquet`: validated label
  aggregate retaining the same teacher-input/output audit fields plus per-drug
  rationales and the catalog compatibility ID.
- `../models/goodoptionchecker_four_point_v2/`: deprecated four-logit checker
  and split/evaluation metadata (written only by the deprecated `train`).

Patient context is never accepted by catalog retrieval or included in public
queries. The patient labeling prompt contains only the patient summary followed
by clean summaries for scoreable drugs; it omits URLs, queries, source IDs,
trial metadata, controls, fetch notices, and raw evidence passages.

To repeat the prototype on all currently completed synthetic label shards without
patient, trial, or drug holdouts, snapshot with `train_good_option_cosplit.py
prepare --all-labels --run-dir <new-run-directory>`, then run its `train` stage
under torchrun with `--num-train-epochs 2`. The prepared manifest records zero
holdouts; invalid statuses and conflicting patient-drug targets retain the
prototype's filtering and audit policy. All label and catalog sources must be
inside `data/no_phi`. Training uses the immutable prepared snapshot, saves the
final epoch, and skips validation when none was reserved.

## Nemotron 3.5 Super VL teacher and trial-space extraction

The shared inference package registers NVIDIA's text sampling profile for
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`, BF16, and the short alias
`nemotron-3.5-super-vl`. Install/select the sibling inference checkout to use it.
The [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
(verified 2026-10-15) recommends temperature 1.0 and top-p 0.95 for text,
`nemotron_v3` for vLLM reasoning parsing, and `qwen3_coder` for tool parsing with
`--enable-auto-tool-choice`. Its serving examples require vLLM 0.32.0 or the
specified nightly. Other sampling filters/penalties are neutralized, not claimed
as NVIDIA recommendations. The vision recommendation is different (temperature
0.6); these training paths are text-only.

`vllm_reasoning_utils.resolve_sampling_defaults` uses the same automatic
model-family detection and sampling defaults as the installed inference package.
This applies to trial screening/extraction, synthetic prompt/note generation,
patient summarization, trial/boilerplate judging, synthetic extractors, and
`TeacherPool`. Gemma 4, Qwen 3.5/3.8, Nemotron, and Reflection Beam profiles are shared with
inference; unregistered models retain each task's previous fallback settings.
Unspecified CLI sampling values select the profile, while explicit values
(including zero) take precedence. Patient summarization's `--run_deterministic`
still overrides sampling when requested. Recognized families use neutral
penalties unless their profile or an explicit override specifies otherwise.

Detection uses the configured model ID. For an opaque server alias in `0b`,
pass the actual Hugging Face ID through `--tokenizer-model` to select both the
tokenizer and family defaults while retaining `--model` in endpoint requests.
The primary runner also selects model-aware reasoning/tool parsers. Registering
a model does not change an existing run's teacher or saved labels.
Use a new run/checkpoint directory when changing models or sampling settings.

Reflection Beam 1.0 detection covers BF16, FP8, NVFP4, quantized reuploads, and
served names such as `Beam-1.0-501B-A23B`, `beam`, and `reflection-beam`.
Its [official vLLM recipe](https://developers.reflection.ai/serving-vllm)
specifies temperature 0.7 and top-p 0.99; the generation config also specifies
repetition penalty 1.0. Top-k/min-p filtering and presence penalty are neutralized.
Both reasoning and tool-call parsers resolve to `beam`. Explicit sampling
overrides still win, and no task output budget is changed. Beam always reasons;
its template uses `reasoning_effort`, not Gemma's boolean thinking switch.
Use a vLLM build with Beam support when serving or parsing raw local completions.
Remote `0b` uses Beam's server-side chat parser, so the client does not need that
specialized vLLM build. It saves the returned reasoning separately from the answer.
`--max-model-len` also caps remote requests: each attempt counts its rendered input,
reserves 256 tokens, and reduces the output allowance if necessary. Source trial
text is never truncated. Set this flag to the endpoint's actual context limit.
These settings are shared from the sibling inference package and were verified
on 2026-10-16; no existing server or saved run is switched automatically.

Example against an existing server (no local model startup):

```bash
PYTHONPATH=../matchminer-ai-inference-kehl/src ~/thisenv/bin/python 0b_create_trial_spaces.py \
  --input ../data/no_phi/ctgov_trials.csv \
  --server_urls http://sn4622130540:8000/v1 \
  --model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --max_concurrent_per_server 256 --request_timeout 7200
```

Remote extraction keeps parsed reasoning separate from the final spaces and
retries truncated or empty answers. Outputs remain `trials_with_spaces.csv` and
`trial_space_lineitems.csv` in `../data/no_phi`, with resumable parquet shards.

### Trial selection in 0a/0b

Run `0a_parse_ctgov_json.py --input <export.json> --output ../data/no_phi/ctgov_trials.csv`
to retain the original trial text plus ClinicalTrials.gov `designModule.studyType`
and `designModule.phases` as `study_type` and a JSON-array `phases` column.
The CSV is replaced atomically only after parsing and writing finish.

0a also retains these fields from the export's `statusModule`:

| CSV column | Registry field |
| --- | --- |
| `overall_status` | `overallStatus` (for example, `RECRUITING` or `COMPLETED`) |
| `status_verified_date` | `statusVerifiedDate` |
| `study_first_post_date` | `studyFirstPostDateStruct.date` |
| `study_start_date` | `startDateStruct.date` |
| `primary_completion_date` | `primaryCompletionDateStruct.date` |
| `study_completion_date` | `completionDateStruct.date` |
| `last_update_post_date` | `lastUpdatePostDateStruct.date` |

Each of the five structured dates also has a corresponding `_type` column,
preserving the supplied `ACTUAL` or `ESTIMATED` value. Date strings retain their
original precision (for example, `2025-11` stays month-only); missing values stay
blank. These are metadata columns and are not appended to the LLM's trial text.
Both 0b output CSVs preserve them when present, including empty outputs.
0b requires `study_first_post_date` for year selection; the other registry
status/date fields remain optional.

Completion dates refer to final outcome-data collection, **not an accrual-closure
date**; see the [NLM definitions](https://support.nlm.nih.gov/knowledgebase/article/KA-03766/en-us).
Use recruitment status to distinguish currently accruing records, rather than
inferring it from completion dates. Existing CSVs acquire the columns only when
0a is rerun; regenerate into a separate output path while a paused run still
references the original file.

By default, **0b keeps trials first posted in 2020 or later**. Use
`--start-year YEAR` to change this inclusive calendar-year cutoff, for example
`--start-year 2023`. Selection uses `study_first_post_date`, not study start or
last-update dates. Missing or invalid first-posted dates are excluded; an input
without the column must be regenerated with 0a.

0b also defaults to **interventional early phase 1 through phase 3 trials**,
including phase 1/2 and phase 2/3. It requires `study_type=INTERVENTIONAL` and a
nonempty phase array containing only `EARLY_PHASE1`, `PHASE1`, `PHASE2`, and/or `PHASE3`.
Phase 4, observational studies, and missing/N/A phases are excluded.
It uses structured metadata, never mentions of phases in the title or eligibility
text. Recruitment status is not filtered.
Use `--trial-filter all` to bypass only the structured study-type/phase filter;
the start-year cutoff and treatment-intent screen below still apply.
Legacy CSVs must be regenerated with 0a for the default filter.

Metadata selection happens before splitting, prompt construction, or LLM calls. A
`trial_selection.json` manifest fingerprints the selected rows, start year,
study-type/phase filter and seed. The year cutoff applies to treatment screening,
space extraction, and both output CSVs consumed downstream.
Use a **new `--work-dir`** when changing inputs or filtering, or when old shards
lack this manifest: checkpoint row numbers cannot safely be reused across
selections. Both final CSVs retain study type, phase, and available registry status
and date columns for audit.


### Direct anti-cancer treatment screen in 0b

The legacy extraction primer and closing instructions in `0b_create_trial_spaces.py`
are complete snapshots of inference-kehl's `trial.user.primer.txt` and
`trial.user.question.txt`, synchronized on 2026-10-17. They require exhaustive
cancer-type/histology spaces for broad solid-tumor or hematologic trials, specific
common histologies instead of "any", and explicit clarification of whether stage
criteria include relapsed/recurrent disease. New outputs use `Disease state allowed:`;
0b and 0c also accept saved spaces using `Cancer burden allowed:`. These labels do
not change the CSV columns. Treatment refractoriness belongs in treatment history,
and boilerplate exclusions must correctly negate inclusion AND/OR conditions. A cross-repository contract test compares
the complete rendered messages with the sibling inference-kehl checkout.
These complete one-shot messages are used only with `--extraction-method one-shot`.
The default hierarchical path below reuses the primer's full semantic instructions
without its one-shot output formatting.
Each `space_extraction/` directory records the rendered prompt fingerprint and
model/sampling configuration in `extraction_manifest.json`. Changing them
requires fresh extraction checkpoints; unchanged treatment-screening results
can be retained. Existing extraction shards without that manifest are rejected.

After metadata selection and deduplication, 0b completes a separate LLM pass
before extracting any trial spaces. It uses the same `--server_urls` or
`--server_urls_file`, `--model`, reasoning parser, sampling settings, concurrency,
and retry configuration as extraction. In local GPU mode it uses the same model
and GPU groups, loading models separately for the two passes.

The distinct prompt in `trial_treatment_screening.py` classifies the studied
intervention's intent using the supplied `trial_text`. It includes direct
anti-cancer therapy (including early-phase safety/dose studies and adjuvant or
maintenance treatment) and excludes diagnostic-only and supportive-care studies.
Anti-cancer treatment studies qualify even when their only outcomes are PK/PD,
safety, tolerability, target engagement, or other surrogate measures. This includes
drug-exposure/food-effect studies and preoperative treatment windows measuring
tumor biomarkers; clinical efficacy endpoints are not required. Biomarker-only
sampling without evaluation of an anti-cancer treatment remains excluded.
Cancer patients receiving background anti-cancer therapy are not sufficient:
the study must evaluate an anti-cancer treatment strategy. Insufficient textual
evidence produces a negative decision with an explanation, so the audit can also
identify records needing richer source text or review.

Only positive decisions proceed to space extraction. `trial_treatment_screening.csv`
is saved beside `--output-trials` by default, with all screened trials, boolean
decisions, rationales, parsed reasoning, final responses, finish reasons, status,
and `treatment_screen_model` (the requested `--model` ID or served alias). The
model is also recorded in new local and remote screening shards. Override the
CSV location with `--output-screening`. This audit covers trials
that passed the metadata filter, not metadata-excluded rows. The two existing
final CSV schemas are unchanged; if every trial is rejected, both contain only
headers. `--screening-max-tokens` defaults to 8192, independently of extraction's
`--max-tokens`.

Use `--screening-only` to save the screening CSV and stop before any trial-space
extraction. This mode creates no extraction checkpoints or trial-space output
CSVs. Later, remove `--screening-only` while retaining the other arguments and
work directory to reuse completed screening decisions and proceed to extraction.

To extract with a different model or endpoint, use
`--screening-input /path/to/trial_treatment_screening.csv` with a new `--work-dir`
and the original input, year/phase filters, and seed. This validates complete
coverage, unchanged trial data and splits, successful responses, boolean
decisions, and screening-model provenance before selecting positive trials.
It makes no screening requests and leaves the source CSV unchanged. A
`screening_input_manifest.json` records its path, content hash and screening
models; extraction records its own model and prompt configuration separately.
`--screening-input` and `--screening-only` are mutually exclusive.

Checkpoints live in separate `treatment_screening/` and `space_extraction/`
subdirectories of `--work-dir`. Resume with the same arguments: successful
screening decisions are reused, including negatives. Malformed or truncated
responses are errors, never exclusions. Remote errors are retried by the pool;
malformed JSON and invalid screening schemas do not lower concurrency or count
as successful generations. The pool treats `OutputValidationError` and
`json.JSONDecodeError` as neutral concurrency feedback, while retaining the same
retry budget and ERROR-on-exhaustion behavior. Other request failures still back
off. Pool status logs separately report `output_err` within the total `err` count.
Remote 0b retries now send the specific validation error and the latest rejected
answer back to the model, preserving the original trial document and task
instructions. JSON syntax errors include the decoder's location and explanation.
Thinking is enabled initially; after every two invalid responses for one trial,
the next retry switches thinking on/off. The completion parser receives the same
thinking mode as the rendered prompt. Transport failures do not advance this
counter. Beam always reasons, so its retries alternate max/low reasoning effort
instead of toggling thinking off. Only the latest answer (up to 12,000 characters) enters retry context;
complete rejected outputs, reasoning, finish reasons, errors, mode changes, and
recovery events are recorded in each stage's `validation_failures.jsonl`.
The same retry feedback applies to truncated or malformed trial-space outputs.
The pool's attempt limit is unchanged. This feedback applies to remote mode;
local worker failures retain the behavior described below.
If attempts are exhausted, the audit records errors and extraction does not
start. Rerunning retries unresolved screens. Local worker failures likewise
stop before extraction and completed compatible screening shards can be reused.
Manifests guard input selection and screening prompt/model/sampling compatibility.
Use a new work directory for pre-screening-era extraction checkpoints or changed
screening configuration.

### Hierarchical trial-space extraction

0b now defaults to `--extraction-method hierarchical`. After the unchanged
treatment screen, it branches through age, cancer primary site, sex, histology,
disease state, required treatment, excluded treatment, required biomarkers and
excluded biomarkers, in that order. Each request receives the whole trial text,
one parent's fixed fields and its internal cohort context, and returns only
alternatives for the next field. Cohort context preserves linked restrictions;
fields are never combined as an independent Cartesian product. One separate
call extracts boilerplate with conditional exclusions kept conditional. Python
assembles, deduplicates and numbers complete leaves into the existing CSV
schemas (training numbering remains one-based).

Each field call supplies the full trial and one candidate's inherited characteristics,
then asks: "For a patient with this set of characteristics ... what sets of [the
current field] are relevant for determining eligibility?" It includes only the
current field's definition and examples verbatim from the active primer, plus
shared anti-boilerplate and omission guidance. Age retains its existing instructions
because the primer has no separate age section. Diagnosis enumeration accompanies
only primary-site/histology calls, and the treatment-defining biomarker example
accompanies only biomarker calls. Shared requirements stay in every child;
cumulative exclusions must not become optional alternative branches.

`trial_hierarchy.py`, `trial_prompts/trial.hierarchy.*.txt`, and
`trial_prompts/trial.user.primer.txt` mirror inference-kehl. Ownership is in
inference; synchronize both copies after edits. Contract tests compare all files
and the embedded one-shot primer. Hierarchy loads each authored field section
verbatim into the corresponding request, leaving out one-shot output formatting. Edits to its definitions
therefore affect the relevant hierarchy field; changed section boundaries fail
explicitly instead of silently dropping guidance. The separate boilerplate call
gets only the authored boilerplate rules, shared omissions and a trial-wide scope.
Biomarker fields retain only positive/negative result restrictions, including
specified alterations and expression thresholds. Testing or known-status
requirements alone are omitted, with `NA` when no result restriction remains;
screening-assessment notes accompany actual result restrictions only. The shared
primer applies this rule to one-shot extraction too. Disease state remains concise; molecular
assay results go in biomarkers, response to specific prior
treatment stays in treatment history, and all-comer cancer/histology enumeration
is retained as field-specific work.

Remote requests use one persistent adaptive pool, with the existing model
sampling, reasoning parsers, retries and per-server concurrency limits. A
validated parent is checkpointed immediately and releases its children without
waiting for siblings or other trials. Independent field levels and trials share
one ready queue; boilerplate can run from the start. Completed trials immediately
admit replacements from a rolling window. `--hierarchy-trial-batch-size`
(default 32) now sets the minimum active trial window, enlarged to at least total
endpoint capacity, rather than a serial batch barrier. One completed-trial shard
is saved per trial, with stable input-order merging. Resume reuses successful
nodes in `space_extraction/hierarchy_nodes/`; completion order cannot change
final numbering. Local GPU workers retain synchronous batches and keep their
loaded model resident across field waves.
In remote batches, a failed trial retains its trace and is listed in
`space_extraction/extraction_failures.json`; other trials continue and save their
completed shards. If any trial still fails, the batch reports an error after
processing the remaining trials and does not write partial final CSVs. A fully
pruned trial remains an extraction failure requiring review; its screening
decision is not changed automatically.
Set the remote ceiling with `--max_concurrent_per_server 32` (CLI default: 100).
That is one shared adaptive ceiling per endpoint across trials, field branches,
boilerplate and retries. With N endpoints the total ceiling is N times the
per-server limit. `--hierarchy-trial-batch-size` does not multiply it. Separate
client processes each have their own limit; set the CLI value before launch.
For tuning a running job, `--concurrency-control-file /path/to/control.json`
accepts `{"max_concurrent_per_server": 256}` and reloads it every
`--server_urls_refresh` seconds. The CLI limit remains the hard ceiling. Atomic
file updates change the active cap without restarting or cancelling requests;
invalid updates retain the previous value and request-error backoff still applies.
The optional `per_server` object overrides the default for exact endpoint URLs,
for example `{"max_concurrent_per_server": 256, "per_server": {"http://worker-b:8001/v1": 384}}`.
Each limit must fit within the CLI ceiling. Total capacity is the sum of active
endpoint limits; removing an override restores that endpoint's default. Combine
this with `--server_urls_file` to add or retire endpoints during a run.
Both paths validate JSON and finish reasons, return validation errors to the
model, and alternate thinking after two invalid responses (Beam alternates
max/low effort). Remote malformed-output errors remain neutral to concurrency.
`--max_attempts` controls the remote per-node pool retry budget and local
per-node validation attempts; choose a small value such as 4 for prompt reviews.

`--hierarchy-max-nodes` (4096) and `--hierarchy-max-calls` (20000 logical requests
per trial, excluding retries) fail explicitly instead of truncating large trees.
Context is recounted for each request/retry and output tokens are capped to fit
`--max-model-len`; source text is never silently shortened. Exhausted or wholly
pruned trials block final output rather than producing partial space lists.
Extraction defaults to `--max-tokens 50000` (reasoning plus final answer) and
`--max-model-len 100000`; explicit caller limits still override these defaults.
The separate treatment-screening budget remains 8192 tokens.
Successful nodes, validation-failure logs, and `hierarchy_traces/` remain
available for review and resumption. `space_reasoning_and_output` contains the
JSON node audit; `space_output_no_reasoning` is the code-assembled answer.

Primary site is fixed before the sex call; final space text retains the established
age, sex, primary-site label order. Existing one-shot, `trial-hierarchy-v1` and
`trial-hierarchy-v2` and `trial-hierarchy-v3` extraction manifests are incompatible
with this `trial-hierarchy-v4` field-specific prompt contract. Use fresh extraction
checkpoints while retaining compatible treatment-screening checkpoints. This
refactor does not clear data or restart a paused run. A single-branch trial
requires ten logical calls, and branching trials require more; correctness still
needs source review, especially for early branching decisions. Do not mix new
spaces/embeddings with artifacts from a previous prompt generation.
