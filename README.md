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

All three shell entrypoints use `run_pipeline.py`. By default they start at
`train_from_summaries.py`, preserving existing training commands and checkpoints.
Select `--start-at` to include data production, and `--stop-after` to stop before
later stages. The training portion expects eight GPUs and a
sibling `matchminer-ai-inference-kehl` checkout pinned to
`75a14df89d27698c504ef81e61d652822f1faea9`. Review the pin and prompt contracts
before adopting changes in that dependency. The default teacher for every
label, catalog stage, exclusion extraction, and distillation request is
`nvidia/Gemma-4-31B-IT-NVFP4`. Eight text-only vLLM workers serve it on localhost;
they are stopped before student training. A separate vLLM executable can be
selected with `--vllm` when serving dependencies need their own environment.
TrialSpace labeling allows up to 96 concurrent requests per teacher GPU
(768 total), and each vLLM worker permits 96 active sequences. The labeler's
adaptive limiter ramps up after successful requests and backs off on errors.
Cataloging, exclusion extraction and distillation share a live endpoint pool.

Trial extraction in `0b_create_trial_spaces.py` accepts
`--hierarchy-always-think` to keep reasoning enabled on all hierarchical calls,
including formatting repairs. This policy is recorded in the extraction
manifest and requires fresh checkpoints when changed. Without the flag, repeated
formatting failures can still use the existing non-thinking repair fallback.

When every trial-population branch is pruned, a separate study-scope review
checks whether the study truly lacks a therapeutic cancer population. A review
that does not confirm rejection now triggers up to two automatic recovery rounds
for the pruned fields, using its explanation and the original field instructions.
Corrective calls have new checkpoint keys; existing successful nodes, screening,
and completed trials remain reusable. Exhausted recovery still flags the trial
for intervention and prevents an incomplete final export. Recovery calls honor
`--hierarchy-always-think` along with ordinary requests.

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
The shared workspace default is `medium`, overriding the upstream template's
`max`, for trial extraction, synthetic seeds/events/notes, and downstream teachers.
Use a vLLM build with Beam support when serving. The shared inference package
parses raw Beam analysis/final channels locally without requiring that build
on the client.
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

When re-extracting an existing sampled cohort, supply an input CSV containing
only those NCT IDs and their saved `split` values, and pass
`--preserve-input-splits`. This keeps the original train/val/test assignments
instead of drawing new ones for the smaller cohort. Missing, invalid, or
conflicting assignments are rejected. If reusing `--screening-input`, subset
the saved screening CSV to the exact same IDs without changing its source text,
decisions, model provenance, or splits. Use fresh extraction checkpoints and
output paths for the new model.

Remote hierarchical extraction prepares prompts on a dedicated tokenizer worker
and parses completion responses off the request event loop. Reasoning parsing
uses the inference package's bounded, synchronized Nemotron parser cache instead
of rebuilding its vocabulary for every response. This keeps ready requests
flowing while preserving prompts, reasoning traces, and existing checkpoints.


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
counter. Beam always reasons, so its retries alternate medium/low reasoning effort
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
disease state, prior treatment requirements, and biomarker requirements, in that order. Each request receives the whole trial text,
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
only primary-site calls, and the treatment-defining biomarker example
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
Remote checkpoint reads/writes run off the request event loop, so shared-filesystem
latency does not stall every server. Each accepted node is still persisted before
its children can run. Resume validates and reuses completed one-trial shards
without replaying their nodes or changing their original completion timestamps.
New node files are spread across hash-prefix subdirectories to avoid contention
on one large network directory; older flat node checkpoints remain readable in place.
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
Both paths validate JSON and finish reasons and return the exact validation error,
latest rejected draft, and a valid JSON structure to the model. Thinking stays off
after two invalid responses, or on the last retry with a smaller budget (Beam
stays at low effort). After repeated failures, the malformed draft is omitted;
the original source, exact error, and valid JSON structure remain. Remote
malformed-output errors remain neutral to concurrency.
`--max_attempts` controls the remote per-node pool retry budget and local
per-node validation attempts; choose a small value such as 4 for prompt reviews.

`--hierarchy-max-nodes` (4096) and `--hierarchy-max-calls` (20000 logical requests
per trial, excluding retries) fail explicitly instead of truncating large trees.
Context is recounted for each request/retry and output tokens are capped to fit
`--max-model-len`; source text is never silently shortened. Exhausted nodes still
block final output rather than producing partial space lists. Wholly pruned trials
receive a separate study-scope review against the full source. A confirmed absence
of a human therapeutic cancer population, supported by a verified verbatim source
excerpt, is a terminal rejection rather than an extraction error. Uncertain reviews,
malformed responses and unsupported excerpts remain errors. This targeted review
does not rescreen the denominator or change the seven-field extraction prompts.
Rejections are checkpointed; remote extraction also records them in
`space_extraction/extraction_rejections.json`.
The trial CSV retains rejected NCT IDs with `space_extraction_status=rejected`,
`space_rejection_reason`, `space_rejection_evidence`, and the full audit; those trials
produce zero space line items. Completed older shards remain reusable. Rejected
trials therefore do not enter 0c or synthetic generation, which consume line items.
Extraction defaults to `--max-tokens 50000` (reasoning plus final answer) and
`--max-model-len 100000`; explicit caller limits still override these defaults.
The separate treatment-screening budget remains 8192 tokens.
Successful nodes, validation-failure logs, and `hierarchy_traces/` remain
available for review and resumption. In `trials_with_spaces.csv`,
`space_reasoning_and_output` contains the full JSON node audit,
`space_output_no_reasoning` is the code-assembled answer, and `space_text`
contains the complete list of spaces. These whole-trial fields are saved once
per trial and omitted from `trial_space_lineitems.csv`. Each line item retains
its own `this_space`, `space_number`, source/registry metadata, boilerplate, and
`reasoning_trials_path`; join its `nct_id` to that trial-level CSV to inspect the
full audit. Original parquet checkpoints and hierarchy trace JSON remain intact.
The shared exporter streams checkpoints and individual rows, so local and remote
extraction (both hierarchical and one-shot) do not assemble a giant expanded
DataFrame or duplicate the entire reasoning tree on every space. Step 0c,
synthetic generation, embedding, and checker consumers continue to use the
space-level `this_space` and boilerplate columns.

Primary site is fixed before the sex call; final space text retains the established
age, sex, primary-site label order. Existing one-shot, `trial-hierarchy-v1` and
`trial-hierarchy-v2`, `trial-hierarchy-v3`, and `trial-hierarchy-v4` extraction manifests are incompatible
with this `trial-hierarchy-v5` combined-requirements prompt contract. Use fresh extraction
checkpoints while retaining compatible treatment-screening checkpoints. This
refactor does not clear data or restart a paused run. A single-branch trial
requires eight logical calls, and branching trials require more; correctness still
needs source review, especially for early branching decisions. Do not mix new
spaces/embeddings with artifacts from a previous prompt generation.

New space text uses `Cancer primary site allowed`, `Prior treatment requirements`,
and `Biomarker requirements`. Treatment and biomarker fields each retain both
positive and negative conditions. Extraction postprocessing, QC, contextualization,
and training's 0c sampling accept these seven fields as well as saved legacy
nine-field spaces; they do not rewrite saved text or mix extraction checkpoints.
TrialChecker and synthetic-patient prompts preserve clause polarity and AND/OR
logic. The structured ontology API and PDQ catalog retain their separate JSON
required/excluded fields, with explicit prompt adapters for the combined input.
Existing trained models can consume the new text, but their quality on this prompt
generation still requires evaluation; regenerate embeddings for changed spaces.

TrialChecker aggregate and component prompts use the inference rubric with combined-field
polarity guidance. The `oncoreasoning-answer-first-v6-combined-space` contract pins
that rubric; preparation prefers the sibling `matchminer-ai-inference-kehl` checkout
when present. Existing prepared distillation data must be regenerated for this
contract. The installed inference package must match it for aggregate labeling.

Hierarchical response parsing accepts one complete outer Markdown code fence
(`json`, case-insensitive, or no language tag). It then applies the same strict
JSON/schema, duplicate-key, and completion-status validation. Prose outside the
fence, malformed JSON, and truncated responses still fail. Raw responses and
reasoning remain unchanged in the audit. This presentation-only parser fix is
compatible with existing v5 prompts and successful node checkpoints.

## Generate event lists and synthetic notes with existing servers

After `0b` has completed, `0c_sample_trial_spaces.py` samples 3,000 trials
(seed 42) and retains all their spaces. Its DFCI holdout excludes trials open
during 2025–2026 using `dfci_trials_summary_10-18-26.csv`. A trial with an
invalid space is excluded as a whole so sampling never silently drops one of
its spaces. Input/output paths, sample size, seed, and holdout years are CLI
options. The sampling receipt records the exact NCT IDs and space count.

`generate_synthetic_data.py` creates five matching and five near-miss patients
per sampled space in two stages. For each class, one call creates five diverse
short prose seeds containing names, sex, diagnoses, and ages at each diagnosis.
Both classes receive a hypothetical prose format example with explicit
instructions not to reproduce the example or reuse its patient identity.
After that seed batch
is saved, five independent calls each expand exactly one patient's chronology.
The clinical instructions, including chronology/anatomy and trial-reference
rules, remain in the import-only modules `1a_make_synthetic_enrollee_prompts.py`
and `1b_make_synthetic_negative_enrollee_prompts.py`. They cannot run generation
standalone; use `generate_synthetic_data.py` or the pipeline's
`generate-synthetic` stage. Their named instruction blocks
feed `build_seed_messages` and `build_patient_messages`; shared stage framing
lives in `synthetic_event_prompts.py`. A positive patient's eligibility takes
precedence over the optional second-primary diversity request.

Both seed and patient prompts express ages as whole years plus additional weeks
(for example, `50 years and 21 weeks`, with weeks from 0 through 51). This applies
to diagnosis ages in seeds and to demographics and every event timestamp in
histories; chronology validation compares both
parts, and downstream synthetic dates preserve the week intervals. The warning
blocks in both prompt modules also prohibit imaging findings, including claims
of absence, for organs outside the specified scan's field of view.

Seed and patient calls share one continuous adaptive endpoint pool. Patient
expansion starts as soon as its own seeds are saved, without waiting for all
seed calls to finish. Each expansion receives only its assigned prose seed and the clinical/formatting
instructions, then generates all tagged events, including demographics and
diagnoses. Trial criteria, general exclusions, and matching/near-miss objectives
are confined to the seed call. Event expansion cannot repair a seed toward the
trial criteria; matching/near-miss labels are generation intentions, not verified
properties of the final histories. The prompt requires preservation of seed facts without verbatim
copying. The demographics line comes first and gives the age at the earliest
event. Clinical events can begin with symptoms, imaging, biopsy, or surgery;
the diagnosis follows the workup establishing it, at its seed-assigned age.
Both event examples illustrate workup before diagnosis. Findings must follow
the procedures or tests that establish them, including within the same week.
The runner validates five distinct prose seeds with diagnosis ages,
one patient per expansion, a diagnosis event at each seed age, nondecreasing
event ages, report events, and common target-trial references. It does not
automatically compare names, sex, or all clinical facts across free prose;
these require output review. Invalid or truncated responses retry
with the specific validation error. These checks do not establish clinical or
biological coherence. Thinking remains enabled and both stages save raw output,
reasoning, model, endpoint, and prompt hashes.
Seeds, histories, and notes use locally rendered prompts through `/v1/completions`,
including Beam. Beam's native analysis/final channel parser is shared with the
inference package and does not depend on the local vLLM version. Full raw text
is saved in `raw_response`, alongside separate reasoning and answer fields.
The pinned Beam fast tokenizer renders prompts locally with medium effort for
context-budget checks and provenance.

`patient_seeds/` contains durable parent responses; `patient_events/` contains
individually checkpointed histories with stable patient numbers and seed hashes.
Resumption reuses both and requests only missing patients. Five histories are
reassembled in patient-number order for the established positive/negative CSV
schema. The run manifest is versioned and pins all prompt/helper source files;
use a fresh output directory after changing them. The v4 seed-only event manifest
rejects reuse of previous trial-conditioned event, tagged-seed, or multi-patient
checkpoints.

After event generation completes, the unchanged step-2 prompt expands clinical,
imaging, pathology, and NGS report events into notes. `--stop-after-events` on the
generator allows an event-only pilot. Seed and individual-history budgets default
to 16,000 and 20,000 tokens (`--seed-max-tokens`, `--event-max-tokens`); the main
pipeline exposes these as `--generation-seed-max-tokens` and
`--generation-event-max-tokens`, including through all `train_all*.sh` wrappers.
Both the generation runner and standalone step 2 parse complete event tags as
boundaries, preserving clinical inequalities such as `PD-L1 <1%` or
`margin < 1 mm` in the event text and downstream note prompts.

```bash
python 0c_sample_trial_spaces.py \
  --input-csv /path/to/completed/trial_space_lineitems.csv \
  --output-csv /path/to/new-run/sample_trial_space_lineitems.csv
python generate_synthetic_data.py \
  --input-csv /path/to/new-run/sample_trial_space_lineitems.csv \
  --output-dir /path/to/new-run \
  --model gemma4-31b \
  --tokenizer-model RedHatAI/Gemma-4-31B-IT-FP8-Dynamic \
  --server_urls_file /path/to/live/servers.json \
  --concurrency-control-file /path/to/live/concurrency.json \
  --max_concurrent_per_server 1024 --request_timeout 7200
```

Outputs include `trial_spaces_with_positive_prompts.csv`,
`trial_spaces_with_negative_prompts.csv`, `all_synthetic_events.parquet`, and
`all_synthetic_notes.parquet`. The combined event parquet is saved before note
generation starts. During events, `status.json` reports saved seed batches and
individual patient histories separately. Preparation
and note assembly are bounded and streamed; all checkpoints and final files
remain on the client machine. The command does not start or stop vLLM services,
shut down VMs, summarize patients, or launch model training. A separately running
pool supervisor must maintain remote tunnels, endpoint health, and VM recovery.

Both `train_all.sh` and `train_all_gcp.sh` accept explicit input overrides:
`--trial-spaces`, `--synthetic-notes`, and
`--patient-summaries`. Defaults retain the canonical filenames under `--data-dir`.
When using a run that exported successful trials under a `completed_` name,
pass that exact space-level CSV; do not rename it or substitute the trial-level
audit CSV. The selected paths are shared by input validation, the run manifest,
all three mining rounds, and OncoReasoning preparation. For example, after notes
have been summarized:

```bash
./train_all.sh \
  --data-dir /path/to/new-synthetic-run \
  --trial-spaces /path/to/extraction/completed_trial_space_lineitems.csv \
  --synthetic-notes /path/to/new-synthetic-run/all_synthetic_notes.parquet \
  --patient-summaries /path/to/new-synthetic-run/patient_summaries_with_spaces.parquet \
  --dry-run
```

Use the same arguments with `train_all_gcp.sh` or `temp_train_gcp.sh`. Without
`--start-at`, these entrypoints still start from completed patient summaries.

## Run the complete pipeline or resume at an intermediate stage

`run_pipeline.py` invokes the existing stage implementations; the event and note
prompts remain owned by `1a`, `1b`, and `2` and are executed by
`generate_synthetic_data.py`. It does not duplicate generation or matching logic.

| Stage name (`--start-at` / `--stop-after`) | Work and outputs |
| --- | --- |
| `parse-trials` | `0a`: CT.gov JSON → `ctgov_trials.csv` |
| `extract-spaces` | `0b`: year/phase filtering, treatment screening, hierarchical extraction; compact `trial_space_lineitems.csv` plus full trial audit |
| `sample-trials` | `0c`: 3,000 complete trials, all their spaces, existing DFCI holdout and sampling receipt |
| `generate-synthetic` | Existing `generate_synthetic_data.py`: event histories, then raw notes; all checkpoints/reasoning preserved |
| `prepare-notes` | `prepare_synthetic_training.py notes`: dated/split notes plus `synthetic_patient_metadata.parquet` |
| `summarize-patients` | Existing `6`: chronological serial summaries, retaining chunk reasoning and one final summary per patient |
| `prepare-training` | Join final summaries to patient/space IDs, class, originating trial space, boilerplate, and source trial split |
| `train` | Existing `train_from_summaries.py`: mining, labeling, TrialSpace training, catalogs, and OncoReasoning |

For a new complete run, first inspect the plan:

```bash
./train_all.sh --start-at parse-trials \
  --ctgov-json ../data/no_phi/ctgov_export.json \
  --data-dir ../data/no_phi/new_dataset \
  --run-dir ../data/no_phi/new_training_run \
  --models-dir ../models/new_training_run \
  --server-urls-file ../data/no_phi/live_pool/servers.json \
  --concurrency-control-file ../data/no_phi/live_pool/concurrency.json \
  --generation-model gemma4-31b \
  --generation-tokenizer RedHatAI/Gemma-4-31B-IT-FP8-Dynamic \
  --dry-run
```

Remove `--dry-run` to execute; `--stop-after prepare-training` produces the data
without starting model training. `--start-year` defaults to 2020, inclusive;
`--sample-trials` defaults to 3000 and `--data-seed` to 42. Supply
`--screening-input` to reuse the previously generated treatment screen instead
of screening with the extraction model. Starting at `sample-trials` accepts an
existing compact `--trial-spaces` file, including `completed_` filenames.

Upstream stages require an existing live `--server-urls-file`. The external pool
supervisor owns VM recovery, tunnels and serving processes. The orchestrator does
not stop that pool. Unless separate teacher URLs are supplied, downstream training
uses the same live registry through `--teacher-servers-file`; declared model roots
allow the explicit mixed FP8/NVFP4 pool. TrialSpace labeling accepts the same live
concurrency-control file; other downstream teacher stages retain their own
`--teacher-concurrency` setting. Mining and fine-tuning still use `--gpus`, so
choose devices appropriate for those stages before executing training.

Raw generated notes stay under `--generation-dir` (default
`DATA_DIR/synthetic_generation`). Prepared training notes go to
`--synthetic-notes` (default `DATA_DIR/all_synthetic_notes.parquet`) without
overwriting raw outputs. Calendar dates absent from the synthetic event histories
are **explicitly fabricated**, using source event ages for approximate intervals
and event numbers for ordering, ending at `--synthetic-date-anchor` (2025-01-01).
Missing ages are interpolated within a patient; a history without ages uses daily
event order. `date_provenance` records the method. Supplied complete dates are
retained; partially dated or contradictory histories fail. These dates are for
synthetic records only. The adapter streams the note text, uses a temporary
on-disk index for metadata, and retains all source patient IDs and trial co-splits.
It requires all five positive and five negative patients per sampled space and
rejects missing/failed final summaries before mining.

To attach to an already-running generation job without restarting it:

```bash
./train_all.sh --start-at generate-synthetic --stop-after prepare-training \
  --wait-for-generation \
  --generation-dir ../data/no_phi/existing_synthetic_run \
  --data-dir ../data/no_phi/prepared_existing_run \
  --run-dir ../data/no_phi/training_existing_run \
  --trial-spaces ../data/no_phi/extraction/completed_trial_space_lineitems.csv \
  --server-urls-file ../data/no_phi/live_pool/servers.json \
  --generation-model gemma4-31b \
  --generation-tokenizer RedHatAI/Gemma-4-31B-IT-FP8-Dynamic \
  --dry-run
```

`--wait-for-generation` only reads that job's manifest/status until completion;
it never signals it, changes its manifest or prompts, or starts another generator.
A generation lock also prevents duplicate clients or resampling a live run.
An explicit wait timeout leaves the original job running. Remove `--dry-run`
when ready to wait and execute the selected downstream stages.

Upstream state lives in `RUN_DIR/upstream` (override with `--pipeline-dir`),
separate from the training runner's existing manifest and locks. Each stage saves
its exact command, input/source hashes, logs and verified output identities.
Rerunning the same command skips completed stages; changed inputs, code, options
or completed outputs fail rather than silently reusing stale checkpoints.
`--dry-run` prints the complete selected plan, including downstream training
commands, without starting work or writing run files. No stage is inferred from
unrelated files or another run directory.

Model-specific reasoning text: the literal `Reasoning: high` system header is
added only for GPT-OSS model identities. Gemma, Nemotron, Qwen, and other families
use their configured chat-template/API thinking controls; task instructions are
preserved without the GPT-OSS header. Unknown model identities receive no header.
