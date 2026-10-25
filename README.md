# matchminer-ai-training

The production TrialSpace path trains **ColBERT with gated score regression**,
starting from `lightonai/GTE-ModernColBERT-v1`. It uses complete patient histories
and cohort eligibility criteria in both retrieval directions. The pipeline then
builds the GoodOption evidence catalog and trains the existing answer-only
OncoReasoning summarization and clinical-QA adapters.

`train_all.sh`, `train_all_gcp.sh`, and `temp_train_gcp.sh` invoke
`run_pipeline.py`; its training stage invokes `train_from_summaries.py`.
`trialspace_training.py` is the production preparation/mining/labeling/training
entrypoint. The native ColBERT encoder, retrieval instruction, and packed MaxSim
scorer are owned by the inference dependency and imported by both production
and [`colbert_experiments`](colbert_experiments/README.md). Production fixes the
objective to gated regression.

Use a local-disk virtual environment, outside the network-mounted checkout, and
install this repository plus the sibling `matchminer-ai-inference-kehl` package.
The inference revision is pinned by `INFERENCE_COMMIT` in
`train_from_summaries.py`. The manifest also hashes the actual production sources
and prompt contract, so an uncommitted code or prompt change cannot silently
resume incompatible work.

```bash
~/thisenv/bin/python train_from_summaries.py --dry-run
PYTHON=~/thisenv/bin/python bash train_all_gcp.sh

# Use an existing teacher pool and explicit prepared data.
~/thisenv/bin/python train_from_summaries.py \
  --patient-summaries ../data/no_phi/patient_summaries_with_spaces.parquet \
  --trial-cohorts ../data/no_phi/synthetic_generation/sample_trial_cohort_lineitems.csv \
  --teacher-servers-file ../data/no_phi/live_pool/servers.json \
  --teacher-served-model gemma4-31b --gpus 2,3
```

Use `--teacher-url` repeatedly for fixed endpoints instead of a managed registry.
External servers remain running; select available local GPUs for mining and
training. Without external endpoints, the runner starts one text-only vLLM
teacher per selected GPU and stops those workers before student computation.
The default teacher is `nvidia/Gemma-4-31B-IT-NVFP4`, with vendor sampling and
thinking enabled. The approved Gemma 4 31B FP8/NVFP4 variants remain traceable by
actual endpoint and model root. `--teacher-max-context` defaults to 250,000;
`--vllm` selects a separate serving executable when needed.

TrialSpace labeling uses the existing adaptive request pool. The default
`--label-concurrency-per-server` ceiling is 96, with a starting limit of 16 and
1,024 pending pair chains (`--trialspace-label-workers`). A failed eligibility
gate ends its chain; eligible pairs ask their specificity questions separately.
`--teacher-concurrency-file` adjusts per-server limits during a run, and the live
registry admits/removes endpoints. Labeling checkpoints await atomic response
writes in background threads, preserving reasoning without blocking other request
chains. `status.json` refreshes every ten seconds with completed questions,
scored pairs, total candidate pairs, score distribution, and the first request
time; use scored-pair throughput for ETA because specificity calls are gated.
Other teacher stages retain the independent
`--teacher-concurrency` setting (0 means automatic capacity-based scheduling).

## Cohort and patient inputs

`0b_extract_trial_cohorts.py` calls the inference package's combined
cancer-treatment/cohort extraction API. `0c_sample_trial_spaces.py` samples
**5,000 cancer-treatment trials**, excluding trials open at DFCI in 2025 or 2026,
and keeps every cohort. When upstream data production is selected, training
consumes this sampled cohort file, not the full extraction denominator.

The production corpus requires native cohort IDs and complete accepted cohort
sets. Multi-cohort text concatenates cohort-specific and common eligibility.
Single-cohort text uses only the original eligibility criteria, without the title
or summary. Patient cancer history and the prior boilerplate field become one
summary with an **Other relevant history:** bridge. Original patient, NCT, cohort,
and source split IDs are preserved. Source assignments for sibling cohorts and
known synthetic patient/source-trial relationships are validated before pooling.
Production pools **all synthetic patients and cohorts** for both retrieval
directions, labeling, and optimization. Original splits become `source_split`
provenance; the effective corpus split is `train`. There are no synthetic patient
or cohort holdouts. Candidate exports retain `patient_source_split` and
`cohort_source_split`, including pairs across original source partitions.

This is the default for direct `trialspace_training.py` commands and both
`train_all*.sh` paths. Production fitting requires labeled query coverage for
every selected patient and cohort. The downstream OncoReasoning adapters also
fit all three source partitions. For a deliberate holdout experiment, use
`--synthetic-training-splits train` in the orchestrator, or prepare a fresh corpus
with `--no-merge-source-splits` and explicitly select training/mining splits.
Changing this policy requires new retrieval and labeling outputs.

The training-only defaults are `DATA_DIR/patient_summaries_with_spaces.parquet`,
`DATA_DIR/trial_cohort_lineitems.csv`, and `DATA_DIR/all_synthetic_notes.parquet`.
Override the cohort path with the sampled file when those differ. Historical
trial-space exports do not supply the new cohort contract.

## Labels and regression objective

For each canonical patient/cohort pair, the teacher first answers:

> Is this patient potentially eligible, meaning there is no clear evidence of ineligibility?

A **no** yields score 0 and skips all specificity calls. A **yes** triggers four
separate calls: matching cancer type, disease burden, prior treatment, and a
required biomarker condition documented in the patient/tumor. Python counts the
yes answers, including potential eligibility, giving scores 1–5 for eligible
pairs. Skipped answers are null; failed calls leave the pair unscored and retry
on resume. The brief answer justification and the model's separate reasoning
trace are both preserved. Unknown information alone does not make the patient
ineligible; specificity requires the support described by its own rubric.

The student minimizes `MSE(mean_MaxSim(query, document), yes_count / 5)`.
The production prediction link is identity: no sigmoid transforms labels or
predictions. This assumes equal spacing between the five positive scores; the
score is not a probability. One aggregate retrieval instruction precedes the
complete query text. The complete opposite-side text is the document. Training
groups labeled candidates by query and direction, retains constant-score groups,
and never treats unjudged/in-batch documents as negatives.

Production fits use **32 query groups per GPU per optimizer update**, with up to
**8 labeled candidates per query**: up to 256 scored pairs per GPU, or 1,024 on
four GPUs. The global update is planned across available direction/score strata,
so every available 0–5 score is represented when the group budget permits. Within
a query, selection keeps its score range and distinct score levels before filling
remaining slots. Rare scores are deliberately oversampled; absent scores are
never invented, and constant-score queries remain eligible for sampling.

Each full query and its candidates are encoded together; gradients accumulate
across query groups to keep 8K texts within memory. This is a large optimizer
batch, not 1,024 full texts held in one forward pass. Configure it with
`--trialspace-accumulation` and `--trialspace-documents-per-query`. Every update
logs its actual pair count, 0–5 histogram and score range. Batch policy and sizes
are checkpointed and cannot silently change on resume. Fixed diagnostic probes
remain unchanged by training's score-balanced sampling.

## Mining, fitting and artifacts

The production round sequence is:

1. Prepare the complete cohort/patient corpus once.
2. Mine with the base ColBERT model, label the canonical pair union, and fit round 1.
3. Mine with round 1, label new candidates, and fit round 2 from round 1 weights
   using only round 2 labels and a fresh optimizer.
4. Mine with round 2 to supply complete cohort/patient tables to catalog and
   OncoReasoning preparation.

Each fit defaults to three **query-group sampling passes in aggregate**, with
score-stratified sampling rather than uniform visitation of each query.
`--trialspace-steps` overrides that
budget. Defaults are learning rate 1e-5, accumulation 4 per GPU, up to four labeled
documents per query, BF16 on CUDA, and gradient checkpointing. `--gpus` controls
both distributed mining and DistributedDataParallel fitting. Checkpoints resume
interruptions within the same fit, with strict source/model/label identities.

Both query and document limits are **8,192 tokens**, including markers. Full text
is never silently truncated. The model uses its published projection, Q/D markers,
and document punctuation mask. All encoding and mining score calculations are
batched. Distributed miners share atomic token-cache shards on disk. By default,
each worker loads the current direction's document vectors once into a packed
GPU cache when they fit an 8-GiB budget, retaining an additional memory reserve
for model inference and scoring. Larger corpora fall back to the bounded CPU
LRU cache. Ragged vectors are stored without corpus-wide padding. Scoring groups
four encoder batches, generates lexical proposals in a batch, then evaluates up
to 512 query/document pairs together, bucketed by token length. Both padding
masks and the original mean-MaxSim rule are retained; normal floating-point
roundoff can differ between kernel batch shapes.

`--trialspace-device-cache-gb`, `--trialspace-score-query-batches`, and
`--trialspace-pair-batch-size` expose these performance controls. The direct
mining command omits the `trialspace-` prefix and also accepts
`--no-device-token-cache`. A zero cache budget selects the old bounded path.
These execution controls are recorded in per-rank runtime files separately from
the model/corpus/candidate identity, so tuning them can resume existing atomic
embeddings and query checkpoints. Queries are checkpointed in atomic batches
under `query_batches/` to avoid thousands of small network-filesystem writes;
older `queries/` files remain readable. A crash redoes only unpublished batches.
Rank count and scoring batch changes preserve all published query results.
These defaults apply through `train_all.sh` and `train_all_gcp.sh`, as well as
direct `trialspace_training.py` execution; no background supervisor is needed.
Encoder batch sizes and text/scoring
contracts remain part of the strict mining identity.

The default candidate pool is 256 lexical/random proposals per query. Exact
MaxSim selects 20 cohorts per patient and 40 patients per cohort within that pool;
five TF-IDF and five random proposals from
the full pooled corpus are also included. Known source-trial relationships and optional
`--trialspace-extra-pairs` add independent/challenge candidates without assuming
positive labels. This is bounded candidate mining, not exhaustive corpus search.
`--trialspace-candidate-limit`, `--trialspace-top-k`, `--trialspace-lexical-k`,
`--trialspace-random-k`, and batching controls expose those choices. Both retrieval
directions are scored independently; directional MaxSim is never transposed.

Every 100 optimizer updates, fixed training groups measure normalized
MSE, MAE/RMSE in 0–5 units, unequal-label ordering accuracy, and tie-aware NDCG.
Zero and training constant-mean baselines expose apparent gains caused solely
by score calibration or zero-heavy labels. These are fixed candidate-pool
training diagnostics, not held-out validation or full-corpus recall estimates.
An explicitly excluded validation partition enables separate held-out probes.

New defaults are `../data/no_phi/colbert_pipeline_v1` and
`../models/colbert_pipeline_v1`. Final models are `trialspace_round1` and
`trialspace_round2` under the models directory, including
`regression_scoring.json` and `experiment_encoding.json`. Load them with the
native ColBERT implementation; pooled SentenceTransformer embeddings and the
legacy remote TrialSpace adapter are incompatible. The older pooled-vector
scripts remain for historical reproduction and are not invoked by production.
Runtime inference presets and published models are not switched by this training
change. Regenerate every token index when adopting a new checkpoint.

Use fresh run/model directories for this architecture. Old pooled-vector stage
markers, label schemas, checkpoints, and upgrade receipts cannot migrate a run
into ColBERT. `--skip-input-validation` only reuses already recorded source hashes
for an otherwise identical run.

## Reasoning archives

All production LLM stages preserve returned reasoning for later distillation:

| Stage | Durable archive |
| --- | --- |
| Trial cohort extraction | Per-trial checkpoints with attempts, answers, reasoning and source metadata; pooled raw-response sidecars |
| Synthetic seeds, events and notes | Original generation checkpoints/shards with reasoning and raw responses |
| Serial patient summarization | Per-chunk summary shards with reasoning, prior summary and source chunk linkage |
| ColBERT labels | `trialspace_round*/teacher/responses/<prefix>/*.json` plus `labels.jsonl`, containing exact messages, rendered prompt, sampling, raw answer/response, reasoning and endpoint identity |
| Catalog and other pooled teacher calls | `llm_traces/<stage>/<prefix>/*.json`, written before validation, including incomplete/malformed attempts |
| Exclusion extraction and OncoReasoning | `teacher_attempts/` raw sidecars plus validated records retaining returned reasoning and retry attempts |

`reasoning_archives.json` records the downstream locations. Upstream pooled
sidecars live under `PIPELINE_DIR/llm_traces`. Raw generation and summarization
artifacts remain intact when prepared tables select only final text. Archives
contain input text and stay in the authorized non-PHI run directories. Headers
and credentials are excluded. A model returning no separate reasoning leaves an
empty field and an intact raw response; no reasoning is invented.

Reasoning retention does **not** change the existing answer-only OncoReasoning
supervision: only the validated final summary or answer/explanation becomes a
training target. Parent summary chains also consume only prior final summaries.
Old distillation responses that discarded reasoning require a fresh generation
directory; they are not silently presented as complete trace archives. See the
[OncoReasoning instructions](oncoreasoning_training/README.md).

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

## Extract trial cohorts with the inference API (0b)

Run `0a_parse_ctgov_json.py` to create the parsed CT.gov CSV, then:

```bash
python 0b_extract_trial_cohorts.py \
  --input ../data/no_phi/ctgov_trials.csv \
  --server_urls http://sn4622130540:8000/v1 \
  --model gemma4-31b --tokenizer-model nvidia/Gemma-4-31B-IT-NVFP4 \
  --max_concurrent_per_server 768
```

The default selection includes trials first posted **in 2020 or later** with
`study_type=INTERVENTIONAL` and early phase I through phase IV, including combined
phases. Missing/N/A phases and invalid/missing first-posted dates are excluded.
Recruitment status is not filtered. Source registry metadata and dates retain
their original precision. Existing train/val/test splits are preserved; otherwise
splits are assigned once per NCT ID with seed 42 and 80/10/10 probabilities.

The script uses the current `trial.cohorts.user.txt` owned by inference-kehl.
Arms and cohorts count only when their eligibility populations differ; treatment
assignment alone does not create a cohort. The API handles strict JSON validation,
outer Markdown fences, retries, single-population source retention, and joining
cohort-specific and common criteria. There is no separate screen or cached-screen
input. Defaults apply the inference package's Gemma 4 sampling profile and a
50,000-token output budget. Use `--tokenizer-model` to identify the actual model
family when `--model` is an endpoint alias.

Outputs under `data/no_phi` are:

- `trial_cohorts.json`: NCT ID → combined decision object, including rejections.
  A rejection contains only `cancer_treatment_trial: "no"`; an accepted single
  population adds `multiple_cohorts: "no"`; multiple populations also have a
  `cohorts` list of specific criteria and `common_eligibility_criteria` string.
- `trials_with_cohorts.csv`: one audit row per selected trial, including its
  source fields, split, combined JSON, decision, cohort count, and error.
- `trial_cohort_lineitems.csv`: one row per accepted population, retaining
  `cohort_trial_id`, `trial_id`/`nct_id`, zero-based `cohort_number`, `cohort_count`,
  `cohort_index`, full `cohort_eligibility_criteria`, separate specific/common
  criteria, and source metadata. Single-population criteria retain the original
  eligibility field verbatim, without the title or brief summary. Rejected trials have no cohort rows.

`--work-dir` stores per-trial compressed checkpoints with raw answers, reasoning,
validation attempts and response endpoint provenance. Resume validates input,
code, prompt, schema and configuration identities, then skips successful decisions
and retries failed trials. Failures produce `.partial` exports and a nonzero exit,
preventing the orchestrator from sampling an incomplete run. Use a fresh work/output
directory for a new input, model, prompt or settings; old space/screening checkpoints
are not reusable.

Extraction uses inference-kehl's shared adaptive endpoint pool, with up to 1,536
in-flight API calls (`--workers`). It ramps concurrency and backs off using request
outcomes and vLLM metrics. Supply `--server_urls_file` for a live endpoint registry;
per-entry `upstream_concurrency` sets each server's ceiling, otherwise
`--max_concurrent_per_server` applies. Registry additions/removals take effect
while running. The external supervisor owns endpoint verification and model
identity; use registry entries with `model` and `model_root` for provenance.
Extraction does not consume the separate training transport's concurrency-control
file. Use registry caps for this stage. Neither serving nor training is launched.

`trial_cohort_format.py` explicitly adapts cohort rows for existing training
internals (`cohort_index` → `space_index`, complete criteria → `this_space`).
It preserves native IDs and marks `trial_representation=trial-treatment-cohorts-v3`.
Common criteria may include inclusions as well as exclusions and are **not** copied
to `trial_boilerplate_text`; that column is empty because this API does not extract
separate boilerplate. Synthetic seeds use the complete cohort criteria. Existing
space-generation artifacts, trained models and saved embeddings are not thereby
converted or validated for this representation. Training-only execution now
requires this native cohort format and the reviewed inference commit pin.

## Legacy trial-space extraction reference

The following historical selection, screening and hierarchy instructions apply to
`legacy_trial_space_extraction.py`, formerly step 0b. They do not describe the active
cohort extractor above; references to “0b” in this legacy section mean that retired
space workflow. Its source and prompt contracts remain available for historical
matching/evaluation consumers.

### Nemotron 3.5 Super VL teacher and legacy trial-space extraction


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
PYTHONPATH=../matchminer-ai-inference-kehl/src ~/thisenv/bin/python legacy_trial_space_extraction.py \
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
to retain the original eligibility field plus ClinicalTrials.gov `designModule.studyType`
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

The legacy extraction primer and closing instructions in `legacy_trial_space_extraction.py`
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

The separate authored screening prompt has been removed.
`trial_treatment_screening.py` is now a compatibility adapter that imports the
inference package's combined `trial.cohorts.user.txt` prompt and JSON validator.
It classifies the studied intervention's intent using the supplied `trial_text`
and, for accepted multi-population trials, extracts cohort-specific criteria and
one shared-criteria string.
The legacy 0b space workflow below still selects positive trials for its explicitly
configured space extraction. New cohort consumers should call the public
`matchminer_ai.trials.summarize_trials` API directly. The installed inference
package must include the `trial-treatment-cohorts-v2` contract. It includes direct
anti-cancer therapy (including early-phase safety/dose studies and adjuvant or
maintenance treatment) and excludes diagnostic-only and supportive-care studies.
Anti-cancer treatment studies qualify even when their only outcomes are PK/PD,
safety, tolerability, target engagement, or other surrogate measures. This includes
drug-exposure/food-effect studies and preoperative treatment windows measuring
tumor biomarkers; clinical efficacy endpoints are not required. Biomarker-only
sampling without evaluation of an anti-cancer treatment remains excluded.
Cancer patients receiving background anti-cancer therapy are not sufficient:
the study must evaluate an anti-cancer treatment strategy. Insufficient textual
evidence produces `{"cancer_treatment_trial": "no"}` without further fields.
Accepted responses add `multiple_cohorts`, and add `cohorts` followed by
`common_eligibility_criteria` only when it is `yes`. Each cohort string together
with the common string contains all applicable eligibility criteria. The public
API combines them in code for complete cohort rows.
The shared validator accepts one complete outer Markdown fence labeled `json`
(case-insensitive) or unlabeled. Enclosed JSON still requires the same schema and
a complete generation; audit columns preserve the original fenced response.

Only positive decisions proceed to space extraction. `trial_treatment_screening.csv`
is saved beside `--output-trials` by default, with all screened trials, boolean
decisions, code-generated decision descriptions, parsed reasoning, final responses, finish reasons, status,
and `treatment_screen_model` (the requested `--model` ID or served alias). The
model is also recorded in new local and remote screening shards. Override the
CSV location with `--output-screening`. This audit covers trials
that passed the metadata filter, not metadata-excluded rows. The two existing
final CSV schemas are unchanged; if every trial is rejected, both contain only
headers. `trial_cohort_output` retains the combined JSON. Historical Gemma audits with
boolean decisions and model rationales, and v1 combined outputs with complete
criteria repeated per cohort, remain readable; live requests require
the new combined schema. `--screening-max-tokens` defaults to 50000, independently of extraction's
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

0b defaults to `--extraction-method hierarchical`. After the combined-prompt
treatment screen, this legacy space path branches through age, cancer primary site, sex, histology,
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

After `0b` has completed, `0c_sample_trial_spaces.py` samples 5,000 distinct NCT IDs
(seed 42) and retains all their cohorts. Its DFCI holdout excludes trials open
during 2025–2026 using `dfci_trials_summary_10-18-26.csv`. A trial with a
missing or invalid cohort is excluded as a whole so sampling never silently drops
one of its populations. Missing DFCI boundary dates are treated as unbounded;
`OPEN TO ACCRUAL` status also triggers the holdout. Input/output paths, sample size,
seed, and holdout years are CLI options. The sampling receipt records exact NCT IDs
and cohort counts. Historical seven-field space files remain supported explicitly.

For cohort inputs, step 0c first keeps only `cancer_treatment_trial="yes"`, then
applies the DFCI holdout and samples distinct trial IDs. The output CSV has one
row per cohort per selected trial. Its `cohort_eligibility_criteria` column joins
the specific and common eligibility strings; for a single cohort it retains the
original `eligibility_criteria` field (without the title or summary). Separate specific/common fields and source metadata also
remain available. The sampler rebuilds the combined field from these sources
when present, rather than trusting a stale combined column.

To sample the completed production JSONL export without making any new LLM calls:

```bash
python 0c_sample_trial_spaces.py \
  --extraction-dir ../data/no_phi/ctgov_treatment_cohorts_since2020_gemma4_31b_20261023 \
  --output-csv ../data/no_phi/sample_trial_cohort_lineitems.csv
```

For production exports made before the single-cohort correction, first run:

```bash
python repair_single_cohort_criteria.py \
  --extraction-dir ../data/no_phi/ctgov_treatment_cohorts_since2020_gemma4_31b_20261023
```

The repair replaces only derived single-cohort text and its source marker, backs
up the original JSONL, and writes `cohort_rows_repair.json` with before/after
hashes. Raw model decisions, reasoning, checkpoints, and the frozen extraction
code remain the historical audit. Regenerate the sample after repair, retaining
the same seed and holdout settings. Older v2 sample files must be refreshed before
synthetic generation; saved seed/chronology checkpoints from that representation
must not be reused with v3 inputs.

This mode checks the saved combined decisions, all cohort rows, and the original
`inputs.parquet` against each other before sampling. It preserves existing source
splits or assigns them once per trial using seed 42. The sampling receipt includes
rejected trial IDs, DFCI exclusions, the extraction run identity, and source hashes.

`generate_synthetic_data.py` creates five matching and five near-miss patients
per sampled cohort in two stages. For each class, one call creates five diverse
short prose seeds containing names, sex, diagnoses, ages at each diagnosis, and
concise patient facts needed to establish the final clinical state against the
criteria. Positive and negative seeds use **two separate LLM calls per cohort**,
with five seeds in each response. Each positive seed must meet all supplied
criteria; each negative seed must contain clinical facts violating at least one
requirement. Multi-cohort inputs join specific and common criteria; single-cohort
inputs use only the original eligibility field. The full trial title and summary
are not supplied as eligibility. Seed checkpoints retain native cohort IDs.
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
histories; chronology checks compare both parts and record reversals as warnings
without rejecting, sorting, or changing the event list. Downstream synthetic
dates preserve the week intervals. The warning
blocks in both prompt modules also prohibit imaging findings, including claims
of absence, for organs outside the specified scan's field of view.

Seed and patient calls share one continuous adaptive endpoint pool. Patient
expansion starts as soon as its own seeds are saved, without waiting for all
seed calls to finish. Before generation, the runner raises its soft open-file
limit as needed to accommodate `--pending-requests`, idle connections, and
checkpoint writes. If the existing hard limit is too low, it exits before
loading the tokenizer or creating output and reports the required limit.
The production generator enables `--scheduler-adaptive-concurrency` by default.
The shared pool uses `adaptive_controller.py` directly, polling each server's
vLLM scheduler metrics every `--server_urls_refresh` seconds. It starts at up to
128 requests per server (`--initial-concurrent-per-server`), grows toward the
configured ceiling, and lowers that limit on queue/cache pressure, preemptions,
or measured throughput regression. Request errors independently reduce active
concurrency; output-validation retries do not. Missing metrics retain the current
limit and error-based adaptation. `concurrency_status.json` records each server's
metrics, effective limit, in-flight requests, and error counts. Live concurrency
files supply upper bounds. Use `--no-scheduler-adaptive-concurrency` when an
external supervisor owns scheduler tuning. The pipeline exposes these settings
as `--generation-scheduler-adaptive-concurrency` (and its `--no-` form) and
`--generation-initial-concurrency-per-server`. Throughput comparisons require uninterrupted, sufficiently loaded
windows; idle workers, missing metrics, and server restarts discard those
comparisons. Cache pressure and request preemptions still trigger backoff.

For the explicitly requested eight-VM RTX Spot fleet, `scripts/gcp_note_fleet.py`
attaches 64 per-GPU Gemma workers to an already-running note job. Its frozen JSON
configuration records project/zone/name, SSH identity and forwarding ports,
the generator PID plus process start identity, the live server registry, pinned
model/image revisions, and the active generation tokenizer-template hash.
`scripts/gcp_note_worker.py` installs the VM-local workers and checks a reasoning
completion before admitting each GPU. API ports bind to loopback and reach the
generator through SSH tunnels. Existing registry entries are preserved.
The controller and VM SSH services reserve 65,536 file descriptors so that
eight busy GPU endpoints do not exhaust a tunnel's connection capacity.

The fleet supervisor retries unavailable/preempted Spot instances, reconnects
changed IP addresses, and removes unhealthy workers from new dispatch. Docker
restarts crashed workers. A stopped or failed generator, completion of note
inference, or an operator `STOP` file triggers removal of this fleet's endpoints
and repeated stop requests until all eight VMs are terminated. Each VM also has
an expiring 15-minute controller lease and powers itself off if renewal stops.
After a reboot, a 15-minute grace period lets startup and SSH renew an expired
lease before the watchdog enforces it again.
Local output assembly can finish after cloud workers shut down. Deployment logs,
original metadata backups, health state, and shutdown receipts stay alongside
the fleet configuration. Run this only for a user-authorized fleet:

```bash
python scripts/gcp_note_fleet.py --config /absolute/path/to/fleet/config.json
```

The same supervisor supports a fresh labeling fleet: set `workflow_mode` to
`labeling`, attach the labeling workflow's PID/start identity and status path,
and point `registry` to its live label-server file. Use a new fleet ID and run
directory, preserving the explicit VM allowlist and pinned worker configuration.
It warms the per-GPU teachers while retrieval finishes, admits healthy endpoints
as they become available, and retries Spot failures throughout labeling. Completed
notes do not stop this fleet; completed or failed labeling workflows do. Endpoint
removal, repeated VM stop requests, and the expiring remote lease still apply.

For a handoff that retains the fleet for patient summarization, attach the
explicit `workflow` record (status path, PID, and process start ticks) to the
fleet config and launch `scripts/post_notes_workflow.py --config <workflow.json>`.
This controller waits for complete notes, runs note preparation and serial
summarization through `run_pipeline.py`, waits for verified termination of all
fleet VMs, then releases only the registered Camus GPU 2/3 serving processes.
It runs corpus preparation and distributed mining, restores the two serving
commands, and labels the mined pairs. It stops after labeling; it does not fit
a model. Commands, output paths, local process identities, restart environments,
and the separate live labeling registry are recorded in its config. GPU 0/1
remain available for their existing work. The fleet watches both the generator
and workflow; either process failing triggers shutdown, with VM leases as a
fallback. Keep both supervisors running until the cloud shutdown receipt exists.

Remote patient summarization uses the shared adaptive scheduler. It checkpoints
every completed patient chunk, preserves the rendered prompt, raw response and
reasoning for every returned attempt, and rejects empty, failed or token-limited
answers before they can become a later chunk's prior summary. An interrupted
round resumes its saved chunks using prompt/sampling identity checks. The
upstream generation concurrency controls also apply to summarization.

Patient-level output has one `patient_summary` text column, containing all
cancer-history sections followed by `Other relevant history:` and, on the next
line, the former boilerplate evidence. Step 6 renders the inference package's
patient prompt directly. The training adapter can also combine historical split
inputs, removes their `patient_boilerplate_text` column, and never duplicates an
already combined history. ColBERT uses the complete unified text. Serial chunk
artifacts retain raw answers and reasoning separately for distillation.

Mining accepts separate `--top-k-patient-to-cohort` and
`--top-k-cohort-to-patient` counts (the training orchestrator prefixes these with
`--trialspace-`). For example, 20 and 40 select that many ColBERT results in the
respective directions, with additional lexical/random/source-trial candidates
retained separately. These rankings are over the configured proposal pool;
the completion manifest records that scope.

Each expansion receives only its assigned prose seed and the clinical/formatting
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
one patient per expansion, diagnosis events at explicitly stated diagnosis ages,
other clinical events at other dated seed facts, valid age formats, report
events, and common target-trial references. Current status, progression scans,
treatment, and procedure ages do not require invented diagnosis tags.
Chronology reversals are saved in `validation_warnings` alongside the validation
revision, including in the exported patient provenance. The validator does not
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
use a fresh output directory after changing them. The v5 cohort-seed event manifest
rejects reuse of previous trial-conditioned event, tagged-seed, or multi-patient
checkpoints.

On resume, failed patients first try their saved rejected LLM attempts against
the current validator. Recovery prefers attempts without chronology warnings,
preserves the original response, endpoint, timestamp, and prompt hash, and writes
an `event_recoveries/` receipt. If none pass, the ordinary LLM retry path runs.
For the v5 validation-only correction, stop the generation coordinator and
worker gracefully, then run
`python scripts/register_event_validation_upgrade.py --run-dir <generation-dir>`
before resuming the original pipeline command. Registration requires both writer
locks, checks that inputs and every prompt/helper except the generator remain
unchanged, and backs up the old pipeline request and failures. The original
generation manifest and patient-ID namespace remain unchanged. This exception
does not authorize resuming with different prompts, sampling, or input data.

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
  --input-csv /path/to/completed/trial_cohort_lineitems.csv \
  --output-csv /path/to/new-run/sample_trial_cohort_lineitems.csv
python generate_synthetic_data.py \
  --input-csv /path/to/new-run/sample_trial_cohort_lineitems.csv \
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
| `extract-cohorts` | `0b_extract_trial_cohorts.py`: year/phase filtering and public `summarize_trials` API; `trial_cohort_lineitems.csv`, trial audit, combined JSON |
| `sample-trials` | `0c`: 5,000 complete trials, all their cohorts, DFCI 2025–2026 holdout and sampling receipt |
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
`--sample-trials` defaults to 5000 and `--data-seed` to 42. Starting at
`sample-trials` accepts an existing complete `--trial-cohorts` file (`--trial-spaces`
is retained as an alias). The default upstream file is `trial_cohort_lineitems.csv`;
its sample is `synthetic_generation/sample_trial_cohort_lineitems.csv`.
`--sampled-cohorts`/`--sampled-spaces` overrides the sample path. Cohort extraction
has separate `--cohort-max-attempts` (3), `--cohort-max-tokens` (50000), and
`--cohort-workers` (1536) controls; generation retry/context limits do not replace
these. Screening and cohort extraction always use the same combined prompt.

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

### ColBERT experiments and shared implementation

[`colbert_experiments`](colbert_experiments/README.md) explores native token-level
late interaction with `lightonai/GTE-ModernColBERT-v1`, complete patient/cohort
text, and bidirectional retrieval. It supports both five independent binary
tasks and a separate gated 0–5 regression experiment: ask potential eligibility
first, ask four specificity questions separately only after yes, then count yes
answers in Python and train MaxSim against `score / 5` with MSE. The production
entrypoint now uses this gated regression implementation; the original binary
experiment remains available for controlled comparisons.
