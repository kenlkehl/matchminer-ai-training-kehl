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
`6e38839e46a61d2583dc6c72dd6c672a19cd6a94`. Review the pin and prompt contracts
before adopting changes in that dependency. The default teacher for every
label, catalog stage, exclusion extraction, and distillation request is
`nvidia/Gemma-4-31B-IT-NVFP4`. Eight text-only vLLM workers serve it on localhost;
they are stopped before student training. A separate vLLM executable can be
selected with `--vllm` when serving dependencies need their own environment.
TrialSpace labeling allows up to 96 concurrent requests per teacher GPU
(768 total), and each vLLM worker permits 96 active sequences. The labeler's
adaptive limiter ramps up after successful requests and backs off on errors.
Catalog and long-context distillation retain their separate global concurrency.

The run starts from `../data/no_phi/patient_summaries_with_spaces.parquet` and
`trial_space_lineitems.csv`, mines candidates with the base embedding model,
labels them, and trains TrialSpace. A second mining/label/training round uses
cumulative labels; a third mining round supplies OncoReasoning candidates.
For each patient, mining samples 500 **distinct NCT IDs**, expands all their
spaces, and keeps 20 spaces. For each trial space it samples up to 20,000
patients and keeps 40. Sampling is within original train/validation co-splits;
test rows are excluded and TrialSpace trains only on the train split.

The pipeline then builds and validates fresh drug and class evidence, extracts
individual exclusions, and distills summarization from `all_synthetic_notes.parquet`
plus all three QA families. Catalog web research receives public trial/drug
information only. All patient-bearing teacher requests stay on the VM.
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
