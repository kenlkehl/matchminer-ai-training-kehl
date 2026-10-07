# matchminer-ai-training

Code for training the MatchMiner-AI pipeline. A full reproduction is intended
for a Linux system with eight H100-class GPUs and takes roughly a week.

```bash
uv sync --group training
bash train_all.sh
```

Training uses vLLM heavily. Restart an interrupted workflow from the last
completed orchestration step rather than rerunning completed stages.

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

The orchestrators place new models and mined/relabeled data under
`../models/trialspace_embeddinggemma2/` and
`../data/no_phi/trialspace_embeddinggemma2/`; the initial checkpoint directory
is `~/models/initial_embeddinggemma2_training`. This avoids reusing Qwen models,
training checkpoints, or candidate-label shards. The final TrialSpace model is
`../models/trialspace_embeddinggemma2/reranker_round2.model`.
An existing run can restart at step 8 using its initial eligibility labels.
Regenerate both patient and trial embeddings for the new model: older vectors
and indexes are incompatible. External inference consumers must likewise use
the saved prefix and the matching regenerated index before adopting this model.

## OncoReasoning distillation

The optional [OncoReasoning workflow](oncoreasoning_training/README.md) defaults
to the text model from `google/gemma-4-E4B-it`. It distills serial patient
summarization with variable 10K–50K-token chunks and individual TrialChecker /
GoodOption component questions. Student targets contain final answers only;
binary QA outputs begin with a single answer letter followed by an explanation,
supporting either one-token or explanatory inference. The documented stages
prepare requests, generate resumable teacher outputs, build masked datasets,
and fine-tune a configurable student.

## GoodOption evidence catalog

GoodOption is an evidence-counting signal for whether a trial's experimental
drug options have support relevant to a synthetic patient's cancer. It is not
an eligibility result, response probability, or treatment recommendation.

The trained four-logit GoodOptionChecker is deprecated. It read one patient and
one drug summary and never saw drug-class evidence, and no trained model was
published. GoodOption scoring now uses the inference package's LLM rubric
(`score_good_options_with_llm`), whose prompt packs drug and class evidence to
the teacher's context. Step 17 of `train_all.sh` and `train_all_gcp.sh`
therefore builds and validates the patient-free catalog only. The `label`,
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
