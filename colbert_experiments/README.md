# Cohort ColBERT experiment

This package contains the shared native `lightonai/GTE-ModernColBERT-v1`
implementation and the original five-question experiments in **both retrieval
directions**. The production entrypoint `../trialspace_training.py` and
`../train_from_summaries.py` use its gated score labels and ordinal-ranking trainer
(CoSENT plus known-negative contrast), with distributed disk-backed mining. The experimental CLI retains
its original binary-task default for controlled comparisons. Use synthetic/non-PHI
inputs unless the selected endpoint is authorized for the data.

## Full production labels on one GB300 (2026-10-26)

Run on the GB300 host from the mounted training checkout, using its training
Python environment and an available GPU:

```bash
python colbert_experiments/train_full_gb300.py --gpu 0
```

An explicit `--gpu 1` always selects physical CUDA GPU 1 and overrides inherited
`CUDA_VISIBLE_DEVICES`. To use an existing visibility mask, omit `--gpu` or use
`--visible-gpu` to index that mask: `CUDA_VISIBLE_DEVICES=1 python
colbert_experiments/train_full_gb300.py --visible-gpu 0` selects physical GPU 1.
With neither option nor a visibility mask, the default is physical GPU 0.
The printed `CUDA_VISIBLE_DEVICES=...` line describes the training subprocess;
the launcher does not retain GPU settings between invocations.

This one-off launcher waits for validated production training scores, then uses
`trialspace_training.py train` on all 2,435,540 repaired patient/cohort pairs,
including all original source splits, in both directions. It starts from the
base ModernColBERT model and uses the production CoSENT plus known-negative
contrast objective for one complete epoch (5,162 optimizer updates), with a linear learning-rate
schedule, learning rate `2e-5` and 1% warmup. The default microbatch has one query
and up to 16 documents; 64 accumulated groups give up to 1,024 directional pairs
per optimizer update. Ranking comparisons span the effective batch. Full inputs
retain the 8,192-token limits and truncation checks.

The encoder combines up to **four query groups** per forward (up to 68 texts),
grouping similar lengths to reduce padding, within a **65,536 padded-token**
packing budget. Longer groups are packed separately without truncation. All 64
query groups still contribute to each update and the full-batch ranking loss.
This cuts encoder calls per pass from 64 to as few as 16. Adjust
`--query-groups-per-forward` and `--encoder-token-budget` for available VRAM.
An eight-million-token CPU LRU cache (`--token-cache-tokens`) avoids repeated
tokenization; both passes reuse the same prepared inputs. It retains no model
outputs or gradients. Token masks are compacted with CPU-derived integer indices
to avoid per-text CUDA boolean-indexing synchronization. Differentiable MaxSim
scores documents in batches. The GB300 launcher disables activation
checkpointing by default to avoid encoder recomputation during backward; add
`--gradient-checkpointing` if memory is insufficient. This trades VRAM for speed
without shortening inputs or reducing pair coverage. Logs include total updates,
per-step time, preparation/cache/loss/backward timings, encoder batches per pass,
token-cache occupancy/hits, and peak allocated GPU memory.

Output defaults to `../data/no_phi/colbert_production_gb300_batched_20261026`, with
checkpoints in `fit/` and the final model in `model/`. Checkpoints save every 25
updates, keeping at most two. Repeat the command to resume; create `OUTPUT/PAUSE`
to pause cleanly and remove it before resuming. This run has no Camus time cutoff.
The new directory starts the one-epoch schedule from scratch and preserves the
earlier runs. Changing epochs or execution settings requires a fresh
output directory.
The launcher refuses to train on Camus and makes no LLM calls. It checks the
complete-score publication receipt and SHA-256 content of the training files,
then reads finalized numeric scores through the production trainer without
inspecting individual reasoning checkpoints. The large distillation archive can
finish separately: a later export/supervisor failure does not invalidate a
complete, independently verified training-score publication. Missing or changed
scores, corpus files, candidate files, or teacher identity still prevent training.
Timestamp differences across host mounts do not invalidate identical content.
Verification streams the four data files sequentially; it does not enumerate
the millions of saved reasoning checkpoints.
Use `--cache-dir` for a local model cache, `--source-run`/`--output` to adjust mount
locations, and `--dry-run` to print the command without starting anything.
End-to-end throughput must be measured on the GB300; microbatch and
accumulation settings can be overridden before the first fit.

For a bounded forward/backward timing comparison without optimizer steps or
saving model weights, run `benchmark_grouped_training.py --probes
PATH/fit/fixed_probes.json --device cuda:0`. It compares one versus four query
groups using the same small synthetic probe subset and reports GPU memory and
cached-score replay errors. A local H100 check with 64 pairs measured 6.82 versus
3.05 seconds (13.0 versus 29.3 GiB allocated); this is not a GB300 production ETA.

## One-off smoke test on two A6000 GPUs while production labels continue

Run this on the machine with the two A6000s, from this training checkout, using
its Python environment with the training dependencies installed:

```bash
python colbert_experiments/smoke_train_completed.py \
  --output ../data/no_phi/colbert_completed_smoke_a6000_all \
  --gpus 0,1
```

The default source is the current
`synthetic_cohorts_5000_gemma4_31b_20261024/post_notes_colbert_20261025` run under
the mounted workspace's `data/no_phi`. Override it with `--source-run PATH`.
The script uses this machine's Python and GPU indices, and Hugging Face's normal
cache location (`HF_HUB_CACHE` or `--cache-dir` can select a local disk cache).
It imports the sibling inference checkout when available. It does not contact
LLM servers or alter the production labeling/training workflow.
The script itself starts `torchrun` with one DDP worker per selected GPU; invoke
it with Python, without an additional distributed launcher. `--gpus 0,1`
therefore already trains on both devices.

Preparation includes **all completed train-patient pairs** and samples up to
**2,000 completed validation-patient pairs**. It reads a compact completed-score
archive sequentially, or the labeler's final `scores.jsonl` when available.
The training host does **not** need to enumerate/open individual question files
or scan the candidate table. A negative eligibility gate is complete;
a positive gate requires all four specificity answers. Missing answers never
become zero scores. Compact-score contracts, corpus hashes, gated answers and
the archive checksum are validated during the sequential read. Source reasoning
traces stay in place and remain linked by question keys and the source archive.
The default `--train-pairs 0` includes every original-training-patient pair in
the archive's fixed cutoff snapshot.
Training defaults to three complete epochs over that snapshot, with no fixed
update cap. An explicit positive `--train-pairs N` can still request a smaller experiment. A bounded,
seeded reservoir is used only for validation and explicitly capped training samples.
Counts, score distributions, the archive cutoff and any shortfall against the
requested sample sizes are printed and recorded. `--prepare-only`
creates the snapshot without loading a model or using GPUs.

While labeling is still running and has not exported final scores, pack its
already-published responses **once on the producing host** (Camus for this run):

```bash
python colbert_experiments/export_completed_scores.py --workers 8
```

The exporter validates question identities, full-text prompt hashes, corpus
identity and completion status, then atomically publishes a compact JSONL plus
manifest under `SOURCE_RUN/labels/completed_scores`. It uses separate CPU
processes and leaves the active labeler and source reasoning untouched. This
one-time producer-side scan avoids repeating millions of small-file reads over
the remote NFS mount. Rerun the exporter when a newer cutoff is needed, then
choose a fresh smoke-test output directory to include those additional labels.
Use `--source-run PATH` on either script to select a different run, or
`--score-archive PATH` on the training script to select a particular compact
archive directory. The new default fails promptly if no compact/final scores
exist; it does not silently start an hours-long inspection. Explicit
`--inspect-checkpoints` retains the old individual-response reconstruction for
diagnostics; `--candidate-multiplier` affects only that fallback's capped samples.

A smoke process already doing the old inspection must be stopped and restarted
to use this path. Existing completed snapshots are reused without any question
inspection; unfinished preparations can restart in the same output directory.

Only patients whose **original `source_split` is `train`** contribute optimizer
updates. Original validation patients supply validation metrics; test patients
are excluded. Production's pooled effective `split=train` is not used as a
holdout assignment. Both patient-to-cohort and cohort-to-patient directions are
trained/evaluated by default (`--direction` can restrict this). Cohorts can appear
in both patient partitions: this is a patient holdout, not a cohort holdout.

The student uses the production base model, complete 8,192-token inputs without
silent truncation, gradient checkpointing, BF16 and two-rank DDP. By default it retains
`MSE(mean_MaxSim, score / 5)`. Complete-pass batching visits every directional
pair exactly once per epoch, mixes scores within each query, and spreads
direction/score strata throughout the epoch without replacement sampling. Each
forward pass encodes one query with up to **two candidates** together; 32
accumulated query groups per GPU give up to **128 labeled pairs per update**.
Validation also batches its encodings. These conservative defaults are intended
for 48 GB A6000s; a particular hardware/software stack is not pre-benchmarked.
For less memory, reduce `--documents-per-query` or `--eval-batch-size` rather than
truncating the clinical text.

Defaults run **3 complete epochs**, matching production's epoch setting. Every
completed training pair contributes once per direction in each epoch. The actual
update count is computed from the complete query chunks, two GPU ranks, and
gradient accumulation, then printed before fitting. The final partial update is
included; DDP padding has zero loss weight and does not duplicate training pairs.
`--epochs` changes the number of full passes; `--steps` is available only as an
explicit update-count override. Batch train loss prints every update.
Before training, every **5 updates**, and at each epoch end, the console prints fixed-sample train
loss, validation loss, and validation AUROC separately for each direction:

```text
[train 00005/TOTAL_UPDATES epoch=1] train_batch_loss=... pairs=...
[eval 00005] train_loss=... val_loss=... val_auroc_patient_to_cohort=... val_auroc_cohort_to_patient=...
```

AUROC uses **raw continuous mean-MaxSim** as its predictor and **`score > 0` as
the positive class**, with `score == 0` negative. Scores are neither thresholded
nor passed through a sigmoid before AUROC. Fixed evaluation samples contain 256
unique pairs per partition, selected independently of student predictions with
approximately prevalence-preserving binary stratification; both classes must be
present. For the default MSE objective, training and evaluation losses both weight actual labeled pairs equally
within their batches; zero-weight DDP padding contributes neither loss nor sample
count. These are diagnostics on the currently
completed, retrieved pairs, not estimates of full-corpus matching performance.
Use `--eval-every`, `--eval-pairs`, `--epochs`, `--train-pairs` and `--val-pairs` to
change the smoke-test size. JSON metrics and individual validation predictions
are saved under `OUTPUT/fit`.

The snapshot is reused unchanged when the same command is repeated. Model and
optimizer checkpoints are saved every 10 updates, keeping at most two, and the
final model goes to `OUTPUT/model`. To pause with a checkpoint, create
`OUTPUT/PAUSE`; remove it and rerun the same command to resume. `--stop-at` also
accepts an ISO timestamp with an explicit timezone offset. Choose a new output
directory for a fresh label snapshot or different training settings, including
switching an already-started replacement-sampled fit to complete epochs. In particular,
use a new output directory if an earlier run already froze a 12,000-pair snapshot;
the script will not silently resume that capped snapshot as an all-pairs run.

### Reusing the previous TrialSpace ranking recipe

The sequence-embedding trainer in `../finetune_embedder.py` uses **CoSENT on
numeric labels plus MultipleNegativesRankingLoss on positive pairs**. Those
numeric labels specify relative order, rather than a target similarity for
pointwise regression. The smoke script can adapt this to mean-MaxSim:

```bash
python colbert_experiments/smoke_train_completed.py \
  --output ../data/no_phi/colbert_completed_ranking_a6000 \
  --gpus 0,1 --loss legacy_ranking \
  --learning-rate 2e-5 --lr-scheduler linear --warmup-ratio .01
```

This starts a separate experiment from the base ColBERT model. It uses all
completed original-train pairs for three full epochs, retains the patient
holdout and both retrieval directions, and makes no new LLM calls. The existing
running MSE experiment remains unchanged. The production entrypoints now default
to this ranking recipe as well. To compare
exactly the same label cutoff and validation pairs, copy **only** the existing
run's `snapshot/` directory into the new output directory before launching;
do not copy `fit/`, since objective/settings changes must not resume an old fit.

`--loss cosent` learns that a pair labeled 5 should score above a pair labeled 0
or 2, without prescribing their absolute MaxSim values. It ignores tied labels
and compares only pairs in the same retrieval direction. Any strictly increasing
label transformation preserves this loss: no sigmoid or label normalization is
needed. `--ranking-scale 20` is its inverse temperature.

`--loss legacy_ranking` adds a within-query softmax loss: each positive (`score
> 0`) competes with that query's **explicitly labeled zeros** in the effective
batch. This is a deliberate adaptation of MNRL: unknown cross-query combinations
do not become negatives, and multiple positives do not repel one another.
`--contrastive-weight 1` controls the added term. CoSENT averages over directions
with unequal labels; the contrastive component averages over queries containing
both positive and zero labels. Zero-only groups still participate in the
cross-query CoSENT comparisons. Batches with no applicable comparison have zero
ranking gradient; JSON training logs include comparison/query counts.

These losses compare across the **entire effective batch**, up to 128 directional
pairs at the default two-GPU settings. Ordinary gradient accumulation alone
would not achieve this. A no-grad pass caches scalar MaxSim scores across ranks,
then a second pass replays the same seeded microbatches and backpropagates the
full-batch loss gradients. This adds a forward pass while keeping activation
memory bounded by one query/candidate microbatch. Replay consistency is checked;
partial batches still have exact coverage and zero-weight padding.
Collective pause polls occur every eight microbatches and at phase boundaries,
rather than synchronizing workers before every short/long-text pair of batches.
The small scalar-score loss runs on CPU, with its gradients copied back to the
encoder GPUs, avoiding many tiny GPU operations. The objective, batch membership,
random seeds and checkpoint identity are unchanged by these execution optimizations.
Workers can still wait at phase/gradient synchronization, evaluation completion,
and rank-zero checkpoint writes; VRAM occupancy alone does not measure GPU work.

The optional linear LR schedule reproduces the older trainer's 2e-5 peak LR and
1% warmup when used with the command above. Scheduler state is checkpointed and
resumed along with optimizer/model state, retaining at most two checkpoints.
The smoke script's original MSE defaults retain constant 1e-5 LR and remain resumable.
`trialspace_training.py train` and both `train_all*.sh` paths default to
`legacy_ranking`, 2e-5 LR and this linear schedule. They share the same loss and
full-batch gradient cache, with 32 groups/GPU and up to 8 candidates/group;
the smoke script uses 2 candidates/group for the A6000 memory budget.
Production includes all synthetic source partitions; the smoke script retains
its original-patient train/validation split. Direct production flags are
`--loss`, `--ranking-scale`, `--contrastive-weight`, `--lr-scheduler`, and
`--warmup-ratio`; the pipeline exposes them with a `--trialspace-` prefix.

Console `train_loss` and `val_loss` now report the selected ranking objective;
AUROCs still use raw MaxSim to predict `score > 0`. MSE is retained separately as
`mse_diagnostic`, not as evidence of calibrated scores. CoSENT magnitude depends
on the number of comparisons, so compare fixed validation probes over time,
not its numeric value against MSE or differently sized batches. Export metadata
marks ranking scores as uncalibrated and does not claim a 0–5 prediction scale.
This objective change needs full-corpus retrieval checks as well as AUROC:
it does not guarantee removal of candidate preferences, especially without
fresh hard negatives from the current model.

## Gated score regression experiment

`--objective gated_regression` selects a separate experiment; the default remains
the five independent binary retrieval tasks described below. The regression
teacher first asks **potential eligibility**, in its own call. A no ends that
pair with score **0**. A yes triggers four separate calls for cancer type, burden,
prior treatment, and required biomarkers. Python counts the yes answers, including
the eligibility answer, yielding **1–5** for eligible pairs. Skipped specificity
answers are saved as null. Failed required calls leave a pair unscored and are
retried on resume; they never become zero labels. A gate error can therefore
suppress useful specificity information, so inspect gate disagreements separately.

The MSE experiment uses one short aggregate retrieval instruction plus the complete
patient or cohort text, in both directions. With `--loss mse`, training minimizes
`MSE(mean_MaxSim, score / 5)` with explicit, labeled candidates grouped by query.
Constant-label groups are retained. There are no implicit in-batch negatives.
Regression update batches balance available direction/score anchors and retain
the sampled query's score range. The experimental CLI keeps its smaller
configurable batch defaults; production ranking uses the complete-pass batching
described above. Encoding is
batched within each query, with gradient accumulation across queries. Rare scores
are intentionally oversampled, not imputed. Actual update histograms and pair
counts are logged, and the sampling policy is part of checkpoint identity.
This is the token-level analogue of Sentence Transformers'
[CosineSimilarityLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss),
whose stock implementation instead compares pooled cosine embeddings.
The integer spacing is an explicit modeling assumption: one matched specificity
criterion adds one point. This composite score is not an eligibility probability.

`--prediction-link sigmoid` optionally changes the prediction to
`sigmoid(20 * (mean_MaxSim - 0.5))`; targets remain `score / 5`. The default is
identity. The link has fixed constants and no learned calibration head; saved
models include `regression_scoring.json` with the instruction and link contract.
Mining uses raw MaxSim because both links preserve ordering. With identity, raw
predictions can be negative; they are not silently clipped during evaluation.
Fixed probes record MSE, MAE/RMSE in 0–5 score units, unequal-label ordering
accuracy, and tie-aware NDCG@10 with linear graded gains. These metrics are
macro-averaged over query groups and directions. Singleton/all-zero groups have
no NDCG; constant-label groups have no ordering comparisons. Counts are reported.
Each sampled variable-label probe includes its lowest and highest available
labels, so it retains an ordering comparison. These fixed diagnostic subsets
do not estimate the deployment score distribution.
Better calibration alone does not establish better retrieval.
`baseline_metrics.json` compares the same probes with zero and a train-only
constant mean, weighted uniformly by direction and query before score-stratified
training sampling.

For an end-to-end fresh round, add `--objective gated_regression` to the
`run_round` command below. It mines with the aggregate instruction, labels with
gating, and trains on `teacher/scores.jsonl`. Existing candidate files can instead
be passed to the standalone stages:

```bash
PY=/data1/ken/envs/gptoss3/bin/python
$PY -m colbert_experiments label --gated-regression \
  --corpus CORPUS --candidates CANDIDATES.jsonl --output NEW_ROUND/teacher \
  --teacher-model gemma4-31b --teacher-tokenizer nvidia/Gemma-4-31B-IT-NVFP4 \
  --server_urls http://sn4622130540:8000/v1
CUDA_VISIBLE_DEVICES=2 $PY -m colbert_experiments train \
  --objective gated_regression --corpus CORPUS \
  --labels NEW_ROUND/teacher/scores.jsonl --output NEW_ROUND/fit \
  --steps 500 --accumulation 4 --eval-every 100 --eval-groups-per-stratum 16
```

For a controlled objective comparison, `aggregate-regression --corpus CORPUS
--candidates CANDIDATES.jsonl --source-teacher OLD_ROUND/teacher --output
NEW_ROUND/teacher` can reuse earlier individual answers **only after verifying**
the exact question contract, corpus, pair list, and each full-text prompt hash.
Specificity answers after a negative gate are discarded. The import manifest
records answer reuse, discarded answers, and zero new LLM calls. Keep these
imported labels separate from fresh stochastic teacher repeats. New contracts
and output directories prevent mixing regression and binary-task checkpoints.

The five positive (`yes`) targets are potential eligibility across **all** criteria,
matching specific cancer type, matching specific burden/stage, satisfying specific
prior treatment, and having the required biomarker condition. Potential eligibility
combines the former reasonable-consideration and other-exclusion questions:
**yes means there is no clear evidence the patient is ineligible**. Missing
information alone yields yes with `evidence_status=insufficient_information`;
clear evidence of an unmet requirement or applicable exclusion yields no. An
unreported biomarker result is not presumed negative just because testing would
normally be standard care. Each question has its own versioned file in `prompts/`.
Each pair produces **five separate LLM calls**, each containing shared formatting
instructions and exactly one question file, returning exactly one binary answer.
`shared.txt` does not request a list of answers. Specificity tasks are independent
of overall potential eligibility. Biomarker eligibility
is about required patient/tumor features, not treatment mechanism. Unknown evidence
is retained in `evidence_status`; it does not silently become a negative finding.

`prepare` joins the cancer summary and older exclusion-evidence field as **Other
relevant history**. It uses the existing cohort adapter to reconstruct specific
plus common criteria; single cohorts use only original eligibility text. Source
IDs and train/val/test splits are retained. Sibling cohorts and known synthetic
patient/source-trial relationships cannot cross splits. Supply lineage in
`source_trial_id` (or `nct_id`); a missing lineage cannot be checked automatically.

The query is a short task instruction plus the **complete** patient or cohort.
The other side is the complete text without the task instruction, allowing reuse
of document token vectors. Both limits default to 8,192 tokens including markers;
overlong inputs fail explicitly. The actual published encoder, 128-dimensional
projection, Q/D markers, and document punctuation mask are loaded with native
Transformers/PyTorch. No PyLate installation or production environment upgrade is
needed. The checkpoint supports 8K positions, but its retrieval training used much
shorter texts: mechanical long-context support does not establish clinical recall.
Every encoder path uses batches: length-sorted document batches (16 by default),
all question queries for a source entity together, and each training query with
its explicitly labeled documents in mixed-role batches grouped by length. Padding is
masked; query/document markers and document-only punctuation masking remain
separate. Batched training retains gradients through all token embeddings.

Scoring is **mean MaxSim**, with chunked score-matrix intermediates but no text
chunking/truncation. Mean normalization limits query-length score scaling; these
scores are ranking signals, not calibrated binary probabilities. The scoring
operation is directional. Training and mining use the same operation in each
direction rather than transposing a patient-centric score matrix.

For v22 data, `sample-v22` reads `patient_summaries_with_spaces.parquet`, samples
patients across distinct source trials, and preserves original splits. Patients
whose source trial conflicts with the new cohort split are excluded before
sampling. Every cohort remains in the corpus, including untouched test cohorts.
The manifest records source hashes, sampling decisions, and the full cohort count.
It also writes source-trial pairs as independent candidate proposals, never as
assumed positive labels.

Mining unions top candidates across all five questions and both directions, adds
TF-IDF and random exploration from the complete split, and accepts additional
challenge or independent-baseline pairs. The teacher answers all five questions
once per canonical patient/cohort pair. The label is reused across directions.
Teacher calls reuse the production adaptive vLLM pool, default vendor sampling,
reasoning parser, strict JSON validation (including outer Markdown fences), and
bounded retries. Successful answers, reasoning, prompt hashes and failures are
saved; changed input/prompt/model settings cannot reuse old responses silently.
The five-task v2 contract invalidates the earlier six-task labels, mining caches,
and training resumes. Use fresh round directories and regenerate labels; prior
experiment artifacts remain historical results under their original contract.

Training groups **one query, one task, one direction** with explicit yes/no
documents. Pairwise logistic loss compares only known positives and negatives in
that group, including multiple positives. Other queries never become implicit
negatives. Groups without both classes are counted and skipped. Sampling balances
available task/direction groups; `group_report.json` makes missing strata visible.
The standalone experiment defaults to train-split labels; `--training-splits`
can include additional partitions. Production `trialspace_training.py` instead
pools every synthetic patient/cohort before mining, labels both directions, and
trains on all records. Original splits are provenance only in that path, and its
metrics are in-sample diagnostics. BF16, gradient checkpointing, accumulation,
and DistributedDataParallel support long inputs and the two selected GPUs.
With `--eval-every N`, the trainer measures fixed, explicitly labeled train and
validation ranking groups before training and every N updates. The same queries,
positive documents, and negative documents are reused at each evaluation. It
records ranking loss, positive/negative ordering accuracy, score margin, and
precision at one, macro-averaged over task/direction strata. These are controlled
candidate-pool diagnostics, not full-corpus recall. Probe definitions, individual
scores, and evaluation history are saved alongside resumable optimizer state.

From the training repository, using the existing local Python environment:

```bash
PY=/data1/ken/envs/gptoss3/bin/python
$PY -m colbert_experiments prepare \
  --patients PATIENT_SUMMARIES.parquet --cohorts SAMPLE_COHORTS.csv \
  --output ../data/no_phi/colbert/corpus

$PY -m colbert_experiments.run_round \
  --corpus ../data/no_phi/colbert/corpus \
  --output ../data/no_phi/colbert/round1 \
  --gpus 2,3 --teacher-url http://sn4622130540:8000/v1 \
  --teacher-model gemma4-31b \
  --teacher-tokenizer nvidia/Gemma-4-31B-IT-NVFP4 \
  --teacher-concurrency 4 --steps 1000
```

Repeat the identical command to resume. Later rounds use a new output directory
and `--model ../data/no_phi/colbert/round1/fit/model`, mining anew and training
against that round's labels. Pass `--extra-pairs` for preserved challenge/known
positive pairs or an independent retriever's output. Input JSONL pairs require
`patient_id`, `cohort_id`, and `split`. Never infer a negative label simply because
an entity was not retrieved. Use `python -m colbert_experiments --help` for the
individual prepare/mine/label/train/evaluate/probe commands.

Endpoint URLs and concurrency may change on resume without changing saved labels.
Use `--teacher-served-model` when the endpoint advertises a different alias from
the recorded teacher identity. The labeler verifies the advertised model root;
Gemma 4 31B's NVFP4 and FP8 variants are treated as the same teacher family, as
specified for this experiment. Prompts, tokenizer, sampling, and output budgets
remain fixed, and each response records its actual endpoint and served model.

If a local teacher occupies one of the selected GPUs, `--training-gpus 2` can
restrict fitting to the available subset of `--gpus 2,3`. Accumulation increases
to preserve the original effective query-group batch. Such a device-count change
is permitted only before fitting has started; established checkpoints keep their
training identity. Before fitting, the coordinator waits for other compute
processes to leave its selected training GPUs. By default it does not stop user-owned servers.
An explicit `--stop-teacher-before-training REGISTRATION.json` can authorize
releasing one local teacher after labeling succeeds. The registration records
`purpose=stop_this_teacher_after_labeling`, the physical GPU, port, and the
`process_identity` result from `local_teacher.py`. The runner checks the selected
GPU, original PID/start time/command, and port, then uses a Linux pidfd to send
SIGTERM only to that registered server. It waits for the GPU to become free
before fitting; other servers remain running. Completed mining resumes validate
their manifest on CPU without occupying a GPU needed by the temporary teacher.

This is an **exact reference miner**, not a production ANN service. Full-corpus
late interaction is expensive. `--candidate-limit N` explicitly samples an
initial candidate pool while lexical/random sources still explore the entire
split. It changes recall and is recorded in the manifest. Token caches are
CPU-resident. Use `--proposal-strategy lexical_random` with that limit to combine
the strongest TF-IDF candidates with random exploration before MaxSim reranking.
`--cohort-query-limit N` bounds cohort queries per split while retaining every
cohort as a retrieval document. Patient queries are all processed. These options
make larger pilot rounds tractable without claiming exhaustive retrieval.
Token caches scale with every document token; plan an ANN/compressed index
and evaluate its recall before attempting large production rounds. Existing
single-vector TrialSpace embeddings are incompatible and are never reused.

For a small fabricated test:

```bash
$PY -m colbert_experiments fixture --output ../data/no_phi/colbert/fixture
CUDA_VISIBLE_DEVICES=2,3 $PY -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module colbert_experiments train \
  --corpus ../data/no_phi/colbert/fixture \
  --labels ../data/no_phi/colbert/fixture/reference_labels.jsonl \
  --output ../data/no_phi/colbert/smoke --steps 12 --accumulation 1
CUDA_VISIBLE_DEVICES=2,3 $PY -m colbert_experiments probe \
  --device cuda:1 --tokens 8000 --output ../data/no_phi/colbert/probe.json
$PY -m pytest -q colbert_experiments/tests
```

The fabricated set covers AND biomarkers, negated brain metastases, neutrophil
thresholds, and progression **on** versus **after** therapy. Train and validation
use different tumor/therapy examples. Hand-specified reference labels exercise
mechanics; review teacher disagreements rather than treating either as clinical
ground truth. `evaluate` reports each task/direction separately and counts missed
positives from all supplied judgments, including queries with no retrieved hits.
Recall is conditional on the available judgments; incomplete annotation is not
full-corpus recall. Broad eligibility and specificity still need separate
validation, calibration and downstream clinical review.

Source model and training references:
[model card](https://huggingface.co/lightonai/GTE-ModernColBERT-v1),
[checkpoint encoding settings](https://huggingface.co/lightonai/GTE-ModernColBERT-v1/blob/main/config_sentence_transformers.json),
[ColBERT](https://arxiv.org/abs/2004.12832).
The model card identifies Apache-2.0 licensing; this training repository's
AGPL-3.0-or-later license continues to apply to its code. Check licenses before
redistributing a trained artifact.
