# Cohort ColBERT experiment

This package contains the shared native `lightonai/GTE-ModernColBERT-v1`
implementation and the original five-question experiments in **both retrieval
directions**. The production entrypoint `../trialspace_training.py` and
`../train_from_summaries.py` now use its gated regression trainer and question
contracts, with distributed disk-backed mining. The experimental CLI retains
its original binary-task default for controlled comparisons. Use synthetic/non-PHI
inputs unless the selected endpoint is authorized for the data.

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

The student uses one short aggregate retrieval instruction plus the complete
patient or cohort text, in both directions. Training minimizes
`MSE(mean_MaxSim, score / 5)` with explicit, labeled candidates grouped by query.
Constant-label groups are retained. There are no implicit in-batch negatives.
Regression update batches balance available direction/score anchors and retain
the sampled query's score range. Production defaults are 32 query groups per GPU
per update and up to eight documents per query (up to 1,024 labeled pairs on four
GPUs); the experimental CLI keeps its smaller configurable defaults. Encoding is
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
its explicitly positive/negative documents in one mixed-role batch. Padding is
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
Only train-split labels update weights. BF16, gradient checkpointing, accumulation,
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
