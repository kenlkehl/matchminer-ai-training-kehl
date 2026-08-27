# matchminer-ai-training

Code for training the MatchMiner-AI pipeline. A full reproduction is intended
for a Linux system with eight H100-class GPUs and takes roughly a week.

```bash
uv sync --group training
bash train_all.sh
```

Training uses vLLM heavily. Restart an interrupted workflow from the last
completed orchestration step rather than rerunning completed stages.

## GoodOptionChecker

GoodOption is an evidence-counting signal for whether a trial's experimental
drug options have support relevant to a synthetic patient's cancer. It is not
an eligibility result, response probability, or treatment recommendation.

The v2 workflow has a hard patient-free research boundary:

1. Collect unique NCT IDs from the configured candidate tables or an explicit
   ID list.
2. Fetch each ClinicalTrials.gov record, classify every active drug's trial
   role, normalize active entities to NCIt where possible, and deduplicate drugs
   across trials.
3. Research each unique drug through authoritative oncology sources and a
   pluggable general-web provider. Fetch bounded document passages rather than
   relying only on result snippets.
4. Synthesize structured facts and clean GoodOption/Help Me Choose summaries.
5. Validate the complete catalog before any patient-bearing LLM request.
6. Label every scoreable patient-drug pair on four independent binary criteria.
7. Train a four-logit ModernBERT checker and aggregate all drug-by-criterion
   probabilities when producing a patient-trial score.

Control, background, and supportive drugs are researched for catalog coverage
but omitted from GoodOption prompts and denominators. A role that remains
uncertain is marked and scored. A completed search with no evidence is a valid
result; an unresolved technical research failure blocks affected labels.

The unchanged criteria are:

1. human clinical benefit from the same drug or a regimen containing it in the
   patient's disease and relevant histology;
2. a directly targeted biomarker present in at least 20% of the full relevant
   disease population, or authoritatively described as common/frequent/highly
   expressed there;
3. explicit documentation of that target in this patient's own tumor; and
4. human benefit from therapeutically targeting that same patient biomarker.

### Commands

Run the complete workflow in sequence with one command. Each stage runs in an
isolated process, validation must succeed before labeling starts, and training
is launched through Accelerate:

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

Build and validate all patient-free evidence first:

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

After validation, label and train:

```bash
python train_good_option_checker.py label \
  --server-urls http://host-a:8000/v1,http://host-b:8000/v1 \
  --model nvidia/Gemma-4-31B-IT-NVFP4

accelerate launch --num_processes 8 train_good_option_checker.py train \
  --patient-validation-fraction 0.20 \
  --drug-validation-fraction 0.20
```

Use `--split-strategy none` to train on all valid synthetic rows when evaluation
will be performed on a separately governed real dataset.

The catalog command also accepts repeated `--nct-id` values or
`--nct-ids-file`. Text files contain one NCT ID per line; CSV/Parquet files use
`nct_id` or `trial_id`. Custom candidate inputs outside `../data/no_phi` require
`--confirm-inputs-are-non-phi` during `label` because their patient summaries
reach the configured teacher endpoint. The `catalog` command reads only the
`nct_id` column and does not accept or load patient context.

### Artifacts

- `../data/no_phi/good_option_catalog_v2/`: versioned Parquet catalog, source
  ledger, retry audit, clean drug summaries, trial-drug index, and manifest.
- `../data/no_phi/good_option_four_point_label_shards_v2/`: resumable
  patient-trial labeling shards.
- `../data/no_phi/good_option_four_point_labels_v2.parquet`: validated label
  aggregate with per-drug rationales and catalog compatibility ID.
- `../models/goodoptionchecker_four_point_v2/`: four-logit checker and split/
  evaluation metadata.

Patient context is never accepted by catalog retrieval or included in public
queries. The patient labeling prompt contains only the patient summary followed
by clean summaries for scoreable drugs; it omits URLs, queries, source IDs,
trial metadata, controls, fetch notices, and raw evidence passages.
