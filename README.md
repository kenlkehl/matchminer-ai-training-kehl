# matchminer-ai-training
Code for training the MatchMiner-AI pipeline.
If you have access to a Linux machine with H100 x 8 GPUs and about a week to spare, you can replicate training by:

Making sure your machine can compile CUDA things:
```
sudo apt update
sudo apt install build-essential python3-dev nvidia-cuda-toolkit
```

Installing uv (https://docs.astral.sh/uv/getting-started/installation/)


Pulling this code and installing dependencies:
```
git clone https://github.com/kenlkehl/matchminer-ai-training
cd matchminer-ai-training
uv sync --group training
```


4. Running the train_all.sh script:
```
bash train_all.sh
```

Note: Training makes heavy use of multiple instances of vllm for efficient parallelized inference. This sometimes causes errors related to race conditions on compile. We try to mitigate this by pre-compiling these at the beginning of the script, but errors during training may still occur and require restarting the script from the last completed step.

Also note: At the time of release (December 2025), we were encountering challenges with gibberish output from gpt-oss-120b when run with vllm on more than one RTX PRO 6000 GPU. Inference on a single RTX PRO 6000 seemed to work well, though.

Original framework and logic were implemented manually; parallelization was vibe-coded with Gemini 2.5 pro and Claude 4.5 Sonnet.

## GoodOptionChecker

`train_good_option_checker.py` builds a separate research signal for how
promising a matched trial's drug option may be for a specific synthetic
patient. This is intentionally distinct from TrialChecker eligibility/match
reasonableness. The label is subjective, is not a response probability, and is
not a treatment or enrollment recommendation.

The component calls the public `matchminer_ai.help_me_choose` drug-research
APIs rather than duplicating their retrieval/privacy logic. In this staging
workspace, install the sibling `matchminer-ai-inference` checkout in editable
mode when testing unpublished changes to those APIs.

The default workflow consumes all six `top_cohorts_tocheck_round*` and
`top_patients_tocheck_round*` files. It deduplicates patient-space pairs and
caches research once per unique NCT ID:

```bash
# 1. ClinicalTrials.gov lookup plus drug-only web research (no patient text)
python train_good_option_checker.py research

# 2a. Label with app-owned local vLLM servers, one per listed GPU by default
python train_good_option_checker.py label \
  --gpus 0,1,2,3,4,5,6,7 \
  --model nvidia/Gemma-4-31B-IT-NVFP4 \
  --reasoning-parser auto \
  --download-dir ~/models

# 2b. Or use existing OpenAI-compatible endpoints
python train_good_option_checker.py label \
  --server_urls http://host-a:8000/v1,http://host-b:8000/v1 \
  --model nvidia/Gemma-4-31B-IT-NVFP4 \
  --reasoning-parser auto

# 3. Fit the single-logit ModernBERT checker (safe to launch with accelerate)
accelerate launch --num_processes 8 train_good_option_checker.py train
```

For an authenticated endpoint, put its key in `OPENAI_API_KEY`, or name another
environment variable with `--api-key-env`. Dynamic endpoint lists written by
`gcp_vllm_orchestrator.py` are accepted through `--server_urls_file`. Labeling
and research are resumable from Parquet shards.
Use `--refresh-research` when intentionally taking a new dated registry/search
snapshot; use a new label output/shard directory if those refreshed records
must replace labels that were already completed.
Use `--max-trials` and `--max-candidates` only for bounded development runs;
labeling refuses candidate NCT IDs that do not yet have a research record. The
full stage is large and must respect the search provider's operational limits.

The generated non-PHI artifacts are:

- `../data/no_phi/good_option_drug_research.parquet`: dated registry metadata,
  structured interventions, drug-only queries, snippets, URLs, notices, and a
  hash of the reused research implementation;
- `../data/no_phi/good_option_labels.parquet`: the 0-100 teacher score, the
  normalized 0-1 training target, confidence, rationale, evidence references,
  prompt/schema versions, source timestamps, and the registry-derived drug
  context used as checker input; and
- `../models/goodoptionchecker`: a one-logit model whose sigmoid is the 0-1
  subjective potential-benefit score.

The privacy boundary is structural: registry calls receive only NCT IDs, and
web queries are built only from non-placebo `DRUG` and `BIOLOGICAL`
intervention names. Patient summaries enter only the later LLM-labeling prompt.
The trained checker consumes patient summary + clinical space + registry drug
context; web snippets supervise the LLM teacher but are not checker inputs.
Custom candidate files outside `../data/no_phi` are rejected unless
`--confirm-inputs-are-non-phi` is supplied. Do not use that override for real
clinical data unless the endpoint and data flow have the required authorization
and controls.
