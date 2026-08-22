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
well-supported a matched trial's drug option is for a specific synthetic
patient. This is intentionally distinct from TrialChecker eligibility/match
reasonableness. The label is an auditable evidence count, not a response
probability or a treatment or enrollment recommendation.

The LLM does not choose a holistic 0-100 score. For every distinct canonical
investigational drug in the trial, it independently awards one binary point for
each of four criteria:

1. human evidence of benefit from the same drug/regimen in the patient's
   disease type;
2. evidence that the drug targets a biomarker commonly present in that disease
   type (at least 20% prevalence in the full relevant disease/histology
   population, or explicitly described as common/frequent in that population);
3. documentation that the patient's own tumor has the targeted biomarker; and
4. human evidence of actual benefit from targeting that documented biomarker,
   including an explicit prior patient response when the target relationship is
   also supported.

Missing, ambiguous, merely mechanistic, or preclinical-only evidence receives
zero where the criterion requires human evidence. Code requires exactly one
four-point assessment per distinct canonical drug, sums every awarded point,
and divides by `4 * number_of_drugs` to produce the 0-1 training target. Repeated
arm strings that normalize to the same investigational drug count once. Drugs
used only as standard-of-care, active-comparator, placebo, sham, supportive-care,
rescue, premedication, or unevaluated background therapy are not scored and do
not enter the denominator. For the population-prevalence point, a rate
calculated only among an already
biomarker-positive or otherwise enriched subgroup does not qualify, nor does
being common relative to other alterations; the evidence must use the full
relevant disease population as its denominator. If the teacher awards a point
without the criterion's required evidence labels, code resets that individual
point to zero and records the validation reason instead of discarding the other
drug assessments.

Before web research, the teacher receives only public ClinicalTrials.gov
DRUG/BIOLOGICAL intervention fields plus their structured arm labels, arm types,
and arm descriptions. It first classifies each drug as investigational,
non-investigational, or uncertain. Code always excludes an intervention assigned
only to control-type arms even if the teacher tries to select it. Uncertain and
non-investigational drugs are not searched. For selected investigational drugs,
the teacher removes arm, cohort, phase, dose, route, and formulation wording and
returns canonical active names that must be textually supported by the registry;
an unsupported normalized name falls back only to that selected intervention's
original registry string. The component then calls the public
`matchminer_ai.help_me_choose` search APIs with those names and adds a second
query per drug for its molecular target and biomarker-expression prevalence
across cancer types. The training wrapper preserves all distinct structured
registry drugs and submits search queries in small chunks so the interactive
helper's per-call result cap cannot starve later investigational drugs of
evidence. In this staging workspace, install the sibling
`matchminer-ai-inference` checkout in editable mode when testing unpublished
changes to those APIs.

When the configured reasoning parser is `qwen3` (including automatic parser
selection for a Qwen model), the intervention-selection prompt explicitly
enables Qwen thinking and the parser separates that reasoning from the final
JSON. The default `--drug-name-max-new-tokens 8000` leaves room for both. Other
model families retain their existing chat-template behavior.

The default workflow consumes all six `top_cohorts_tocheck_round*` and
`top_patients_tocheck_round*` files. It deduplicates at the patient--trial level;
candidate trial-space text is mining provenance and is not part of this label or
the trained checker input. Research is cached once per unique NCT ID. During
labeling, the default `--patients-per-request 8` shares one trial and web-evidence
payload across as many as eight patient--trial cases while preserving an
independent response object and validation result for each patient. For multi-drug
trials, `--max-drug-assessments-per-request 16` automatically reduces the patient
count to bound response size. The request pool also dispatches these grouped
prompts concurrently across all configured vLLM endpoints:

```bash
# 1a. Normalize names, research drugs, and label with one app-owned vLLM pool
python train_good_option_checker.py generate \
  --gpus 0,1,2,3,4,5,6,7 \
  --model nvidia/Gemma-4-31B-IT-NVFP4 \
  --reasoning-parser auto \
  --download-dir ~/models

# 1b. Or use existing OpenAI-compatible endpoints for both teacher calls
python train_good_option_checker.py generate \
  --server_urls http://host-a:8000/v1,http://host-b:8000/v1 \
  --model nvidia/Gemma-4-31B-IT-NVFP4 \
  --reasoning-parser auto

# 2. Fit the single-logit ModernBERT checker (safe to launch with accelerate)
accelerate launch --num_processes 8 train_good_option_checker.py train
```

The `research` and `label` subcommands remain available for separate resumable
runs. Because canonicalization now precedes search, `research` accepts the same
model, endpoint, or local-GPU arguments as `generate`; no patient context is
sent during that stage.

For an authenticated endpoint, put its key in `OPENAI_API_KEY`, or name another
environment variable with `--api-key-env`. Dynamic endpoint lists written by
`gcp_vllm_orchestrator.py` are accepted through `--server_urls_file`. Labeling
and research are resumable from Parquet shards.
Use `--refresh-research` when intentionally taking a new dated registry/search
snapshot. Research caches whose implementation/query fingerprint is stale are
automatically refreshed. Label resume state is restricted to the current
prompt and schema versions.
ClinicalTrials.gov request starts are globally paced, and HTTP 429, transient
5xx, timeout, and transport errors use shared exponential cooldowns that honor
`Retry-After`. The defaults allow ten attempts. Research aborts before web search
or patient labeling if exhausted registry failures exceed 5%, intervention-name
selection failures exceed 10%, or drug selection unexpectedly collapses to zero.
The limits and pacing are configurable with the `--registry-*` and
`--max-*-failure-fraction` options.
Regardless of the aggregate limits, labeling stops if any in-scope trial still
has unavailable registry or teacher research; those failures are never expanded
into patient-level labels. A successful registry record with no structured
DRUG/BIOLOGICAL intervention is instead a confirmed no-experimental-drug result.

If an older run cached widespread registry or empty Qwen-answer failures, rerun
the same `generate` command with `--refresh-research`. The current research
prompt/fingerprint forces fresh trial records, and the current label schema
ignores the old unscored shards without deleting them. The quality gate runs
again before any patient-bearing teacher request.
Use `--max-trials` and `--max-candidates` only for bounded development runs;
labeling refuses candidate NCT IDs that do not yet have a research record. The
full stage is large and must respect the search provider's operational limits.
Increase `--patients-per-request` only when the endpoint context window and
completion limit can accommodate the larger grouped response; use
`--max-batch-new-tokens` and `--max-drug-assessments-per-request` to cap that
response.

The generated non-PHI artifacts are:

- `../data/no_phi/good_option_drug_research.parquet`: dated registry metadata,
  raw intervention strings, teacher-normalized canonical names and mappings,
  efficacy/safety and target-expression queries, snippets, URLs, notices, and
  implementation/query/prompt fingerprints;
- `../data/no_phi/good_option_four_point_labels.parquet`: one patient--trial
  record containing a four-point assessment per canonical investigational drug,
  criterion-specific rationales and evidence references, drug count,
  code-derived total and maximum points, the normalized 0-1 training target,
  prompt/schema versions, source timestamps, and the registry-derived drug
  context used as checker input; and
- `../models/goodoptionchecker_four_point`: a one-logit model whose sigmoid is
  the predicted fraction of all per-drug evidence criteria satisfied.

The privacy boundary is structural: registry calls receive only NCT IDs; the
name-normalization teacher call receives only public registry intervention and
arm fields; and web queries are built only from its textually supported canonical
investigational-drug names. Even the target-expression query asks generically
about prevalence across cancer types; it does not contain the patient's disease.
Patient summaries and their disease context enter only the later LLM-labeling
prompt. The trained checker consumes patient summary + registry
investigational-drug context. Neither candidate-space text nor web snippets are
checker inputs; web snippets supervise only the LLM teacher.
Custom candidate files outside `../data/no_phi` are rejected unless
`--confirm-inputs-are-non-phi` is supplied. Do not use that override for real
clinical data unless the endpoint and data flow have the required authorization
and controls.
