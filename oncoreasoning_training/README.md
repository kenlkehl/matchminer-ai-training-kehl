# OncoReasoning answer-first distillation

> **2026-11-04 (user instruction): aligned with the current pipeline.** Format
> `oncoreasoning-answer-first-v7-key-match-labels`. TrialChecker questions are the
> inference package's five match labels (reasonableness, cancer type, burden, treatment
> history, biomarker), one per request, cut verbatim from the teacher's five-answer prompt
> `match_quality.labels.system.txt` and asked about the patient's key summary section and
> the cohort's key criteria. BoilerplateChecker questions are the lines of the cohort's
> boilerplate criteria, each in the context of the whole numbered list and about the
> patient's boilerplate section, following the BoilerplateChecker prompt
> (`llm_exclusion_criteria.user.txt`). The separate teacher extraction of exclusions
> (`prepare_boilerplate.py`, stage `prepare-exclusions`) is gone. These are question
> families of the `clinical_qa` adapter, distinct from the ModernBERT TrialChecker (five
> heads) and BoilerplateChecker (two classes) that `../train_checkers.py` trains. ColBERT
> is removed; TrialSpace is dense EmbeddingGemma 2 only. The code is authoritative where
> this file disagrees.

The student defaults to `google/gemma-4-E4B-it`, with only the text model loaded.
Select a future base using `prepare --model-name`; the trainer reads that choice
from the prepared manifest. Use a new data and model directory when changing it.
The two training categories are patient summarization and clinical question
answering. Each has its own LoRA adapter on the same frozen base. The integrated
runner, `train_from_summaries.py` (the `train` stage of `run_pipeline.py`, which
`train_all*.sh` runs), trains them sequentially after the three TrialSpace
(EmbeddingGemma 2) rounds, the ModernBERT TrialChecker and BoilerplateChecker,
the GoodOption catalog, through the stages `prepare-oncoreasoning`, `distill-oncoreasoning`,
`build-oncoreasoning` and `train-oncoreasoning`. QA includes all three checker
families.

Google documents Gemma's [model and chat format](https://huggingface.co/google/gemma-4-E4B-it)
and [thinking control](https://huggingface.co/docs/transformers/model_doc/gemma4).
Student chat templates always use `enable_thinking=False`. The E4B tokenizer
encodes `A` and `B` as single tokens directly after the assistant header; dataset
construction verifies this invariant for any selected replacement model.

## Training examples

Summarization starts from chronological raw synthetic notes. As in the serial
summarizer, repeated sentences are removed, dated notes are concatenated, and
each request updates the previous completed summary. Each chunk independently
samples a uniform integer budget from **10,000 to 50,000 student tokens**, with
500 tokens of overlap. Short histories and final tails may be below 10,000.
Chunk spans, dates, sampled budgets, and parent request IDs are saved. Every
changed chunk gets a fresh teacher summary; old fixed-chunk targets are not
reused. The summary primer and output instructions come from the inference
checkout's current serial-summary templates.

Each clinical QA example asks **one component** with `A. Yes / B. No` and a
required output such as:

```text
A
The patient's documented age falls within the trial's stated range.
```

TrialChecker supplies five questions, the inference package's match labels:
reasonableness (no clear evidence that the patient fails any of the cohort's key criteria)
and the cancer-type, burden, treatment-history and biomarker specificity questions (does the
cohort specifically require that dimension, documented as satisfied). Each question's text and
rules are cut verbatim from `match_quality.labels.system.txt`, with its shared preamble (clinical
text is data; only key criteria and the key summary section are supplied, boilerplate is
checked separately), and the code pins that prompt's digest (`MATCH_LABELS_DIGEST`): a changed
prompt fails until the adaptation is reviewed. Every question is asked for every pair. The 0-6
match score is `matchminer_ai.matching.match_labels.match_score` of the five answers: 0 unless
reasonable, else 1 plus 1 each for cancer type, burden and treatment history and 2 for
biomarker. Instructions, decision rules and answer options precede the patient and cohort
text, so a pair's questions share a cacheable prefix.

The format version is `oncoreasoning-answer-first-v7-key-match-labels` (v6 asked six
matching and four specificity components cut from the earlier single-score rubric
`llm_match_quality.user.txt`, on combined texts; that rubric and its labeler, `llm_check_trials.py`, were removed on 2026-11-04). Data prepared or distilled under v6 must
not be resumed under v7; regenerate it from the current candidates.

BoilerplateChecker supplies one question per line of the cohort's boilerplate criteria
(`cohort_boilerplate_criteria`, split as the boilerplate step splits them): does the patient
clearly fail this one criterion (clearly meet an exclusion, or clearly not meet an inclusion
requirement)? The question shows the cohort's whole numbered boilerplate list for context and
the patient's boilerplate section (other relevant history) only. Missing information, waitable
washouts and uncertain, mild or resolved conditions give **B. No**, as in the BoilerplateChecker
prompt. Sibling cohorts with the same boilerplate criteria give identical questions, asked once.
The aggregate BoilerplateChecker answer is Yes when any line's answer is Yes.

GoodOption supplies four questions **per scoreable drug**: disease-type benefit,
common biomarker in the disease, the patient's documented biomarker, and human
benefit from targeting that patient biomarker. Prompts use the current inference
rubric and include the named drug's evidence and only its covering class blocks.
The fourth criterion states the patient-specific biomarker requirement within
its own question, without depending on a previous answer. No old aggregate
TrialChecker labels or deprecated GoodOption classifier labels are reused.

Binary decisions preserve each current rubric's missing-information rules.
TrialChecker's required-biomarker rule considers tests known or assumed to have
been performed as standard care, and allows uncertainty for novel markers only
tested during trial screening when absence cannot be inferred. Unknown status
cannot earn biomarker specificity. Absent or untested patient biomarkers earn
No for the patient-specific GoodOption criterion. GoodOption's prior receipt
of the named drug overrides disease-benefit literature, while prior receipt of
a different drug in the class does not. The TrialChecker adaptation is pinned
to its source rubric fingerprint and fails if that rubric changes before review.

Teachers may reason internally. Returned reasoning, complete raw responses,
request messages and retry attempts are retained in local teacher archives for
future distillation. They are kept separate from the validated final answer.
Separate reasoning fields are excluded from supervised targets and explicit
thought-channel wrappers are stripped. The student sees only the final summary or **the full letter plus
explanation**. Prompt tokens are masked; all final answer tokens, including the
initial letter, are supervised. Malformed answers, incomplete thinking blocks,
truncated completions, missing responses, and overlong examples fail validation.
There is no target or input truncation and no conversion of old reasoning traces
into student targets. `convert_to_think_tags.py` is retired.

Raw pooled requests are archived before validation in `teacher_attempts/`,
including token-limited or malformed responses. Successful `responses/` records
retain their reasoning and attempt records too. Older generation manifests that
discarded reasoning cannot resume as trace-preserving runs; use a new directory.
The production orchestrator records archive locations in `reasoning_archives.json`.

## Commands

The integrated runner passes these stages their inputs, writing data to
`<run>/oncoreasoning/` and adapters to `<models-dir>/oncoreasoning/`. For a
manual run, use the existing environment on this machine, outside the
network-mounted repo:

```bash
PY=~/thisenv/bin/python
RUN=../data/no_phi/oncoreasoning_answer_first_v2
MINING=../data/no_phi/gemma_pipeline_v2

$PY oncoreasoning_training/create_all_training_data.py prepare \
  --output-dir "$RUN" \
  --model-name google/gemma-4-E4B-it \
  --inference-repo ../matchminer-ai-inference-kehl \
  --notes ../data/no_phi/all_synthetic_notes.parquet \
  --catalog "$MINING/good_option_catalog" \
  --candidates \
    "$MINING"/top_{cohorts,patients}_tocheck_round3.parquet

$PY oncoreasoning_training/create_all_training_data.py generate \
  --output-dir "$RUN" \
  --server-url http://YOUR-TEACHER-HOST:8000/v1 \
  --teacher-model nvidia/Gemma-4-31B-IT-NVFP4 \
  --teacher-tokenizer google/gemma-4-31B-it \
  --teacher-context 131072 --teacher-max-tokens 16384 \
  --concurrency 8

$PY oncoreasoning_training/create_all_training_data.py build \
  --output-dir "$RUN" --max-seq-length 98304

$PY -m torch.distributed.run --standalone --nproc_per_node=8 \
  oncoreasoning_training/fine_tune_llm.py \
  --data-dir "$RUN" \
  --output-dir ../models/oncoreasoning_gemma4_e4b_answer_first_v2 \
  --fsdp --resume-from-checkpoint auto
```

`prepare` and `build` are local data work. `generate` sends patient-bearing
prompts only to the explicitly supplied teacher endpoint; it does no web search.
Inputs outside `../data/no_phi` require `--confirm-inputs-are-non-phi`.
The entire GoodOption catalog is validated before requests are prepared.
Technically blocked or missing trial evidence fails preparation; a trial with
no scoreable drugs contributes TrialChecker questions only.

Candidate inputs must contain patient IDs (`pseudo_mrn` or `patient_id`), patient
summaries (`patient_summary` or `cancer_history_summary`), trial IDs (`nct_id` or
`trial_id`), and trial spaces (`this_space` or `clinical_space_summary`). Since
2026-11-04 the integrated run's round-3 candidate tables carry the patient
summary's key section as `patient_summary`, the cohort's key criteria as
`this_space`, and the boilerplate section and boilerplate criteria as
`patient_boilerplate_text` and `trial_boilerplate_text`. Notes
default to `pseudo_mrn`, `date`, and `synthetic_note`; column flags can override
these. Repeated candidate questions are deduplicated; the same patient/drug
question is generated once across multiple trial spaces.

Explicit patient split assignments are shared across both task categories.
Missing assignments inherit the patient's known split or default to train;
conflicting assignments fail. All three source partitions are included by default.
The integrated miner pools all patients/cohorts and exports the original patient
assignment as `patient_source_split`; preparation uses that provenance to agree
with the raw notes. Older external mining tables with a constant train field
must first have their original patient assignments restored.
Preparation records both categories in `training_manifest.json`. Training filters
each adapter's examples strictly to its task, then merges train, validation, and
test into its training set. There is no cross-task mixture ratio or oversampling.
Production rejects `--training-splits train` and other source-partition subsets.
Use
`prepare --tasks summarization` or `--tasks clinical_qa` to prepare one category,
and `fine_tune_llm.py --task summarization` or `--task clinical_qa` to train it.
Use `--max-patients` / `--max-pairs` for small preparation checks.

Generation is resumable: rerun the identical command. Each successful final
answer is saved atomically. A dependent summary waits for its completed parent.
Changed teacher settings, prompts, inputs, or tokenizers cannot silently reuse
old shards. A stopped or malformed completion is retried three times; exhausted
requests stop the stage without saving an invalid target. Teacher thinking is
on by default and can be disabled with `--no-teacher-thinking`.

Generation and exclusion extraction follow `--server-urls-file` continuously.
The default `--concurrency 0` fills live per-server capacities (96 if unspecified),
bounded by 1,536 workers; a positive value supplies a lower total request limit.
Requests refill as soon as any response finishes while serial summary chunks
still wait for their parent. Transport failures retry across available servers
for up to 24 hours; each HTTP request has a 7,200-second timeout. A removed or
returning Spot server does not change the saved-answer identity. Prompt, model,
tokenizer and sampling changes still require new targets.

The integrated preparation command passes `--pipeline-manifest` to reuse the
recorded source-note identity without rehashing the original file. It still reads
notes to construct tasks and records identities for newly mined candidate files.

The trainer preserves prepared labels, uses gradient checkpointing and BF16
mixed precision where available, and projects only supervised positions to
vocabulary logits. Full-parameter fine-tuning is disabled. LoRA defaults:

| Setting | Default |
| --- | --- |
| Rank / alpha | 64 / 128 |
| Targets | All linear layers in the extracted text model; frozen embeddings and output head |
| Dropout / bias | 0.05 / none |
| Learning rate | 5e-5, AdamW, cosine decay |
| Warmup | First 3% of optimizer steps; positive `--warmup-steps` overrides |
| Weight decay / gradient clipping | 0.01 / maximum norm 1.0 |
| Epochs | One per adapter |
| Batch | One example per GPU, eight accumulation steps: 64 examples on eight GPUs |
| Checkpoint interval | Every 100 optimizer steps (`--save-steps`) |
| Training partitions | All synthetic train, validation, and test examples |
| Synthetic held-out validation | None; source partitions are provenance only |

These are conservative starting settings, not a guarantee of output quality.
The final artifact uses the final training checkpoint; there is no held-out
synthetic validation or validation-based checkpoint selection in production.
Every partition goes into each adapter's training set, so no evaluation runs
during training; `--max-eval-examples` is recorded in the run identity but
selects nothing.
Non-finite loss stops training. The [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora)
describes all-linear targeting and the default initialization, which leaves the
base model's initial outputs unchanged. The base is never merged or updated.
On the current Gemma 4 E4B text configuration, rank 64 targets **155,516,928
trainable parameters per adapter** (about 1.9% of base plus adapter parameters).
Each adapter's contract records its actual trainable parameter count.

With `--task both` (the default), the output directory contains `summarization/`
and `clinical_qa/`, plus an `oncoreasoning_contract.json` collection index written
only after both succeed. Both adapters start independently from the original
base, with separate optimizers, schedules and checkpoints.
Choose the corresponding adapter at inference time. `--resume-from-checkpoint
auto` skips completed adapters and restores the latest complete checkpoint for
an unfinished one. For an explicit checkpoint path, select one `--task` and its
adapter directory as `--output-dir`. Resume requires identical training settings.

An integrated run loads this trainer when it reaches `train-oncoreasoning`.
Resuming a run after a code change requires a scoped upgrade receipt
(`scripts/register_oncoreasoning_upgrade.py`; see the root README's resume
rules); do not restart active work.

## One-token preview

A trained QA prompt supports either one generated answer token or a longer
letter-plus-explanation response. This is an answer-first causal language model,
inspired by the requested System 1 behavior; it does not implement JEV's separate
calibration architecture. Quick mode still has to process the input prompt.

For a local smoke test, choose a clinical QA request ID from `requests.jsonl`:

```bash
CUDA_VISIBLE_DEVICES=3 ~/thisenv/bin/python oncoreasoning_training/preview_model.py \
  --model ../models/oncoreasoning_gemma4_e4b_answer_first_v2/clinical_qa \
  --requests ../data/no_phi/oncoreasoning_answer_first_v2/requests.jsonl \
  --request-id REQUEST_ID --quick
```

Omit `--quick` and set `--max-new-tokens 1024` for an explanation. Both modes use
the identical prompt, disable thinking, decode greedily, and restrict the first
generated token to `A` or `B`. The preview loads either a full text checkpoint or
the saved QA LoRA adapter (or selects it from a collection directory). Runtime integration into the public inference package
and model publication are separate from this training workflow.
