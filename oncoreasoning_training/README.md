# OncoReasoning answer-first distillation

The student defaults to `google/gemma-4-E4B-it`, with only the text model loaded.
Select a future base using `prepare --model-name`; the trainer reads that choice
from the prepared manifest. Use a new data and model directory when changing it.
The two training categories are patient summarization and clinical question
answering. The integrated `train_all*.sh` pipeline runs both categories after TrialSpace and catalog construction.

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

TrialChecker supplies six independent matching questions: age, sex, cancer type
and histology, burden/stage, treatment history, and biomarkers. Four additional
questions cover its cancer-type, burden, prior-treatment, and biomarker
specificity points. The reasonableness gate and final aggregate score are not
asked in these component examples. To reconstruct the old trial score, all six
matching answers must be Yes; otherwise the score is zero. When they are all
Yes, the score is one plus the four specificity points.

BoilerplateChecker supplies one question per trial exclusion: does this specific
exclusion clearly apply? Unknown or uncertain evidence gives **B. No**. A
patient-free teacher stage extracts verbatim rules, preserving all thresholds,
exceptions, and AND/OR clauses; extraction rejects omitted source text. General
exclusion evidence accompanies the patient summary. The original aggregate
exclusion flag can be reconstructed as Yes when any individual answer is Yes.

GoodOption supplies four questions **per scoreable drug**: disease-type benefit,
common biomarker in the disease, the patient's documented biomarker, and human
benefit from targeting that patient biomarker. Prompts use the current inference
rubric and include the named drug's evidence and only its covering class blocks.
The fourth criterion states the patient-specific biomarker requirement within
its own question, without depending on a previous answer. No old aggregate
TrialChecker labels or deprecated GoodOption classifier labels are reused.

Binary decisions preserve the current missing-information rules. For example,
an unknown required biomarker is not itself a clear TrialChecker mismatch, but
cannot earn a biomarker-specificity point; absent or untested patient biomarkers
earn No for the patient-specific GoodOption criterion. GoodOption's prior receipt
of the named drug overrides disease-benefit literature, while prior receipt of
a different drug in the class does not. The TrialChecker adaptation is pinned
to its source rubric fingerprint and fails if that rubric changes before review.

Teachers may reason internally. Only completed final response content is saved;
separate reasoning fields are ignored and explicit thought-channel wrappers are
stripped. The student sees only the final summary or **the full letter plus
explanation**. Prompt tokens are masked; all final answer tokens, including the
initial letter, are supervised. Malformed answers, incomplete thinking blocks,
truncated completions, missing responses, and overlong examples fail validation.
There is no target or input truncation and no conversion of old reasoning traces
into student targets. `convert_to_think_tags.py` is retired.

## Commands

Use the existing environment on this machine, outside the network-mounted repo:

```bash
PY=~/thisenv/bin/python
RUN=../data/no_phi/oncoreasoning_answer_first_v2
MINING=../data/no_phi/gemma_pipeline_v2

$PY oncoreasoning_training/prepare_boilerplate.py \
  --candidates "$MINING"/top_{cohorts,patients}_tocheck_round3.parquet \
  --output-dir "$MINING/boilerplate_components" \
  --server-urls-file "$MINING/teacher_servers.json"

$PY oncoreasoning_training/create_all_training_data.py prepare \
  --output-dir "$RUN" \
  --model-name google/gemma-4-E4B-it \
  --inference-repo ../matchminer-ai-inference-kehl \
  --notes ../data/no_phi/all_synthetic_notes.parquet \
  --catalog "$MINING/good_option_catalog" \
  --boilerplate-components "$MINING/boilerplate_components" \
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

~/thisenv/bin/accelerate launch oncoreasoning_training/fine_tune_llm.py \
  --data-dir "$RUN" \
  --output-dir ../models/oncoreasoning_gemma4_e4b_answer_first_v2 \
  --lora-rank 64
```

`prepare` and `build` are local data work. `generate` sends patient-bearing
prompts only to the explicitly supplied teacher endpoint; it does no web search.
Inputs outside `../data/no_phi` require `--confirm-inputs-are-non-phi`.
The entire GoodOption catalog is validated before requests are prepared.
Technically blocked or missing trial evidence fails preparation; a trial with
no scoreable drugs contributes TrialChecker questions only.

Candidate inputs must contain patient IDs (`pseudo_mrn` or `patient_id`), patient
summaries (`patient_summary` or `cancer_history_summary`), trial IDs (`nct_id` or
`trial_id`), and trial spaces (`this_space` or `clinical_space_summary`). Notes
default to `pseudo_mrn`, `date`, and `synthetic_note`; column flags can override
these. Repeated candidate questions are deduplicated; the same patient/drug
question is generated once across multiple trial spaces.

Explicit patient split assignments are shared across both task categories.
Missing assignments inherit the patient's known split or default to train;
conflicting assignments fail. Test patients are excluded. The integrated miner
preserves the original patient/trial co-splits. Older external mining tables
with a constant train field must first have their original splits restored.
Training and validation retain their observed category proportions, recorded in
`training_manifest.json`; examples are shuffled without oversampling. Use
`--tasks summarization` or `--tasks clinical_qa` for a single category, and
`--max-patients` / `--max-pairs` for small preparation checks.

Generation is resumable: rerun the identical command. Each successful final
answer is saved atomically. A dependent summary waits for its completed parent.
Changed teacher settings, prompts, inputs, or tokenizers cannot silently reuse
old shards. A stopped or malformed completion is retried three times; exhausted
requests stop the stage without saving an invalid target. Teacher thinking is
on by default and can be disabled with `--no-teacher-thinking`.

The trainer preserves prepared labels, uses gradient checkpointing and BF16
mixed precision where available, and projects only supervised positions to
vocabulary logits. Full fine-tuning is the default (`--lora-rank 0`); the example
uses LoRA to reduce trainable memory. Long-context training still needs suitable
GPU memory and, for full tuning, distributed sharding configured in Accelerate.
It is not launched by dataset preparation. Resume training only with the same
settings and an explicit `--resume-from-checkpoint PATH`.

## One-token preview

A trained QA prompt supports either one generated answer token or a longer
letter-plus-explanation response. This is an answer-first causal language model,
inspired by the requested System 1 behavior; it does not implement JEV's separate
calibration architecture. Quick mode still has to process the input prompt.

For a local smoke test, choose a clinical QA request ID from `requests.jsonl`:

```bash
CUDA_VISIBLE_DEVICES=3 ~/thisenv/bin/python oncoreasoning_training/preview_model.py \
  --model ../models/oncoreasoning_gemma4_e4b_answer_first_v2 \
  --requests ../data/no_phi/oncoreasoning_answer_first_v2/requests.jsonl \
  --request-id REQUEST_ID --quick
```

Omit `--quick` and set `--max-new-tokens 1024` for an explanation. Both modes use
the identical prompt, disable thinking, decode greedily, and restrict the first
generated token to `A` or `B`. The preview loads either a full text checkpoint or
the saved LoRA adapter. Runtime integration into the public inference package
and model publication are separate from this training workflow.
