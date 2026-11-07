# Synthetic training policy (user instruction, 2026-10-27)

Trial sample (user instruction, November 4): the production path is the CT.gov export, then the
ctgov rule (first posted since 2020, interventional, early phase I-IV), then the DFCI holdout
(trials open at DFCI 2025-2026), then the cohort workflow's cancer-treatment screen on every remaining
listing (`screen-trials`, `0b_screen_trials.py`; every answer and its reasoning saved), then a seed-42
draw of exactly 5,000 accepted trials (`sample-trials`). A synthetic protocol is generated for each of
the 5,000. Later attrition is reported from these counts; never sample from an already-filtered
older table or drop sampled trials without replacement silently. The first run on this path,
`training_run_screened_sample5000_20261104`, screened 29,005 listings (21,789 cancer treatment trials,
7,216 not, 0 errors) and drew 5,000 with seed 42; its synthetic protocols are generating.

Two trial versions (user instruction, October 31; the new default path): every
sampled trial is represented by its synthetic-protocol cohorts and by cohorts
extracted from the ctgov record its protocol was written from
(`0d_extract_ctgov_cohorts.py --trials-jsonl`), combined by
`synthetic_protocols/retrieval_cohorts.py`. Since November 3 (user instruction)
both versions go through the inference package's three-step workflow with the
same prompts: a cancer-treatment screen, the cohort names, and one request per
cohort for its complete criteria (schema `trial-cohorts-stepwise-v1`, rows
`trial-treatment-cohorts-v5`); a listing is its title, brief summary and
eligibility criteria as one document. Since November 4 (user instruction) the criteria step copies every criterion verbatim, boilerplate included; training then merges each trial's equivalent cohorts (the merge ignores differences of extraction) and removes the boilerplate from each merged cohort (`remove_boilerplate_criteria`, which returns the boilerplate separately in `cohort_boilerplate_criteria`; since late November 4 (user instruction) it splits the criteria into lines, classifies each line as boilerplate or not in its own request, and sorts the unchanged lines in code, so the model never re-transcribes criteria; broadened November 4 to prognosis, performance status, organ function, brain metastases, prior-treatment toxicities and non-cancer conditions unless central to the trial, then (user, November 4) unrelated other cancers, simultaneous participation in another trial, washout periods of 30 days or less from radiation, systemic or trial therapy, and recovery from recent major surgery; whether a treatment was ever received is not boilerplate). ctgov listings get the same steps as protocols. A cohort's key criteria are its criteria that are not boilerplate: the de-boilerplated tables keep `cohort_eligibility_criteria` (all criteria) and add `cohort_key_criteria` and `cohort_boilerplate_criteria`. Since November 5 (user instruction) the same per-line request also tags each line `inclusion` or `exclusion` (schema `trial-cohorts-deboilerplate-v4`), and every line's tags are kept in `cohort_criteria_items` (a JSON string), which the corpus keeps as each cohort's `items`. Retrieval, match labels and the decision model's match questions read the key criteria of both versions (option 2, user, November 4) and its boilerplate question reads the boilerplate criteria; since late November 4 (user instruction) patients are seeded from the merged protocol cohorts with all their criteria, boilerplate included (`cohorts.parquet` via `patient_cohorts.py`, which also writes the same rows with the key and boilerplate columns for retrieval); both steps are optional at inference and keep their reasoning. Mining represents each trial by one
version: chosen at random per patient-trial pair within each patient's 500-trial
sample (each mining round redraws that sample and those versions, user
instruction November 2; round 1 keeps its original draws), and per trial for
cohort queries; source-trial pairs include both
versions; a trial accepted in only one version uses it. Patients are generated
from protocol cohorts only. Do not revert to single-version retrieval without
the user's instruction.

Removed components (user instruction, November 5): OncoReasoning distillation (`oncoreasoning_training/`,
its pipeline stages and the peft/trl/bitsandbytes/liger dependencies), the training GoodOption catalog stages and
scripts (`train_good_option_checker.py`, `train_good_option_cosplit.py`; GoodOption stays in the inference
package, scored with LLMs), `14_check_boilerplate.py` and the real-time evaluations' optional LLM-boilerplate
stage that ran it (question-6 labels come from `trialspace_label.py --task boilerplate`), and the scoped
upgrade receipts (`scripts/register_{oncoreasoning,trialspace,downstream}_upgrade.py`, `pipeline_upgrade.py`):
a changed run identity now always needs a fresh run directory. Do not restore them without the user's instruction.
Also removed (user instruction, November 6): the legacy pre-cohort trial-space extractor
(`legacy_trial_space_extraction.py`, `trial_hierarchy.py`, `trial_hierarchy_runner.py`, `trial_prompts/`,
`trial_space_export.py`, `trial_treatment_screening.py`). The real-data evaluations build their trial line items
from cohorts instead: `real_eval_code/eval_cohort_lineitems.py` runs `0d_extract_ctgov_cohorts.py` (remote only)
and writes each cohort's key criteria as `this_space` and its boilerplate criteria as `trial_boilerplate_text`
(used by `pull_spacify_and_embed_ctgov`, `real_time_eval/spacify_trials_freshly.py` and the enrollment
evaluation's `spacify` stage).

Stage scripts (user instruction, November 6): every TrialSpace and decision-model stage is its own script
(`trialspace_prepare.py`, `trialspace_mine.py`, `trialspace_label.py --task match|boilerplate|criteria`,
`trialspace_project.py`, `finetune_embedder.py`, `train_decision_model.py prepare|train`; `trialspace_evaluate.py`
outside the plan) that writes `stage_receipt.json` last (`trialspace/stage.py`: an identity with no paths, hosts or
devices, outputs by name and byte size, provenance) and answers `--check` (0 complete, 10 not yet, 20 changed: use a
fresh directory). `train_from_summaries.py` only orders them (`--start-at`, `--stop-after`), so a stage can run on
another machine (for example mining on a cloud VM) and be adopted when its directory is copied back; the run-wide
fingerprint over git commits and source files is gone. Training uses the inference package's teacher labeler and
request pool (`matchminer_ai.labeling`, `matchminer_ai.llm.teacher_pool`, moved there unchanged: the label contracts
still hash the same, so earlier label directories resume), its encoder (`matchminer_ai.embedding.encoder`: one GPU,
several GPUs with one GPU's batches and vectors, or vLLM >= 0.32 `/v1/embeddings` servers) and its exact batched
search (`matchminer_ai.matching.retrieval`). `trialspace_training.py`, `trialspace_comparison.py`,
`remote_vllm_pool.py`, `adaptive_controller.py`, `label_workspace.py` and the labeler modules of `trialspace/` were
removed; training keeps its mining policy (trial sampling, lexical and random extras), candidate builders and fitting.

Sampling (user instruction, November 5): every teacher, generation and summarization script must be
Nemotron-aware. Resolve sampling from the served model's HF id through the inference package's profiles
(`vllm_reasoning_utils.resolve_sampling_defaults` / `sampling_from_args`, `apply_model_profile`, or a section
with `sampling_profile: auto`); never hard-code another family's values. Nemotron 3.5 Super gets NVIDIA's
sampling (temperature 1.0, top_p 0.95, top_k -1, min_p 0, repetition_penalty 1.0, thinking on) even in
inference sections whose presets fix greedy decoding (its profile's `replaces_task_sampling`). Free-text
JSON answers from Nemotron often lose the opening quote of a string field, so the JSON teacher tasks (match
labels and per-criterion labels) are decoded under their schemas (`trialspace.contracts.answer_schema`, vLLM
`structured_outputs`; part of each label contract, user instruction November 5); constrain any new one too.

Retriever, labels and checkers (user instruction, November 4): every ColBERT retrieval path is
removed from both repositories; TrialSpace (dense `google/embeddinggemma-2`) is the only retriever.
`colbert_experiments/` is now `trialspace/` without its ColBERT-only modules, and its October run
logs are in `docs/history/`. Patient summaries keep two sections: `patient_summary` is the key
section (the cancer history, before `Other relevant history:`) and `patient_boilerplate_text` the
boilerplate section. Retrieval embeds key texts only: the patient's key section and the cohort's key
criteria (finetune cap 4,096 tokens; prompt profile `trialspace`). The match teacher asks all five
match labels (reasonableness, cancer_type, burden, treatment_history, biomarker; each yes/no with
evidence_status and reason) in ONE request per pair with the inference prompt
`match_quality.labels.system.txt`, on the key texts; score 0-6 = 0 unless reasonable, else 1 +
cancer type + burden + treatment history + 2 x biomarker. The gated one-question-per-request design
(`potentially_eligible` first) is gone.

Decision model (user instruction, November 5): TrialChecker and BoilerplateChecker (ModernBERT) are
replaced in both repositories by one "System One" decision model, `LiquidAI/d1-2B` (revision
`652f920e`, LFM Open License v1.0), usable zero-shot and fine-tuned by `train_decision_model.py`
(full fine-tune with transformers' Trainer; no LoRA, Unsloth or TRL). Every question is yes/no
(`noul`), in seven families: (1-5) the five match labels on the key texts (one state, five
questions; `reasonableness` yes = potentially eligible); (6) boilerplate on the boilerplate sections
(yes = potentially eligible; the boilerplate teacher's `excluded` inverted); (7) one question per
criteria line on the complete summary and the cohort's complete numbered criteria, the question
restating the line's number, type and text. **Question 7 polarity: yes = the patient does not
clearly fail the line; no = the patient clearly fails it (an inclusion line clearly not met, or an
exclusion line clearly met).** Every round the teacher labels all seven families: match labels per
pair, boilerplate labels per new unique boilerplate combination (`label-boilerplate-N`,
`--skip-labeled` earlier rounds), and each tagged criteria line of a seeded
`--criteria-pair-fraction` (default 0.01) of the pairs (`label-criteria-N`,
`criterion_check.labels.user.txt`). d1-2B records `d1_attention: "set"` (the state read both ways
with its questions on full-attention layers), so training never appends the answer: **the loss is
the cross-entropy of the yes/no token at each answer slot only, never on prompt tokens**; questions
are packed after their state by the checkpoint's own serving code (16 per sequence, 10,000 tokens).
The states and question texts live in the inference package (`matching/decision_model.py`); the
reasonableness, boilerplate and per-criterion questions open "Is this patient potentially eligible
... If potentially eligible, answer yes; if clearly ineligible, answer no." vLLM 0.33 cannot serve
d1-2B (no LFM3 architecture; its d1 reader is causal and d1-3B only), so inference uses
Transformers. Do not restore ColBERT, the gated labels or the ModernBERT checkers without the
user's instruction.

HISTORICAL (October 26-31 comparison run state, kept as recorded): ColBERT was removed on
November 4, so mentions below of ModernColBERT files, ColBERT scoring or deferred ColBERT-only
labeling, gated labels and the deleted `scripts/trialspace_comparison_workflow.py` describe that run,
not current code. Its comparison, handoff and memory documents are in `docs/history/`
(`COMPARISON_GB300_20261026.md`, `HANDOFF_GB300_TRAINING_20261026.md`,
`GB300_MEMORY_RESULTS_20261026.*`, `colbert_experiments_README.md`). The standing policies among
these paragraphs still apply: all synthetic data with no holdouts, judgment-reuse integrity, no
idle cloud GPUs, file identity and the test policy.

STOPPED at user request (October 28): the batch-96 training run and its full
controller service were terminated so the user can make adjustments. Trainer,
launcher, controller and loader workers exited; GB300 is idle. Its last
checkpoint, 11,600 (epoch 0.39647), was deleted from the comparison
`recovery/user_stop_20261028_checkpoint11600/` at the user's request on
October 31 (record in that directory); the GB300 SSD copy may remain. Do not
resume this run. The schema-v2 corpus retrieves with base `google/embeddinggemma-2`.
See comparison `operator_stop.json` for shutdown/publication evidence. The unit
has `Restart=no`; its failed result reflects the deliberate termination. Do not
restart training or downstream retrieval/labeling without the user's revised
instructions. The ACTIVE RESTART description below is now historical.

Length defaults (user instruction, October 28): `finetune_embedder.py` and
`make_top_matches.py` default to a 3,000-token cap (0.4% of the 6,680 named
cohorts still exceed it), and the teacher label stage defaults to 32,768 output
tokens. The batch-96 restart below runs frozen source at 2,500 tokens. Label
`max_tokens` is part of the labeling identity: directories labeled at 8,192
resume or are reused only with an explicit `--max-tokens 8192`, which the
comparison workflow's label commands already passed (that workflow is deleted). Since November 4
the `finetune_embedder.py` default, which `make_top_matches.py` shares, is 4,096 tokens on key
texts only.

ACTIVE RESTART (October 28, 01:19 UTC): the user authorized a fresh start.
EmbeddingGemma 2 is training from the pinned base model, not checkpoint 200:
full loss batch 96, gradient-cache encoder chunks 48, token cap 2,500, BF16
parameters plus BF16 autocast, four persistent loader workers/prefetch two,
expandable CUDA segments. Keep known-positive masking (eligible score >=1),
CoSENT, proportional sampling, all source partitions, 3 epochs, LR 2e-5 and
1% warmup. Save every 200 optimizer steps; log every 25. Single-process
Accelerate uses GB300 physical GPU 1. All eight cloud VMs were live-verified
TERMINATED. Controller unit `mmai-trialspace-comparison-batch96-20261028.service`;
use comparison `launch.json` for identity. Old controller was killed without
SIGCONT. Old shared checkpoint/logs and original metadata are preserved in
`recovery/restart96_20261028/`; old SSD state is
`/tmp/mmai-training-restart96-20261028/previous_batch128_ssd`.
New SSD workspace reuses `/tmp/mmai-training/af03c2d69805c49782b23d47` with
fresh inputs/settings/checkpoints. Active source is frozen at
`/tmp/mmai-training-restart96-20261028/source`, published under comparison
`source/restart96_20261028/matchminer-ai-training-kehl`. Public model destination
is unchanged. Round-two retrieval uses the trained model with the 2,500-token
cap, then local/LAN-only labeling; ColBERT remains deferred. Earlier pause and
benchmark paragraphs below are historical. Saved v22 code is unchanged.

Data-loading audit (October 28, 01:02 UTC): the working `finetune_embedder.py`
now defaults to four native persistent DataLoader workers, two prefetched batches
per worker, pinned memory and nonblocking CUDA transfers. Inputs are already
SSD-staged, loaded into RAM, and dynamically padded. Loader-only benchmarks
delivered 1,752–1,810 pairs/s, with exact token/label/batch-order agreement versus
zero workers; ten existing training/prompt/resume tests passed. Training remains
paused at checkpoint 200. Its saved args and the frozen controller's source
snapshot still have zero workers; refresh the snapshot/settings deliberately
when resuming, never SIGCONT that controller. See comparison recovery
`dataloader_audit_20261028/`. Saved v22 code remains unchanged.

Latest diagnostic (October 28, 00:54 UTC): encoder chunks **48**, full loss
batch **96**, 2,500-token cap, native BF16 weights/autocast, expandable segments.
Retaining the four chunk graphs without replay OOMed before update 1 at
246.76 GiB. Gradient caching completed 25 updates (20 timed),
averaging **22.79 pairs/s**, **4.213 seconds/update**, peak
**212.43 GiB**. This successful option DOES replay the encoder.
MNRL uses native CachedMultipleNegativesRankingLoss, mini_batch_size=48;
CoSENT uses the existing CachedEmbeddingLoss wrapper. Runtime assertions verified
two MNRL 48x96 score blocks, CoSENT comparisons over all 96 pairs, and four initial
48-text forwards plus four replay forwards per update. Gradient accumulation is
1; these are full 96-pair comparisons, not two independent 48-pair losses.
Production remains paused at checkpoint 200. No configuration/weights were
promoted, and saved v22 code is unchanged. Benchmark-only stock MNRL does not
remove production known-positive masking. Artifacts and exact commands:
comparison run `recovery/encoder48_effective96_20261028/`.

Latest diagnostic (October 28, 00:46 UTC): actual encoder batch **64** with a
**2,500-token cap** also OOMed after 0 completed optimizer updates. This used
native v22 Trainer/losses, BF16 weights/autocast, one GB300 Accelerate worker,
expandable segments, no gradient cache and no activation checkpointing.
Failed input shapes: `[[64, 505], [64, 2500]]`; peak allocated memory was 247.10 GiB.
Production remains paused at checkpoint 200, and the saved v22 source is unchanged.
Evidence and exact command: comparison run `recovery/actual_encoder64_2500_20261028/`.

Latest diagnostic (October 28, 00:43 UTC): an actual 128-text encoder batch
was tested with the native v22 Trainer/losses, BF16 weights/autocast, one GB300
Accelerate worker, no gradient cache and no activation checkpointing. At the
2,500-token cohort limit it OOMed before its first optimizer step, even with
`PYTORCH_ALLOC_CONF=expandable_segments:True` (247.3 GiB allocated). With only
cohorts capped at 512 and that allocator setting, 25 steps completed: 82.69 pairs/s
after five warmups, 240.44 GiB peak. A matched actual-batch-10 control gave
29.42 pairs/s and 19.47 GiB: batch 128 improved throughput 2.81x at the same cap.
The default allocator failed after three capped steps with about 30 GiB reserved
but unused; expandable segments avoided that failure in this short test.
Production remains paused at checkpoint 200. Do not promote the artificial
512-token cap or infer full-corpus/long-patient stability from this benchmark.
The saved v22 script remains byte-for-byte and mtime unchanged. Artifacts and
commands are in the comparison run's `recovery/actual_encoder128_20261028/`.

Latest diagnostic (October 28, 00:31 UTC): an isolated copy of v22's
`finetune_embedder.py` was benchmarked on 8,192 sampled current pairs using its
original prompt, 2,500-token cap, batch 10 and native SentenceTransformers losses.
Forty timed steps after five warmups: EmbeddingGemma 2 direct launch 17.69 pairs/s;
Accelerate BF16 17.48; Accelerate BF16 with FP32 weights 17.06; original Qwen3
0.6B with Accelerate BF16 17.34. All completed with finite logged gradients.
The v22 source bytes and mtime were verified unchanged. This used current
libraries, not the historical v22 environment. Production remains paused at
checkpoint 200; no benchmark weights were exported and no production settings
were changed. Artifacts and exact commands are in the comparison run's
`recovery/v22_baseline_20261028/`. See docs/history/COMPARISON_GB300_20261026.md for details.

Latest diagnostic (October 28, 00:22 UTC): a benchmark-only 512-token cohort
cap leaves patient inputs at their 8,192-token maximum (dynamic padding).
Without gradient caching, retained chunks or activation checkpointing, batch 128
completed only one measured ordinary batch at 42.2 pairs/sec before OOM. Batch
64 completed 20 ordinary batches at 24.2 pairs/sec, then OOMed on the first
longest-observed-patient stress batch. Stock CoSENT alone averaged 21.2 pairs/sec;
the MNRL subclass with known-positive masking averaged 28.2. These are short,
unstable benchmark configurations, not a full-run ETA or production settings.
A separate 8-pair stock-CoSENT profile used BF16 memory-efficient SDPA kernels
(no math fallback observed) and retained about 13.4 GiB of unique saved activation
storage, with many FP32 normalization/residual/rotary intermediates. Current
loading explicitly requests FP32 master weights; v22's saved weights are BF16.
Their performance impact has not been isolated. Production still uses the standard
SentenceTransformerTrainer, with custom optional loss wrappers; no custom Trainer
or optimizer loop. The timing harness alone drives forward/backward manually.
Evidence is published under `recovery/cohort512_20261028/` in the comparison run.
Production remains paused at checkpoint 200; do not apply the cap or restart it.

Current status (October 28, 00:01 UTC): the user stopped training to test
execution without encoder replay. Training remains **stopped** after the tests;
checkpoint 200 is retained at
`/tmp/mmai-training/af03c2d69805c49782b23d47/checkpoints/checkpoint-200`.
The ordinary 128-pair forward ran out of memory at about 247 GiB. Length-sorted
encoder chunks retaining all graphs also ran out of memory at 128, on ordinary
batch 6. The retained 64-pair benchmark passed both losses and full 8,192-token
stress inputs, peaking at 197.5 GiB allocated, but averaged only 11.4 pairs/sec
on ordinary batches. No replay and no activation checkpointing were enabled.
The preceding cached-128 benchmark was 13.3 pairs/sec; the smaller stable
configuration did not improve throughput. Do not silently restart the cached run.

The optional no-replay implementation is `RetainedEmbeddingLoss` in
`trialspace_embedding_losses.py`; use `--loss-mini-batch-size 0
--encoder-mini-batch-size 8 --encoder-token-budget 4096` plus the selected full
loss `--batch-size`. It encodes each text once, retains all activation graphs,
restores pair order, and computes the loss over the entire sampled batch.
Six gradient/encoder-call checks and ten tiny-model training/prompt checks passed.
Benchmarks, process identities and resume information are preserved in the
comparison run's `recovery/no_replay_20261028/`. All eight cloud VMs remain off.

The controller recorded in `launch.json` (PID 205256) is SIGSTOP-frozen; its old
training child has exited. CPU label publication was left running. Resume needs
a deliberate controller handoff and preserved checkpoint/settings migration;
**do not SIGCONT the old controller**, which would see its terminated child as a
failure. The prior training configuration below is historical, not a live run.

Latest batch update (October 27): EmbeddingGemma 2 uses **128 pairs per optimizer
step**, with both ranking losses computed over that full batch. Encoder chunks
of eight use gradient caching with RNG/autocast replay; texts are sorted by
length only within each sampled batch and restored to original pair order before
the loss. This is not gradient accumulation of independent eight-pair losses.
MNRL masks off-diagonal pairs known to be eligible (score >= 1), including repeated
patient/cohort text identities, so those are not false negatives. CoSENT, the
positive threshold, proportional dataset sampling, all-row coverage, three epochs,
learning rate 2e-5 and 1% warmup remain the same. No synthetic holdouts.

Live flags: `--batch-size 128 --loss-mini-batch-size 8 --mask-known-positives
--no-gradient-checkpointing --save-steps 200 --logging-steps 25`. Checkpoint and
logging intervals count optimizer steps, not encoder chunks. The initial batch-10
run was stopped before its first checkpoint; its logs and SSD state are retained
under the comparison run's `recovery/larger_batches_20261027/` audit and the
referenced local archive. The larger-batch run starts a fresh optimizer schedule
and retains the original public model/checkpoint destinations. The controller
unit is `mmai-trialspace-comparison-batch128-20261027.service`; consult `launch.json`
for current identity. All cloud VMs remain off; the round-two local/LAN plan holds.

Production uses **all synthetic patients and cohorts** for retrieval, labeling,
and optimizer updates. Do not reserve synthetic validation/calibration/test
partitions or accept a train-only override. Historical partition assignments
are provenance (`source_split`), never candidate filters or training exclusions.
Both retrieval directions must search the full opposite-side corpus. At the
October 5,000-trial sample this was 66,800 patients and 6,680 extracted cohorts.

The source-partition-restricted comparison rankings are invalid and archived.
Rebuild both directions: the old implementation filtered cohorts for a patient
as well as patients for a cohort. Unchanged encodings can be reused only after
verifying complete entity/text/model identity; never reuse those rankings.
Completed judgments may be imported only for current candidate pairs after
verifying clinical text, exact prompts, teacher settings and response integrity.
Preserve complete original prompts, answers, reasoning and old artifacts.

Historical (October 27 comparison; ColBERT removed November 4):
Keep separate EmbeddingGemma 2 and ModernColBERT candidate and label files.
Overlapping pairs are judged once and copied into both exports. Train
EmbeddingGemma 2 only on its own candidates, using all source patients.
Execution order (user update, October 27): finish EmbeddingGemma-only inference
and immediately shut down all eight cloud VMs while label exports continue.
Export its independent training pairs and verify all VMs are TERMINATED, then
train and assess EmbeddingGemma 2 on the GB300 **before ColBERT-only labeling**.
Latest plan (October 27): after round-one EmbeddingGemma 2 training and its
in-sample diagnostics, retrieve again with the **round-one trained model** over
all 66,800 patients and 6,680 cohorts in both directions, then label those round-two
candidates using **only the GB300 and authorized LAN teachers**. Keep all eight
cloud VMs TERMINATED throughout round-two retrieval and labeling; do not restart
the cloud fleet after round-one training. ColBERT-only labeling is deferred.
Round-two outputs are separate under `round2_embeddinggemma2/`; round-one and
ColBERT artifacts remain intact. Reuse only intersecting pair judgments after
verifying unchanged corpus, prompts, teacher settings and full response integrity.
Keep original answers/reasoning unchanged and physically include reused scores
in round-two exports. Stop after round-two label exports for review unless the
user authorizes the next optimizer pass. Active retrieval/label state stays on
local SSD and final products are verified at the shared no_phi destination.

Latest storage update (October 27): training-pair preparation and EmbeddingGemma
optimizer work use local SSD too. A complete `scores_ready.json` with matching
corpus, candidates, teacher manifest and file stamps can release local training
before the large reasoning archive finishes publication. Copy those immutable
inputs into an independent SSD workspace before the labeler can clean its own
workspace. Keep the label publisher supervised; join and verify its publication
before diagnostics/shared-artifact consumption or round-two judgment reuse.
Publish independent training-pair tables concurrently with fitting. Optimizer
checkpoints, input Parquets, model cache and final weights stay on SSD during
training; copy retained checkpoints and final weights back to the original
shared no_phi destinations with size checks (no SHA256 readback). Retain local state for resume
on failure. All eight cloud VMs must be live-verified TERMINATED before training.

Never keep cloud GPUs running for CPU-only export, copying, reports or local training.
Metrics over these synthetic training examples are in-sample diagnostics,
not held-out validation or full-corpus recall estimates. This policy does not
remove independent real-data evaluation or DFCI trial exclusion rules.

Historical (October 27; ColBERT removed November 4). Operational request at the
time: shut down the eight RTX Spot VMs until corrected
candidates are ready for labeling, then restart them with preemption/startup
retries. Verify all eight are TERMINATED again before student training.
Immediately after ColBERT scoring completes, restart the registered GB300
vLLM teacher on port 8000 (nvidia/Gemma-4-31B-IT-NVFP4), verify its health/model,
and add it back to the live labeling registry with a concurrency ceiling of 768.
The controller verifies completed local retrieval products and sufficient free
GPU memory, then restores the teacher while shared-storage publication continues.
Keep the publisher supervised and require verified publication before consuming
its outputs. Do not wait for copy-back or dense labeling to return the GB300.
Use local SSD for active preparation, retrieval and label checkpoints; publish
and verify final products at their intended shared no_phi destinations.
Read docs/history/COMPARISON_GB300_20261026.md and
docs/history/HANDOFF_GB300_TRAINING_20261026.md for paths.

Latest pool update (October 27, 18:28 UTC): the user restored Camus ports 8000
and 8001. All four Camus ports 8000–8003 are authorized for labeling, through
loopback alias proxies 18100–18103. Keep the old retirement cutoff disabled.

File identity and test policy (user instruction, October 27): never hash whole
data files or read copies back to verify them; that is slow on NFS. Use
`trial_cohort_format.file_stamp` (size plus whole-second mtime) for same-path
change detection, compare sizes for copies (preserve the source mtime when the
stamp must survive the copy), and keep content hashes only for in-memory values
and small source files. `tests/` holds only the critical prompt, parsing,
masking, CSV-literal, pipeline-wiring, scoring-contract and LLM-client tests; do
not add tests for operational scaffolding, supervisors, fleets or receipts.
