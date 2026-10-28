# Synthetic training policy (user instruction, 2026-10-27)

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
`recovery/v22_baseline_20261028/`. See COMPARISON_GB300_20261026.md for details.

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
current 5,000-trial sample this is 66,800 patients and 6,680 extracted cohorts.

The source-partition-restricted comparison rankings are invalid and archived.
Rebuild both directions: the old implementation filtered cohorts for a patient
as well as patients for a cohort. Unchanged encodings can be reused only after
verifying complete entity/text/model identity; never reuse those rankings.
Completed judgments may be imported only for current candidate pairs after
verifying clinical text, exact prompts, teacher settings and response integrity.
Preserve complete original prompts, answers, reasoning and old artifacts.

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

Current operational request: shut down the eight RTX Spot VMs until corrected
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
Read colbert_experiments/COMPARISON_GB300_20261026.md and the handoff for paths.

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
