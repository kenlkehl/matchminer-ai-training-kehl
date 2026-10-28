# Synthetic training policy (user instruction, 2026-10-27)

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
