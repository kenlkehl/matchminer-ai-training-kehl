# Current correction: full synthetic pool (2026-10-27)

Data-loading update (October 28, 01:02 UTC): working-repo dense training defaults
to four native persistent workers with prefetch factor two, pinned memory and
nonblocking CUDA copies. Loader-only measurement: 1,752–1,810 pairs/s; tokens,
labels and sampled order match the zero-worker loader exactly. Ten existing
training/prompt/resume tests pass. Production remains paused; the old source
snapshot/checkpoint args still use zero workers. Refresh source/settings during
the deliberate controller handoff, not by resuming the frozen controller.
Evidence: comparison `recovery/dataloader_audit_20261028/`. v22 is unchanged.

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

Latest storage update (October 27): training-pair preparation and EmbeddingGemma
optimizer work use local SSD too. A complete `scores_ready.json` with matching
corpus, candidates, teacher manifest and file hashes can release local training
before the large reasoning archive finishes publication. Copy those immutable
inputs into an independent SSD workspace before the labeler can clean its own
workspace. Keep the label publisher supervised; join and verify its publication
before diagnostics/shared-artifact consumption or round-two judgment reuse.
Publish independent training-pair tables concurrently with fitting. Optimizer
checkpoints, input Parquets, model cache and final weights stay on SSD during
training; copy retained checkpoints and final weights back to the original
shared no_phi destinations with SHA256 readback. Retain local state for resume
on failure. All eight cloud VMs must be live-verified TERMINATED before training.


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

This section supersedes source-partition and runtime claims in the historical
entries below. User policy is permanent in `AGENTS.md`: no synthetic train/cal/
validation/test holdouts in production retrieval, labeling, or optimization.

The corrected corpus contains 66,800 patients and 6,680 cohorts. Active `split`
is `train`; `source_split` preserves original assignments. Both old candidate
rankings were partition-restricted and must be rebuilt. The old outputs are in
`archive/split_restricted_20261027/` under the comparison run. The portable
label snapshot preserves **151,728 completed pair scores and 237,266 full
question responses**; it is a complete storage snapshot of unfinished labeling.
New labeling imports only current pairs with identical clinical text, rubric,
teacher settings and verified responses. Encodings are reused after exact
identity/hash checks; rankings are never imported.

All eight cloud VMs were verified TERMINATED at the user's request. The new
controller restarts the fleet only after corrected dense retrieval finishes,
then supports Spot retries throughout labeling. It verifies shutdown again
before EmbeddingGemma 2 training. The new unit is
`mmai-trialspace-comparison-ssd-training-20261027.service`; use the run's `launch.json`
and `status.json` for actual live identity/state. Recovery audits/source backups
are in `recovery/full_pool_20261027/`. The old controller and split-restricted
ColBERT miner have stopped. Summaries and generation are complete and unchanged.

User reminder, October 27: after the active full-pool ColBERT scoring finishes,
restart the GB300 teacher on port 8000 and return it to labeling immediately.
The controller's `mine_colbert_and_restore_teacher()` verifies complete local
products and free GPU memory, then restores the teacher while shared copy-back
continues. It joins and verifies publication before downstream consumption.
Do not wait for dense labels. It launches the pinned
NVIDIA Gemma 4 31B IT NVFP4 model, verifies health/model readiness, and publishes
`http://127.0.0.1:8000/v1` to the live registry with a 768 concurrency ceiling.
Latest pool instruction, October 27 at 18:28 UTC: Camus 8000/8001 are restored
at the user's request. All four Camus ports are in the labeling registry via
18100–18103. The returning pair starts at concurrency 32, ceiling 128 per server.
Do not rearm the old retirement cutoff. This supersedes earlier removal notes.

The publication handoff preserved labeler PID 3884183, publisher PID 3884301,
all eight SSH tunnels and running VMs. New controller PID: 4000615; fleet PID:
4000985. Audits/backups are in `recovery/retrieval_publication_overlap_20261027/`.
Copy-back finished at 15:57:40 UTC during the handoff, before teacher startup;
the overlap capability is installed but did not overlap the teacher in this run.

Subsequent teacher startup exposed a missing virtualenv bin directory in the
service PATH (`ninja` was installed but not discoverable). Failure cleanup stopped
labeling and the eight VMs. Recovery under `recovery/teacher_path_20261027/`
resumes the existing SSD checkpoint and restarts the fleet. Current PIDs:
controller 4006530, labeler 4007097, GB300 teacher 4007111, fleet 4007512.
The production teacher launcher preserves its virtualenv bin directory in PATH;
the active configuration explicitly includes CUDA_HOME and the CUDA compiler bin.
Resumed labeling starts before waiting for teacher readiness. Consult current
status and registry files rather than assuming startup has completed.

See `COMPARISON_GB300_20261026.md` for current artifacts and policy.

---

# GB300 ColBERT training optimization handoff

Prepared 2026-10-26 from Camus for a Codex session running directly on
`sn4622130540`. This is an implementation/debugging handoff, not a published
model release. Paths below assume the same shared mount.

## Labeling moved to SSD (latest, 2026-10-27)

The dense labeler was paused at 12:53 UTC to move its hot working set off the
shared mount. The verified snapshot preserves 24,909 committed pair scores and
37,312 full question responses. Every committed score's question keys were
checked against the compressed SQLite database; response bytes were verified
during import. Original shared checkpoints remain intact. The snapshot is
`/tmp/mmai-label-recovery-20261027/state/`, and the active workspace is
`/tmp/mmai-labels/9165dcaf456d4b0c42622fdb/`.

The replacement controller is
`mmai-trialspace-comparison-label-ssd-20261027.service`; `launch.json` and
`processes/labeling_dense.json` contain current identities. It adopts the exact
unchanged ColBERT miner, PID 3799303/start ticks 540389493. The miner never stopped.
Fleet handoff preserves all eight SSH control masters and the remote vLLM workers;
the supervisor remains responsible for Spot recovery and final VM shutdown.
Recovery configuration/source backups and audits are under
`recovery/label_ssd_20261027/`. Earlier service names below are historical.

The normal `trialspace_training.py label` entrypoint now uses
`label_workspace.py`: corpus, candidates, reused labels, response checkpoints,
score journals, and metrics stay local. Responses use zlib-compressed SQLite
with durable commits and content hashes, preserving complete prompt/answer/
reasoning objects. Status is mirrored outside the request loop. Final
`labels.jsonl` streams to the original shared destination; the portable database,
scores, and metadata are copied back with checksum verification before scratch
cleanup. `storage_complete.json` records publication. Failed runs retain scratch.
The later union labeler uses this same production path and reads only the active
published response products, avoiding a second copy of retired JSON checkpoints.

Validation covers concurrent durable writes, corruption/disk reserve failures,
legacy/SQLite reuse, complete reasoning export, interrupted resume, copy-back
failure, prompt identity rejection, PID reuse, GPU ownership, and verified miner
completion. The normal labeling entrypoint was exercised end to end. Clinical
rubrics, candidates, denominators, sampling, and separate model label exports
are unchanged. The GB300 teacher's absent endpoint was removed only from the
temporary tuning target list while retrieval owns the GPU; its 768 concurrency
ceiling remains configured for its return.

## Retrieval overlaps summary archive publication (2026-10-27)

The user requested overlapping retrieval and labeling with copy-back. The active
comparison config now enables `overlap_summary_publication: true`. The controller
waits until the summary callback has closed its final patient Parquets, retains a
local snapshot, checks complete patient/metadata coverage, and publishes the
smaller raw/attached patient tables first. It then prepares the full corpus and
runs exhaustive EmbeddingGemma 2 retrieval on the GB300 while the original summary
process continues archiving and publishing all traces/checkpoints. Dense labeling
starts when its candidate set is complete, alongside ColBERT retrieval. No smaller
patient denominator or partial corpus is used.

`summary_retrieval_ready.json` records the exact early input hashes and whether
archive publication remains pending. The original summary pipeline stays under
supervision; failures stop the workflow. Before final union labeling, the
controller joins that pipeline and verifies the canonical published patient
products against the early retrieval hashes. All final outputs still go to the
original shared destinations. Cloud shutdown is still verified before training.

The new unit is `mmai-trialspace-comparison-overlap-20261027.service`; read
`launch.json` for its current PID. It adopted the unchanged running upstream
pipeline PID 3739646 and summary process PID 3739718. The handoff moved service
ownership of these processes and the eight live SSH control masters; generation
and active network requests were not restarted. Temporary holding/bridge units
are removed after fleet reattachment. Config/source backups, process identities,
and test evidence are under `recovery/summary_publication_overlap_20261027/`.

New reusable modules are `summary_overlap.py` and `adopted_pipeline.py`. The
latter requires exact PID/start-time/command identity and verifies upstream
completion receipts/output hashes; a dead process alone is never success.
25 targeted tests passed, including full-denominator gating, unchanged final
products, adopted-process failures, stage order, and copy-back failure barriers.
A real isolated service-group transfer test also confirmed that adopted children
survive stopping their former parent service.

## Summary checkpoint recovery (2026-10-27)

The shared mount blocked summary checkpoint I/O while all 72 inference servers
were healthy and idle. The normal `6_summarize_patients.py` entrypoint now uses
`summary_workspace.py` to keep prepared chunks, attempts, chunk checkpoints,
metrics, and output assembly on local SSD. Shared status mirroring runs outside
the asyncio request loop. Final Parquets retain the original `prepared/` paths;
traces/checkpoints are bulk-published as `summaries/checkpoints.tar.gz`, with
hash/row verification recorded in `storage_complete.json` before SSD cleanup.
Failures retain scratch for resume. 39 summary recovery and pipeline tests passed.

The old workflow was frozen and its 70,060 immutable files verified onto SSD:
33,574 chunk checkpoints, 36,485 attempt traces, and the prepared-patient cache.
Valid orphaned attempts are reusable only with exact prompt/sampling matches;
legacy attempts also require the verified template/context/budget recovery
contract. The snapshot and contract live in
`/tmp/mmai-summary-recovery-20261027/`. The original shared checkpoints remain
in `upstream_local_scratch_20261027/summaries/`.

The new upstream directory is `upstream_summary_ssd_20261027/`. Active checkpoint
storage is `/tmp/mmai-summaries/a40e75c33194e9009b8c1932/checkpoints/`; its live
`status.json` and `concurrency_status.json` are also mirrored to the new shared
`summaries/` directory. Read `launch.json` for current process identities. The
new unit is `mmai-trialspace-comparison-summary-ssd-20261027.service`.

At 2026-10-27T11:04:32.019027+00:00, all 72 endpoints were actively generating. Resume reused
33,574 chunk checkpoints and recovered 2,910 valid orphaned attempts. A further
161 fresh summaries were already saved locally; three fresh traces passed
identity, finish-reason, and quality checks. Shared progress mirroring was
verified. The detailed receipt is `recovery/summary_local_ssd_20261027/verified.json`.
Recovery configs, source backups, and receipts are under
`recovery/summary_local_ssd_20261027/` in the comparison directory.

The fleet supervisor was rebound through a bounded temporary bridge without
stopping VM inference workers. It retains its Spot retry logic and automatic
shutdown after final union labeling or workflow failure. Cloud shutdown is still
verified before training. The GB300 teacher and all LAN servers were preserved.

## Retrieval on local SSD (2026-10-27)

The user authorized moving retrieval work off the shared filesystem. Both
`trialspace_training.py mine` and `trialspace_comparison.py dense-mine` now
stage the corpus, vectors/token shards, query checkpoints, and exports in a
persistent private workspace under `/tmp/mmai-retrieval/`. The default reserves
16 GiB free disk; actual network scratch mounts are rejected. CLI overrides are
`--retrieval-scratch-dir`, `--retrieval-min-free-gb`, and `--no-local-retrieval`.
Single-host distributed ranks share one workspace and publish through rank zero.

All resulting files, including token shards and query checkpoints, are copied
back to the original `candidates/{embeddinggemma2,colbert_modernbert}/` directories.
Every copy receives SHA-256 readback verification and an atomic rename. The
native `complete.json` is published last, before scratch cleanup. Failures retain
local work for resume; input/settings changes are rejected. Each candidate
directory has `retrieval_status.json` for the local workspace and current phase,
and `retrieval_storage.json` records all published file hashes.

The live controller does not need restarting: it will launch the updated frozen
retrieval entrypoints after summaries finish. The source overlay and archived
originals are recorded in `retrieval_storage_overlay_20261027.json` and its
recovery directory, including test results and unchanged active-source hashes.
At that deployment, active summary sources and process identities were unchanged;
the later summary recovery above supersedes those identities. Model revisions,
retrieval scoring, source partitions, candidate counts, and label exports are unchanged. Local disk had about 233 GiB available before retrieval starts.

## Cloud resumed by user (latest, 2026-10-27 09:30 UTC)

The user explicitly requested bringing VMs back into the active pool. The fleet
supervisor was relaunched as PID 3663563/start ticks 539493875 and rebound to the
current comparison controller PID 3609705/start ticks 538751693. Read the live
fleet `launch.json` for authoritative identities. The old STOP and shutdown
receipts were archived under fleet `recovery/resume_20261027T093012Z/`; STOP is
now cleared. The comparison receipt is `cloud_resume_20261027.json`.

The same eight allowlisted RTX Pro 6000 x8 Spot VMs and frozen NVIDIA Gemma 4
31B IT NVFP4 worker configuration are used. Unavailable/preempted VMs retry at
five-minute intervals; healthy per-GPU endpoints automatically join the live
registry while all eight local/LAN servers continue working. This resumed
fleet stays attached through final union labeling and shuts down before student
training, or on workflow failure/controller lease expiry. Initial resume state
is warming; consult fleet status and live summary metrics for admitted workers.
This authorization supersedes the stop-only instructions below.

At 2026-10-27T09:37:28.577463+00:00 the fleet had 40 healthy cloud
GPU servers; summary metrics showed 24 admitted cloud endpoints, 2304
cloud requests in flight, and 0 successful cloud summaries. GCP confirmed
`rtxpro` and `rtxpro-2` were immediately preempted after the initial starts;
five-minute retries remain active. Live counts are in
`cloud_resume_progress_20261027.json` and the fleet/summary status files.

## Local scratch recovery (latest, 2026-10-27 07:26 UTC)

The user authorized restarting preparation on local SSD, then summarizing on
the eight non-cloud endpoints. They explicitly require all finished products
at the original shared `data/no_phi/` destinations. The comparison controller
was restarted as PID 3609705/start ticks 538751693; use `launch.json` for the
current identity. Its upstream directory is now `upstream_local_scratch_20261027`.
Recovery receipts, original config/source, and the incomplete old SQLite DB
are retained in `recovery/local_scratch_20261027T0724Z/`.

Production `prepare_synthetic_training.py` now performs SQLite/date/output
preparation in a private temporary directory. `run_pipeline.py` forwards
`--note-preparation-scratch-dir` and `--stage-note-preparation-input`. This run
uses `/tmp/mmai-note-preparation-20261027/`. Completed notes and patient metadata
are copied back to `prepared/`, verified by SHA-256 readback and Parquet row
count, and atomically published before temporary files are deleted. Failed
publication retains local products. Progress is in
`prepared/all_synthetic_notes.preparation.status.json`; durable checksums and
counts are in `prepared/all_synthetic_notes.preparation.json` on completion.
All 49 preparation/pipeline/cohort tests passed, including content/date/split
parity and failure/corruption handling. The frozen source snapshot has an
explicit two-file overlay; the old source/manifest is in the recovery directory.

Preparation completed in 355.7 seconds at 07:32 UTC, including input staging
and verified publication to the shared mount. The final notes contain 1,137,527
rows; patient metadata contains 66,800 patients. The private scratch directory
was removed only after both products were verified. Patient summarization then
started (child PID 3613070); as of 07:47 UTC it is still concatenating and
tokenizing patient histories before sending requests. Read the current
`upstream_local_scratch_20261027/logs/summarize-patients.log` and eventual
`summaries/status.json` for progress. All summaries/checkpoints use shared paths.

All eight local/LAN endpoints passed model/context checks. Initial summary
concurrency is 128 on GB300 and 32 per 172.24.216.113/Camus endpoint, with
adaptive ceilings 768/256/128. At 09:25 UTC the four requested Camus caps were
corrected from 16 to the controller minimum 32; the invalid cap had rejected
the entire tuning document. The live JSON was validated with `apply_tuning`
and updated without restarting inference; the previous configuration and
scheduler state are in `recovery/concurrency_minimum_20261027T0925Z/`. The cloud fleet remains stopped;
do not restart it without a new instruction. Other comparison settings,
including separate pair exports and EmbeddingGemma-only training candidates,
are retained.

At 09:23 UTC summarization was active on all eight endpoints, with 9,888 of
66,800 patient chunks complete (one chunk per patient). No request or
validation errors were recorded. Recent 30-minute throughput was about 7,400
patients/hour, roughly 7.7 hours remaining at that pace. Shared chunk
checkpoints and response traces are under the active upstream `summaries/`.

## Cloud stopped by user (2026-10-27 07:14 UTC)

The user requested stopping the VMs while `prepare-notes` was blocked on
network-mounted SQLite I/O. All eight allowlisted RTX VMs in `profile-notes`
and `kehllab-caia-v2` were verified `TERMINATED`. The fleet `STOP` marker is
present and its service is inactive; do not restart cloud workers without a
new user instruction. Receipt: comparison run `user_vm_stop_20261027.json`.
The comparison controller and current note preparation were left running.
All 1,137,527 raw notes were completed/assembled at 06:17 UTC. LLM patient
summarization had not started as of 07:14 UTC. The scratch database is under
`prepared/all_synthetic_notes.preparation.sqlite` on the shared mount; the
existing adapter deletes/recreates that scratch database on startup, so merely
copying it does not provide resume support.

## Queued comparative experiment (latest, 2026-10-27 00:35 UTC)

User service `mmai-trialspace-comparison-20261026.service` is waiting for the
current notes to finish. Its run directory is
`data/no_phi/trialspace_comparison_namedcohorts_5000_20261026/`; read its
`config.json`, `status.json`, and `launch.json` for authoritative commands and
process identities. Initial controller PID 3230485/start ticks 536278864.
Full plan and artifact contract: [COMPARISON_GB300_20261026.md](COMPARISON_GB300_20261026.md).

The fleet is now attached to this workflow in **labeling mode**, keeping cloud
inference alive through summaries, exhaustive GB300 retrieval for both models,
and shared-union labeling. Its supervisor-only handoff preserved all 64 healthy
cloud servers and active note generation. New fleet PID 3231589; verify its
current launch receipt before signaling. The temporary handoff override was
removed and normal `KillMode=control-group`/30-second restart delay restored.
Receipts are in the comparison's `fleet_handoff/` directory. Cloud shutdown is
required after final label inference, with live GCP termination verified before
student training. Earlier note-completion shutdown descriptions below are historical.

Separate `pair_labels/embeddinggemma2/` and `pair_labels/colbert_modernbert/`
exports are mandatory; overlaps appear in both with identical shared judgments.
Each includes all pairs plus patient-centric and cohort-centric Parquet files
and `scores.jsonl`. The union and reasoning traces remain available separately.
The user selected **only EmbeddingGemma 2 candidates** for its training;
`training_candidates.json` records this choice and is read when training starts.
Both retrievals preserve source partitions and search every document in each
partition, with identical top-k/exploration settings and clinical instruction.

Camus 8000/8001 rejoined at 00:08 UTC; the user cancelled their cutoff. All four
Camus ports, 172.24.216.113 ports 8001–8003, and GB300 8000 remain in the pool.
Do not rearm the old retirement timer. The comparison freezes source snapshots
from both repos. New comparison changes are in the training working tree;
inference remains at 9b1bf81. Existing note generation was not restarted.

## Fresh synthetic generation (latest request, 2026-10-26 19:25 UTC)

The user subsequently authorized fresh cohort-seeded patient generation through
seeds, event lists, and synthetic notes, using the six existing endpoints and a
live pool for future additions. Production `run_pipeline.py` is running only the
`generate-synthetic` stage in
`data/no_phi/synthetic_namedcohorts_5000_gemma4_31b_20261026/`.
After checkpoint recovery at 23:07 UTC, the coordinator launch PID is 3148418
and generation child PID is 3148482;
verify their recorded start identities in `launch.json` and
`generator_process.json` before signaling them. The run stops after raw synthetic
notes; summarization, retrieval, labeling, and model training remain paused.

The run has a verified, independent copy of the latest seven cohort exports and
receipts, retaining source hash `b19a836fe402b509cef9713f53c27b9c63ac5bc9877e5d3a01e9cb7978a0778e`.
All 6,680 cohort criteria match the promoted extraction exactly, including the
separate name/criteria representation and original single-cohort eligibility.
Five matching and five near-miss patients per cohort give 66,800 patients from
13,360 seed batches. Each saved seed batch precedes its individual patient
histories; notes start after all histories validate. No archived patient
checkpoints are reused. Source archives and provenance capture training
`609ecd5` and inference `9b1bf81`.

Pool: GB300 port 8000, `172.24.216.113:8001–8003`, and Camus ports 8002–8003. The four
Camus alias proxies use loopback ports 18100–18103, but 18100/18101 were removed
from the registry at 23:06 UTC after both upstream ports refused connections;
their configs and launch identities are under `pool/`. Initial adaptive targets
are 256 for GB300, 64 for 172.24.216.113, and 32 per Camus endpoint; configured
ceilings are 768, 256, and 128 respectively. `servers.json` and `concurrency.json`
reload live, with `servers.fleet.lock` for coordinated edits. The pending queue
is 8,192; per-server limits control actual requests.

At 22:13 UTC the user authorized attaching all eight RTX Pro 6000 x8 Spot VMs
in `profile-notes` and `kehllab-caia-v2` through both event and synthetic-note
generation. The current controller/configuration is
`gcp_generation_fleet_20261026/` inside this generation run, managed by user
service `mmai-synthetic-gcp-fleet-20261026.service`. It uses the existing tested
`scripts/gcp_note_fleet.py` and `scripts/gcp_note_worker.py`, with frozen copies
and source hashes in that fleet directory. Each GPU serves
`nvidia/Gemma-4-31B-IT-NVFP4` as `gemma4-31b`, with the pinned model revision,
container digest, and tokenizer template verified against the active run.
Cloud endpoints use SSH-forwarded loopback ports 22400–22477 (eight ports per
ten-port VM block), initial concurrency 64 and configured ceiling 256 per GPU.
Only healthy endpoints join the shared registry. VM starts retry every 300
seconds; preempted workers recover through startup metadata and container
restart policies. A VM-local 900-second controller lease guards against orphans.
The controller watches this run's generation PID/start identity and status;
it stops all eight VMs after note inference completes (`assembling`), or if
generation fails/exits. No downstream summarization/labeling workflow is attached.
Consult fleet `status.json` and `vm_status.json` for current capacity rather than
assuming all 64 GPUs started. Fleet regressions passed (29 tests) before launch.

The user also scheduled Camus ports 8000 and 8001 to stop accepting new prompts
at 18:50 Pacific on October 26 (October 27 01:50 UTC). User timer
`mmai-retire-camus-8000-8001-20261026.timer` removes their aliases from the live
registry, with up to 15 seconds for reload; in-flight requests finish normally.
Its script and receipt are in `scheduled_actions/`. Both endpoints were already
removed for connection failures at 23:06 UTC; the scheduled retirement remains
idempotent. Fleet registry merges preserve that removal and all unrelated endpoints.

At 23:06 UTC event generation saved 66,796/66,800 histories with all 13,360 seed
batches complete, then exited because four histories exhausted transport retries
on the offline Camus endpoints. No validated history was discarded. Recovery
receipts and the old failure/status files are under `recovery/20261026T230655Z/`.
The identical pipeline command resumed against the six healthy existing endpoints.
The fleet safety shutdown completed, then the same fleet service was relaunched
at 23:07 UTC with its configuration bound to the new generator PID/start identity;
Spot retry and automatic shutdown at note-inference completion remain enabled.
At 23:15 UTC all four retries succeeded: 66,800 histories saved, all 13,360 seed
batches complete, zero unresolved items. The worker is exporting events before
preparing notes. Retired Camus entries were also removed from concurrency tuning
overrides so remaining-server tuning validates after resume.

Use `status.json` for saved seed/history/note counts, `concurrency_status.json`
for endpoint activity, and `pipeline/logs/generate-synthetic.log` for the worker
log. Exact resume command: `command.json`; set `PYTHONPATH` to the sibling
`matchminer-ai-inference-kehl/src` plus this repository before launching the pipeline.
Full instructions:
`GENERATION_OPERATIONS.txt`. On pause, interrupt the generation child first and
allow checkpoint flushing and coordinator exit before stopping alias proxies.

### Next-run note preparation optimization (current run left untouched)

The event export finished at 23:35 UTC; note preparation finished at 23:37 UTC
and the current process is generating 1,137,527 notes. At the user's request,
working-tree changes optimize preparation only for subsequent launches. No
process, endpoint, fleet configuration, or active checkpoint was changed by this
optimization. The active generator retains its already-loaded original code.

`create_masked_dataset` now joins each patient history once and splices the
event marker at recorded offsets, preserving the old first-equal-event rule.
`Generator.prepare_notes` uses a bounded spawn process pool (up to eight CPUs
by default), assigns deterministic row-ID ranges before dispatch, reuses completed
chunks, and reports preparation progress. Override with
`--generation-note-preparation-workers N` in the pipeline or
`--note-preparation-workers N` in the generator; `1` selects serial preparation.

An isolated local benchmark copied one completed synthetic event chunk and
repeated it across 1,024 cohort indices (175,184 note inputs). The original path
took 3.83 s, optimized serial 3.46 s, and four workers 2.05 s, including process
startup and parquet writes. Every event/note dataframe matched exactly. These
measurements use local `/tmp`, not the shared filesystem used by the live run.
Benchmark code, results, and pre-edit source snapshots are under
`/tmp/mmai-note-preparation-20261026/`. The live run's frozen source archives also
remain under its `source/` directory. If that run needs restarting, use its
original source snapshot: the edited working tree intentionally fails its
existing source-hash checks; do not weaken or rewrite that manifest.

## Previous pause and archive (2026-10-26 19:08 UTC)

The user paused labeling to reconsider synthetic patients generated under an
earlier cohort prompt, and requested archiving the experiment except its latest
5,000-trial cohort extraction. The labeler exited gracefully at 19:06 UTC;
its workflow and four Camus alias proxies are stopped. All six teacher endpoints
reported zero running and queued requests afterward. The existing vLLM servers
remain available; no new generation, labeling, retrieval, or model training is
authorized by this pause.

Archive: `data/no_phi/archives/trialspace_reset_20261026T190832Z/`.
Its `manifest.json` maps all 32 original paths to their archived locations,
including old cohort extractions, synthetic patients/notes/summaries, ColBERT
retrieval and labels, model outputs, and audits. Each move was a same-filesystem
rename preserving the file/directory identity. Historical manifests inside the
archive retain their original paths as provenance; no redirect symlinks exist.
The completed-score journal contains 13,058 valid records after graceful drain;
the last progress snapshot reported 19,249 completed questions. Partial answers
and reasoning remain archived.
Raw references and unrelated catalogs/experiments remain outside this archive.

Still active and unchanged: `data/no_phi/cohort_reextract_named_sample5000_gemma4_31b_20261026/`
and its seven promoted root sample files/receipts. The sample remains 5,000
trials, 4,981 accepted and 19 rejected, with 6,680 cohort rows. Root export hashes
and the extraction snapshot's exported files were verified against promotion
receipts. `data/no_phi/trialspace_archive_latest.json` records the current state.

The default `train_full_gb300.py` patient input is now archived. Do not restore
or silently reuse it; future work needs an explicit choice of patient data and
fresh run/output paths. The following sections describe historical work.

## Named-cohort format regeneration (history)

Subsequent authorization: run full retrieval, then labeling on this GB300's
existing port 8000 only. The workflow stops after labeling; cloud workers and
model training remain paused. The live run is
`data/no_phi/colbert_namedcohorts_gb300_20261026/`; consult `gb300_status.json`,
`labeling_launch.json`, and `LABELING_OPERATIONS.txt` there for current state.
Its endpoint registry supports adding servers later. No old-cohort labels are
reused and no synthetic patients are regenerated or subsampled.

Initial preparation exposed pandas treating the literal criteria text `None`
as missing. `trial_cohort_format.read_clinical_csv` now preserves pandas' usual
NA-like words and treats only empty cells as missing. Extraction, sampling,
generation/preparation, standalone retrieval, and the production corpus builder
share this reader. Standalone retrieval also retains `cohort_name` for schema
validation. The production source manifest includes this shared adapter.
228 affected tests passed, including a direct production `prepare` CLI and
CSV round trips through each consumer. The promoted input files were unchanged.
The failed preparation receipt is retained as `labeling_launch.prepare_failed.json`.

The user authorized support for separate `cohort_name` and
`cohort_specific_eligibility_criteria` fields, commits of both repositories, and
re-extraction of the same 5,000 trials on the GB300's existing port 8000. The new
inference commit is `9b1bf81`, with response schema v3 and row representation v4.
Historical string outputs retain their prior versions and cannot satisfy a new
live response. Both public inference embeddings and training use criteria text,
with names retained only as metadata. Source splits and patient data are unchanged.

New extraction snapshot: `data/no_phi/cohort_reextract_named_sample5000_gemma4_31b_20261026/`.
Completed and promoted at 17:01 UTC: all 5,000 original trials validated, with
4,981 accepted, 19 rejected, and 6,680 rows (2,706 named multi-cohort rows plus
3,974 single-cohort rows). No failed trials remain. Full sample coverage, source
splits, original single-cohort eligibility, separate name/criteria fields, output
hashes, and unchanged synthetic artifacts passed validation. No exact-name or
generic cohort/arm/part headings were flagged in the final criteria fields.

Extraction ran from frozen training `87906be` and inference `9b1bf81` archives.
The actual request trace confirms literal single-brace JSON examples; doubled
braces exist only in the Python format template. NCT07738367 received one extra
same-prompt extraction to replace a repeated descriptive name before a colon.
Both original and replacement responses are preserved under `format_retries/`,
and the replacement keeps the driver-gene and treatment-history requirements.
Root sample exports and receipts were replaced after validation; the earlier
immutable extraction and patient-generation snapshots remain for provenance.
The launcher now
uses fresh `colbert_namedcohorts_gb300_20261026/` source artifacts and
`colbert_production_gb300_namedcohorts_20261026/` model output, so the previous
partial retrieval/labels cannot be mistaken for the new cohort generation.
At extraction completion, labeling, retrieval, model training, and all cloud VMs
were paused; the subsequent authorization above supersedes that state for local
retrieval and labeling. Run receipts contain completion, exact counts,
source hashes, and launch details; check them rather than the historical counts below.

## Current state after cohort refresh (2026-10-26)

**Latest: user paused the run at 16:22 UTC for further cohort-prompt and output-format changes.**
The labeling client, retrieval workflow, and host-key helper are stopped. All
eight cloud VMs were confirmed TERMINATED at 16:24 UTC; the supervisor exited and
the fleet STOP marker is set. Preserve
partial answers and reasoning, and do not restart without user authorization.
The active-run details below describe the state immediately before this pause.

This section supersedes the older optimization handoff below. GB300 benchmarking
selected replay as the default; see `GB300_MEMORY_RESULTS_20261026.md`.
The pre-refresh code was committed as training `b0bccdc` and inference `1e57f66`.

The same 5,000 trials were re-screened/extracted using the updated cohort prompt
on local port 8000. All decisions validated: 4,981 accepted trials, 19 rejected,
6,762 cohort rows (previously 6,848). The refreshed exports are promoted to the
main `data/no_phi` directory. Previous root sample files are preserved under
`archives/cohort_sample_before_reextract_20261026T152425Z/`; the root
`sample_trial_cohort_lineitems.promotion.json` records hashes and exact paths.
Synthetic generation snapshots and patient data are unchanged.

`train_full_gb300.py` now orchestrates `prepare → mine → label → train`, defaulting
to the promoted `sample_trial_cohort_lineitems.csv` and all 68,480 existing repaired
patient summaries from the old `post_notes_colbert_20261025/prepared/` directory.
New source artifacts go to `data/no_phi/colbert_reextract_gb300_20261026/` and model
training to `data/no_phi/colbert_production_gb300_reextract_20261026/`. Old labels,
retrieval artifacts, and paused checkpoints stay in their dated directories.
The new run must retrieve and label against the new cohort definitions; the old
2,435,540 pair count and 5,162-update estimate are historical.

The new CPU corpus has been prepared and checked: its `patients.jsonl` is
byte-identical to the repaired prior corpus, with SHA-256
`53c4c541dda0746a94744b8cb7689589cd7426dd322e313f90d3e76005b5f7c5`.
`colbert_reextract_gb300_20261026/configuration_validation.json` records promotion,
archive, and corpus checks, and `launch_plan.txt` contains the dry-run command plan.
Retrieval and labeling are now active under the user's subsequent authorization.
The invocation stops after labeling; model training has not started. The launch
receipt, live progress, and complete operating instructions are in that run's
`labeling_launch.json`, `gb300_status.json`, and `LABELING_OPERATIONS.txt`.

Full retrieval shares the GB300 with its existing vLLM process, using batches of
8, 128-pair scoring, a 4 GiB token cache, and a 7 GiB PyTorch allocator ceiling.
While retrieval runs, `source_trial_labeling/` labels the 135,580 source-trial
pairs guaranteed to appear in the final union. No patients are subsampled. The
final `labels/reuse_sources.json` imports completed pairs only after the source
labeler releases its lock, verifies identical corpus/prompt/teacher contracts and
candidate membership, and preserves each reasoning response. Old-cohort labels
are never imported.

The live pool uses the existing local port 8000, Camus ports 8002/8003 through
model-name proxies on 18102/18103, and 172.24.216.113:8001. The run-owned cloud
supervisor manages only the eight requested RTX Spot VMs in `profile-notes` and
`kehllab-caia-v2`, up to eight NVIDIA Gemma 4 31B NVFP4 servers per VM. Healthy
servers join automatically; unavailable or preempted VMs retry every 300 seconds.
At 16:12 UTC, 56 cloud GPU endpoints were admitted; `rtxpro-3` was repeatedly
preempted and remained under retry. Treat this as a timestamped snapshot.

`label_servers.json` and `label_concurrency.json` are live controls. Runtime
ceilings are GB300 768, Camus 128 each, other existing host 256, cloud 256 each,
within CLI ceiling 1024. Adaptive targets start at 96. Fleet shutdown follows
completion of final label inference, main workflow failure/exit, or lease expiry;
source-subset completion alone does not stop the fleet.

The cohort prompt audit compared the actual stored extraction request with the
inference file as it stood at the initial audit: SHA-256
`013c4376609cf7e8b05aaeeb425e2d2955e05cbaa8bb659af6afda01c69db965`.
The revised examples were sent and the old “First population:” example was absent.
Cohort-name headings remain in raw model responses (e.g. NCT04215978's “Phase 1a
(dose escalation):”), rather than being restored by CSV processing. The prompt
still asks the string to identify its population and does not prohibit headings.
No prompt edits or new extraction were performed by the agent as part of this audit.
At 16:20 UTC a subsequent user edit appeared, with SHA-256
`6f16c11aa8661bcd905eaadeb7f1eabe31b87fea82626d2a9c104d01fa4965ed`.
It requests dictionaries containing `cohort_name` and
`cohort_specific_eligibility_criteria`; that newer edit was not used in this run.
Its output-format section and example still specify strings, a format conflict
reported to the user. Preserve these ongoing user edits.

The teacher defaults to `http://127.0.0.1:8000/v1`, served model `gemma4-31b`,
root `nvidia/Gemma-4-31B-IT-NVFP4`, with adaptive concurrency capped at 320.
The launcher does not manage server processes. The GPU flag selects the single
device used for retrieval and training. The previous verified training device was
`GPU-b9b14780-9af1-c0ab-46e5-fa6faeef64d9`; check availability before launch.
Use `--dry-run` to inspect the plan or `--stop-after prepare|mine|label` to stop
before later stages. Repeating a command resumes with source and content checks.
Training keeps replay, 64 groups × 16 documents, group cap 8, token budget 131,072,
checkpoints every 25 updates (two retained), and AUROC every 100 updates.

## Historical optimization handoff

Make the complete, one-epoch TrialSpace training run substantially faster on the
GB300 without discarding training examples or weakening its effective-batch
ranking signal. Larger encoder microbatches helped, but retaining all activation
graphs exhausted VRAM. The user can restart from scratch: **do not spend effort
preserving the handful of completed updates or migrating checkpoints between
execution modes.** Ordinary checkpoints should still be saved, keeping at most two.

The immediate research question is whether saved activations can remain in CPU
RAM and be **read directly by GPU kernels over the coherent CPU–GPU interconnect**.
The user correctly distinguished this from copying activations to CPU and then
copying them back into HBM for backward. **The code currently implements the
latter, not the former.** Investigate and benchmark on the actual GB300; Camus
H100 results cannot establish the answer.

The prior session did not have usable SSH access to the GB300. The user runs its
training commands. Do not launch production training on Camus, restart cloud
VMs, or restart labeling as part of this optimization. Check local processes
before benchmarking: a previous replay run may still be active.

## Repositories and current state

Workspace: `/ksg/kehl_mm_data/mmai/v23`.

| Repository | Branch | HEAD when this handoff was written |
| --- | --- | --- |
| `matchminer-ai-training-kehl` | `v24_no_spaces` | `dba5ec0` — Retain ranking activations for one-forward GB300 training |
| `matchminer-ai-inference-kehl` | `v24_no_spaces` | `3a12dad764b4ba9f8111f2802612bad764f7adc6` — Prepare cached CPU tokens and encode multiple ColBERT query groups |

Both working trees were clean before adding this document. Read local instructions
and check their current status before editing. The workspace's older architecture
description is partly stale: the checked-out TrialSpace implementation is now
ColBERT, not the previous single-vector Qwen embedder.

Relevant earlier training commits:

- `a0e9fdb`: pack query groups per forward and reuse CPU tokens.
- `ed4dbd5`: one GB300 epoch, faster encoding, step timing.
- `2bba1d7`: explicit, unambiguous physical CUDA device selection.

The training inference pin is `3a12dad...` in both `train_from_summaries.py` and
`scripts/bootstrap_gcp.sh`. Shared encoder/scoring changes belong in inference;
training scheduling and activation-storage logic belong in training. Propagate
reusable improvements through the standard training entrypoints, not only the
one-off launcher. Do not push or publish without a separate user request.

## Data, readiness, and complete-epoch coverage

The repaired, completed production source is:

```text
/ksg/kehl_mm_data/mmai/v23/data/no_phi/
  synthetic_cohorts_5000_gemma4_31b_20261024/
    post_notes_colbert_20261025/
      corpus_all/patients.jsonl
      corpus_all/cohorts.jsonl
      mining_all/candidates.jsonl
      labels/scores.jsonl
      labels/scores_ready.json
```

These are synthetic/non-PHI artifacts. Keep them unchanged during optimization.
Two previously faulty patient summaries were repaired, retrieval updated, and
affected/new pairs relabeled. Do not use pre-repair artifacts.

- **2,435,540 unique labeled patient/cohort pairs**; both directions yield
  **4,871,080 directional training examples per full epoch**.
- At 16 documents/query chunk and 64 chunks/update on one GPU, there are
  **330,337 query chunks and 5,162 optimizer updates per epoch**. Tail groups
  are retained; updates contain up to 1,024 directional pairs.
- All source splits (`train val test`) are intentionally included in training.
  Accordingly, production validation probes are **in-sample diagnostics**, not
  a held-out estimate of generalization.
- `scores_ready.json` records complete candidate coverage, 2,435,540 pairs, and
  3,650,668 question responses. Score file size: 1,662,430,326 bytes. SHA-256:
  `8f05f36c2581e99d274c00701ce77ee46ac237e59997314dedd24fd93c8fcac7`.

The launcher checks this publication by content hashes, not host-dependent NFS
mtimes. It does not replay millions of reasoning checkpoints. Historical workflow
failure status during export does not invalidate a verified score publication.
Reasoning responses remain separate for later distillation; preserve them if
touching labeling, but no LLM calls are needed for this training task.

## Model, loss, and settings that must survive optimization

Base: `lightonai/GTE-ModernColBERT-v1`, using the native ModernBERT encoder,
published 128-dimensional projection, query/document markers, and document
punctuation mask. Queries combine a short retrieval instruction with the full
patient summary or cohort eligibility; the opposite side is the full document.
The query and document limits are both 8,192 tokens including markers. Do not
silently truncate clinical text.

The CLI says `--objective gated_regression` because that selects the gated 0–5
teacher-label schema. **The active loss is `--loss legacy_ranking`, not MSE**:
CoSENT ranking comparisons within each retrieval direction plus a contrastive
term against explicitly labeled zero candidates within a query. It uses full
effective-batch score derivatives. Multiple positive candidates must not become
implicit negatives. The raw mean-MaxSim output is a ranking score, not a
calibrated probability or predicted integer 0–5 score.

Current GB300 defaults:

| Setting | Value |
| --- | --- |
| Epochs | 1 |
| Documents per query chunk | 16 |
| Query chunks per optimizer update (`--accumulation`) | 64 |
| Maximum groups per encoder forward | 8 |
| Padded-token packing budget | 131,072 |
| CPU token-cache capacity | 8,000,000 tokens |
| Ranking backward mode | `retained` |
| Activation storage | `auto`, GPU budget 0 = calculated soft limit |
| Encoder gradient checkpointing | Disabled |
| Precision | BF16 autocast |
| Learning rate / scheduler | 2e-5 / linear, 1% warmup |
| Ranking scale / known-negative weight | 20 / 1 |
| Evaluation | Every 100 updates |
| Checkpointing | Every 25 updates; maximum two checkpoints |

**Physical microbatch size and effective batch are different controls.** Reducing
groups per forward or the padded-token packing budget preserves examples and
ranking comparisons while changing execution packing. Minor floating-point and
dropout-realization differences are possible. Reducing `--accumulation` changes
the ranking comparison set and training signal; do not silently do that to make
memory fit. The token budget is a packing heuristic, not a hard VRAM bound:
an oversized complete query group is kept intact in its own batch.

## Launching and device selection

From `matchminer-ai-training-kehl`, inspect the generated command without starting
training or reading the full dataset:

```bash
python colbert_experiments/train_full_gb300.py --gpu 1 --dry-run
```

`1` above is an example of the user's prior selection, **not a verified GB300
device index**. Inspect actual CUDA device names, capacity, UUIDs, and visibility.
Explicit `--gpu N` overrides inherited `CUDA_VISIBLE_DEVICES`; the child sees that
physical CUDA ordinal as `cuda:0`. Omit `--gpu` to use the first inherited visible
entry, or use `--visible-gpu N` to index the inherited mask. CUDA ordinals and
NVML/`nvidia-smi` indices can differ. Prefer a verified GPU UUID when appropriate.

The launcher uses `torch.distributed.run --standalone --nproc_per_node=1`, adds the
mounted sibling inference `src` to `PYTHONPATH`, and refuses to train on Camus.
Use the GB300's own suitable local Python environment; the Camus test environment
was `/data1/ken/envs/gptoss3/bin/python`. Training metadata requires Python >=3.13;
record the actual torch/CUDA/transformers versions on the GB300 before debugging.

Current default output:
`../data/no_phi/colbert_production_gb300_retained_20261026`.
Use a **new output directory** for changed execution settings and a fresh model.
The retained default already produced an OOM for the user; the dry-run command
above is not a recommendation to repeat that full run unchanged.

Earlier output directories under `data/no_phi`:

- `colbert_production_gb300_20261026`: original three-epoch run.
- `colbert_production_gb300_one_epoch_20261026`: first one-epoch optimization.
- `colbert_production_gb300_batched_20261026`: working grouped replay baseline.
- `colbert_production_gb300_retained_20261026`: latest retained attempt.

At handoff inspection, the retained `fit/` directory had its initial
`checkpoint-00000000.pt`, manifests, and initial evaluation outputs, but **no
`status.json` and no recorded completed update**. The user reported VRAM exhaustion;
we have not received its traceback or identified the failing allocation/phase.
Do not treat retained graphs or activation restoration as a confirmed OOM cause
without profiling. Saved status files do not establish whether a process is live.

## Implementation map and the memory distinction

Shared inference code:

- `src/matchminer_ai/embedding/colbert.py`: length-bucketed encoding,
  `prepare_groups`, `encode_prepared`, `prepared_batch_tokens`, multi-group
  `forward`, and differentiable document-batched `maxsim_batch`.
- `src/matchminer_ai/embedding/colbert_inputs.py`: bounded CPU token/mask cache,
  CPU-derived integer keep indices. Learned vectors are always recomputed;
  no stale embeddings are reused during training.

Training code under `colbert_experiments/`:

- `ranking_microbatches.py`: greedily packs complete query groups subject to
  group and padded-token limits; preserves effective-batch membership.
- `ordinal_ranking.py::cached_ranking_backward`: replay implementation. First
  forward has no gradients; compute full-batch scalar loss derivatives; repeat
  forward with the same seeded dropout and backpropagate each packed batch.
  Each example gets two forwards and one model backward, with one optimizer
  update for the full effective batch. Replay consistency is checked.
- `retained_ranking.py::retained_ranking_backward`: forward each packed batch
  once, retain graphs, compute the same scalar loss derivatives, then invoke
  `torch.autograd.backward(predictions, gradients)` once over all saved graphs.
  No forward replay. Gradient checkpointing is rejected in this mode because
  that would reintroduce forward recomputation.
- `retained_ranking.py::ActivationStorage`: the current saved-tensor hooks.
  `pack()` allocates pinned CPU storage and calls
  `host.copy_(tensor.detach(), non_blocking=True)`.
  `unpack()` waits on an event and calls **`host.to(device, non_blocking=True)`**.
  Therefore backward **restores the activation into HBM**. It does not execute
  kernels directly against CPU-resident saved activations.
- `training.py`: optimizer, scheduler, clipping, checkpoints, metrics, and mode
  selection. Retained DDP uses `broadcast_buffers=False` to avoid invalidating
  saved position-buffer versions across multiple forwards.
- `__main__.py`, `train_from_summaries.py`: execution flags exposed through
  standard training. Generic training defaults remain replay; the GB300 launcher
  explicitly selects retained.

Current storage modes are `gpu`, `cpu`, and `auto`. Auto's default soft allocation
limit is current allocation plus half of available memory, including reusable
allocator-reserved space. It is **not a hard cap**: temporaries and restoration
can exceed it. Parameters/buffers are excluded from offloading. Repeated saves of
the same tensor/version share a host copy via weak identity; in-place changes
and inference tensors are handled separately. Host memory is checked periodically
to leave at least 8 GiB available.

Copies currently use the compute stream. `non_blocking=True` alone does not
provide an independent copy/prefetch pipeline or guarantee useful overlap. There
are per-tensor events and host allocations. Joint backward over many graphs may
also have a different peak working set from sequential retained-graph backward;
measure it. Sequential backward calls could still preserve **one forward per
example**, but require careful DDP and gradient-equivalence validation.

Corpus and numeric labels are loaded into CPU RAM at startup. Training does not
stream each batch's texts from NFS. NFS still matters for initial validation/load,
pause-file checks, status, evaluation outputs, and checkpoints. Treat filesystem
bottlenecks as a measurement question rather than the default explanation for
every slow GPU step.

## Measurements already available

Saved production snapshots, not controlled identical-batch comparisons:

| Execution | Update | Pairs | Time/update | Peak allocated VRAM |
| --- | --- | --- | --- | --- |
| Initial one-epoch implementation | 8 | 942 | 54.459 s | 41.08 GiB |
| Grouped replay | 170 | 930 | 23.326 s | 48.33 GiB |
| Grouped replay, latest observed | 221 | 911 | 20.051 s | 48.33 GiB |

At update 221: 64 query chunks packed into 22 encoder batches per pass;
preparation 0.792 s, no-grad forward 6.603 s, scalar ranking loss 0.0097 s,
replay-forward/backward 12.637 s. Replay error was zero. The 5,460.58 s cumulative
time includes more than a single steady-state step. Do not turn one fast update
into a promised completion ETA.

Bounded native-model tests on **Camus H100**, with other workloads present, used
real synthetic texts but never optimizer updates:

- 64 pairs, 8 query groups: group cap 1 versus 4 measured 6.817 versus 3.054 s,
  with 12.986 versus 29.327 GiB allocated.
- 64 pairs, cap 8, token budget 131,072: replay 3.656 s / 54.019 GiB;
  retained GPU-only 1.540 s / 54.013 GiB. One timed repeat, not a production-scale
  retained-memory test or a GB300 throughput forecast.
- Forced CPU storage on four actual encoder pairs completed in 3.112 s,
  allocated 1.353 GiB GPU memory, and copied 2.516 GiB saved activations.
  This preceded later host-copy deduplication and does not measure coherent C2C.

## Suggested next investigation on the GB300

1. Identify the actual target device, running processes, CPU/GPU free memory,
   NUMA placement, driver/CUDA/PyTorch versions, and coherence mode. Inspect
   `/proc/driver/nvidia/params` (for example `rg Coherent ...`) and, where exposed,
   CUDA attributes `ConcurrentManagedAccess`, `PageableMemoryAccess`, and
   `PageableMemoryAccessUsesHostPageTables`. Do not change drivers or reboot merely
   to investigate this optimization.
2. Obtain the OOM traceback or reproduce with a bounded batch. Measure peak HBM
   in forward, loss preparation, backward, and optimizer step separately; inspect
   allocator-reserved memory and host RAM too. Check whether another live job
   consumes the presumed free GPU memory.
3. Establish working replay and smaller-physical-microbatch retained baselines.
   For example, compare group caps 2/4/8 and token budgets 32,768/65,536/131,072
   while retaining 64 effective query chunks and 16 documents/chunk. Smaller
   physical batches may reduce temporaries but cannot alone remove all retained
   activation storage for the full effective batch.
4. Compare three distinct approaches: (A) replay/recompute, (B) current staged
   CPU copies and restoration, (C) proposed GPU-direct access to CPU-resident
   saved activations. C is **not implemented**. ATS addressability at the CUDA
   level does not make an ordinary PyTorch CPU tensor a CUDA tensor: investigate
   a compatible allocator/view/extension, autograd lifetime, device metadata,
   streams, and kernels before proposing that it works.
5. Start with selective saved activations; keep weights, optimizer state, and
   ordinary kernel workspaces in HBM. Distinguish system-memory direct access
   from managed memory that migrates pages. Verify real allocation residence
   and transfers/page faults with profiling. A global managed allocator could
   alter much more than saved activations and obscure the result.
6. Profile repeated warm representative updates, ideally with Nsight Systems
   if available. Compare pairs/second, phase times, peak HBM, CPU memory,
   C2C traffic, transfer overlap, and gradient equivalence. Direct CPU reads
   have different latency/bandwidth and reuse costs from HBM; they may or may
   not beat staged prefetch or replay for this encoder. Do not promise speedup
   based on hardware bandwidth alone.
7. Integrate the winning supported option into standard production training and
   the launcher, document its hardware requirements, and test explicit fallback
   behavior. Preserve full coverage, ranking semantics, and model export.

NVIDIA's documented CDMM mode keeps system allocations CPU-resident and permits
GPU access over C2C; behavior depends on platform/driver configuration. The
advertised 900 GB/s NVLink-C2C figure is **aggregate bidirectional bandwidth**, not
900 GB/s each way or guaranteed application throughput. HBM remains faster.

## Bounded benchmark and correctness checks

Existing benchmark (from the training checkout; select a verified free device):

```bash
CUDA_VISIBLE_DEVICES=<verified-GPU-UUID-or-CUDA-ordinal> \
python colbert_experiments/benchmark_grouped_training.py \
  --probes ../data/no_phi/colbert_production_gb300_batched_20261026/fit/fixed_probes.json \
  --device cuda:0 --groups 8 --documents 8 --repeats 3 \
  --group-counts 4 8 --backward-modes replay retained \
  --encoder-token-budget 131072 --activation-storage auto
```

Replace the visibility placeholder before running. This benchmark only performs
forward/backward, with no optimizer update or checkpoint writes. It is not a
memory-safe guarantee; start smaller if necessary and eventually test the full
64-chunk effective batch. CLI accepts `gpu`, `cpu`, or `auto` activation storage
and an explicit `--activation-gpu-budget-gb`. Its probes are a bounded sample,
not necessarily the longest examples in the full production corpus.

Targeted regression tests already passed at the recorded revisions:

```bash
python -m pytest \
  colbert_experiments/tests/test_grouped_encoder_training.py \
  colbert_experiments/tests/test_retained_ranking.py \
  tests/test_full_gb300_launcher.py \
  tests/test_trialspace_production.py -q

# On an explicitly selected available CUDA GPU:
MMAI_TEST_RETAINED_CUDA=1 python -m pytest \
  colbert_experiments/tests/test_retained_ranking.py -q
```

Coverage includes replay/retained loss and gradient equivalence, dropout,
unequal DDP packing and empty ranks, retained forward counts, pause without a
partial update, full-epoch pair coverage, launcher selection, checkpoints,
GPU/CPU/auto storage, host-copy deduplication, tensor versions, and host-memory
guards. The final explicit CUDA retained-storage run passed all nine tests.
Re-run affected tests after changes; allocation-path changes particularly need
real CUDA validation, not only CPU mocks.

## Primary references for the memory investigation

These informed the preceding discussion; verify against the GB300's installed
software and actual platform before relying on specific behavior.

- [NVIDIA: Understanding Memory Management on Hardware-Coherent Platforms](https://developer.nvidia.com/blog/understanding-memory-management-on-hardware-coherent-platforms/)
  — system allocations, CDMM, direct access versus migration.
- [NVIDIA DGX Station: Mixed Coherency Environment](https://docs.nvidia.com/dgx/dgx-station-development-guide/coherency.html)
  — ATS/coherent GB300 versus discrete GPU paths and placement concerns.
- [CUDA Programming Guide: Understanding Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
  — unified addressing, system/managed memory, and capability distinctions.
- [NVIDIA: Inside Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era)
  — hardware interconnect specifications; not a measured training benchmark.
- [NVIDIA: Reducing HBM Bottlenecks in JAX-Based LLM Training with Host Offloading](https://developer.nvidia.com/blog/reducing-high-bandwidth-memory-bottlenecks-in-jax-based-llm-training-with-host-offloading/)
  — selective, pipelined staged offload; different model/framework/workload.
- [Poolside: C2C Activation Offloading on Grace Blackwell](https://poolside.ai/blog/tools-of-the-trade-c2c-activation-offloading-on-grace-blackwell)
  — workload-specific offload/recompute comparisons; not proof of direct-access
  PyTorch saved activations or a speedup for this model.

GB300 serving was verified at 2026-10-27T16:08:38.126272+00:00: 260 successful requests, 96 in flight, no request errors, ceiling 768. All 696,592 saved pair judgments were restored and new scoring resumed. `recovery/teacher_path_20261027/verified_serving.json` records evidence. The active controller retains the cosmetic `restoring_gb300` phase while waiting for dense labels; registry/metrics confirm serving. That display is corrected for subsequent launches.
