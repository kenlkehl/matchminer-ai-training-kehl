# EmbeddingGemma 2 / ModernColBERT comparison

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


Current policy, corrected October 27: use all **66,800 synthetic patients and
6,680 cohorts** for both retrieval directions, labeling, and fitting. Historical
train/validation/test assignments remain `source_split` provenance only.
There are no synthetic holdouts. Production entrypoints reject subset overrides.

Run directory: `/ksg/kehl_mm_data/mmai/v23/data/no_phi/trialspace_comparison_namedcohorts_5000_20261026`.
The controller is `mmai-trialspace-comparison-ssd-training-20261027.service`;
`launch.json`, `status.json`, `logs/` and `completed/` record live execution.
`config.json` contains exact commands; `source/workspace/` freezes both repos.
The inference prompt snapshot is unchanged by this correction.
The round-two controller handoff and its 94 passing tests are recorded in
`recovery/embeddinggemma_round_two_20261027/`. Read `launch.json` for the active
controller identity; older process descriptions below are historical.

The earlier October 27 ordering handoff is recorded in `recovery/dense_training_first_20261027/`.
Controller PID 4080014 adopted labeler PID 4007097 and teacher PID 4007111;
fleet supervisor PID 4080324 retained all eight SSH connections and 64 cloud
endpoints. Labeling and inference continued throughout the controller transfer.
The new controller now reports `waiting_dense_labels` while this pass runs.

The old split-restricted rankings are invalid in **both** directions: the old
implementation also restricted the cohort denominator for patient queries.
They are archived with the old corpus/config in
`archive/split_restricted_20261027/`. The interrupted label snapshot preserves
151,728 completed pair judgments and 237,266 full question responses, including
prompts, raw answers and reasoning. `checkpoint_snapshot.json` records hashes.
It marks completed snapshot publication, not completed labeling.

`retrieval_cache_import.py` verifies identical entity order, clinical text,
provenance and encoding identity before importing unchanged vectors/token
shards. It never imports rankings or query checkpoints. Both full-pool rankings
are recomputed. `colbert_experiments/full_pool_reuse.py` imports only judgments
and partial answers for current candidate pairs with identical clinical text,
exact prompts and teacher settings. Originals remain in the archive. Recovery
audits and source/config backups are in `recovery/full_pool_20261027/`.

Both retrievers score every opposite-side entity, retaining 20 cohorts per
patient and 40 patients per cohort, plus five lexical and five random proposals
and original-trial lineage pairs. The same ColBERT clinical rubric supplies the
retrieval text and gated five-question teacher labeling. EmbeddingGemma 2 uses
Google's symmetric `task: sentence similarity | query: ` prefix, applied once
in retrieval and training. Both models use an 8,192-token limit and pinned
revisions recorded in `model_versions.json` and mining manifests.

All eight cloud RTX VMs were verified TERMINATED during this correction. The
controller restarts the fleet **only when corrected dense candidates are ready
to label**, honoring the recorded STOP identity. Spot startup/preemption retries
continue while dense labeling overlaps GB300 ColBERT retrieval. The GB300 teacher
returns after scoring, overlapping verified copy-back to shared storage when
there is sufficient free GPU memory. The publisher stays supervised and its
published completion marker remains the downstream gate.

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

The active `dense_round_two` plan blocks the former post-training fleet restart
and the ColBERT union-labeling transition. Round-two mining uses
`models/embeddinggemma2`, fresh embeddings and fresh exhaustive rankings with the
same retrieval counts, prompts and sampling as round one. Its manifest hashes
local model files so changed weights cannot silently reuse an old embedding cache.
The dedicated `round2_embeddinggemma2/servers.json` follows only the eight
explicitly authorized local/LAN URLs and excludes fleet-tagged entries. Live
cloud termination is checked before round-two mining and labeling. Separate
stage receipts make the new pass resumable; `awaiting_round_two_review` marks
completion of its labeling/export, not completion of the deferred comparison.

Immediate cloud shutdown is supervised independently of the labeler's export.
Registered process identities and their status files are checked for complete
pair counts, `inference_complete: true`, and no failures. Live labelers that still
need inference prevent shutdown; stale completed stages cannot stop a later pass.
The supervisor joins pending Spot-start operations and retries stops until all
eight VMs are TERMINATED. Local/LAN inference servers and exporting processes
remain running. This also covers dense labeling's `waiting_dense_labels` phase,
whose workflow `child_pid` is intentionally empty.
The immediate-shutdown watcher was deployed at 19:58 UTC on October 27. Fleet
supervisor PID 33723 replaced its predecessor while retaining all eight SSH
connections; workflow PID 4080014, labeler PID 4007097, and GB300 teacher PID
4007111 were not restarted. The independent fleet watcher handles shutdown while
that workflow still awaits the labeler's full publication. Backups, 57 passing
tests, and live verification are in `recovery/immediate_cloud_shutdown_20261027/`.

Latest pool update, October 27 at 18:28 UTC: the user restored Camus 8000/8001.
The LAN pool now includes all four Camus ports 8000–8003, through loopback alias
proxies 18100–18103, plus 172.24.216.113:8001/8002/8003. The returning endpoints
were checked against RedHatAI/Gemma-4-31B-IT-FP8-Dynamic with 262,144-token
contexts and added live at initial concurrency 32, ceiling 128 each. Fleet
publication preserves these entries. Do not rearm the old retirement cutoff.
Registry/config backups and health evidence are in
`pool_updates/camus_8000_8001_restore_20261027T182817Z/`.

Retrieval uses `/tmp/mmai-retrieval/` and labeling uses `/tmp/mmai-labels/` for
active work. Final outputs return to the shared run directory with checksums;
failed runs retain local checkpoints. Label responses use durable compressed
SQLite preserving complete traces; final `labels.jsonl` streams to shared
storage. `retrieval_storage.json` / `storage_complete.json` record publication.

Independent exports under `pair_labels/`:

| Directory | Candidate membership |
| --- | --- |
| `embeddinggemma2/` | EmbeddingGemma 2 only |
| `colbert_modernbert/` | ModernColBERT only |
| `union/` | Deduplicated combined pairs |

Each contains `pairs.parquet`, `patient_to_cohort.parquet`,
`cohort_to_patient.parquet` and `scores.jsonl`. Overlapping pairs appear separately
in both model directories with identical judgments. Full text, source provenance,
question keys and retrieval provenance are retained; full traces reside in
`labels_union/responses`. Separate candidate rankings remain under
`candidates/{embeddinggemma2,colbert_modernbert}/`.

The independent pretraining export is `dense_pair_labels/embeddinggemma2/`;
the later combined projection writes `pair_labels/` without changing that input.
EmbeddingGemma 2 training uses **only its own candidates**, across all source
patients. The single-vector TrialSpace recipe is unchanged: positive pairs
(score >= 1) use MultipleNegativesRankingLoss, valid pairs use CoSENTLoss with
scores mapped to [-1, 1], proportional dataset sampling, batch size 10, three
epochs, learning rate 2e-5 and linear scheduling with 1% warmup.

`retrieval_comparison.json` reports teacher-assessed top-k quality and overlap.
`training_diagnostics/metrics.json` reports before/after Spearman and AUROC at
scores >= 1 and >= 3 on the EmbeddingGemma synthetic candidates. These are **in-sample**
diagnostics conditional on candidate selection, not held-out validation or
full-corpus recall estimates. Independent real-data evaluation is unaffected.

The isolated EmbeddingGemma environment serves dense retrieval/training; vLLM
and ColBERT retain `/home/klkehl/thisenv`. Generation and summaries are complete;
this correction does not regenerate them or modify clinical prompts.

October 27 publication handoff: the controller adopted the existing labeler
PID 3884183 and ColBERT publisher PID 3884301 without stopping either. Its new
PID is 4000615; fleet supervisor PID 4000985 retains the eight live SSH tunnels.
Recovery evidence, source/config backups and tests are under
`recovery/retrieval_publication_overlap_20261027/`. ColBERT's current copy-back
finished at 15:57:40 UTC during this handoff, before the teacher launch; actual
teacher/copy overlap did not occur in this run. Future runs release unused CUDA
cache before publication and use `retrieval_publication.py` to verify the exact
publisher, complete local artifacts and checksums before returning the GPU.

The first teacher restart failed at 16:00 UTC because the service PATH omitted
the existing virtualenv `ninja` executable. Failure cleanup stopped the labeler
and all eight VMs. Recovery is under `recovery/teacher_path_20261027/`; the new
controller is PID 4006530, labeler PID 4007097, teacher PID 4007111 and fleet
supervisor PID 4007512. The labeler resumes its original SSD journal and response
database. The fleet has requested all eight VM restarts with normal Spot retries.
The teacher environment now includes the virtualenv bin and CUDA compiler paths;
the production launcher also prepends its configured interpreter's bin directory.
On resume, labeling starts before teacher initialization so the other endpoints
can work during GB300 startup. Use live status/registry files for readiness.

GB300 serving was verified at 2026-10-27T16:08:38.126272+00:00: 260 successful requests, 96 in flight, no request errors, ceiling 768. All 696,592 saved pair judgments were restored and new scoring resumed. `recovery/teacher_path_20261027/verified_serving.json` records evidence. The active controller retains the cosmetic `restoring_gb300` phase while waiting for dense labels; registry/metrics confirm serving. That display is corrected for subsequent launches.

The October 27 SSD-training handoff is audited in
`recovery/ssd_training_20261027/`. `local_dense_training` records the independent
SSD snapshot/pair directory under `/tmp/mmai-dense-training/`. The ordinary
`finetune_embedder.py` entrypoint defaults to `--local-training-storage`, with
`/tmp/mmai-training/` scratch and a 32 GiB free-space reserve. Public
`embeddinggemma2_training/training_storage.json` reports its actual local paths
and phase; `training_storage_complete.json` verifies final model/checkpoint
publication. A failed copy can resume publication without rerunning training.
All clinical text, score filtering, prompts, losses, sampling and optimization
settings are unchanged. `dense_pair_labels/embeddinggemma2/` remains the shared
independent pair export; round-two candidates/labels remain separate and local/LAN-only.
