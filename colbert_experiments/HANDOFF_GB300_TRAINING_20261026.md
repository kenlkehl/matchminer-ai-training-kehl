# GB300 ColBERT training optimization handoff

Prepared 2026-10-26 from Camus for a Codex session running directly on
`sn4622130540`. This is an implementation/debugging handoff, not a published
model release. Paths below assume the same shared mount.

## What the user wants next

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
