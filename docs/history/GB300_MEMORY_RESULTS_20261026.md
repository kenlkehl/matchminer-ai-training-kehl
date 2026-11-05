# GB300 activation-memory investigation — 2026-10-26

Continuation of [the handoff](HANDOFF_GB300_TRAINING_20261026.md), performed directly
on `sn4622130540`. Numerical records, probe hash, environment and copy counters
are in [the results JSON](GB300_MEMORY_RESULTS_20261026.json).

## Result and current recommendation

**GPU kernels can read pinned CPU activations without an explicit restoration
copy in this PyTorch/GB300 environment. That does not make full-batch retained
training fit, or establish a throughput advantage.** Keep replay for the complete
64-query-chunk training batch. The launcher now defaults to replay and a fresh
`../data/no_phi/colbert_production_gb300_replay_20261026` output directory.

The ranking objective, 64 effective chunks, 16 documents per chunk, both retrieval
directions, complete-epoch coverage, 8,192-token limits, learning-rate schedule,
and two-checkpoint retention policy are unchanged. No optimizer update, model
export, labeling call or full production training launch was performed.

On the same 64-group / 1,024-pair synthetic probe batch, with one full warmup and
three timed repetitions:

| Execution | Group cap / padded-token budget | Mean seconds | Peak allocated HBM |
| --- | --- | ---: | ---: |
| Replay, recommended | 8 / 131,072 | 16.835 | 90.46 GiB |
| Replay, larger physical batches | 16 / 262,144 | 16.075 | 173.02 GiB |

The larger setting gained about 4.5% on this batch while nearly doubling peak
allocated HBM; it was not made the default. Repeats were 16.812/16.828/16.864
seconds and 16.057/16.060/16.109 seconds, respectively. These are forward/backward
times, excluding optimizer updates, evaluation, checkpoints and startup. They
are not an epoch ETA or a comparison against identical batches from older runs.

Both staged `auto` and experimental `direct` retention at cap 8 / 131,072 with
the automatic GPU budget stopped at the host-memory guard **during forward**.
This is a controlled finding, not a reconstruction of the user's earlier VRAM
OOM; its original traceback remains unavailable. A temporary shared pinned arena,
cap 4 / 65,536 and 160 GiB GPU budget also failed during forward after reserving
524.5 GiB of offload storage. That arena implementation was removed: reducing
allocation overhead did not solve complete-batch capacity.

## Hardware and placement

- GB300 UUID: `GPU-b9b14780-9af1-c0ab-46e5-fa6faeef64d9`; NVML index 1.
  Use the UUID to avoid ambiguity with CUDA ordinals or inherited visibility.
- Python 3.13.15, PyTorch 2.13.0+cu130, CUDA runtime 13.0,
  Transformers 5.17.0, driver 595.58.03. No environment upgrade was made.
- ATS was reported by NVML. CUDA pageable access, registered-host-pointer use,
  concurrent managed access and host-page-table access were available.
- This installation is in NUMA mode: `CoherentGPUMemoryMode` is empty, CPU RAM is
  node 0 and HBM is exposed as memory-only node 1. Roughly 744 GiB of total Linux
  memory includes about 249 GiB of GPU memory; it is not all CPU offload capacity.
- Small mapped pinned allocations were observed on CPU node 0 in
  `/proc/self/numa_maps`, with identical host and CUDA-view pointers. This does
  not prove CPU residence for every allocation under full-run memory pressure.
  No driver configuration or machine restart was attempted.

NVIDIA distinguishes NUMA-mode placement/migration from CDMM's CPU-resident
system allocations; do not infer physical placement solely from PyTorch's CPU
device label. See [NVIDIA's memory-management explanation](https://developer.nvidia.com/blog/understanding-memory-management-on-hardware-coherent-platforms/)
and [the DGX Station coherency guide](https://docs.nvidia.com/dgx/dgx-station-development-guide/coherency.html).

## Implemented changes and correctness

`--activation-storage direct` is explicit and experimental, available through
the experiment CLI, production trainer, GB300 launcher and integrated runner
(`--trialspace-activation-storage direct`). It uses auto's GPU retention threshold
and exposes spilled pinned storage through CUDA array interface views. Parameters,
optimizer state and ordinary workspaces still use their normal GPU allocations.
The capability check rejects non-ATS devices without silently changing modes.
Owners survive all asynchronous GPU reads through synchronized backward.

The [CUDA array interface](https://nvidia.github.io/numba-cuda/user/cuda_array_interface.html)
accepts device-accessible pointers; [PyTorch's implementation](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/utils/tensor_numpy.cpp#L363)
constructs a CUDA tensor over that memory and retains its Python owner. The
implementation uses a byte view then restores the saved dtype and strides,
including BF16, without an extension or a global allocator replacement.

The investigation also fixed two problems in staged offload:

1. Compacting saved attention masks lost their padded row strides. Deterministic
   backward exposed `attn_bias is not correctly aligned (strideM)`. Both staged
   and direct storage now preserve the bounded physical span and strides,
   including broadcast views and versioned snapshots.
2. `/proc/meminfo` counted HBM as available host memory. The guard now also checks
   free memory on NUMA nodes with CPUs and accounts for pinned-allocation rounding.
   It remains a conservative guard, not a hard process-memory limit.

Ordinary BF16 replay differed from a repeat of itself by 0.625% relative gradient
L2 in a preliminary check. GPU-retained versus replay differed by 0.618%; the
initial direct comparison was therefore not an isolated correctness test.
`--check-gradients` now enables deterministic algorithms automatically. With
aligned storage, the native 128-pair direct/replay comparison had relative L2
error **5.34e-8**, maximum absolute error **1.19e-7**. The final 16-pair staged
and direct comparisons both matched replay exactly. Deterministic-kernel timing
is separate from the ordinary throughput table above.

Final validation: **61 CPU/integration tests passed**, with 15 opt-in CUDA cases
skipped in that invocation; **the 20-test CUDA-enabled suite passed** on the GB300 (15 CUDA cases and five CPU cases).
Coverage includes full-batch ranking and dropout, uneven DDP packing, empty ranks,
pair coverage, launch/entrypoint controls, ownership, non-default streams,
strides/alignment, tensor versions and host-memory guards. The UUID-selected
launcher dry run passed without creating its output directory.

## Copy-path evidence

Two groups / 16 pairs, forced spill, same deterministic encoder configuration:

| Profiler counter | Staged `cpu` | `direct` |
| --- | ---: | ---: |
| Device → pinned bytes | 13,654,927,336 | 13,654,927,336 |
| Pinned → device bytes | 13,717,109,848 | **0** |
| Pinned → device operations | 2,522 | **0** |
| Pageable → device bytes (inputs/score derivatives) | 347,488 | 347,488 |

Both modes used about 1.4 GiB peak allocated HBM and 20.29 GiB of pinned allocator
storage for 12.72 GiB of copied activation spans. The warmed samples averaged
0.791 seconds staged and 1.342 seconds direct (two repeats each). This small test
does not establish production throughput, but provides no reason to prefer direct
reads here. Removing explicit copies does not remove C2C traffic from kernels.
These traces do not measure page migration or full interconnect traffic.

Local detailed traces: `/tmp/colbert_cpu.trace.json` and
`/tmp/colbert_direct.trace.json`; durable aggregate counters are in the results JSON.

## Commands

Reviewed production command, **not launched**:

```bash
/home/klkehl/thisenv/bin/python colbert_experiments/train_full_gb300.py \
  --gpu GPU-b9b14780-9af1-c0ab-46e5-fa6faeef64d9
```

It starts a new one-epoch replay run in the new output directory. Add `--dry-run`
to inspect it. It continues to validate the repaired publication before training.

Reproduce the recommended bounded baseline, after checking the device is free:

```bash
CUDA_VISIBLE_DEVICES=GPU-b9b14780-9af1-c0ab-46e5-fa6faeef64d9 HF_HUB_OFFLINE=1 \
/home/klkehl/thisenv/bin/python colbert_experiments/benchmark_grouped_training.py \
  --probes ../data/no_phi/colbert_production_gb300_batched_20261026/fit/fixed_probes.json \
  --groups 64 --documents 16 --group-counts 8 --encoder-token-budget 131072 \
  --backward-modes replay --warmup 1 --repeats 3
```

For a bounded direct correctness/profile check, use `--groups 2 --documents 8
--group-counts 2 --encoder-token-budget 32768 --backward-modes retained
--activation-storage direct --activation-gpu-budget-gb 0.000001
--check-gradients --profile /tmp/colbert_direct.trace.json` with the same executable,
visibility and probe file. The tiny positive GPU budget deliberately forces spill.

Next work should first measure which saved tensors dominate total retention and
whether their storage can be shared safely. Replacing restoration copies alone
has now been tested. Keep full-batch ranking semantics and verify gradients before
adopting selective retention, different attention kernels, or another memory plan.
