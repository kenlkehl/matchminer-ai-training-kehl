# OncoReasoning full fine-tune: a plan (not started)

User, 2026-11-16: "too expensive for now all around. save what we've staged as a plan for how we could train, in the
training repo, but do not start training. shut down all b200s." Before that, the user had asked for "a full-fledged
training run on all my data", with the task mix left as it is.

This file records the pilot, the full data pool, what is already built, and the steps to launch later. Paths in
`L/...` are relative to `data/no_phi/synthetic_dataset_n5000_screened_20261106/`. Everything is synthetic.

## 1. Pilot (mmai-b200-6, 2026-11-16 15:02–15:55 UTC)

The pilot ran on an a4-highgpu-8g Spot VM (8× B200, 178 GiB each), right after the decision model released it.
- Environment: `~/dmenv` with torch 2.13 + cu130, transformers 5.20.2, triton, and no flash-attn.
- Data: `pilot_data_20261116` (27,645 examples, 241M tokens, 122M supervised; labels seek-sampled).
- Every phase started fresh from `LiquidAI/LFM3-1.7B` @ `1da06bec`.
- Common settings: lr 1e-5, cosine schedule, 2% warmup, fp32 master weights with bf16 autocast, fused AdamW, weight
  decay 0.01, `--ce-chunk 4096`.
- Logs: `L/oncoreasoning/b200_pilot_20261116/` (`pilot.log.jsonl`, `logs_1794844481/`, `pilot_trends.json`).

| Phase | Attention | Row tokens | Grad. ckpt | Steps × tokens/step | Result | Real tokens/s (8 GPUs) | Peak alloc/GPU |
|---|---|---|---|---|---|---|---|
| flex64k | flex | 65,536 | on | 80 × 0.52M | finished, 11.2 min | 140–147k | 67.0 GiB |
| flex128k | flex | 131,072 | on | 25 × 1.05M | finished | ~147k | 100.7 GiB |
| nogc64k | flex | 65,536 | off | — | **OOM before step 1** (175.9 of 178.3 GiB) | — | — |
| nogc32k | flex | 32,768 | off | — | stopped after ~1 min, at the user's request (no OOM by then; not measured) | — | — |
| loss | flex | 65,536 | on | 240 × 1.05M (252M tokens, 1.04 passes) | finished, 30.1 min | ~146k | 67.0 GiB |

- Packing fills 99.8% of each 64k row. The longest example is 63k tokens, so nothing is dropped.
- Startup to the first step takes 1–2 min, including the flex compile. The remote-code module cache was filled
  beforehand by a single process: 8 ranks racing to fill it failed the 2026-11-15 decision run.
- Doubling the row length did not change throughput.

**Validated settings for the full run.**
- Model setup: flex attention, 64k-token rows, gradient checkpointing **on**.
- Batch: gradient accumulation 2, i.e. 16 rows = 1.05M tokens per optimizer step.
- Optimization: lr 1e-5 cosine over the whole epoch with 2% warmup, bf16 autocast with fp32 master weights, fused
  AdamW.
- Runtime: 4 loader workers, `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- Throughput for planning: 145k real tokens/s.

**Per-task loss in the loss run.** The loss is the mean cross-entropy per token over the completion (reasoning + answer).
"Last 40%" is a linear fit over the last 40% of the run's logs: falling if the slope's 95% CI is below 0. Each log
point averages only tens of examples per task, so the CIs are wide.

| Task | Start | End (last 3 logs) | Last 40% |
|---|---|---|---|
| label_match | 1.119 | 0.507 | falling (−0.065/epoch, CI −0.128 to −0.001) |
| label_boilerplate | 0.692 | 0.334 | flat/noisy |
| label_criteria | 1.032 | 0.526 | probably falling (CI −0.14 to +0.008) |
| summary | 0.634 | 0.323 | probably falling (CI −0.056 to +0.004) |
| cohort_screen | 1.129 | 0.421 | noisy |
| cohort_list | 0.830 | 0.473 | probably falling |
| cohort_criteria | 0.337 | 0.247 | flat (mostly verbatim copying) |
| cohort_merge | 0.701 | 0.408 | noisy |
| cohort_deboilerplate | 0.976 | 0.394 | probably falling |
| **overall** | **0.872** | **0.417** | **falling** (−0.068/epoch, CI −0.12 to −0.015) |

Most of the drop comes in the first ~50 steps, as the model learns the teacher's format and style. The loss keeps falling
after that, more slowly, even as the learning rate decays to 0. Phases flex64k and flex128k agree: 0.872 → 0.481 and
0.892 → 0.525 overall. These are training-loss diagnostics, not held-out validation.

## 2. The full pool (one epoch over every stored teacher output)

The **common** tasks are counted exactly from the built data (section 3). For **match** and **boilerplate**:
- The example counts are exact: each label directory's `status.json` and receipt, cross-checked against file bytes per
  line to within 1.4%.
- The token counts are the means of 358k and 258k built examples multiplied by those counts.
- Match keeps each patient-cohort pair once, from the **earliest** round. Rounds 2 and 3 re-judged 1.34M of 9,820,771
  match records. Boilerplate and criteria labels have no repeats across rounds.
- Question 7 uses only contract `7a7eb230`.

Hours are at 145k tokens/s on 8× B200.

| Task | Examples | Tokens | Supervised | Hours |
|---|---|---|---|---|
| label_match | 8,485,495 | 85.17B (est.) | 67.71B | 163.2 |
| label_boilerplate | 8,359,851 | 58.22B (est.) | 42.99B | 111.5 |
| label_criteria | 2,929,586 | 16.11B | 4.22B | 30.9 |
| summary | 87,890 | 2.39B | 0.67B | 4.6 |
| cohort_deboilerplate (every line) | 550,582 | 2.34B | 0.33B | 4.5 |
| cohort_criteria | 22,363 | 0.48B | 0.29B | 0.9 |
| cohort_screen | 34,005 | 0.11B | 0.01B | 0.2 |
| cohort_list | 9,950 | 0.09B | 0.03B | 0.2 |
| cohort_merge | 5,050 | 0.04B | 0.01B | 0.1 |
| **total** | **20.48M** | **164.9B** | **116.3B** | **316 (13.2 days)** |

The common tasks (everything but match and boilerplate) come to 3,639,426 examples, 21.55B tokens (5.56B supervised),
41.3 h. They are already built.

**Options.** Every option keeps all of the common set. "k" is the number of match and boilerplate partitions kept, out
of 100 (section 4). Steps assume 1.0465M real tokens per step. Disk is int32 token files. Staging is the copy from
/ksg to the VM at the 25 MB/s copy pace.

| Option | k | Tokens | Supervised | Training | Steps | Disk | Staging |
|---|---|---|---|---|---|---|---|
| A, everything | 100 | 164.9B | 116.3B | 316 h = 13.2 d | ~158k | 660 GB | 7.3 h |
| B | 55 | 100.4B | 66.4B | 192 h = 8.0 d | ~96k | 402 GB | 4.5 h |
| C | 29 | 63.1B | 37.7B | 121 h = 5.0 d | ~60k | 253 GB | 2.8 h |
| D | 11 | 37.3B | 17.7B | 72 h = 3.0 d | ~36k | 149 GB | 1.7 h |

**Disk on the VM.**
- mmai-b200-6 had a 1.5 TB persistent boot disk with 345 GiB free, plus 32 unmounted 375 GB local SSDs (12 TB of scratch,
  wiped by any stop or preemption).
- C and D fit on the boot disk next to the checkpoints (about 21 GB each, 2 kept). A preemption then needs no
  re-staging.
- B and A need either a boot-disk resize (`gcloud compute disks resize` + `resize2fs`; costs money, needs the user's
  approval) or the local-SSD RAID0. The driver creates the RAID0 and re-stages after every restart, which takes
  4.5–7.3 h each time.

## 3. Already built (left in place)

Every directory has `manifest.json`, `build.log.jsonl`, `preview.jsonl` and a `stage_receipt.json` (kind
`oncoreasoning-data`). Published copies are in `L/oncoreasoning/data/` (size listings compared). Local copies are on
camus `/data1/ken/oncoreasoning/`.

| Directory | Where | Size | Examples | Tokens | Contents |
|---|---|---|---|---|---|
| `pilot_data_20261116` | both | 1.0 GB | 27,645 | 0.24B | all 9 tasks, sampled (pilot only; not part of a full run) |
| `full_a2_20261116` | both | 13.5 GB | 284,216 | 3.36B | 80,016 summaries; ETL capped near 90–95% (19,000 criteria, 140,000 per-line, 32,000 screen, 9,400 list, 3,800 merge) |
| `full_b1_20261116` | both | 9.6 GB | 314,429 | 2.40B | labels, block-sampled (seed 4201); 1,970 within-chunk duplicates, dropped by `train.py` |
| `full_b2_20261116` | both | 19.2 GB | 630,061 | 4.80B | labels, block-sampled (seed 4202), excluding B1 |
| `eval2_20261116` | both | 0.2 GB | 2,750 | 0.03B | 300 summaries, 300 screen, 200 list, 200 criteria, 150 merge, 400 per-line, 400 of each label family; seek-sampled, excluding all of the above; `--save-text` for `evaluate.py` |
| `full_common_20261116` | /ksg only | 65.7 GB | 3,027,197 | 16.38B | the rest of the common pool: 7,574 summaries, every remaining ETL example (410,182 per-line), 2,603,123 criteria labels |

- B1's and B2's **match and boilerplate** parts must not be used in a full run. They are not partition-filtered and keep
  later-round repeats; the partition build replaces them.
- Training policy: all synthetic data, no holdouts. So eval2 is part of the pool, and its metrics are in-sample
  diagnostics.
- The local caches `/data1/ken/oncoreasoning/oncoreasoning_*.parquet` (2.3 GB: the notes' and serial summaries' needed
  columns) are rebuilt automatically if deleted.

## 4. Match and boilerplate partitions (coded, not built)

The code is in `oncoreasoning_training/sources.py`:
- `partition(task, patient_id, cohort_id)` = `sha256("oncoreasoning-20261116:<task>:<patient>:<cohort>")[:8] mod 100`.
- `earlier_round_pairs` and `iter_label_blocks`.

`build_data.py` options:
- `--label-partitions label_match=k label_boilerplate=k` keeps only pairs in partitions 0..k−1. Any option's cap is its
  first k partitions, so the selection is reproducible and nested.
- `--match-earliest-round` drops a match record whose pair was labeled in an earlier round. The main process computes
  `match_pairs_before_round{2,3}.npy` once from `scores.jsonl`.
- Both need block sampling (`--label-block-mb 64`). Shards split one block order.

Partition mode reads every block: about 1.47 TB of match and boilerplate `labels.jsonl`, about 6.8 h at 60 MB/s for any
k. For a small k, a token budget is cheaper (`--tokens label_match=… label_boilerplate=…` with block sampling and
`--match-earliest-round`). It reads only about k% of the files, but its selection is not a prefix of a larger one.

Smoke-tested 2026-11-16: k=3 kept only partitions 0–2, and the earliest-round rule removed 13 later-round repeats.

## 5. The full-run driver and the VM contract

The driver is `L/source/scripts/oncoreasoning_full_b200_20261116.py` (operational, so it stays outside the repo). Only
its paced send was tested: 229 MB at 17.6 MB/s, size listing matched. It was never launched.
- **Heartbeat:** it touches `<control-dir>/driver.heartbeat` every 2 min, from start to finish.
- **Keeper:** the VM's fleet controller watches a keeper such as `L/source/scripts/oncoreasoning_full_keeper_20261116.py`,
  which the coordinator wrote. The keeper exits, and the controller stops the VM, on any of these:
  - `done` or `abort` in the control dir;
  - a heartbeat older than 90 min;
  - no heartbeat within 8 h of the keeper starting;
  - 240 h after the keeper starts.
- **Repointing a controller:** a new VM's controller is pointed at a fresh keeper by SIGKILLing and restarting it on the
  same config, as `L/source/scripts/repoint_to_keeper_20261113.py` does. Never SIGTERM a controller: SIGTERM stops its
  VM.
- **Staging:**
  - holds the image's vLLM worker (`/etc/mmai-train-hold`, survives restarts) and waits for free GPUs;
  - ships the code, the pinned model snapshot and every data directory. From /ksg it paces at 25 MB/s; from /data1 it
    doesn't. A directory whose size listing already matches is skipped;
  - fills the module cache once;
  - with `--vm-data /mnt/oncoreasoning/...`, first builds the local-SSD RAID0.
- **Training:** `train.py` runs as the transient unit `oncoreasoning-full` (torchrun, 8 GPUs, the validated settings,
  `--save-steps 500` ≈ 1 h). Checkpoints go to `~/oncoreasoning/full_work` on the persistent boot disk, the last 2
  kept.
- **Resume:** `train.py` resumes from its newest checkpoint. Packing and the sampler are seeded, so a resumed run skips
  exactly the rows it had seen.
- **Preemption:** an unreachable VM is waited for, up to 2 h per wait, while the controller restarts the Spot VM. The
  driver then re-stages local-SSD data and restarts the unit. A failed unit is restarted from its checkpoint at most 6
  times.
- **Checkpoint copies:** the newest complete checkpoint is copied to `<model>_training/checkpoint-N` on camus (size
  listing compared, never hashed or read back), 2 kept.
- **Finish:** the model and its stage receipt (`oncoreasoning-model`) are copied to `--model`, sizes compared, and the
  driver touches `done`.
- **Failure:** if the driver fails, its heartbeat stops and the keeper stops the VM 90 min later. Touch `abort` to stop
  sooner.
- Events: `<control-dir>/full.log.jsonl`.

## 6. Launching later

1. **Choose.** The user picks A–D (k). For A or B they also choose between a boot-disk resize and local-SSD staging.
2. **Build match and boilerplate** on camus (CPU only; camus GPUs are off-limits) into
   `L/oncoreasoning/data/full_labels_k<k>_<date>`:
   ```bash
   cd matchminer-ai-training-kehl
   HF_HUB_CACHE=/data1/ken/models HF_HUB_OFFLINE=1 ONCOREASONING_CACHE=/data1/ken/oncoreasoning \
   /data1/ken/envs/gptoss3/bin/python oncoreasoning_training/build_data.py --output-dir L/oncoreasoning/data/full_labels_k<k>_<date> \
       --seed 6001 --examples label_match=1e12 label_boilerplate=1e12 \
       --label-partitions label_match=<k> label_boilerplate=<k> --match-earliest-round \
       --label-block-mb 64 --shards label_match=24 label_boilerplate=16 --mb-per-s 60
   ```
   This takes about 7 h at 60 MB/s. Check its manifest: match examples ≈ 8,485,495 × k/100, boilerplate ≈ 8,359,851 ×
   k/100.
3. **Hunt a B200.** Use image `mmai-nemotron-fleet-a4-20261107` with `create_b200_vm_20261107.py`, or restart a stopped
   B200 with its controller, following `decision_move_20261115.py`'s hunt across fleets. Give the winner's controller a
   fresh keeper with a new control dir (copy `oncoreasoning_full_keeper_20261116.py`) and repoint it. Ask the
   coordinator or user first: B200s are paid by the hour.
4. **Smoke run** (about 15 min), into a separate model directory:
   ```bash
   cd L/source/scripts
   /data1/ken/envs/gptoss3/bin/python oncoreasoning_full_b200_20261116.py --fleet <fleet> --control-dir <control dir> \
       --model L/oncoreasoning/models/oncoreasoning_lfm3_smoke_<date> --train-arg=--max-steps --train-arg=20 \
       --data /data1/ken/oncoreasoning/full_a2_20261116 /data1/ken/oncoreasoning/full_b1_20261116@label_criteria \
              /data1/ken/oncoreasoning/full_b2_20261116@label_criteria \
              /data1/ken/oncoreasoning/eval2_20261116@summary,cohort_screen,cohort_list,cohort_criteria,cohort_merge,cohort_deboilerplate,label_criteria \
              L/oncoreasoning/data/full_common_20261116 L/oncoreasoning/data/full_labels_k<k>_<date>
   ```
   The smoke run touches `done` when it ends, and that stops the VM. Either give it its own keeper, or remove `done`
   and restart the keeper before the full run.
5. **Full run:** the same command with `--model L/oncoreasoning/models/oncoreasoning_lfm3_full_<date>`, without
   `--train-arg`. Add `--vm-data /mnt/oncoreasoning/full_data` for local-SSD staging with B or A. Report milestones:
   training started, first checkpoint copied back, every 25%, and any preemption.
6. **Evaluate:** `evaluate.py generate --eval-data L/oncoreasoning/data/eval2_20261116 --model <model> --gpus 8`
   (on a GPU host), then `evaluate.py score`. This gives agreement with the teacher's answers per task. eval2 is in the
   training pool, so these are in-sample diagnostics.

**Not yet done:**
- the driver has never run end to end (staging, RAID, resume, checkpoint copy);
- `evaluate.py generate` has never run on a GPU;
- nothing was trained beyond the pilot, which used `--no-save`.
