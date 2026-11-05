#!/usr/bin/env python3
"""Resumable TrialSpace → TrialChecker / BoilerplateChecker → catalog → OncoReasoning pipeline.

Existing non-PHI summaries seed mining; the NVIDIA teacher provides all fresh labels,
catalog evidence and distillation. TrialSpace (EmbeddingGemma 2) is mined and refit over
three rounds on the patients' key summary sections and the cohorts' key criteria; every
round's five-answer match labels train the multi-head TrialChecker, and boilerplate labels
of round-3 pairs train the separate BoilerplateChecker. OncoReasoning is distilled last.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from oncoreasoning_training.contracts import DEFAULT_MODEL_NAME, MATCH_LABELS_DIGEST, digest
from oncoreasoning_training.create_all_training_data import atomic_json, source_record
from oncoreasoning_training.teacher import DEFAULT_TEACHER
from vllm_reasoning_utils import resolve_parser_name, resolve_tool_call_parser
from trialspace.label_concurrency import DEFAULT_CEILING, DEFAULT_INITIAL, DEFAULT_AUTO_WORKERS_MAX

ROOT = Path(__file__).resolve().parent
DEFAULT_EMBEDDER = 'google/embeddinggemma-2'
DEFAULT_CHECKER = 'answerdotai/ModernBERT-large'
EMBEDDINGGEMMA2_REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'

TRIALSPACE_OPERATIONAL_OPTIONS = {
    'trialspace_save_steps', 'trialspace_save_total_limit', 'trialspace_stop_at', 'trialspace_pause_file',
}


def training_identity_arguments(args):
    return {key: str(value) for key, value in vars(args).items()
            if key not in {'dry_run', 'skip_input_validation', *TRIALSPACE_OPERATIONAL_OPTIONS}
            and not (key in ('trial_spaces', 'patient_summaries', 'synthetic_notes',
                             'teacher_servers_file', 'teacher_concurrency_file') and value is None)}


def training_identity_command(command):
    """Exclude only checkpoint cadence and pause controls from upstream resume identity."""
    flags = {'--' + key.replace('_', '-') for key in TRIALSPACE_OPERATIONAL_OPTIONS}
    result, index = [], 0
    while index < len(command):
        if command[index] in flags:
            index += 2
        else:
            result.append(command[index])
            index += 1
    return result


@dataclass
class Stage:
    name: str
    command: list[str]
    outputs: list[Path]
    teacher: bool = False


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT.parent / "data/no_phi")
    p.add_argument("--trial-spaces", "--trial-cohorts", dest="trial_spaces", type=Path,
                   help="Complete accepted cohort CSV/parquet; default: DATA_DIR/trial_cohort_lineitems.csv")
    p.add_argument("--patient-summaries", type=Path,
                   help="Summarized synthetic patients; default: DATA_DIR/patient_summaries_with_spaces.parquet")
    p.add_argument("--synthetic-notes", type=Path,
                   help="Generated synthetic notes; default: DATA_DIR/all_synthetic_notes.parquet")
    p.add_argument("--run-dir", type=Path, default=ROOT.parent / "data/no_phi/embeddinggemma2_pipeline_v1")
    p.add_argument("--models-dir", type=Path, default=ROOT.parent / "models/embeddinggemma2_pipeline_v1")
    p.add_argument("--inference-repo", type=Path, default=ROOT.parent / "matchminer-ai-inference-kehl")
    p.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    p.add_argument("--teacher-served-model", help="API model alias; --teacher-model remains the underlying HF model")
    p.add_argument("--teacher-url", action="append", default=[],
                   help="Existing OpenAI-compatible teacher URL; repeat for a pool. Skips local vLLM startup.")
    p.add_argument("--teacher-servers-file", type=Path,
                   help="Externally managed live endpoint registry. Never rewritten or shut down by this runner.")
    p.add_argument("--teacher-concurrency-file", type=Path,
                   help="Live per-server limits for TrialSpace labeling only; its short requests "
                        "need higher limits than generation or summary stages.")
    p.add_argument("--student-model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDER, help="Round-1 TrialSpace base model")
    p.add_argument('--embedding-revision', help='Base model revision; default: the pinned EmbeddingGemma 2 revision')
    p.add_argument('--embedding-python', type=Path,
                   help='Python for dense mining, projection and fits (needs EmbeddingGemma 2 support); default: this one')
    p.add_argument('--synthetic-training-splits', nargs='+', choices=['train', 'val', 'test'],
                   default=['train', 'val', 'test'], help='Synthetic source partitions used by every student fit')
    p.add_argument('--trialspace-context', type=int, default=8192)
    p.add_argument('--trialspace-steps', type=int, help='Explicit optimizer updates per fit; otherwise derive from complete epochs')
    p.add_argument('--trialspace-epochs', type=float, default=1, help='Complete passes over the labeled pairs per fit')
    p.add_argument('--trialspace-batch-size', type=int, default=96, help='EmbeddingGemma 2 pairs per GPU (one rank per --gpus entry); each rank ranks its own pairs, '
                        'and an optimizer step covers this times the GPU count')
    p.add_argument('--trialspace-loss-mini-batch-size', type=int, default=32,
                   help='EmbeddingGemma 2 gradient-cache encoder chunk; the loss still spans each rank\'s full batch '
                        '(48 peaked at 212 GiB on the GB300 with a 2,500-token cap; the cap is now 4,096 on key text)')
    p.add_argument('--trialspace-save-steps', type=int, default=200)
    p.add_argument('--trialspace-save-total-limit', type=int, default=2)
    p.add_argument('--trialspace-stop-at', help='Pause student training at an ISO timestamp with timezone offset')
    p.add_argument('--trialspace-pause-file', type=Path, help='Pause student training when this file exists')
    p.add_argument('--trialspace-top-k', type=int, default=20)
    p.add_argument('--trialspace-top-k-patient-to-cohort', type=int)
    p.add_argument('--trialspace-top-k-cohort-to-patient', type=int, default=40)
    p.add_argument('--trialspace-lexical-k', type=int, default=5)
    p.add_argument('--trialspace-random-k', type=int, default=5)
    p.add_argument('--trialspace-patient-trial-sample', type=int, default=500,
                   help="Trials sampled per patient for patient-to-cohort mining, like one center's open trials (0 = all); "
                        'redrawn for each mining round')
    p.add_argument('--trialspace-label-workers', type=int, default=0,
                   help='Concurrent pair chains; 0 scales automatically with the live teacher pool')
    p.add_argument('--trialspace-label-max-auto-workers', type=int, default=DEFAULT_AUTO_WORKERS_MAX)
    p.add_argument('--checker-model', default=DEFAULT_CHECKER, help='Base model of TrialChecker and BoilerplateChecker')
    p.add_argument('--checker-max-length', type=int, default=8192,
                   help='Checker context: key or boilerplate criteria plus the patient section (ModernBERT maximum)')
    p.add_argument('--checker-epochs', type=float, default=2)
    p.add_argument('--checker-batch-size', type=int, default=8, help='Pairs per GPU per step')
    p.add_argument('--checker-learning-rate', type=float, default=2e-5)
    p.add_argument('--trialspace-cache-dir', type=Path, help='Local Hugging Face model cache')
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--vllm", default=str(Path(sys.executable).parent / "vllm"))
    p.add_argument("--teacher-max-context", type=int, default=250000)
    p.add_argument('--teacher-max-num-seqs', type=int, choices=range(1, 1025),
                   metavar='1..1024', default=256,
                   help='Sequence capacity for locally launched vLLM servers; request concurrency adapts below it')
    p.add_argument("--teacher-port", type=int, default=8100)
    p.add_argument("--teacher-concurrency", type=int, default=0,
                   help="Total downstream teacher request limit; 0 uses live per-server capacities")
    p.add_argument("--label-concurrency-per-server", type=int, choices=range(1, 1025),
                   metavar="1..1024", default=DEFAULT_CEILING,
                   help="Adaptive TrialSpace labeling ceiling per endpoint; live per-server limits can lower it.")
    p.add_argument('--label-initial-concurrency-per-server', type=int, choices=range(1, 1025),
                   metavar='1..1024', default=DEFAULT_INITIAL,
                   help='Starting target before automatic queue/cache/throughput adjustment')
    p.add_argument("--teacher-startup-timeout", type=int, default=1800)
    p.add_argument("--skip-input-validation", action="store_true",
                   help="Reuse source records from this run's existing manifest without reading or rehashing input files.")
    p.add_argument("--dry-run", action="store_true")
    return p


def input_paths(args):
    """Use the same explicit filenames in validation, manifests, and every stage."""
    defaults = dict(patient_summaries="patient_summaries_with_spaces.parquet",
                    synthetic_notes="all_synthetic_notes.parquet", trial_spaces="trial_cohort_lineitems.csv")
    return {key: (getattr(args, key, None) or args.data_dir / filename).expanduser().resolve()
            for key, filename in defaults.items()}


def teacher_args(args, servers):
    """Teacher endpoint, model and concurrency options shared by every labeling stage."""
    teacher_model = args.teacher_served_model or args.teacher_model
    command = ['--server_urls_file', str(servers), '--teacher-model', teacher_model,
        '--teacher-tokenizer', args.teacher_model, '--teacher-context', str(args.teacher_max_context),
        '--max_concurrent_per_server', str(args.label_concurrency_per_server),
        '--initial-concurrent-per-server', str(min(args.label_initial_concurrency_per_server, args.label_concurrency_per_server)),
        '--pending-requests', str(args.trialspace_label_workers),
        '--max-auto-pending-requests', str(args.trialspace_label_max_auto_workers), '--max_attempts', '16',
        '--request_timeout', '7200', '--scheduler-adaptive-concurrency']
    if args.trialspace_cache_dir:
        command += ['--cache-dir', str(args.trialspace_cache_dir)]
    if args.teacher_concurrency_file:
        command += ['--concurrency-control-file', str(args.teacher_concurrency_file)]
    return command


def stages(args):
    run, models = args.run_dir, args.models_dir
    inputs = input_paths(args)
    py = sys.executable
    servers = getattr(args, 'teacher_servers_file', None) or run / "teacher_servers.json"
    gpu_count = len(args.gpus.split(','))
    teacher_model = args.teacher_served_model or args.teacher_model
    launch = [py, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpu_count}"]
    corpus = run / 'trialspace_corpus'
    result = [Stage('prepare-trialspace', [py, 'trialspace_training.py', 'prepare',
        '--patients', str(inputs['patient_summaries']), '--cohorts', str(inputs['trial_spaces']),
        '--output', str(corpus),
        '--merge-source-splits'], [corpus / 'prepared.json'])]
    from synthetic_pool import require_all_sources
    require_all_sources(args.synthetic_training_splits)
    training_splits = args.synthetic_training_splits
    mining_splits = ['train', 'val', 'test'] if 'test' in training_splits else ['train', 'val']
    dense_py = str(args.embedding_python or py)
    dense_launch = [dense_py, *launch[1:]]  # One finetune_embedder rank per GPU in --gpus.
    embedding = args.embedding_model
    revision = args.embedding_revision or (EMBEDDINGGEMMA2_REVISION if embedding == DEFAULT_EMBEDDER else None)
    # Three mining and labeling rounds; TrialSpace is refit after rounds 1 and 2, every
    # round's labels train TrialChecker, and round 3's candidates supply the other stages.
    teacher_dirs = []
    for iteration in (1, 2, 3):
        round_dir = run / f'trialspace_round{iteration}'
        mining = round_dir / 'mining'
        candidates = [mining / f'{direction}.parquet' for direction in ('patient_to_cohort', 'cohort_to_patient')]
        mine = [dense_py, 'trialspace_comparison.py', 'dense-mine', '--corpus', str(corpus),
            '--output', str(mining), '--model', str(embedding), '--max-tokens', str(args.trialspace_context),
            '--patient-k', str(args.trialspace_top_k_patient_to_cohort or args.trialspace_top_k),
            '--cohort-k', str(args.trialspace_top_k_cohort_to_patient or args.trialspace_top_k),
            '--lexical-k', str(args.trialspace_lexical_k), '--random-k', str(args.trialspace_random_k),
            '--patient-trial-sample', str(args.trialspace_patient_trial_sample)]
        if iteration == 1 and revision:
            mine += ['--revision', revision]
        if args.trialspace_cache_dir:
            mine += ['--cache-dir', str(args.trialspace_cache_dir)]
        if iteration > 1:  # A fresh trial sample per patient each round; round 1 keeps the original draws.
            mine += ['--trial-sample-round', str(iteration)]
        result.append(Stage(f'mine-{iteration}', mine, [mining / 'complete.json', mining / 'candidates.jsonl', *candidates]))
        directory = round_dir / 'teacher'
        teacher_dirs.append(directory)
        scores = directory / 'scores.jsonl'
        label = [py, 'trialspace_training.py', 'label', '--task', 'match', '--corpus', str(corpus),
            '--candidates', str(mining / 'candidates.jsonl'), '--output', str(directory), *teacher_args(args, servers)]
        if iteration > 1:  # Pairs judged in the previous round are imported, not judged again.
            previous = run / f'trialspace_round{iteration - 1}'
            label += ['--reuse-labels', str(previous / 'teacher'),
                      '--reuse-candidates', str(previous / 'mining' / 'candidates.jsonl')]
        result.append(Stage(f'label-trialspace-{iteration}', label,
                            [scores, directory / 'labels.jsonl', directory / 'labeling_manifest.json'], True))
        if iteration == 3:
            break
        previous_embedding = embedding
        embedding = models / f"trialspace_round{iteration}"
        pairs = round_dir / 'pairs'
        result.append(Stage(f'project-trialspace-{iteration}', [dense_py, 'trialspace_comparison.py',
            'project-dense-labels', '--corpus', str(corpus), '--embeddinggemma', str(mining),
            '--labels', str(directory), '--output', str(pairs)],
            [pairs / 'embeddinggemma2' / 'complete.json', pairs / 'embeddinggemma2' / 'pairs.parquet']))
        command = dense_launch + ['finetune_embedder.py', '-i', str(pairs / 'embeddinggemma2' / 'pairs.parquet'),
            '-c', str(round_dir / 'fit'), '-o', str(embedding), '--base-model', str(previous_embedding),
            '--prompt-profile', 'trialspace', '--model-dtype', 'bfloat16',
            '--batch-size', str(args.trialspace_batch_size),
            '--loss-mini-batch-size', str(args.trialspace_loss_mini_batch_size), '--mask-known-positives',
            '--logging-steps', '25', '--save-steps', str(args.trialspace_save_steps)]
        command += (['--max-steps', str(args.trialspace_steps)] if args.trialspace_steps is not None
                    else ['--epochs', str(args.trialspace_epochs)])
        if iteration == 1 and revision:
            command += ['--model-revision', revision]
        if args.trialspace_cache_dir:
            command += ['--model-cache-dir', str(args.trialspace_cache_dir)]
        result.append(Stage(f'train-trialspace-{iteration}', command,
                            [embedding / 'modules.json', embedding / 'model.safetensors']))
    # BoilerplateChecker labels: the separate boilerplate prompt on every unique combination of patient
    # boilerplate section and cohort boilerplate criteria among the rounds 1-3 candidate pairs.
    boilerplate = run / 'boilerplate_teacher'
    result.append(Stage('label-boilerplate', [py, 'trialspace_training.py', 'label', '--task', 'boilerplate',
        '--corpus', str(corpus), '--candidates', str(mining / 'candidates.jsonl'), '--output', str(boilerplate),
        '--combine-candidates', *(str(run / f'trialspace_round{n}' / 'mining' / 'candidates.jsonl') for n in (1, 2)),
        *teacher_args(args, servers)],
        [boilerplate / 'labels.jsonl', boilerplate / 'labeling_manifest.json'], True))
    checker = [*launch, 'train_checkers.py']
    settings = ['--corpus', str(corpus), '--base-model', args.checker_model, '--max-length', str(args.checker_max_length),
                '--epochs', str(args.checker_epochs), '--batch-size', str(args.checker_batch_size),
                '--learning-rate', str(args.checker_learning_rate)]
    trial_checker, boilerplate_checker = models / 'trial_checker', models / 'boilerplate_checker'
    result.append(Stage('train-trial-checker', checker + ['trial', *settings, '--output-model', str(trial_checker),
        '--labels', *map(str, teacher_dirs)], [trial_checker / 'checker_contract.json', trial_checker / 'model.safetensors']))
    result.append(Stage('train-boilerplate-checker', checker + ['boilerplate', *settings,
        '--output-model', str(boilerplate_checker), '--labels', str(boilerplate)],
        [boilerplate_checker / 'checker_contract.json', boilerplate_checker / 'model.safetensors']))
    catalog = run / "good_option_catalog"
    result.append(Stage("catalog", [py, "train_good_option_checker.py", "catalog",
        "--catalog", str(catalog), "--catalog-checkpoint-dir", str(run / "catalog_checkpoints"),
        "--candidate-files", *map(str, candidates), "--model", teacher_model,
        "--server-urls-file", str(servers), "--max-concurrent-requests", str(args.teacher_concurrency)],
        [catalog / "manifest.json"], True))
    result.append(Stage("validate-catalog", [py, "train_good_option_checker.py", "validate-catalog",
        "--catalog", str(catalog)], [catalog / "manifest.json"]))
    # OncoReasoning asks the five match labels on key texts and each cohort boilerplate criterion
    # line on the patients' boilerplate sections, so no separate exclusion extraction runs.
    data = run / "oncoreasoning"
    program = [py, "oncoreasoning_training/create_all_training_data.py"]
    result.append(Stage("prepare-oncoreasoning", program + ["prepare", "--output-dir", str(data),
        "--notes", str(inputs['synthetic_notes']),
        "--pipeline-manifest", str(run / "pipeline_manifest.json"),
        "--candidates", *map(str, candidates), "--catalog", str(catalog),
        "--model-name", args.student_model,
        "--inference-repo", str(args.inference_repo),
        '--splits', *['validation' if split == 'val' else split for split in mining_splits]], [data / "prepared.json"]))
    result.append(Stage("distill-oncoreasoning", program + ["generate", "--output-dir", str(data),
        "--server-urls-file", str(servers), "--teacher-model", teacher_model,
        "--teacher-tokenizer", args.teacher_model,
        "--teacher-context", str(args.teacher_max_context), "--teacher-thinking",
        "--concurrency", str(args.teacher_concurrency)], [data / "generation_manifest.json"], True))
    result.append(Stage("build-oncoreasoning", program + ["build", "--output-dir", str(data)],
                        [data / "training_manifest.json", data / "tokenized_dataset"]))
    student = models / "oncoreasoning"
    result.append(Stage("train-oncoreasoning", launch + ["oncoreasoning_training/fine_tune_llm.py",
        "--data-dir", str(data), "--output-dir", str(student),
        "--gradient-accumulation-steps", str(max(1, 64 // gpu_count)),
        '--training-splits', *['validation' if split == 'val' else split for split in training_splits],
        "--fsdp", "--resume-from-checkpoint", "auto"],
        [student / "oncoreasoning_contract.json"]))
    return result


# Hybrid Mamba serving flags shared by the labeling fleet and the camus Nemotron servers.
NEMOTRON_SERVE_FLAGS = ["--trust-remote-code", "--async-scheduling", "--mamba-backend", "flashinfer",
    "--mamba-ssm-cache-dtype", "float16", "--enable-mamba-cache-stochastic-rounding",
    "--mamba-cache-philox-rounds", "5", "--chat-template-content-format", "string", "--mamba-cache-mode", "align"]


def teacher_command(args, index):
    nemotron = "nemotron" in args.teacher_model.lower()
    command = [args.vllm, "serve", args.teacher_model, "--host", "127.0.0.1",
        "--port", str(args.teacher_port + index), "--tensor-parallel-size", "1",
        *([] if nemotron else ["--quantization", "modelopt"]),
        "--reasoning-parser", resolve_parser_name(args.teacher_model),
        "--language-model-only", "--max-model-len", str(args.teacher_max_context),
        "--max-num-seqs", str(args.teacher_max_num_seqs), "--gpu-memory-utilization", "0.90",
        "--no-enable-log-requests", *(NEMOTRON_SERVE_FLAGS if nemotron else [])]
    tool_parser = resolve_tool_call_parser(args.teacher_model)
    if tool_parser:
        command += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
    return command


def teacher_environment(args):
    """Expose the pip CUDA toolkit when using a driver-only accelerator image."""
    env = dict(os.environ)
    executable = Path(shutil.which(args.vllm) or args.vllm).expanduser()
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    if not env.get("CUDA_HOME") and not shutil.which("nvcc"):
        python = executable.parent / "python"
        probe = (
            "import sysconfig; from pathlib import Path; "
            "root=Path(sysconfig.get_paths()['purelib'])/'nvidia'; "
            "print(next((str(p) for p in sorted(root.glob('cu*'), reverse=True) "
            "if (p/'bin/nvcc').is_file()), ''))"
        )
        cuda_home = subprocess.check_output([str(python), "-c", probe], text=True).strip()
        if cuda_home:
            env["CUDA_HOME"] = cuda_home
            env["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + env.get("PATH", "")
    return env


@contextmanager
def teacher_pool(args):
    processes, handles, servers = [], [], []
    server_file = args.run_dir / "teacher_servers.json"
    atomic_json(server_file, {"ready": False, "servers": []})
    try:
        environment = teacher_environment(args)
        for index, gpu in enumerate(args.gpus.split(',')):
            port = args.teacher_port + index
            handle = (args.run_dir / "logs" / f"teacher-{index}.log").open("a")
            handles.append(handle)
            env = {**environment, "CUDA_VISIBLE_DEVICES": gpu}
            processes.append(subprocess.Popen(teacher_command(args, index), env=env,
                             stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
            servers.append({"url": f"http://127.0.0.1:{port}/v1", "gpus": [int(gpu)], "kind": "local"})
        deadline = time.monotonic() + args.teacher_startup_timeout
        ready = set()
        while len(ready) != len(processes):
            for index, process in enumerate(processes):
                if process.poll() is not None:
                    raise RuntimeError(f"Teacher {index} exited; see its log")
                if index in ready:
                    continue
                try:
                    with urllib.request.urlopen(servers[index]["url"] + "/models", timeout=3) as response:
                        models = json.load(response)["data"]
                    if args.teacher_model not in [item["id"] for item in models]:
                        raise RuntimeError("Teacher endpoint reports an unexpected model")
                    ready.add(index)
                except (OSError, TimeoutError):
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError("Teacher pool startup timed out; see teacher logs")
            if len(ready) < len(processes):
                print(f"Waiting for teacher pool: {len(ready)}/{len(processes)} ready", flush=True)
                time.sleep(15)
        atomic_json(server_file, {"ready": True, "servers": servers})
        yield
    finally:
        atomic_json(server_file, {"ready": False, "servers": []})
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 45
        for process in processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()


@contextmanager
def managed_teacher_pool(args):
    """Read an external registry without taking ownership of it or its servers."""
    from oncoreasoning_training.teacher import server_urls
    path = args.teacher_servers_file
    registry = json.loads(path.read_text())
    entries = registry.get('servers', [])
    if not registry.get('ready', True) or not entries:
        raise ValueError('The external teacher registry has no ready servers')
    for entry in entries:
        url = entry if isinstance(entry, str) else entry['url']
        url = server_urls(url=url)[0].rstrip('/')
        with urllib.request.urlopen(url + '/models', timeout=15) as response:
            models = json.load(response)['data']
        served = args.teacher_served_model or args.teacher_model
        matches = [model for model in models if model['id'] == served]
        if not matches:
            raise ValueError(f'Teacher {url} does not serve {served}')
        root = entry.get('model_root', args.teacher_model) if isinstance(entry, dict) else args.teacher_model
        if args.teacher_served_model and matches[0].get('root') != root:
            raise ValueError(f'Teacher {url} does not report the declared model root {root}')
        if int(matches[0].get('max_model_len') or args.teacher_max_context) < args.teacher_max_context:
            raise ValueError(f'Teacher {url} has insufficient context')
    yield


@contextmanager
def external_teacher_pool(args):
    """Publish validated external endpoints without owning their processes."""
    from oncoreasoning_training.teacher import server_urls

    registry = args.run_dir / "teacher_servers.json"
    atomic_json(registry, {"ready": False, "servers": []})
    servers = []
    try:
        for url in dict.fromkeys(args.teacher_url):
            url = url.rstrip('/')
            if not url.endswith('/v1'):
                url += '/v1'
            url = server_urls(url=url)[0]
            with urllib.request.urlopen(url + '/models', timeout=15) as response:
                models = json.load(response)['data']
            served = args.teacher_served_model or args.teacher_model
            matches = [model for model in models if model['id'] == served]
            if not matches:
                raise ValueError(f"Teacher {url} does not serve {served}")
            if args.teacher_served_model and matches[0].get('root') != args.teacher_model:
                raise ValueError(f"Teacher alias {served} does not identify {args.teacher_model} as its root")
            context = matches[0].get('max_model_len')
            if context and int(context) < args.teacher_max_context:
                raise ValueError(f"Teacher {url} supports {context} tokens; lower --teacher-max-context")
            servers.append({"url": url, "kind": "remote",
                            # Other teacher stages use this as a fixed demand
                            # hint; only the adaptive labeler uses the ceiling.
                            "upstream_concurrency": min(args.label_initial_concurrency_per_server,
                                                        args.label_concurrency_per_server)})
        if not servers:
            raise ValueError("Supply at least one --teacher-url")
        atomic_json(registry, {"ready": True, "servers": servers})
        yield
    finally:
        atomic_json(registry, {"ready": False, "servers": []})


def git_revision(directory):
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def input_records(files, manifest_path, skip_validation=False):
    if not skip_validation:
        return [source_record(path) for path in files]
    if not manifest_path.is_file():
        raise ValueError("--skip-input-validation requires an existing run manifest")
    records = json.loads(manifest_path.read_text())["inputs"]
    if [record["path"] for record in records] != list(map(str, files)):
        raise ValueError("Existing source records refer to different input paths")
    print("[inputs] Reusing recorded input identities; input validation skipped", flush=True)
    return records


def main():
    args = parser().parse_args()
    for key in ("data_dir", "run_dir", "models_dir", "inference_repo"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    for key in ("trial_spaces", "patient_summaries", "synthetic_notes", "teacher_servers_file", "teacher_concurrency_file",
                'trialspace_cache_dir'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.teacher_servers_file and args.teacher_url:
        raise ValueError('Use either --teacher-servers-file or --teacher-url')
    plan = stages(args)
    if args.dry_run:
        print(json.dumps([vars(stage) for stage in plan], default=str, indent=2))
        return
    if not args.data_dir.is_relative_to(ROOT.parent / "data/no_phi"):
        raise ValueError("The integrated runner requires the authorized data/no_phi directory")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "logs").mkdir(exist_ok=True)
    lock = (args.run_dir / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = list(input_paths(args).values())
    if any(not path.is_relative_to(ROOT.parent / "data/no_phi") for path in files):
        raise ValueError("The integrated runner requires inputs in the authorized data/no_phi directory")
    manifest_path = args.run_dir / "pipeline_manifest.json"
    identity = {"args": training_identity_arguments(args),
                "training_commit": git_revision(ROOT), "inference_commit": git_revision(args.inference_repo),
                "match_labels_digest": MATCH_LABELS_DIGEST,
                "inputs": input_records(files, manifest_path, args.skip_input_validation)}
    from trialspace.regression import score_contract
    identity['trialspace_contract'] = score_contract()
    source_paths = [ROOT / 'train_from_summaries.py', ROOT / 'trialspace_training.py',
                    ROOT / 'retrieval_workspace.py', ROOT / 'label_workspace.py', ROOT / 'synthetic_pool.py', ROOT / 'retrieval_cache_import.py',
                    ROOT / 'trial_cohort_format.py',
                    ROOT / 'remote_vllm_pool.py', ROOT / 'adaptive_controller.py',
                    ROOT / 'train_good_option_checker.py', ROOT / 'train_checkers.py',
                    *sorted((ROOT / 'trialspace').glob('*.py')),
                    ROOT / 'oncoreasoning_training/create_all_training_data.py',
                    ROOT / 'oncoreasoning_training/teacher.py',
                    args.inference_repo / 'src/matchminer_ai/llm/traces.py',
                    args.inference_repo / 'src/matchminer_ai/llm/async_processing.py',
                    args.inference_repo / 'src/matchminer_ai/llm/remote_inference.py',
                    args.inference_repo / 'src/matchminer_ai/llm/endpoint_pool.py',
                    args.inference_repo / 'src/matchminer_ai/llm/dispatch.py',
                    *sorted((args.inference_repo / 'src/matchminer_ai/embedding').glob('*.py')),
                    args.inference_repo / 'src/matchminer_ai/matching/match_labels.py',
                    *(args.inference_repo / 'src/matchminer_ai/prompts' / name for name in (
                        'match_quality.labels.system.txt', 'llm_exclusion_criteria.user.txt',
                        'match_quality_checker_template.txt', 'exclusion_criteria_checker_template.txt'))]
    identity['production_sources'] = [source_record(path) for path in source_paths]
    if manifest_path.exists():
        from oncoreasoning_training.pipeline_upgrade import compatible_identity
        identity = compatible_identity(json.loads(manifest_path.read_text()), identity, args.run_dir, ROOT)
    atomic_json(manifest_path, identity)
    atomic_json(args.run_dir / 'trialspace_runtime_controls.json',
                {key: str(getattr(args, key)) for key in TRIALSPACE_OPERATIONAL_OPTIONS})
    atomic_json(args.run_dir / 'reasoning_archives.json', dict(
        policy='Preserve returned reasoning and complete responses; answer-only targets remain separate.',
        pooled_teacher_traces=str(args.run_dir / 'llm_traces'),
        match_label_traces=[str(args.run_dir / f'trialspace_round{i}/teacher/responses') for i in (1, 2, 3)],
        boilerplate_label_traces=str(args.run_dir / 'boilerplate_teacher/responses'),
        distillation_traces=str(args.run_dir / 'oncoreasoning/teacher_attempts'),
        distillation_validated_responses=str(args.run_dir / 'oncoreasoning/responses')))
    fingerprint = digest(identity)
    os.environ["PYTHONPATH"] = str(args.inference_repo / "src") + os.pathsep + str(ROOT)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ["PYTHONUNBUFFERED"] = "1"
    # Distributed mining and fitting use ranks within the selected visible GPUs.
    pool = None
    try:
        for stage in plan:
            stamp = args.run_dir / "completed" / (stage.name + ".json")
            if stamp.exists():
                if json.loads(stamp.read_text())["fingerprint"] != fingerprint or not all(p.exists() for p in stage.outputs):
                    raise ValueError(f"Invalid completion marker for {stage.name}")
                print(f"[skip] {stage.name}", flush=True)
                continue
            if pool and not stage.teacher:
                pool.__exit__(None, None, None)
                pool = None
            if stage.teacher and not pool:
                pool = (managed_teacher_pool(args) if args.teacher_servers_file else
                        external_teacher_pool(args) if args.teacher_url else teacher_pool(args))
                pool.__enter__()
            atomic_json(args.run_dir / "status.json", {"stage": stage.name, "status": "running"})
            print(f"[start] {stage.name}", flush=True)
            # Each command emits progress but no clinical rows to its own log.
            with (args.run_dir / "logs" / (stage.name + ".log")).open("a") as log:
                env = dict(os.environ)
                env['MMAI_LLM_TRACE_DIR'] = str(args.run_dir / 'llm_traces' / stage.name)
                env['MMAI_LLM_TRACE_STAGE'] = stage.name
                if "torch.distributed.run" in stage.command:
                    env["CUDA_VISIBLE_DEVICES"] = args.gpus
                subprocess.run(stage.command, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            if not all(path.exists() for path in stage.outputs):
                raise RuntimeError(f"Missing outputs from {stage.name}")
            atomic_json(stamp, {"fingerprint": fingerprint, "outputs": list(map(str, stage.outputs))})
            print(f"[done] {stage.name}", flush=True)
        atomic_json(args.run_dir / "status.json", {"status": "complete"})
    except BaseException:
        atomic_json(args.run_dir / "status.json", {"stage": stage.name, "status": "failed"})
        raise
    finally:
        if pool:
            pool.__exit__(None, None, None)


if __name__ == "__main__":
    main()
