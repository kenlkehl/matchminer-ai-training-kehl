#!/usr/bin/env python3
"""Resumable TrialSpace → catalog → answer-first OncoReasoning pipeline.

Only two student models are trained. Existing non-PHI summaries seed mining;
the NVIDIA teacher provides all fresh labels, catalog evidence, and distillation.
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

from oncoreasoning_training.contracts import DEFAULT_MODEL_NAME, TRIAL_RUBRIC_DIGEST, digest
from oncoreasoning_training.create_all_training_data import atomic_json, source_record
from oncoreasoning_training.teacher import DEFAULT_TEACHER
from vllm_reasoning_utils import resolve_parser_name, resolve_tool_call_parser

ROOT = Path(__file__).resolve().parent
INFERENCE_COMMIT = "1f17ef0ee29d039befb1f13f4780674dc634dd2f"


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
    p.add_argument("--run-dir", type=Path, default=ROOT.parent / "data/no_phi/colbert_pipeline_v1")
    p.add_argument("--models-dir", type=Path, default=ROOT.parent / "models/colbert_pipeline_v1")
    p.add_argument("--inference-repo", type=Path, default=ROOT.parent / "matchminer-ai-inference-kehl")
    p.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    p.add_argument("--teacher-served-model", help="API model alias; --teacher-model remains the underlying HF model")
    p.add_argument("--teacher-url", action="append", default=[],
                   help="Existing OpenAI-compatible teacher URL; repeat for a pool. Skips local vLLM startup.")
    p.add_argument("--teacher-servers-file", type=Path,
                   help="Externally managed live endpoint registry. Never rewritten or shut down by this runner.")
    p.add_argument("--teacher-concurrency-file", type=Path,
                   help="Live per-server limits for TrialSpace labeling.")
    p.add_argument("--student-model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--embedding-model", default="lightonai/GTE-ModernColBERT-v1")
    p.add_argument('--synthetic-training-splits', nargs='+', choices=['train', 'val', 'test'],
                   default=['train', 'val', 'test'], help='Synthetic source partitions used by every student fit')
    p.add_argument('--trialspace-context', type=int, default=8192)
    p.add_argument('--trialspace-steps', type=int, help='Explicit optimizer updates per fit; otherwise derive from query count')
    p.add_argument('--trialspace-epochs', type=float, default=3, help='Aggregate query-group sampling passes per fit')
    p.add_argument('--trialspace-accumulation', type=int, default=32,
                   help='Query groups per GPU per optimizer update, sampled across available scores')
    p.add_argument('--trialspace-documents-per-query', type=int, default=8,
                   help='Full-text labeled candidates encoded together for each training query')
    p.add_argument('--trialspace-learning-rate', type=float, default=1e-5)
    p.add_argument('--trialspace-candidate-limit', type=int, default=256)
    p.add_argument('--trialspace-top-k', type=int, default=20)
    p.add_argument('--trialspace-top-k-patient-to-cohort', type=int)
    p.add_argument('--trialspace-top-k-cohort-to-patient', type=int, default=40)
    p.add_argument('--trialspace-lexical-k', type=int, default=5)
    p.add_argument('--trialspace-random-k', type=int, default=5)
    p.add_argument('--trialspace-encode-batch-size', type=int, default=16)
    p.add_argument('--trialspace-score-batch-size', type=int, default=16)
    p.add_argument('--trialspace-pair-batch-size', type=int, default=512)
    p.add_argument('--trialspace-score-query-batches', type=int, default=4)
    p.add_argument('--trialspace-device-cache-gb', type=float, default=8)
    p.add_argument('--trialspace-label-workers', type=int, default=1024)
    p.add_argument('--trialspace-extra-pairs', type=Path, help='Additional independent/challenge candidate JSONL')
    p.add_argument('--trialspace-cache-dir', type=Path, help='Local Hugging Face model cache')
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--vllm", default=str(Path(sys.executable).parent / "vllm"))
    p.add_argument("--teacher-max-context", type=int, default=250000)
    p.add_argument("--teacher-port", type=int, default=8100)
    p.add_argument("--teacher-concurrency", type=int, default=0,
                   help="Total downstream teacher request limit; 0 uses live per-server capacities")
    p.add_argument("--label-concurrency-per-server", type=int, choices=range(1, 1025),
                   metavar="1..1024", default=96,
                   help="TrialSpace labeling concurrency per endpoint; use 288 for the shared sn4622130540 teacher.")
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
        '--merge-source-splits' if set(args.synthetic_training_splits) == {'train', 'val', 'test'}
        else '--no-merge-source-splits'], [corpus / 'prepared.json'])]
    common = ['--corpus', str(corpus), '--query-length', str(args.trialspace_context),
              '--document-length', str(args.trialspace_context)]
    training_splits = args.synthetic_training_splits
    mining_splits = ['train', 'val', 'test'] if 'test' in training_splits else ['train', 'val']
    if args.trialspace_cache_dir:
        common += ['--cache-dir', str(args.trialspace_cache_dir)]
    embedding = args.embedding_model
    for iteration in (1, 2, 3):
        round_dir = run / f'trialspace_round{iteration}'
        mining = round_dir / 'mining'
        candidates = [mining / f'{direction}.parquet' for direction in ('patient_to_cohort', 'cohort_to_patient')]
        mine = launch + ['trialspace_training.py', 'mine', *common, '--model', str(embedding),
            '--output', str(mining), '--candidate-limit', str(args.trialspace_candidate_limit),
            '--proposal-strategy', 'lexical_random', '--top-k', str(args.trialspace_top_k),
            '--lexical-k', str(args.trialspace_lexical_k), '--random-k', str(args.trialspace_random_k),
            '--document-batch-size', str(args.trialspace_encode_batch_size),
            '--query-batch-size', str(args.trialspace_encode_batch_size),
            '--score-batch-size', str(args.trialspace_score_batch_size),
            '--pair-batch-size', str(args.trialspace_pair_batch_size),
            '--score-query-batches', str(args.trialspace_score_query_batches),
            '--device-cache-gb', str(args.trialspace_device_cache_gb)]
        mine += ['--splits', *mining_splits]
        if args.trialspace_extra_pairs:
            mine += ['--extra-pairs', str(args.trialspace_extra_pairs)]
        for direction in ('patient-to-cohort', 'cohort-to-patient'):
            value = getattr(args, 'trialspace_top_k_' + direction.replace('-', '_'), None)
            if value is not None:
                mine += ['--top-k-' + direction, str(value)]
        result.append(Stage(f'mine-{iteration}', mine, [mining / 'complete.json', mining / 'candidates.jsonl', *candidates]))
        if iteration == 3:
            break
        directory = round_dir / 'teacher'
        scores = directory / 'scores.jsonl'
        label = [py, 'trialspace_training.py', 'label', '--gated-regression', '--corpus', str(corpus),
            '--candidates', str(mining / 'candidates.jsonl'), '--output', str(directory),
            '--server_urls_file', str(servers), '--teacher-model', teacher_model,
            '--teacher-tokenizer', args.teacher_model, '--teacher-context', str(args.teacher_max_context),
            '--max_concurrent_per_server', str(args.label_concurrency_per_server),
            '--initial-concurrent-per-server', str(min(16, args.label_concurrency_per_server)),
            '--pending-requests', str(args.trialspace_label_workers), '--max_attempts', '5']
        if args.trialspace_cache_dir:
            label += ['--cache-dir', str(args.trialspace_cache_dir)]
        if args.teacher_concurrency_file:
            label += ['--concurrency-control-file', str(args.teacher_concurrency_file)]
        result.append(Stage(f'label-trialspace-{iteration}', label,
                            [scores, directory / 'labels.jsonl', directory / 'labeling_manifest.json'], True))
        previous_embedding = embedding
        embedding = models / f"trialspace_round{iteration}"
        command = launch + ['trialspace_training.py', 'train', *common, '--model', str(previous_embedding),
            '--labels', str(scores), '--output', str(round_dir / 'fit'), '--output-model', str(embedding),
            '--objective', 'gated_regression', '--prediction-link', 'identity',
            '--accumulation', str(args.trialspace_accumulation), '--learning-rate', str(args.trialspace_learning_rate),
            '--documents-per-query', str(args.trialspace_documents_per_query),
            '--training-splits', *training_splits,
            '--eval-every', '100', '--eval-groups-per-stratum', '32']
        command += (['--steps', str(args.trialspace_steps)] if args.trialspace_steps is not None
                    else ['--epochs', str(args.trialspace_epochs)])
        result.append(Stage(f'train-trialspace-{iteration}', command,
                            [embedding / 'regression_scoring.json', embedding / 'model.safetensors']))
    catalog = run / "good_option_catalog"
    result.append(Stage("catalog", [py, "train_good_option_checker.py", "catalog",
        "--catalog", str(catalog), "--catalog-checkpoint-dir", str(run / "catalog_checkpoints"),
        "--candidate-files", *map(str, candidates), "--model", teacher_model,
        "--server-urls-file", str(servers), "--max-concurrent-requests", str(args.teacher_concurrency)],
        [catalog / "manifest.json"], True))
    result.append(Stage("validate-catalog", [py, "train_good_option_checker.py", "validate-catalog",
        "--catalog", str(catalog)], [catalog / "manifest.json"]))
    exclusions = run / "boilerplate_components"
    result.append(Stage("prepare-exclusions", [py, "oncoreasoning_training/prepare_boilerplate.py",
        "--candidates", *map(str, candidates), "--output-dir", str(exclusions),
        "--server-urls-file", str(servers), "--teacher-model", teacher_model,
        "--concurrency", str(args.teacher_concurrency)], [exclusions / "manifest.json"], True))
    data = run / "oncoreasoning"
    program = [py, "oncoreasoning_training/create_all_training_data.py"]
    result.append(Stage("prepare-oncoreasoning", program + ["prepare", "--output-dir", str(data),
        "--notes", str(inputs['synthetic_notes']),
        "--pipeline-manifest", str(run / "pipeline_manifest.json"),
        "--candidates", *map(str, candidates), "--catalog", str(catalog),
        "--boilerplate-components", str(exclusions), "--model-name", args.student_model,
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


def teacher_command(args, index):
    command = [args.vllm, "serve", args.teacher_model, "--host", "127.0.0.1",
        "--port", str(args.teacher_port + index), "--tensor-parallel-size", "1",
        "--quantization", "modelopt", "--reasoning-parser", resolve_parser_name(args.teacher_model),
        "--language-model-only", "--max-model-len", str(args.teacher_max_context),
        "--max-num-seqs", "96", "--gpu-memory-utilization", "0.90",
        "--no-enable-log-requests"]
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
                            "upstream_concurrency": args.label_concurrency_per_server})
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
                'trialspace_extra_pairs', 'trialspace_cache_dir'):
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
    if git_revision(args.inference_repo) != INFERENCE_COMMIT:
        raise ValueError("Use the pinned inference revision; review prompt/catalog changes before a new run")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "logs").mkdir(exist_ok=True)
    lock = (args.run_dir / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = list(input_paths(args).values())
    if any(not path.is_relative_to(ROOT.parent / "data/no_phi") for path in files):
        raise ValueError("The integrated runner requires inputs in the authorized data/no_phi directory")
    manifest_path = args.run_dir / "pipeline_manifest.json"
    identity = {"args": {key: str(value) for key, value in vars(args).items()
                         if key not in ("dry_run", "skip_input_validation")
                         and not (key in ("trial_spaces", "patient_summaries", "synthetic_notes", "teacher_servers_file", "teacher_concurrency_file") and value is None)},
                "training_commit": git_revision(ROOT), "inference_commit": INFERENCE_COMMIT,
                "trial_rubric_digest": TRIAL_RUBRIC_DIGEST,
                "inputs": input_records(files, manifest_path, args.skip_input_validation)}
    from colbert_experiments.regression import score_contract
    identity['trialspace_contract'] = score_contract()
    source_paths = [ROOT / 'train_from_summaries.py', ROOT / 'trialspace_training.py',
                    ROOT / 'train_good_option_checker.py',
                    *sorted((ROOT / 'colbert_experiments').glob('*.py')),
                    *sorted((ROOT / 'colbert_experiments/prompts').glob('*.txt')),
                    ROOT / 'oncoreasoning_training/create_all_training_data.py',
                    ROOT / 'oncoreasoning_training/prepare_boilerplate.py',
                    ROOT / 'oncoreasoning_training/teacher.py',
                    args.inference_repo / 'src/matchminer_ai/llm/traces.py',
                    args.inference_repo / 'src/matchminer_ai/llm/endpoint_pool.py',
                    *sorted((args.inference_repo / 'src/matchminer_ai/embedding').glob('*.py'))]
    identity['production_sources'] = [source_record(path) for path in source_paths]
    if manifest_path.exists():
        from oncoreasoning_training.pipeline_upgrade import compatible_identity
        identity = compatible_identity(json.loads(manifest_path.read_text()), identity, args.run_dir, ROOT)
    atomic_json(manifest_path, identity)
    atomic_json(args.run_dir / 'reasoning_archives.json', dict(
        policy='Preserve returned reasoning and complete responses; answer-only targets remain separate.',
        pooled_teacher_traces=str(args.run_dir / 'llm_traces'),
        colbert_question_traces=[str(args.run_dir / f'trialspace_round{i}/teacher/responses') for i in (1, 2)],
        distillation_traces=str(args.run_dir / 'oncoreasoning/teacher_attempts'),
        distillation_validated_responses=str(args.run_dir / 'oncoreasoning/responses'),
        exclusion_traces=str(args.run_dir / 'boilerplate_components/teacher_attempts')))
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
