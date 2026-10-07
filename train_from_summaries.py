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

from oncoreasoning_training.contracts import DEFAULT_MODEL_NAME, digest
from oncoreasoning_training.create_all_training_data import atomic_json, source_record
from oncoreasoning_training.teacher import DEFAULT_TEACHER

ROOT = Path(__file__).resolve().parent
INFERENCE_COMMIT = "6e38839e46a61d2583dc6c72dd6c672a19cd6a94"


@dataclass
class Stage:
    name: str
    command: list[str]
    outputs: list[Path]
    teacher: bool = False


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT.parent / "data/no_phi")
    p.add_argument("--run-dir", type=Path, default=ROOT.parent / "data/no_phi/gemma_pipeline_v2")
    p.add_argument("--models-dir", type=Path, default=ROOT.parent / "models/gemma_pipeline_v2")
    p.add_argument("--inference-repo", type=Path, default=ROOT.parent / "matchminer-ai-inference-kehl")
    p.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    p.add_argument("--student-model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--embedding-model", default="google/embeddinggemma-2")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--vllm", default=str(Path(sys.executable).parent / "vllm"))
    p.add_argument("--teacher-max-context", type=int, default=262144)
    p.add_argument("--teacher-port", type=int, default=8100)
    p.add_argument("--teacher-concurrency", type=int, default=32)
    p.add_argument("--teacher-startup-timeout", type=int, default=1800)
    p.add_argument("--dry-run", action="store_true")
    return p


def stages(args):
    run, models = args.run_dir, args.models_dir
    py = sys.executable
    servers = run / "teacher_servers.json"
    gpu_count = len(args.gpus.split(','))
    launch = [py, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpu_count}"]
    result, labels = [], []
    embedding = args.embedding_model
    for iteration in (1, 2, 3):
        candidates = [run / f"top_{direction}_tocheck_round{iteration}.parquet" for direction in ("cohorts", "patients")]
        result.append(Stage(f"mine-{iteration}", [py, "make_top_matches.py",
            "--patients-parquet", str(args.data_dir / "patient_summaries_with_spaces.parquet"),
            "--trials-file", str(args.data_dir / "trial_space_lineitems.csv"),
            "--model", str(embedding), "--gpus", args.gpus,
            "--sample_trials_per_patient", "500", "--sample_patients_per_trial", "20000",
            "--top_k_spaces", "20", "--top_k_patients", "40",
            "--encode_batch_size", "32", "--max_seq_length", "2500",
            "--out_cohorts_parquet", str(candidates[0]), "--out_patients_parquet", str(candidates[1])], candidates))
        if iteration == 3:
            break
        for candidate in candidates:
            directory = run / (candidate.stem + "_labels")
            output = directory / "labels.parquet"
            result.append(Stage(f"label-{candidate.stem}", [py, "llm_check_trials.py",
                "--input_parquet", str(candidate), "--out_dir", str(directory),
                "--final_output", "labels.parquet", "--server_urls_file", str(servers),
                "--model", args.teacher_model, "--reasoning-parser", "gemma4",
                "--max_concurrent_per_server", "64", "--max_attempts", "5"], [output], True))
            labels.append(output)
        embedding = models / f"trialspace_round{iteration}"
        command = launch + ["finetune_embedder.py", "--base-model", args.embedding_model,
            "--ckpt-dir", str(models / f"trialspace_round{iteration}_checkpoints"),
            "--output-model", str(embedding)]
        for path in labels:
            command += ["--input-parquet", str(path)]
        result.append(Stage(f"train-trialspace-{iteration}", command, [embedding / "modules.json"]))
    catalog = run / "good_option_catalog"
    result.append(Stage("catalog", [py, "train_good_option_checker.py", "catalog",
        "--catalog", str(catalog), "--catalog-checkpoint-dir", str(run / "catalog_checkpoints"),
        "--candidate-files", *map(str, candidates), "--model", args.teacher_model,
        "--server-urls-file", str(servers), "--max-concurrent-requests", str(args.teacher_concurrency)],
        [catalog / "manifest.json"], True))
    result.append(Stage("validate-catalog", [py, "train_good_option_checker.py", "validate-catalog",
        "--catalog", str(catalog)], [catalog / "manifest.json"]))
    exclusions = run / "boilerplate_components"
    result.append(Stage("prepare-exclusions", [py, "oncoreasoning_training/prepare_boilerplate.py",
        "--candidates", *map(str, candidates), "--output-dir", str(exclusions),
        "--server-urls-file", str(servers), "--teacher-model", args.teacher_model,
        "--concurrency", str(args.teacher_concurrency)], [exclusions / "manifest.json"], True))
    data = run / "oncoreasoning"
    program = [py, "oncoreasoning_training/create_all_training_data.py"]
    result.append(Stage("prepare-oncoreasoning", program + ["prepare", "--output-dir", str(data),
        "--notes", str(args.data_dir / "all_synthetic_notes.parquet"),
        "--candidates", *map(str, candidates), "--catalog", str(catalog),
        "--boilerplate-components", str(exclusions), "--model-name", args.student_model,
        "--inference-repo", str(args.inference_repo)], [data / "prepared.json"]))
    result.append(Stage("distill-oncoreasoning", program + ["generate", "--output-dir", str(data),
        "--server-urls-file", str(servers), "--teacher-model", args.teacher_model,
        "--teacher-context", str(args.teacher_max_context), "--teacher-thinking",
        "--concurrency", str(args.teacher_concurrency)], [data / "generation_manifest.json"], True))
    result.append(Stage("build-oncoreasoning", program + ["build", "--output-dir", str(data)],
                        [data / "training_manifest.json", data / "tokenized_dataset"]))
    student = models / "oncoreasoning"
    result.append(Stage("train-oncoreasoning", launch + ["oncoreasoning_training/fine_tune_llm.py",
        "--data-dir", str(data), "--output-dir", str(student), "--fsdp", "--resume-from-checkpoint", "auto"],
        [student / "oncoreasoning_contract.json"]))
    return result


def teacher_command(args, index):
    return [args.vllm, "serve", args.teacher_model, "--host", "127.0.0.1",
        "--port", str(args.teacher_port + index), "--tensor-parallel-size", "1",
        "--quantization", "modelopt", "--reasoning-parser", "gemma4",
        "--language-model-only", "--max-model-len", str(args.teacher_max_context),
        "--max-num-seqs", "64", "--gpu-memory-utilization", "0.90",
        "--no-enable-log-requests"]


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


def git_revision(directory):
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def main():
    args = parser().parse_args()
    for key in ("data_dir", "run_dir", "models_dir", "inference_repo"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
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
    files = [args.data_dir / name for name in ("patient_summaries_with_spaces.parquet",
             "all_synthetic_notes.parquet", "trial_space_lineitems.csv")]
    identity = {"args": {key: str(value) for key, value in vars(args).items() if key != "dry_run"},
                "training_commit": git_revision(ROOT), "inference_commit": INFERENCE_COMMIT,
                "inputs": [source_record(path) for path in files]}
    manifest_path = args.run_dir / "pipeline_manifest.json"
    if manifest_path.exists():
        from oncoreasoning_training.pipeline_upgrade import compatible_identity
        identity = compatible_identity(json.loads(manifest_path.read_text()), identity, args.run_dir, ROOT)
    atomic_json(manifest_path, identity)
    fingerprint = digest(identity)
    os.environ["PYTHONPATH"] = str(args.inference_repo / "src") + os.pathsep + str(ROOT)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ["PYTHONUNBUFFERED"] = "1"
    # Mining selects physical devices itself; torchrun selects the visible ranks.
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
                pool = teacher_pool(args)
                pool.__enter__()
            atomic_json(args.run_dir / "status.json", {"stage": stage.name, "status": "running"})
            print(f"[start] {stage.name}", flush=True)
            # Each command emits progress but no clinical rows to its own log.
            with (args.run_dir / "logs" / (stage.name + ".log")).open("a") as log:
                env = dict(os.environ)
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
