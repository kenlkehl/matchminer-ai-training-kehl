#!/usr/bin/env python3
"""
GCP-based vLLM orchestrator for the matchminer-ai-training pipeline.

This program is invoked by train_all_gcp.sh around each vLLM step. It:

  - Ensures a configurable list of GCP instances (worker VMs) are running,
    reusing workers that are already RUNNING;
  - SSHes into each instance to count GPUs and launch one vLLM OpenAI
    server per (gpus_per_server) GPU group;
  - Writes the canonical list of healthy server URLs to a JSON file
    that the modified inference scripts re-read via
    remote_vllm_pool.DynamicServerRegistry;
  - Maintains health by re-launching dead vLLM processes and re-starting
    stopped/preempted instances on a periodic loop;
  - On SIGTERM/SIGINT, kills the SSH-launched vLLM processes and the
    orchestrator's own local vLLM servers (if --include-self was used)
    so the bash wrapper can move on.

GCP transport: all instance and SSH operations go through the `gcloud`
CLI (subprocess); no extra Python dependencies. The orchestrator
assumes orchestrator and workers share a VPC: the inference scripts
connect to workers via their *internal* IPs. If you need external IPs,
replace `networkInterfaces[0].networkIP` with
`networkInterfaces[0].accessConfigs[0].natIP` in _describe_instance.

Subcommands
-----------
  start-instances  Ensure every instance in the file is RUNNING (parallel),
                   skipping workers that are already RUNNING.
  stop-instances   Stop every instance in the file (parallel).
  serve            The main loop. Ensure instances are running, launch vLLM
                   servers, write/maintain the servers JSON, restart dead
                   things. Runs until SIGTERM/SIGINT.
  wait-for-ready   Poll the servers JSON until at least one healthy
                   server is present (or timeout). Used by bash.

Instances file format (`#` comments allowed):

    # name:zone
    worker-a100-1:us-central1-a
    worker-a100-2:us-central1-a
    worker-h100-1:us-east1-b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import urllib.request
import urllib.error


# -------------------------
# Constants / defaults
# -------------------------

DEFAULT_BASE_PORT = 8000
DEFAULT_DOWNLOAD_DIR = "~/models"
DEFAULT_HEALTH_INTERVAL = 20.0
DEFAULT_RESTART_INTERVAL = 120.0
DEFAULT_SERVER_READY_TIMEOUT = 1800  # 30 min: weights can be huge
DEFAULT_STARTUP_GRACE = 900.0  # 15 min: vLLM model load can take this long

PID_FILE_FMT = "/tmp/mmai_vllm_{port}.pid"
LOG_FILE_FMT = "/tmp/mmai_vllm_{port}.log"


# -------------------------
# Instance file parsing
# -------------------------

@dataclass
class InstanceSpec:
    name: str
    zone: str


@dataclass
class EnsureInstanceResult:
    ok: bool
    started: bool
    status: str
    message: str = ""


def parse_instances_file(path: str) -> List[InstanceSpec]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"instances file not found: {path}")
    out: List[InstanceSpec] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 2:
                raise ValueError(
                    f"bad line in {path}: {line!r} "
                    "(expected 'name:zone' optionally followed by extras)"
                )
            out.append(InstanceSpec(name=parts[0].strip(), zone=parts[1].strip()))
    if not out:
        raise ValueError(f"{path} contains no instance entries")
    return out


# -------------------------
# gcloud subprocess helpers
# -------------------------

async def _run(cmd: List[str], *, timeout: Optional[float] = None) -> Tuple[int, str, str]:
    """Run a subprocess, return (rc, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return (124, "", f"timeout after {timeout}s")
    return (proc.returncode or 0, stdout.decode("utf-8", "replace"),
            stderr.decode("utf-8", "replace"))


async def gcloud_start(name: str, zone: str) -> Tuple[bool, str]:
    rc, _, err = await _run(
        ["gcloud", "compute", "instances", "start", name, f"--zone={zone}", "--quiet"],
        timeout=600,
    )
    return rc == 0, err.strip()


async def gcloud_stop(name: str, zone: str) -> Tuple[bool, str]:
    rc, _, err = await _run(
        ["gcloud", "compute", "instances", "stop", name, f"--zone={zone}", "--quiet"],
        timeout=600,
    )
    return rc == 0, err.strip()


async def gcloud_describe(name: str, zone: str) -> Optional[dict]:
    rc, out, _ = await _run(
        ["gcloud", "compute", "instances", "describe", name,
         f"--zone={zone}", "--format=json"],
        timeout=60,
    )
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


async def wait_for_instance_running(
    spec: InstanceSpec,
    *,
    timeout: float = 600.0,
    poll_interval: float = 10.0,
) -> Tuple[bool, str]:
    """Poll GCE until the instance reports RUNNING, or return the last status."""
    deadline = time.monotonic() + timeout
    last_status = "UNKNOWN"
    while True:
        desc = await gcloud_describe(spec.name, spec.zone)
        last_status = (desc or {}).get("status", "UNKNOWN")
        if last_status == "RUNNING":
            return True, last_status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, last_status
        await asyncio.sleep(min(poll_interval, remaining))


async def wait_while_instance_status(
    spec: InstanceSpec,
    statuses: set[str],
    *,
    timeout: float = 120.0,
    poll_interval: float = 10.0,
) -> str:
    """Poll until the instance leaves one of the given statuses."""
    deadline = time.monotonic() + timeout
    last_status = "UNKNOWN"
    while True:
        desc = await gcloud_describe(spec.name, spec.zone)
        last_status = (desc or {}).get("status", "UNKNOWN")
        if last_status not in statuses:
            return last_status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return last_status
        await asyncio.sleep(min(poll_interval, remaining))


async def ensure_instance_running(spec: InstanceSpec) -> EnsureInstanceResult:
    """Make the instance RUNNING, without issuing a start for an already-running VM."""
    desc = await gcloud_describe(spec.name, spec.zone)
    status = (desc or {}).get("status", "UNKNOWN")

    if status == "RUNNING":
        return EnsureInstanceResult(
            ok=True,
            started=False,
            status=status,
            message="already RUNNING; reusing",
        )

    if status in {"PROVISIONING", "STAGING", "REPAIRING"}:
        ok, final_status = await wait_for_instance_running(spec)
        return EnsureInstanceResult(
            ok=ok,
            started=False,
            status=final_status,
            message=(
                f"already starting ({status}); reached RUNNING"
                if ok else f"already starting ({status}); timed out at {final_status}"
            ),
        )

    if status in {"STOPPING", "SUSPENDING"}:
        final_status = await wait_while_instance_status(
            spec,
            {"STOPPING", "SUSPENDING"},
        )
        if final_status == "RUNNING":
            return EnsureInstanceResult(
                ok=True,
                started=False,
                status=final_status,
                message=f"was {status}; reached RUNNING",
            )
        if final_status in {"STOPPING", "SUSPENDING"}:
            return EnsureInstanceResult(
                ok=False,
                started=False,
                status=final_status,
                message=f"timed out waiting for {status} to finish",
            )
        status = final_status

    ok, err = await gcloud_start(spec.name, spec.zone)
    if not ok:
        return EnsureInstanceResult(
            ok=False,
            started=False,
            status=status,
            message=f"start failed from {status}: {err}",
        )

    ok, final_status = await wait_for_instance_running(spec)
    return EnsureInstanceResult(
        ok=ok,
        started=True,
        status=final_status,
        message=(
            "started"
            if ok else f"start command returned but status is {final_status}"
        ),
    )


def _internal_ip(desc: dict) -> Optional[str]:
    try:
        return desc["networkInterfaces"][0]["networkIP"]
    except (KeyError, IndexError, TypeError):
        return None


async def gcloud_ssh(name: str, zone: str, remote_cmd: str,
                    *, timeout: float = 120.0) -> Tuple[int, str, str]:
    """Run a one-shot SSH command via `gcloud compute ssh`."""
    cmd = [
        "gcloud", "compute", "ssh", name, f"--zone={zone}",
        "--tunnel-through-iap", "--quiet",
        f"--command={remote_cmd}",
    ]
    return await _run(cmd, timeout=timeout)


# -------------------------
# Health checks
# -------------------------

def _http_health(url: str, *, timeout: float = 5.0) -> Tuple[bool, Optional[str]]:
    """Probe /health and return (ok, error_msg). error_msg is None on success."""
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    health_url = base.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return (True, None)
            return (False, f"HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        return (False, f"HTTP {e.code}")
    except urllib.error.URLError as e:
        return (False, f"URLError: {e.reason}")
    except OSError as e:
        return (False, f"OSError: {e}")


async def http_health(url: str, *, timeout: float = 5.0) -> Tuple[bool, Optional[str]]:
    return await asyncio.to_thread(_http_health, url, timeout=timeout)


# -------------------------
# vLLM server lifecycle
# -------------------------

def _build_vllm_cmd(
    python_env: str,
    *,
    model: str,
    download_dir: str,
    cuda_visible_devices: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    reasoning_parser: str,
    port: int,
    extra_args: Optional[List[str]] = None,
) -> str:
    """Returns a shell-quoted command line for launching vLLM in the
    server's OpenAI mode under nohup, writing a PID file."""
    python_bin = os.path.join(python_env.rstrip("/"), "bin", "python")
    pid_file = PID_FILE_FMT.format(port=port)
    log_file = LOG_FILE_FMT.format(port=port)

    args = [
        python_bin, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--download-dir", download_dir,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--port", str(port),
        "--host", "0.0.0.0",
        "--reasoning-parser", reasoning_parser,
    ]
    if extra_args:
        args.extend(extra_args)

    # Single command: clean any prior PID file, exec vLLM in background,
    # record PID, then exit so SSH session can close.
    quoted = " ".join(_shquote(a) for a in args)
    env_bin = os.path.join(python_env.rstrip("/"), "bin")
    return (
        f"rm -f {pid_file}; "
        f"PATH={_shquote(env_bin)}:$PATH "
        f"CUDA_VISIBLE_DEVICES={_shquote(cuda_visible_devices)} "
        f"nohup {quoted} > {log_file} 2>&1 < /dev/null & "
        f"echo $! > {pid_file}; "
        f"disown; "
        f"sleep 1; "
        f"cat {pid_file}"
    )


def _shquote(s: str) -> str:
    """Tiny shell quoting: wraps in single quotes, escapes embedded ones.
    Treats '~' as safe so leading-tilde paths (e.g. ~/models) expand on the
    remote shell rather than being passed verbatim."""
    if not s or re.search(r"[^A-Za-z0-9_./@:=,+\-~]", s):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


async def remote_kill_vllm(spec: InstanceSpec, port: int) -> None:
    pid_file = PID_FILE_FMT.format(port=port)
    cmd = (
        f"if [ -f {pid_file} ]; then "
        f"pid=$(cat {pid_file}); "
        f"kill $pid 2>/dev/null; sleep 2; kill -9 $pid 2>/dev/null; "
        f"rm -f {pid_file}; fi"
    )
    await gcloud_ssh(spec.name, spec.zone, cmd, timeout=30.0)


async def remote_kill_all_vllm(spec: InstanceSpec) -> None:
    # Kill anything matching our launch pattern, in case PID files are stale.
    cmd = (
        "pkill -9 -f 'vllm.entrypoints.openai.api_server' || true; "
        "rm -f /tmp/mmai_vllm_*.pid"
    )
    await gcloud_ssh(spec.name, spec.zone, cmd, timeout=30.0)


# -------------------------
# Local server lifecycle (orchestrator GPUs)
# -------------------------

def _local_stop_vllm(proc, *, timeout: float = 15.0) -> None:
    """Stop a local vLLM subprocess and any engine workers it spawned.

    vLLM is launched with start_new_session=True, so the leader is the
    process group leader for a brand-new session. proc.terminate() on
    the leader alone does NOT propagate to the multiprocessing engine
    workers vLLM forks — they get orphaned, keep holding GPU memory,
    and block the next GPU allocation. Kill the whole process group
    instead, then SIGKILL anything still alive after the timeout."""
    import errno
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None

    def _signal_group(sig):
        if pgid is None:
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except OSError as e:
            if e.errno != errno.ESRCH:
                raise

    _signal_group(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except Exception:
        pass
    if proc.poll() is None:
        _signal_group(signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def _local_start_vllm(
    *,
    model: str,
    download_dir: str,
    cuda_visible_devices: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    reasoning_parser: str,
    port: int,
    extra_args: Optional[List[str]] = None,
):
    import subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--download-dir", download_dir,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--port", str(port),
        "--host", "0.0.0.0",
        "--reasoning-parser", reasoning_parser,
    ]
    if extra_args:
        cmd.extend(extra_args)
    log_path = LOG_FILE_FMT.format(port=port).replace("/tmp/", "/tmp/local_")
    log_handle = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_handle = log_handle  # type: ignore[attr-defined]
    return proc


# -------------------------
# Orchestrator state
# -------------------------

@dataclass
class ServerSlot:
    """One vLLM server launched on an instance (or locally)."""
    instance: str         # display name
    zone: str
    host: str             # internal IP or "127.0.0.1" for local
    port: int
    gpus: List[int]
    url: str
    kind: str             # "remote" or "local"
    healthy: bool = False
    local_proc: object = None  # subprocess.Popen for local kind
    launched_ts: float = 0.0    # monotonic time the vLLM process was launched
    last_healthy_ts: float = 0.0  # monotonic time of most recent OK health check
    last_health_error: Optional[str] = None  # most recent /health failure text


@dataclass
class WorkerState:
    spec: InstanceSpec
    internal_ip: Optional[str] = None
    gpu_count: Optional[int] = None
    slots: List[ServerSlot] = field(default_factory=list)
    last_attempt_ts: float = 0.0
    status: str = "unknown"   # unknown|stopped|running|down|error
    bring_up_task: Optional["asyncio.Task"] = None


@dataclass
class ServeConfig:
    instances_file: str
    python_env: str
    servers_file: str
    model: str
    reasoning_parser: str
    gpus_per_server: int
    base_port: int
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    download_dir: str
    include_self: Optional[str]            # comma-separated GPU IDs or None
    server_ready_timeout: int
    health_interval: float
    restart_interval: float
    startup_grace_seconds: float
    extra_vllm_args: List[str]


# -------------------------
# Servers file writer
# -------------------------

def _atomic_write_json(path: str, payload: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".srv_", dir=d)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_servers_file(
    servers_file: str,
    workers: Dict[str, WorkerState],
    local_slots: List[ServerSlot],
) -> None:
    entries = []
    for w in workers.values():
        for s in w.slots:
            if s.healthy:
                entries.append({
                    "url": s.url,
                    "instance": s.instance,
                    "zone": s.zone,
                    "gpus": s.gpus,
                    "kind": s.kind,
                })
    for s in local_slots:
        if s.healthy:
            entries.append({
                "url": s.url,
                "instance": s.instance,
                "zone": s.zone,
                "gpus": s.gpus,
                "kind": s.kind,
            })
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ready": len(entries) > 0,
        "expected_servers": sum(len(w.slots) for w in workers.values()) + len(local_slots),
        "servers": entries,
    }
    _atomic_write_json(servers_file, payload)


# -------------------------
# Discovery / launch
# -------------------------

async def discover_gpu_count(spec: InstanceSpec) -> Optional[int]:
    rc, out, _ = await gcloud_ssh(
        spec.name, spec.zone,
        "nvidia-smi -L | wc -l",
        timeout=60.0,
    )
    if rc != 0:
        return None
    try:
        return int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


async def ensure_internal_ip(state: WorkerState) -> Optional[str]:
    if state.internal_ip:
        return state.internal_ip
    desc = await gcloud_describe(state.spec.name, state.spec.zone)
    if not desc:
        return None
    ip = _internal_ip(desc)
    if ip:
        state.internal_ip = ip
    return ip


async def launch_one_remote_server(
    state: WorkerState,
    cfg: ServeConfig,
    gpus: List[int],
    port: int,
) -> Optional[ServerSlot]:
    cuda = ",".join(str(g) for g in gpus)
    cmd = _build_vllm_cmd(
        cfg.python_env,
        model=cfg.model,
        download_dir=cfg.download_dir,
        cuda_visible_devices=cuda,
        tensor_parallel_size=len(gpus),
        max_model_len=cfg.max_model_len,
        max_num_seqs=cfg.max_num_seqs,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        reasoning_parser=cfg.reasoning_parser,
        port=port,
        extra_args=cfg.extra_vllm_args or None,
    )
    rc, out, err = await gcloud_ssh(state.spec.name, state.spec.zone, cmd, timeout=90.0)
    if rc != 0:
        print(f"[orch] {state.spec.name}: SSH launch (port {port}) failed: {err.strip()}")
        return None
    ip = await ensure_internal_ip(state)
    if not ip:
        print(f"[orch] {state.spec.name}: no internal IP after launch")
        return None
    url = f"http://{ip}:{port}/v1"
    return ServerSlot(
        instance=state.spec.name,
        zone=state.spec.zone,
        host=ip,
        port=port,
        gpus=gpus,
        url=url,
        kind="remote",
        launched_ts=time.monotonic(),
    )


async def launch_one_local_server(cfg: ServeConfig, gpus: List[int], port: int) -> Optional[ServerSlot]:
    cuda = ",".join(str(g) for g in gpus)
    try:
        proc = _local_start_vllm(
            model=cfg.model,
            download_dir=os.path.expanduser(cfg.download_dir),
            cuda_visible_devices=cuda,
            tensor_parallel_size=len(gpus),
            max_model_len=cfg.max_model_len,
            max_num_seqs=cfg.max_num_seqs,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            reasoning_parser=cfg.reasoning_parser,
            port=port,
            extra_args=cfg.extra_vllm_args or None,
        )
    except Exception as e:
        print(f"[orch] local server (port {port}) failed to spawn: {e}")
        return None
    return ServerSlot(
        instance="local",
        zone="",
        host="127.0.0.1",
        port=port,
        gpus=gpus,
        url=f"http://127.0.0.1:{port}/v1",
        kind="local",
        local_proc=proc,
        launched_ts=time.monotonic(),
    )


async def bring_up_worker(
    state: WorkerState,
    cfg: ServeConfig,
    cluster_base_port: int,
) -> None:
    """Idempotently ensure the worker is RUNNING, has its GPU count, and
    has one vLLM server per GPU group. Leaves slots in `state.slots`."""
    state.last_attempt_ts = time.monotonic()

    ensure_result = await ensure_instance_running(state.spec)
    print(f"[orch] {state.spec.name}: {ensure_result.message}")
    if not ensure_result.ok:
        state.status = "error"
        return
    if ensure_result.started:
        # SSH may not be ready immediately; loop a couple times on nvidia-smi.
        await asyncio.sleep(10.0)

    ip = await ensure_internal_ip(state)
    if not ip:
        state.status = "error"
        return

    if state.gpu_count is None:
        gpus = None
        for attempt in range(6):
            gpus = await discover_gpu_count(state.spec)
            if gpus is not None and gpus > 0:
                break
            await asyncio.sleep(10.0)
        if not gpus:
            print(f"[orch] {state.spec.name}: could not discover GPUs")
            state.status = "error"
            return
        state.gpu_count = gpus
        print(f"[orch] {state.spec.name}: {gpus} GPU(s) at {ip}")

    state.status = "running"

    # Decide GPU groups and ports
    gps = max(1, cfg.gpus_per_server)
    if state.gpu_count % gps != 0:
        usable = (state.gpu_count // gps) * gps
        if usable == 0:
            print(f"[orch] {state.spec.name}: not enough GPUs ({state.gpu_count}) for gpus_per_server={gps}")
            state.status = "error"
            return
        print(f"[orch] {state.spec.name}: WARNING using {usable}/{state.gpu_count} GPUs (gpus_per_server={gps})")
        gpu_count = usable
    else:
        gpu_count = state.gpu_count
    n_servers = gpu_count // gps

    desired_ports = [cluster_base_port + i for i in range(n_servers)]
    desired_gpus = [list(range(i * gps, (i + 1) * gps)) for i in range(n_servers)]

    # Anything in state.slots not in desired_ports we kill
    existing_ports = {s.port: s for s in state.slots}
    for port in list(existing_ports.keys()):
        if port not in desired_ports:
            await remote_kill_vllm(state.spec, port)
            existing_ports.pop(port)
    state.slots = [s for s in state.slots if s.port in existing_ports]

    # Launch missing
    new_slots: List[ServerSlot] = list(state.slots)
    for port, gpus in zip(desired_ports, desired_gpus):
        if port in existing_ports:
            continue
        slot = await launch_one_remote_server(state, cfg, gpus, port)
        if slot is not None:
            new_slots.append(slot)
    state.slots = new_slots


# -------------------------
# Health loop
# -------------------------

async def health_pass(
    workers: Dict[str, WorkerState],
    local_slots: List[ServerSlot],
    cfg: ServeConfig,
) -> bool:
    """Check health of all known servers. Returns True if anything changed."""
    changed = False
    now = time.monotonic()

    def _record(slot: ServerSlot, label: str, healthy: bool, err: Optional[str]) -> bool:
        nonlocal changed
        if healthy:
            slot.last_healthy_ts = now
            slot.last_health_error = None
        else:
            # Log the first failure and every change in failure reason so the
            # user can distinguish "still loading" (HTTP failed/Connection
            # refused) from "firewall blocking" (timeout/no route) from
            # "vLLM up but unhealthy" (HTTP 5xx).
            if err and err != slot.last_health_error:
                print(f"[orch] {label} probe failed: {err}  ({slot.url})")
                slot.last_health_error = err
        if healthy != slot.healthy:
            slot.healthy = healthy
            marker = "OK" if healthy else "DOWN"
            print(f"[orch] {label} -> {marker}")
            return True
        return False

    for state in workers.values():
        for slot in state.slots:
            healthy, err = await http_health(slot.url, timeout=5.0)
            if _record(slot, f"{slot.instance}:{slot.port}", healthy, err):
                changed = True
    for slot in local_slots:
        healthy, err = await http_health(slot.url, timeout=5.0)
        if _record(slot, f"local:{slot.port}", healthy, err):
            changed = True
    return changed


async def relaunch_dead_remote_slots(
    workers: Dict[str, WorkerState],
    cfg: ServeConfig,
) -> bool:
    """For unhealthy remote slots, try to relaunch the vLLM process on the
    same instance. Returns True if anything was attempted.

    Skips slots that have never been healthy yet and are still within the
    configured startup grace — vLLM model loading routinely takes several
    minutes and a hot-restart loop would prevent it from ever finishing."""
    attempted = False
    now = time.monotonic()
    for state in workers.values():
        if state.status != "running":
            continue
        for slot in list(state.slots):
            if slot.healthy:
                continue
            if slot.last_healthy_ts == 0.0 and \
               (now - slot.launched_ts) < cfg.startup_grace_seconds:
                # Still in initial model-load window; leave it alone.
                continue
            attempted = True
            reason = "was healthy, now down" if slot.last_healthy_ts else \
                     "never healthy past grace"
            print(f"[orch] relaunching {slot.instance}:{slot.port} ({reason})")
            await remote_kill_vllm(state.spec, slot.port)
            new = await launch_one_remote_server(state, cfg, slot.gpus, slot.port)
            if new is not None:
                # Replace slot in place to preserve list order
                idx = state.slots.index(slot)
                state.slots[idx] = new
    return attempted


async def relaunch_dead_local_slots(
    local_slots: List[ServerSlot],
    cfg: ServeConfig,
) -> bool:
    attempted = False
    now = time.monotonic()
    for i, slot in enumerate(local_slots):
        if slot.healthy:
            continue
        proc = slot.local_proc
        proc_alive = proc is not None and proc.poll() is None
        # If the process is still running and either we're inside the startup
        # grace window or it was healthy at some point recently, leave it
        # alone — vLLM may still be loading or temporarily unresponsive.
        if proc_alive and slot.last_healthy_ts == 0.0 and \
           (now - slot.launched_ts) < cfg.startup_grace_seconds:
            continue
        reason = "process exited" if not proc_alive else \
                 ("was healthy, now down" if slot.last_healthy_ts else
                  "never healthy past grace")
        print(f"[orch] relaunching local:{slot.port} ({reason})")
        _local_stop_vllm(proc, timeout=10)
        attempted = True
        new = await launch_one_local_server(cfg, slot.gpus, slot.port)
        if new is not None:
            local_slots[i] = new
    return attempted


# -------------------------
# Subcommand: start-instances
# -------------------------

async def cmd_start_instances(args) -> int:
    specs = parse_instances_file(args.instances_file)
    print(f"[orch] ensuring {len(specs)} instance(s) are RUNNING...")
    results = await asyncio.gather(
        *[ensure_instance_running(s) for s in specs],
        return_exceptions=True,
    )
    n_ok = 0
    n_started = 0
    for s, r in zip(specs, results):
        if isinstance(r, Exception):
            print(f"[orch] {s.name}: exception {r}")
            continue
        if r.ok:
            n_ok += 1
            if r.started:
                n_started += 1
            print(f"[orch] {s.name}: {r.message}")
        else:
            print(f"[orch] {s.name}: {r.message}")
    print(f"[orch] {n_ok}/{len(specs)} instance(s) RUNNING ({n_started} newly started)")
    return 0 if n_ok > 0 else 1


# -------------------------
# Subcommand: stop-instances
# -------------------------

async def cmd_stop_instances(args) -> int:
    specs = parse_instances_file(args.instances_file)
    print(f"[orch] stopping {len(specs)} instance(s)...")
    # Best-effort kill of vLLM first, so the stop is clean.
    await asyncio.gather(
        *[remote_kill_all_vllm(s) for s in specs],
        return_exceptions=True,
    )
    results = await asyncio.gather(
        *[gcloud_stop(s.name, s.zone) for s in specs],
        return_exceptions=True,
    )
    n_ok = 0
    for s, r in zip(specs, results):
        if isinstance(r, Exception):
            print(f"[orch] {s.name}: exception {r}")
            continue
        ok, err = r
        if ok:
            n_ok += 1
            print(f"[orch] {s.name}: stopped")
        else:
            print(f"[orch] {s.name}: stop failed: {err}")
    return 0


# -------------------------
# Subcommand: serve
# -------------------------

async def cmd_serve(args) -> int:
    cfg = ServeConfig(
        instances_file=args.instances_file,
        python_env=args.python_env,
        servers_file=args.servers_file,
        model=args.model,
        reasoning_parser=args.reasoning_parser,
        gpus_per_server=args.gpus_per_server,
        base_port=args.base_port,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        download_dir=args.download_dir,
        include_self=args.include_self,
        server_ready_timeout=args.server_ready_timeout,
        health_interval=args.health_interval,
        restart_interval=args.restart_interval,
        startup_grace_seconds=args.startup_grace_seconds,
        extra_vllm_args=args.extra_vllm_args or [],
    )

    specs = parse_instances_file(cfg.instances_file)
    workers: Dict[str, WorkerState] = {
        s.name: WorkerState(spec=s) for s in specs
    }
    local_slots: List[ServerSlot] = []
    local_setup_tasks: List[asyncio.Task] = []

    # Best-effort cleanup of leftover processes from a prior run.
    await asyncio.gather(
        *[remote_kill_all_vllm(s) for s in specs],
        return_exceptions=True,
    )

    # Publish an empty (but valid) servers file immediately so wait-for-ready
    # can begin polling without "file not found" noise while bring-up runs.
    write_servers_file(cfg.servers_file, workers, local_slots)

    # Fire-and-forget remote bring-up. Each worker publishes its slots into
    # state.slots as it completes; the maintenance loop re-publishes the JSON
    # every health_interval seconds, so downstream code sees servers appear
    # as soon as they're ready instead of waiting for the slowest one.
    print(f"[orch] launching bring-up for {len(workers)} worker(s) in background")
    for w in workers.values():
        w.bring_up_task = asyncio.create_task(
            bring_up_worker(w, cfg, cfg.base_port)
        )

    # Local servers on orchestrator GPUs (optional) — also fire-and-forget.
    if cfg.include_self:
        local_gpus = [int(g.strip()) for g in cfg.include_self.split(",") if g.strip() != ""]
        gps = max(1, cfg.gpus_per_server)
        if len(local_gpus) % gps != 0:
            usable = (len(local_gpus) // gps) * gps
            print(f"[orch] local: WARNING using {usable}/{len(local_gpus)} GPUs")
            local_gpus = local_gpus[:usable]
        n_local = len(local_gpus) // gps
        # Use a port range that won't collide with remote ports; pick a high base.
        local_base = cfg.base_port + 100

        async def _spawn_local(grp: List[int], port: int) -> None:
            slot = await launch_one_local_server(cfg, grp, port)
            if slot is not None:
                local_slots.append(slot)

        for i in range(n_local):
            grp = local_gpus[i * gps:(i + 1) * gps]
            local_setup_tasks.append(
                asyncio.create_task(_spawn_local(grp, local_base + i))
            )

    # Set up signal handling
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal():
        if not stop_event.is_set():
            print("[orch] received signal; shutting down...")
            stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            pass

    last_restart_attempt = 0.0
    print("[orch] entering maintenance loop")
    try:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=cfg.health_interval)
                break
            except asyncio.TimeoutError:
                pass

            changed = await health_pass(workers, local_slots, cfg)

            # Relaunch any dead vLLM processes on running workers
            relaunched_remote = await relaunch_dead_remote_slots(workers, cfg)
            relaunched_local = await relaunch_dead_local_slots(local_slots, cfg)

            # Try to recover errored/down instances on a slower cadence. Skip
            # any worker whose bring-up task is still in flight, otherwise the
            # restart would race the initial spawn.
            now = time.monotonic()
            if now - last_restart_attempt >= cfg.restart_interval:
                last_restart_attempt = now
                to_recover = [
                    w for w in workers.values()
                    if (w.bring_up_task is None or w.bring_up_task.done())
                    and (w.status in ("error", "stopped", "unknown")
                         or not any(s.healthy for s in w.slots))
                ]
                if to_recover:
                    print(f"[orch] restart-loop: attempting {len(to_recover)} worker(s)")
                    for w in to_recover:
                        w.bring_up_task = asyncio.create_task(
                            bring_up_worker(w, cfg, cfg.base_port)
                        )

            if changed or relaunched_remote or relaunched_local:
                await health_pass(workers, local_slots, cfg)
            write_servers_file(cfg.servers_file, workers, local_slots)
    finally:
        # Cancel any still-pending bring-up tasks so half-started workers
        # don't race the teardown that follows.
        pending_setup = []
        for w in workers.values():
            if w.bring_up_task is not None and not w.bring_up_task.done():
                w.bring_up_task.cancel()
                pending_setup.append(w.bring_up_task)
        for t in local_setup_tasks:
            if not t.done():
                t.cancel()
                pending_setup.append(t)
        if pending_setup:
            await asyncio.gather(*pending_setup, return_exceptions=True)

        print("[orch] tearing down vLLM processes...")
        # Best-effort: kill remote vLLM processes
        await asyncio.gather(
            *[remote_kill_all_vllm(w.spec) for w in workers.values()],
            return_exceptions=True,
        )
        # Stop local processes (kill the whole session, not just the leader)
        for slot in local_slots:
            _local_stop_vllm(slot.local_proc, timeout=15)
        # Final servers file: nothing ready
        try:
            _atomic_write_json(cfg.servers_file, {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "ready": False,
                "servers": [],
            })
        except Exception:
            pass
        print("[orch] serve exit")
    return 0


# -------------------------
# Subcommand: wait-for-ready
# -------------------------

async def cmd_wait_for_ready(args) -> int:
    deadline = time.monotonic() + args.timeout
    target = max(1, args.min_servers)
    while time.monotonic() < deadline:
        try:
            with open(args.servers_file) as fh:
                data = json.load(fh)
            servers = data.get("servers") or []
            if args.require_all:
                target = max(target, int(data.get("expected_servers") or 0), 1)
            ready = bool(data.get("ready")) and len(servers) >= target
            if ready:
                print(f"[orch] {len(servers)} server(s) ready")
                return 0
            print(f"[orch] waiting: {len(servers)}/{target} ready")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            print(f"[orch] waiting: servers file not yet readable ({args.servers_file})")
        await asyncio.sleep(5.0)
    print(f"[orch] timeout waiting for servers ready after {args.timeout}s")
    return 1


# -------------------------
# CLI
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # shared
    def add_instances(sp):
        sp.add_argument("--instances-file", required=True,
                        help="File with one 'name:zone' line per worker.")

    sp = sub.add_parser("start-instances", help="Ensure every instance in the file is RUNNING.")
    add_instances(sp)

    sp = sub.add_parser("stop-instances", help="Stop every instance in the file.")
    add_instances(sp)

    sp = sub.add_parser("serve", help="Run vLLM cluster + maintenance loop.")
    add_instances(sp)
    sp.add_argument("--python-env", required=True,
                    help="Path to the python env on each instance "
                         "(e.g. /home/kenneth_kehl/thisenv). Same on all workers.")
    sp.add_argument("--servers-file", required=True,
                    help="Path to write the JSON list of healthy server URLs.")
    sp.add_argument("--model", required=True, help="HF model id for vLLM.")
    sp.add_argument("--reasoning-parser", required=True,
                    help="vLLM --reasoning-parser value (or 'auto').")
    sp.add_argument("--gpus-per-server", type=int, default=1)
    sp.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    sp.add_argument("--max-model-len", type=int, required=True)
    sp.add_argument("--max-num-seqs", type=int, default=900)
    sp.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    sp.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR)
    sp.add_argument("--include-self", default=None,
                    help="Comma-separated GPU IDs on the orchestrator to use "
                         "as additional local vLLM servers (e.g. '0,1,2,3').")
    sp.add_argument("--server-ready-timeout", type=int,
                    default=DEFAULT_SERVER_READY_TIMEOUT)
    sp.add_argument("--health-interval", type=float, default=DEFAULT_HEALTH_INTERVAL)
    sp.add_argument("--restart-interval", type=float, default=DEFAULT_RESTART_INTERVAL)
    sp.add_argument("--startup-grace-seconds", type=float, default=DEFAULT_STARTUP_GRACE,
                    help="Don't relaunch a never-yet-healthy slot until this "
                         "many seconds after launch; lets vLLM finish loading "
                         "the model before being declared dead.")
    sp.add_argument("--extra-vllm-args", nargs=argparse.REMAINDER, default=[],
                    help="Extra args appended verbatim to the vLLM server command.")

    sp = sub.add_parser("wait-for-ready",
                        help="Wait until the servers file shows >=N healthy servers.")
    sp.add_argument("--servers-file", required=True)
    sp.add_argument("--timeout", type=float, default=1800.0)
    sp.add_argument("--min-servers", type=int, default=1)
    sp.add_argument("--require-all", action="store_true",
                    help="Wait for every server slot discovered by the serve process.")

    return p


async def _main_async(args) -> int:
    cmd = args.cmd
    if cmd == "start-instances":
        return await cmd_start_instances(args)
    if cmd == "stop-instances":
        return await cmd_stop_instances(args)
    if cmd == "serve":
        # Resolve reasoning_parser auto via the same helper inference scripts use.
        if args.reasoning_parser in (None, "", "auto"):
            from vllm_reasoning_utils import resolve_parser_name
            args.reasoning_parser = resolve_parser_name(args.model, args.reasoning_parser)
            print(f"[orch] resolved reasoning_parser={args.reasoning_parser}")
        return await cmd_serve(args)
    if cmd == "wait-for-ready":
        return await cmd_wait_for_ready(args)
    print(f"unknown command {cmd!r}")
    return 2


def main() -> None:
    args = build_parser().parse_args()
    try:
        rc = asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
