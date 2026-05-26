from __future__ import annotations

import argparse
import atexit
import asyncio
import csv
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_ID_FIELDS = (
    "record_id",
    "id",
    "report_id",
    "note_id",
    "document_id",
    "encounter_id",
    "accession",
    "accession_number",
)

DEFAULT_TEXT_FIELDS = (
    "text",
    "report_text",
    "note_text",
    "document_text",
    "content",
    "impression",
    "assessment_plan",
)


@dataclass(frozen=True)
class Record:
    input_index: int
    record_id: str
    text: str
    data: dict[str, Any]


@dataclass
class ManagedServer:
    process: subprocess.Popen[bytes]
    base_url: str
    log_file: Any
    log_path: Path
    gpus: list[str]


_MANAGED_SERVERS: list[ManagedServer] = []


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_text_field: str,
    default_output: str,
    default_input: str | None = None,
    default_id_fields: list[str] | None = None,
    default_metadata_fields: list[str] | None = None,
) -> None:
    input_group = parser.add_argument_group("input/output")
    input_group.add_argument(
        "--input",
        default=default_input,
        required=default_input is None,
        help="Input JSONL, JSON, CSV, TSV, Parquet, TXT file, or directory of TXT files.",
    )
    input_group.add_argument("--output", default=default_output, help="Output JSONL path.")
    input_group.add_argument("--input-format", choices=("auto", "jsonl", "json", "csv", "tsv", "parquet", "txt"), default="auto")
    input_group.add_argument("--text-field", default=default_text_field, help="Text field for structured inputs.")
    input_group.add_argument("--id-field", default=None, help="Optional single record id field. Auto-detected if omitted.")
    input_group.add_argument(
        "--id-fields",
        default=",".join(default_id_fields) if default_id_fields else None,
        help="Comma-separated list of fields to join (with '-') into a composite record_id. Overrides --id-field.",
    )
    input_group.add_argument(
        "--metadata-fields",
        default=",".join(default_metadata_fields) if default_metadata_fields else None,
        help="Comma-separated whitelist of input columns to render into the prompt metadata. If unset, all non-text fields are included.",
    )
    input_group.add_argument("--encoding", default="utf-8", help="Input/output text encoding.")
    input_group.add_argument("--limit", type=int, default=None, help="Only process the first N input records.")
    input_group.add_argument("--resume", action="store_true", help="Append to output and skip record_ids already present.")
    input_group.add_argument("--include-input", action="store_true", help="Include original non-text input fields in each output row.")
    input_group.add_argument(
        "--parquet-output",
        default=None,
        help="Path to write a parquet copy of the JSONL output. Defaults to the --output path with .parquet extension.",
    )
    input_group.add_argument("--no-parquet", action="store_true", help="Skip writing the parquet copy of the output JSONL.")
    input_group.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help=(
            "Buffer this many completed labeling rows before writing+flushing to the output JSONL. "
            "Larger values reduce NFS write load; smaller values lose less work on crash. "
            "Set to 1 to flush after every record."
        ),
    )

    model_group = parser.add_argument_group("model/server")
    model_group.add_argument("--model", required=True, help="Model name/path. For existing servers, this must match the served model name.")
    model_group.add_argument("--served-model-name", default=None, help="Optional alias to use when starting vLLM, and in requests.")
    model_group.add_argument(
        "--server-url",
        action="append",
        default=[],
        help="OpenAI-compatible vLLM base URL. Repeat or comma-separate for multiple already-running servers.",
    )
    model_group.add_argument("--start-vllm", action="store_true", help="Start one or more vLLM OpenAI API servers.")
    model_group.add_argument(
        "--gpus",
        default="",
        help="GPU ids to use when --start-vllm is set, e.g. 0,1,2,3 or 0-3.",
    )
    model_group.add_argument(
        "--gpus-per-server",
        type=int,
        default=1,
        help="Tensor-parallel GPU count per vLLM server when starting servers.",
    )
    model_group.add_argument("--host", default="127.0.0.1", help="Host for started vLLM servers.")
    model_group.add_argument("--base-port", type=int, default=8000, help="First port for started vLLM servers.")
    model_group.add_argument("--startup-timeout", type=float, default=900.0, help="Seconds to wait for each started server.")
    model_group.add_argument("--vllm-log-dir", default="vllm_logs", help="Directory for started server logs.")
    model_group.add_argument(
        "--vllm-arg",
        action="append",
        default=[],
        help="Extra argument(s) passed to each vLLM server. May be repeated.",
    )
    model_group.add_argument(
        "--reasoning-parser",
        default="auto",
        help=(
            "vLLM reasoning parser name to enable when --start-vllm. 'auto' (default) guesses from "
            "the model name (qwen3, gpt_oss, gemma, deepseek_r1, ...). Pass 'none' to disable, or a "
            "specific parser name to force it. Ignored when --vllm-arg already supplies "
            "--reasoning-parser."
        ),
    )

    remote_group = parser.add_argument_group("Remote vLLM pool (GCP orchestrator)")
    remote_group.add_argument(
        "--server_urls",
        type=str,
        default=None,
        help=(
            "Comma-separated URLs of existing vLLM servers, e.g. "
            "'http://10.0.0.5:8000/v1,http://10.0.0.6:8000/v1'. "
            "Use this or --server_urls_file for the remote pool workflow."
        ),
    )
    remote_group.add_argument(
        "--server_urls_file",
        type=str,
        default=None,
        help="Path to a JSON file maintained by gcp_vllm_orchestrator.py with healthy server URLs.",
    )
    remote_group.add_argument(
        "--server_urls_refresh",
        type=float,
        default=15.0,
        help="Seconds between re-reads of --server_urls_file.",
    )
    remote_group.add_argument(
        "--max_concurrent_per_server",
        type=int,
        default=50,
        help="Remote-pool per-server adaptive concurrency ceiling.",
    )
    remote_group.add_argument(
        "--concurrency_success_threshold",
        type=int,
        default=3,
        help="Successful remote requests needed before increasing a server concurrency limit.",
    )
    remote_group.add_argument(
        "--concurrency_increase_step",
        type=int,
        default=2,
        help="Adaptive remote concurrency slots to add after a clean success streak.",
    )
    remote_group.add_argument(
        "--concurrency_backoff_factor",
        type=float,
        default=0.5,
        help="Multiplier applied to a remote server's concurrency limit after request errors.",
    )
    remote_group.add_argument(
        "--results_per_shard",
        type=int,
        default=200,
        help="Completed remote labeling records per JSONL shard.",
    )
    remote_group.add_argument(
        "--remote-shard-dir",
        default=None,
        help="Directory for remote labeling JSONL shards. Defaults to <output stem>_remote_shards.",
    )
    remote_group.add_argument(
        "--max_attempts",
        type=int,
        default=200,
        help="Remote-pool max attempts per record before writing an error placeholder.",
    )
    remote_group.add_argument(
        "--status-interval",
        type=float,
        default=30.0,
        help="Remote-pool status logging interval in seconds.",
    )

    request_group = parser.add_argument_group("request")
    request_group.add_argument("--workers", type=int, default=4, help="Concurrent labeling requests.")
    request_group.add_argument("--temperature", type=float, default=0.0)
    request_group.add_argument("--max-tokens", type=int, default=10000)
    request_group.add_argument("--request-timeout", "--request_timeout", type=float, default=240.0)
    request_group.add_argument("--retries", type=int, default=2, help="Retries per record after the first attempt.")
    request_group.add_argument("--retry-sleep", type=float, default=2.0, help="Base seconds to sleep between retries.")
    request_group.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        help="Optional API key if the server requires one. Defaults to VLLM_API_KEY or OPENAI_API_KEY.",
    )
    request_group.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Do not send OpenAI response_format={type: json_object}. Use this for older vLLM servers.",
    )


def normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        raise ValueError("Empty server URL")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def request_model_name(args: argparse.Namespace) -> str:
    return args.served_model_name or args.model


def resolve_server_urls(args: argparse.Namespace) -> tuple[list[str], list[ManagedServer]]:
    if args.start_vllm and args.server_url:
        raise SystemExit("Use either --start-vllm or --server-url, not both.")

    if args.start_vllm:
        servers = start_vllm_servers(args)
        return [server.base_url for server in servers], servers

    urls = flatten_server_urls(args.server_url)
    if not urls:
        urls = ["http://127.0.0.1:8000/v1"]
        print("No --server-url supplied; using http://127.0.0.1:8000/v1", file=sys.stderr)
    return [normalize_base_url(url) for url in urls], []


def flatten_server_urls(values: list[str]) -> list[str]:
    urls: list[str] = []
    for value in values:
        urls.extend(part.strip() for part in value.split(",") if part.strip())
    return urls


def parse_remote_urls(value: str | None) -> list[str]:
    if not value:
        return []
    return [normalize_base_url(url) for url in value.split(",") if url.strip()]


def remote_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "server_urls", None) or getattr(args, "server_urls_file", None))


def validate_server_mode_args(args: argparse.Namespace) -> None:
    if getattr(args, "server_urls", None) and getattr(args, "server_urls_file", None):
        raise SystemExit("Specify only one of --server_urls / --server_urls_file.")
    if remote_mode(args) and (args.start_vllm or args.server_url):
        raise SystemExit("Remote mode uses --server_urls or --server_urls_file; do not combine it with --start-vllm or --server-url.")


def parse_gpu_ids(gpus: str) -> list[str]:
    ids: list[str] = []
    for raw_part in gpus.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            if start_s.strip().isdigit() and end_s.strip().isdigit():
                start = int(start_s)
                end = int(end_s)
                step = 1 if end >= start else -1
                ids.extend(str(i) for i in range(start, end + step, step))
                continue
        ids.append(part)
    return ids


def chunk_gpu_ids(gpus: list[str], gpus_per_server: int) -> list[list[str]]:
    if gpus_per_server < 1:
        raise SystemExit("--gpus-per-server must be >= 1.")
    if not gpus:
        raise SystemExit("--gpus is required when --start-vllm is set.")
    if len(gpus) % gpus_per_server != 0:
        raise SystemExit(
            f"{len(gpus)} GPU ids were provided, which is not divisible by --gpus-per-server={gpus_per_server}."
        )
    return [gpus[i : i + gpus_per_server] for i in range(0, len(gpus), gpus_per_server)]


def split_vllm_args(values: list[str]) -> list[str]:
    parts: list[str] = []
    for value in values:
        parts.extend(shlex.split(value, posix=os.name != "nt"))
    return parts


# Substring → vLLM reasoning parser name. Checked in order against a lowercased
# model identifier (path or HF repo). First match wins. Add new entries here as
# new model families ship; the user can always override with --reasoning-parser.
REASONING_PARSER_HEURISTICS: list[tuple[str, str]] = [
    ("qwen3", "qwen3"),
    ("qwen-3", "qwen3"),
    ("gpt-oss", "gpt_oss"),
    ("gpt_oss", "gpt_oss"),
    ("deepseek-r1", "deepseek_r1"),
    ("deepseek_r1", "deepseek_r1"),
    ("r1-distill", "deepseek_r1"),
    ("gemma-4", "gemma"),
    ("gemma4", "gemma"),
    ("granite", "granite"),
    ("glm-4.5", "glm45"),
    ("glm4.5", "glm45"),
]


def guess_reasoning_parser(model_name: str) -> str | None:
    name = model_name.lower()
    for needle, parser_name in REASONING_PARSER_HEURISTICS:
        if needle in name:
            return parser_name
    return None


def inject_reasoning_args(args: argparse.Namespace, extra_args: list[str]) -> list[str]:
    if "--reasoning-parser" in extra_args:
        return extra_args

    requested = (args.reasoning_parser or "").strip()
    if requested.lower() in {"none", "off", "false", "disable", "disabled"}:
        print("Reasoning parser disabled (--reasoning-parser none).", file=sys.stderr)
        return extra_args

    parser_name: str | None
    if requested.lower() in {"", "auto"}:
        parser_name = guess_reasoning_parser(args.model)
        if parser_name is None:
            print(
                f"Could not auto-detect a reasoning parser for model {args.model!r}. "
                "Pass --reasoning-parser <name> to enable, or --reasoning-parser none to silence this.",
                file=sys.stderr,
            )
            return extra_args
        print(f"Auto-detected reasoning parser {parser_name!r} for model {args.model!r}.", file=sys.stderr)
    else:
        parser_name = requested
        print(f"Using reasoning parser {parser_name!r} for model {args.model!r}.", file=sys.stderr)

    return ["--reasoning-parser", parser_name, *extra_args]


def start_vllm_servers(args: argparse.Namespace) -> list[ManagedServer]:
    gpus = parse_gpu_ids(args.gpus)
    groups = chunk_gpu_ids(gpus, args.gpus_per_server)
    log_dir = Path(args.vllm_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    servers: list[ManagedServer] = []
    extra_args = split_vllm_args(args.vllm_arg)
    extra_args = inject_reasoning_args(args, extra_args)
    for index, gpu_group in enumerate(groups):
        port = args.base_port + index
        base_url = normalize_base_url(f"http://{args.host}:{port}")
        log_path = log_dir / f"vllm_server_{index}_port_{port}.log"
        log_file = log_path.open("ab")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.model,
            "--host",
            args.host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(len(gpu_group)),
        ]
        if args.served_model_name:
            cmd.extend(["--served-model-name", args.served_model_name])
        cmd.extend(extra_args)

        print(
            f"Starting vLLM server {index} on {base_url} with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}",
            file=sys.stderr,
        )
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        server = ManagedServer(process=proc, base_url=base_url, log_file=log_file, log_path=log_path, gpus=gpu_group)
        servers.append(server)
        _MANAGED_SERVERS.append(server)

    atexit.register(stop_managed_servers)
    for server in servers:
        wait_for_server(server, args.startup_timeout)
    return servers


def wait_for_server(server: ManagedServer, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(
                f"vLLM server at {server.base_url} exited with code {server.process.returncode}. "
                f"See {server.log_path}."
            )
        try:
            http_json("GET", f"{server.base_url}/models", timeout=10.0)
            print(f"vLLM server ready: {server.base_url}", file=sys.stderr)
            return
        except Exception as exc:  # noqa: BLE001 - include connection and HTTP failures in readiness polling
            last_error = str(exc)
            time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for {server.base_url}. Last error: {last_error}. See {server.log_path}.")


def stop_managed_servers() -> None:
    while _MANAGED_SERVERS:
        server = _MANAGED_SERVERS.pop()
        if server.process.poll() is None:
            server.process.terminate()
            try:
                server.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.process.kill()
                server.process.wait(timeout=30)
        server.log_file.close()


def http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
    api_key: str | None = None,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    return json.loads(response_body)


def chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    api_key: str | None,
    json_mode: bool,
) -> tuple[str, str | None, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = http_json("POST", f"{base_url}/chat/completions", payload=payload, timeout=timeout, api_key=api_key)
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        reasoning_content = json.dumps(reasoning_content, ensure_ascii=False)
    return content, reasoning_content, data.get("usage") or {}


def read_records(args: argparse.Namespace) -> list[Record]:
    path = Path(args.input)
    if path.is_dir():
        return read_text_directory(path, args.encoding)

    input_format = args.input_format
    if input_format == "auto":
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            input_format = "jsonl"
        elif suffix == ".json":
            input_format = "json"
        elif suffix == ".csv":
            input_format = "csv"
        elif suffix in {".tsv", ".tab"}:
            input_format = "tsv"
        elif suffix in {".parquet", ".pq"}:
            input_format = "parquet"
        else:
            input_format = "txt"

    if input_format == "jsonl":
        records = read_jsonl(path, args)
    elif input_format == "json":
        records = read_json(path, args)
    elif input_format in {"csv", "tsv"}:
        records = read_delimited(path, args, delimiter="\t" if input_format == "tsv" else ",")
    elif input_format == "parquet":
        records = read_parquet(path, args)
    elif input_format == "txt":
        text = path.read_text(encoding=args.encoding)
        records = [Record(input_index=0, record_id=path.stem, text=text, data={"path": str(path)})]
    else:
        raise AssertionError(input_format)

    if args.limit is not None:
        records = records[: args.limit]
    return records


def read_text_directory(path: Path, encoding: str) -> list[Record]:
    records: list[Record] = []
    for index, item in enumerate(sorted(path.rglob("*.txt"))):
        records.append(
            Record(
                input_index=index,
                record_id=str(item.relative_to(path)).replace("\\", "/"),
                text=item.read_text(encoding=encoding),
                data={"path": str(item)},
            )
        )
    if not records:
        raise SystemExit(f"No .txt files found under directory: {path}")
    return records


def read_jsonl(path: Path, args: argparse.Namespace) -> list[Record]:
    records: list[Record] = []
    with path.open("r", encoding=args.encoding) as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            data = json.loads(line)
            records.append(record_from_mapping(index, data, args))
    return records


def read_json(path: Path, args: argparse.Namespace) -> list[Record]:
    data = json.loads(path.read_text(encoding=args.encoding))
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        rows = data["records"]
    elif isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = [data]
    else:
        raise SystemExit(f"Unsupported JSON root in {path}: expected object, list, or object with records list.")
    return [record_from_mapping(index, row, args) for index, row in enumerate(rows)]


def read_delimited(path: Path, args: argparse.Namespace, *, delimiter: str) -> list[Record]:
    records: list[Record] = []
    with path.open("r", encoding=args.encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for index, row in enumerate(reader):
            records.append(record_from_mapping(index, row, args))
    return records


def read_parquet(path: Path, args: argparse.Namespace) -> list[Record]:
    rows = _load_parquet_rows(path)
    return [record_from_mapping(index, row, args) for index, row in enumerate(rows)]


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        pq = None  # type: ignore[assignment]
    if pq is not None:
        table = pq.read_table(str(path))
        return table.to_pylist()

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Reading Parquet input requires pyarrow or pandas. Install with `pip install pyarrow` "
            "(recommended) or `pip install pandas pyarrow`."
        ) from exc
    frame = pd.read_parquet(path)
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def record_from_mapping(index: int, data: Any, args: argparse.Namespace) -> Record:
    if not isinstance(data, dict):
        raise SystemExit(f"Input row {index + 1} is not an object/mapping.")

    text = first_present(data, [args.text_field, *DEFAULT_TEXT_FIELDS])
    if text is None or not str(text).strip():
        raise SystemExit(
            f"Input row {index + 1} has no text. Tried --text-field={args.text_field!r} and common text fields."
        )

    id_fields = parse_csv_arg(getattr(args, "id_fields", None))
    if id_fields:
        parts: list[str] = []
        for field in id_fields:
            value = data.get(field)
            if value is None or str(value) == "":
                raise SystemExit(
                    f"Input row {index + 1} has no value for --id-fields entry {field!r}."
                )
            parts.append(str(value))
        record_id: Any = "-".join(parts)
    elif args.id_field:
        record_id = data.get(args.id_field)
        if record_id is None or not str(record_id).strip():
            raise SystemExit(f"Input row {index + 1} has no value for --id-field={args.id_field!r}.")
    else:
        record_id = first_present(data, DEFAULT_ID_FIELDS) or str(index + 1)

    return Record(input_index=index, record_id=str(record_id), text=str(text), data=dict(data))


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def first_present(data: dict[str, Any], fields: tuple[str, ...] | list[str]) -> Any | None:
    for field in fields:
        if field in data and data[field] not in (None, ""):
            return data[field]
    return None


def metadata_for_prompt(
    record: Record,
    text_field: str,
    *,
    max_chars: int = 4000,
    metadata_fields: list[str] | None = None,
) -> str:
    omit = {text_field, *DEFAULT_TEXT_FIELDS}
    if metadata_fields:
        items = [(field, record.data.get(field)) for field in metadata_fields]
        metadata = {
            key: value
            for key, value in items
            if key not in omit and value not in (None, "")
        }
    else:
        metadata = {
            key: value
            for key, value in record.data.items()
            if key not in omit and value not in (None, "")
        }
    if not metadata:
        return "{}"
    rendered = json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
    if len(rendered) > max_chars:
        return rendered[:max_chars] + "\n... [metadata truncated]"
    return rendered


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value, None
        return None, "Model returned JSON, but the top-level value is not an object."
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            if isinstance(value, dict):
                return value, None
        except json.JSONDecodeError:
            continue
    return None, "Could not parse a JSON object from the model response."


def completed_record_ids(output_path: Path, encoding: str) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding=encoding) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = row.get("record_id")
            if record_id is not None:
                done.add(str(record_id))
    return done


def iter_jsonl_rows(path: Path, encoding: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding=encoding) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def build_label_row(
    *,
    record: Record,
    args: argparse.Namespace,
    server_url: str | None,
    content: str,
    reasoning_content: str | None,
    usage: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    parsed, parse_error = parse_json_object(content) if content else (None, None)
    row: dict[str, Any] = {
        "record_id": record.record_id,
        "input_index": record.input_index,
        "model": request_model_name(args),
        "server_url": server_url,
        "labels": parsed,
        "raw_response": content,
        "reasoning_content": reasoning_content,
        "request_error": error,
        "parse_error": parse_error,
        "usage": usage,
    }
    if args.include_input:
        text_keys = {args.text_field, *DEFAULT_TEXT_FIELDS}
        row["input"] = {key: value for key, value in record.data.items() if key not in text_keys}
    return row


def label_one_record(
    *,
    record: Record,
    base_url: str,
    args: argparse.Namespace,
    build_messages: Callable[[Record, argparse.Namespace], list[dict[str, str]]],
) -> dict[str, Any]:
    messages = build_messages(record, args)
    content = ""
    reasoning_content: str | None = None
    usage: dict[str, Any] = {}
    error = None

    for attempt in range(args.retries + 1):
        try:
            content, reasoning_content, usage = chat_completion(
                base_url=base_url,
                model=request_model_name(args),
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
                api_key=args.api_key,
                json_mode=not args.no_json_mode,
            )
            error = None
            break
        except Exception as exc:  # noqa: BLE001 - preserve per-record failure in JSONL output
            error = str(exc)
            if attempt >= args.retries:
                break
            time.sleep(args.retry_sleep * (attempt + 1))

    return build_label_row(
        record=record,
        args=args,
        server_url=base_url,
        content=content,
        reasoning_content=reasoning_content,
        usage=usage,
        error=error,
    )


def remote_shard_dir_for(args: argparse.Namespace, output_path: Path) -> Path:
    if args.remote_shard_dir:
        return Path(args.remote_shard_dir)
    return output_path.with_name(f"{output_path.stem}_remote_shards")


def remote_shard_paths(shard_dir: Path) -> list[Path]:
    if not shard_dir.exists():
        return []
    return sorted(shard_dir.glob("label_shard_*.jsonl"))


def clear_remote_shards(shard_dir: Path) -> None:
    if not shard_dir.exists():
        return
    for path in remote_shard_paths(shard_dir):
        path.unlink()


def completed_record_ids_in_shards(shard_dir: Path, encoding: str) -> set[str]:
    done: set[str] = set()
    for path in remote_shard_paths(shard_dir):
        for row in iter_jsonl_rows(path, encoding):
            record_id = row.get("record_id")
            if record_id is not None:
                done.add(str(record_id))
    return done


def merge_remote_jsonl(
    *,
    output_path: Path,
    shard_dir: Path,
    encoding: str,
    include_existing_output: bool,
) -> int:
    rows_by_id: dict[str, dict[str, Any]] = {}
    sources: list[Path] = []
    if include_existing_output and output_path.exists():
        sources.append(output_path)
    sources.extend(remote_shard_paths(shard_dir))

    for source in sources:
        for row in iter_jsonl_rows(source, encoding):
            record_id = row.get("record_id")
            if record_id is not None:
                rows_by_id[str(record_id)] = row

    def sort_key(row: dict[str, Any]) -> tuple[int, int | str]:
        value = row.get("input_index")
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(row.get("record_id") or ""))

    rows = sorted(rows_by_id.values(), key=sort_key)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with tmp_path.open("w", encoding=encoding, newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp_path, output_path)
    return len(rows)


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return json.loads(json.dumps(usage, default=str))


async def run_remote_labeling_pool(
    *,
    args: argparse.Namespace,
    records: list[Record],
    build_messages: Callable[[Record, argparse.Namespace], list[dict[str, str]]],
    shard_dir: Path,
    starting_shard_idx: int,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from remote_vllm_pool import build_registry_from_args, run_pool  # type: ignore[import-not-found]

    if args.server_urls:
        args.server_urls = ",".join(parse_remote_urls(args.server_urls))

    async def work_fn(client: Any, record: Record) -> dict[str, Any]:
        request_payload: dict[str, Any] = {
            "model": request_model_name(args),
            "messages": build_messages(record, args),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if not args.no_json_mode:
            request_payload["response_format"] = {"type": "json_object"}

        response = await asyncio.wait_for(
            client.chat.completions.create(**request_payload),
            timeout=float(args.request_timeout),
        )
        message = response.choices[0].message
        content = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is None:
            reasoning_content = (getattr(message, "model_extra", None) or {}).get("reasoning_content")
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            reasoning_content = json.dumps(reasoning_content, ensure_ascii=False)

        return build_label_row(
            record=record,
            args=args,
            server_url=str(getattr(client, "base_url", "")).rstrip("/") or None,
            content=content,
            reasoning_content=reasoning_content,
            usage=_usage_to_dict(getattr(response, "usage", None)),
            error=None,
        )

    def error_placeholder(record: Record, err: Exception) -> dict[str, Any]:
        return build_label_row(
            record=record,
            args=args,
            server_url=None,
            content="",
            reasoning_content=None,
            usage={},
            error=str(err),
        )

    def shard_writer(payload: list[tuple[Any, Any]], shard_idx: int) -> None:
        shard_dir.mkdir(parents=True, exist_ok=True)
        out_path = shard_dir / f"label_shard_{shard_idx:06d}.jsonl"
        tmp_path = out_path.with_suffix(".jsonl.part")
        with tmp_path.open("w", encoding=args.encoding, newline="\n") as handle:
            for _record_id, row in payload:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp_path, out_path)
        print(f"[remote] wrote {out_path.name} ({len(payload)} rows)", file=sys.stderr)

    registry = build_registry_from_args(args)
    try:
        await run_pool(
            work_items=[(record.record_id, record) for record in records],
            work_fn=work_fn,
            registry=registry,
            shard_writer=shard_writer,
            results_per_shard=int(args.results_per_shard),
            starting_shard_idx=starting_shard_idx,
            max_attempts=int(args.max_attempts),
            status_interval=float(args.status_interval),
            error_placeholder=error_placeholder,
        )
    finally:
        await registry.stop()


def run_labeling_remote(
    *,
    args: argparse.Namespace,
    records: list[Record],
    output_path: Path,
    build_messages: Callable[[Record, argparse.Namespace], list[dict[str, str]]],
    all_records: list[Record] | None = None,
) -> None:
    if all_records is None:
        all_records = list(records)
    shard_dir = remote_shard_dir_for(args, output_path)
    shard_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        clear_remote_shards(shard_dir)

    done: set[str] = set()
    if args.resume:
        done.update(completed_record_ids(output_path, args.encoding))
        done.update(completed_record_ids_in_shards(shard_dir, args.encoding))
        records = [record for record in records if record.record_id not in done]
        print(f"Skipping {len(done)} completed record_id(s); {len(records)} remain.", file=sys.stderr)

    existing_indices = []
    for path in remote_shard_paths(shard_dir):
        stem = path.stem
        try:
            existing_indices.append(int(stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    starting_shard_idx = max(existing_indices, default=-1) + 1

    if records:
        print(
            f"Remote-labeling {len(records)} record(s); shards={shard_dir}, "
            f"results_per_shard={args.results_per_shard}.",
            file=sys.stderr,
        )
        asyncio.run(
            run_remote_labeling_pool(
                args=args,
                records=records,
                build_messages=build_messages,
                shard_dir=shard_dir,
                starting_shard_idx=starting_shard_idx,
            )
        )
    else:
        print("No new records to label in remote mode.", file=sys.stderr)

    merged = merge_remote_jsonl(
        output_path=output_path,
        shard_dir=shard_dir,
        encoding=args.encoding,
        include_existing_output=bool(args.resume),
    )
    print(f"Merged {merged} labeled record(s) to {output_path}", file=sys.stderr)

    if not args.no_parquet:
        parquet_path = resolve_parquet_path(args)
        write_parquet_from_jsonl(
            output_path,
            parquet_path,
            encoding=args.encoding,
            input_records=all_records,
        )
        print(f"Wrote parquet copy to {parquet_path}", file=sys.stderr)


def run_labeling(
    *,
    args: argparse.Namespace,
    build_messages: Callable[[Record, argparse.Namespace], list[dict[str, str]]],
) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_server_mode_args(args)

    records = read_records(args)
    all_records = list(records)
    if remote_mode(args):
        run_labeling_remote(
            args=args,
            records=records,
            output_path=output_path,
            build_messages=build_messages,
            all_records=all_records,
        )
        return

    if args.resume:
        done = completed_record_ids(output_path, args.encoding)
        records = [record for record in records if record.record_id not in done]
        print(f"Skipping {len(done)} completed record_id(s); {len(records)} remain.", file=sys.stderr)
    else:
        done = set()

    if not records:
        print("No records to label.", file=sys.stderr)
        return

    server_urls, _servers = resolve_server_urls(args)
    max_workers = max(1, args.workers)
    mode = "a" if args.resume and output_path.exists() else "w"

    print(
        f"Labeling {len(records)} record(s) with {len(server_urls)} server(s), workers={max_workers}.",
        file=sys.stderr,
    )
    flush_every = max(1, int(getattr(args, "flush_every", 1) or 1))
    with output_path.open(mode, encoding=args.encoding, newline="\n") as handle:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for position, record in enumerate(records):
                futures.append(
                    executor.submit(
                        label_one_record,
                        record=record,
                        base_url=server_urls[position % len(server_urls)],
                        args=args,
                        build_messages=build_messages,
                    )
                )

            pending_lines: list[str] = []

            def flush_pending() -> None:
                if not pending_lines:
                    return
                handle.write("".join(pending_lines))
                handle.flush()
                pending_lines.clear()

            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                pending_lines.append(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                if len(pending_lines) >= flush_every or completed == len(records):
                    flush_pending()
                status = "ok" if row.get("labels") and not row.get("request_error") else "failed"
                print(f"[{completed}/{len(records)}] {status} {row['record_id']}", file=sys.stderr)

            flush_pending()

    if not args.no_parquet:
        parquet_path = resolve_parquet_path(args)
        write_parquet_from_jsonl(
            output_path,
            parquet_path,
            encoding=args.encoding,
            input_records=all_records,
        )
        print(f"Wrote parquet copy to {parquet_path}", file=sys.stderr)


def resolve_parquet_path(args: argparse.Namespace) -> Path:
    if args.parquet_output:
        return Path(args.parquet_output)
    output_path = Path(args.output)
    if output_path.suffix.lower() in {".jsonl", ".ndjson", ".json"}:
        return output_path.with_suffix(".parquet")
    return output_path.with_name(output_path.name + ".parquet")


LABEL_OUTPUT_COLUMNS = (
    "record_id",
    "input_index",
    "model",
    "server_url",
    "labels_json",
    "raw_response",
    "reasoning_content",
    "request_error",
    "parse_error",
    "usage_json",
    "input_json",
)


def _flatten_label_value(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for subkey, subvalue in value.items():
            col_name = f"{key}_{subkey}"
            if isinstance(subvalue, (dict, list)):
                result[col_name] = json.dumps(subvalue, ensure_ascii=False, default=str)
            else:
                result[col_name] = subvalue
        return result
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return {key: value}
        return {f"{key}_json": json.dumps(value, ensure_ascii=False, default=str)}
    return {key: value}


def flatten_labels(labels: dict[str, Any] | None) -> dict[str, Any]:
    if not labels:
        return {}
    flat: dict[str, Any] = {}
    for key, value in labels.items():
        if key == "record_id":
            continue
        flat.update(_flatten_label_value(key, value))
    return flat


def classify_label_schema(label_dicts: Any) -> dict[str, tuple[str, ...]]:
    """Decide a single column shape per top-level label key from a sample of rows.

    Resolution priority: any row seen as dict wins over list-of-dicts over
    list-of-primitives over scalar. Subkey set for dict-typed fields is the
    union across rows.
    """
    seen_dict_subkeys: dict[str, set[str]] = {}
    seen_list_dict: set[str] = set()
    seen_list_prim: set[str] = set()
    seen_scalar: set[str] = set()
    for labels in label_dicts:
        if not isinstance(labels, dict):
            continue
        for key, value in labels.items():
            if key == "record_id":
                continue
            if isinstance(value, dict):
                seen_dict_subkeys.setdefault(key, set()).update(value.keys())
            elif isinstance(value, list):
                if any(isinstance(item, (dict, list)) for item in value):
                    seen_list_dict.add(key)
                else:
                    seen_list_prim.add(key)
            else:
                seen_scalar.add(key)

    schema: dict[str, tuple[str, ...]] = {}
    all_keys = set(seen_dict_subkeys) | seen_list_dict | seen_list_prim | seen_scalar
    for key in all_keys:
        if key in seen_dict_subkeys:
            subkeys = tuple(sorted(seen_dict_subkeys[key]))
            schema[key] = ("dict",) + subkeys
        elif key in seen_list_dict:
            schema[key] = ("list_dict",)
        elif key in seen_list_prim:
            schema[key] = ("list_prim",)
        else:
            schema[key] = ("scalar",)
    return schema


# ICD-O-3 topography prefixes used to derive organ-specific metastatic-site
# columns for imaging reports. Keys match the trailing token used in the
# manual-label columns (e.g. "brain" -> derived_brain_met) so the input and
# derived columns line up for comparison.
SITE_CODE_PREFIXES: dict[str, tuple[str, ...]] = {
    "brain":      ("C70.", "C71."),      # meninges + brain parenchyma
    "bone":       ("C40.", "C41."),      # limb + axial bones (incl. spine, ribs, pelvis)
    "adrenal":    ("C74.",),
    "liver":      ("C22.",),
    "lung":       ("C34.",),
    "node":       ("C77.",),
    "peritoneal": ("C48.",),             # peritoneum + retroperitoneum
}


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _field_code(labels: dict[str, Any], key: str) -> int | None:
    field = labels.get(key)
    if isinstance(field, dict):
        return _safe_int(field.get("code"))
    return _safe_int(field)


def detect_label_kind(label_schema: dict[str, tuple[str, ...]]) -> str | None:
    keys = set(label_schema.keys())
    if keys & {"image_ca", "image_overall", "image_cancer_sites"}:
        return "imaging"
    if keys & {"md_ca", "md_ca_status"}:
        return "medonc"
    return None


def derive_manual_label_columns(
    labels: dict[str, Any] | None,
    kind: str | None,
) -> dict[str, Any]:
    """Compute manual-label-style derived columns from a labels dict.

    Always emits the same column set for a given `kind`. When `labels` is
    not a dict (e.g. labeling failed) every derived column is null.
    Otherwise: "1 if the model labeled cancer ..., else 0" per the user's
    spec; downstream organ-met flags additionally require the cancer flag.
    """
    if kind == "imaging":
        cols = [
            "derived_any_cancer",
            "derived_response",
            "derived_progression",
        ] + [f"derived_{site}_met" for site in SITE_CODE_PREFIXES]
        out: dict[str, Any] = {col: None for col in cols}
        if not isinstance(labels, dict):
            return out
        ca = _field_code(labels, "image_ca")
        overall = _field_code(labels, "image_overall")
        has_cancer = ca == 1
        out["derived_any_cancer"] = int(has_cancer)
        out["derived_response"] = int(has_cancer and overall == 1)
        out["derived_progression"] = int(has_cancer and overall in (3, 4))
        codes: list[str] = []
        sites = labels.get("image_cancer_sites")
        if isinstance(sites, list):
            for entry in sites:
                if isinstance(entry, dict):
                    code = entry.get("icdo_topography_code")
                    if isinstance(code, str):
                        codes.append(code.strip().upper())
        for site_name, prefixes in SITE_CODE_PREFIXES.items():
            present = any(any(code.startswith(p) for p in prefixes) for code in codes)
            out[f"derived_{site_name}_met"] = int(has_cancer and present)
        return out
    if kind == "medonc":
        cols = ["derived_any_cancer", "derived_response", "derived_progression"]
        out = {col: None for col in cols}
        if not isinstance(labels, dict):
            return out
        ca = _field_code(labels, "md_ca")
        status = _field_code(labels, "md_ca_status")
        has_cancer = ca == 1
        out["derived_any_cancer"] = int(has_cancer)
        out["derived_response"] = int(has_cancer and status == 1)
        out["derived_progression"] = int(has_cancer and status in (3, 4))
        return out
    return {}


def flatten_labels_with_schema(
    labels: dict[str, Any] | None,
    schema: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    src = labels if isinstance(labels, dict) else {}
    for key, info in schema.items():
        kind = info[0]
        value = src.get(key)
        if kind == "dict":
            subkeys = info[1:]
            d = value if isinstance(value, dict) else {}
            for subkey in subkeys:
                col = f"{key}_{subkey}"
                subvalue = d.get(subkey)
                if isinstance(subvalue, (dict, list)):
                    flat[col] = json.dumps(subvalue, ensure_ascii=False, default=str)
                else:
                    flat[col] = subvalue
        elif kind == "list_dict":
            col = f"{key}_json"
            flat[col] = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, list) else None
        elif kind == "list_prim":
            flat[key] = value if isinstance(value, list) else None
        else:
            if isinstance(value, (dict, list)):
                flat[key] = json.dumps(value, ensure_ascii=False, default=str)
            else:
                flat[key] = value
    return flat


def write_parquet_from_jsonl(
    jsonl_path: Path,
    parquet_path: Path,
    *,
    encoding: str,
    input_records: list[Record] | None = None,
) -> None:
    input_lookup: dict[str, dict[str, Any]] = {}
    if input_records:
        for record in input_records:
            input_lookup[str(record.record_id)] = record.data

    all_input_keys: list[str] = []
    seen_keys: set[str] = set()
    if input_records:
        for record in input_records:
            for key in record.data.keys():
                if key in seen_keys or key in LABEL_OUTPUT_COLUMNS:
                    continue
                seen_keys.add(key)
                all_input_keys.append(key)

    raw_rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding=encoding) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_rows.append(raw)

    label_schema = classify_label_schema(raw.get("labels") for raw in raw_rows)
    label_kind = detect_label_kind(label_schema)

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        flat: dict[str, Any] = {}
        input_data = input_lookup.get(str(raw.get("record_id"))) if input_lookup else None
        for key in all_input_keys:
            flat[key] = input_data.get(key) if input_data else None
        flat.update(
            {
                "record_id": raw.get("record_id"),
                "input_index": raw.get("input_index"),
                "model": raw.get("model"),
                "server_url": raw.get("server_url"),
                "labels_json": json.dumps(raw["labels"], ensure_ascii=False) if raw.get("labels") is not None else None,
                "raw_response": raw.get("raw_response"),
                "reasoning_content": raw.get("reasoning_content"),
                "request_error": raw.get("request_error"),
                "parse_error": raw.get("parse_error"),
                "usage_json": json.dumps(raw["usage"], ensure_ascii=False) if raw.get("usage") else None,
            }
        )
        if "input" in raw:
            flat["input_json"] = json.dumps(raw["input"], ensure_ascii=False, default=str)
        flat.update(flatten_labels_with_schema(raw.get("labels"), label_schema))
        flat.update(derive_manual_label_columns(raw.get("labels"), label_kind))
        rows.append(flat)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        pa = None  # type: ignore[assignment]
        pq = None  # type: ignore[assignment]
    if pa is not None and pq is not None:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, str(parquet_path))
        return

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Writing the parquet output requires pyarrow or pandas. Install pyarrow (`pip install pyarrow`) "
            "or rerun with --no-parquet to skip this step."
        ) from exc
    pd.DataFrame(rows).to_parquet(parquet_path)
