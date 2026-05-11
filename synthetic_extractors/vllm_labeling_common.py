from __future__ import annotations

import argparse
import atexit
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

    request_group = parser.add_argument_group("request")
    request_group.add_argument("--workers", type=int, default=4, help="Concurrent labeling requests.")
    request_group.add_argument("--temperature", type=float, default=0.0)
    request_group.add_argument("--max-tokens", type=int, default=2200)
    request_group.add_argument("--request-timeout", type=float, default=240.0)
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
    ("gemma-4", "gemma4"),
    ("gemma4", "gemma4"),
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

    parsed, parse_error = parse_json_object(content) if content else (None, None)
    row: dict[str, Any] = {
        "record_id": record.record_id,
        "input_index": record.input_index,
        "model": request_model_name(args),
        "server_url": base_url,
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


def run_labeling(
    *,
    args: argparse.Namespace,
    build_messages: Callable[[Record, argparse.Namespace], list[dict[str, str]]],
) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_records(args)
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

            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                status = "ok" if row.get("labels") and not row.get("request_error") else "failed"
                print(f"[{completed}/{len(records)}] {status} {row['record_id']}", file=sys.stderr)

    if not args.no_parquet:
        parquet_path = resolve_parquet_path(args)
        write_parquet_from_jsonl(output_path, parquet_path, encoding=args.encoding)
        print(f"Wrote parquet copy to {parquet_path}", file=sys.stderr)


def resolve_parquet_path(args: argparse.Namespace) -> Path:
    if args.parquet_output:
        return Path(args.parquet_output)
    output_path = Path(args.output)
    if output_path.suffix.lower() in {".jsonl", ".ndjson", ".json"}:
        return output_path.with_suffix(".parquet")
    return output_path.with_name(output_path.name + ".parquet")


def write_parquet_from_jsonl(jsonl_path: Path, parquet_path: Path, *, encoding: str) -> None:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding=encoding) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            flat: dict[str, Any] = {
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
            if "input" in raw:
                flat["input_json"] = json.dumps(raw["input"], ensure_ascii=False, default=str)
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

