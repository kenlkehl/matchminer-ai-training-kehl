"""Generate registry-seeded synthetic protocols with matchminer_ai.trials.generate_synthetic_protocols.

Seeds are a JSON-lines file of registry records (nct_id, title, brief_summary,
detailed_description, eligibility_criteria). Requests go through the inference
package's endpoint pool with adaptive concurrency and vLLM load feedback. The
servers come from <output-dir>/servers.json, written from --server on first run
and re-read while running: edit it to add or remove a server without restarting.
Each server entry carries its own served model ID. Reruns skip completed trials.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import httpx
import pandas as pd

from matchminer_ai import load_default_preset
from matchminer_ai.trials import generate_synthetic_protocols


def served_model(url):
    response = httpx.get(url.rstrip("/") + "/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = args.output_dir / "servers.json"
    if not registry.exists():
        servers = []
        for spec in args.server:
            url, _, cap = spec.partition("=")
            url = url.rstrip("/") + ("" if url.rstrip("/").endswith("/v1") else "/v1")
            servers.append({"url": url, "model": served_model(url), "upstream_concurrency": int(cap or 64)})
        registry.write_text(json.dumps({"ready": True, "servers": servers}, indent=2))
    servers = json.loads(registry.read_text())["servers"]
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] servers: " + ", ".join(
        f"{s['url']} ({s['model'].split('/')[-1]}, cap {s['upstream_concurrency']})" for s in servers), flush=True)

    seeds = pd.read_json(args.seeds, lines=True).rename(columns={"nct_id": "trial_id"})
    if args.limit:
        seeds = seeds.head(args.limit)
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=[], server_urls_file=str(registry), request_timeout=7200)
    config.remote["adaptive_concurrency"] = {**config.remote.get("adaptive_concurrency", {}),
                                             "enabled": True, "metrics_interval": 10}
    settings = config.synthetic_protocol_generation
    settings["chunk_size"] = args.chunk_size
    # The runtime model selects the sampling profile; each server's own model ID comes from the registry.
    settings["remote"]["model_name"] = args.model_name
    settings["remote"]["request_params"]["max_tokens"] = args.max_tokens
    started = time.time()
    result, metadata = generate_synthetic_protocols(
        seeds, config=config, output_dir=args.output_dir / "protocols", return_metadata=True,
        progress_callback=lambda stage, done, total, detail: print(
            f"[{datetime.now(timezone.utc):%H:%M:%S} +{(time.time() - started) / 60:,.1f} min] {stage} {detail}",
            flush=True),
    )
    metadata.pop("config_snapshot", None)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(
        {k: v for k, v in metadata.items() if k != "trial_attempts"}, indent=2, default=str))
    (args.output_dir / "trial_attempts.json").write_text(json.dumps(metadata["trial_attempts"], default=str))
    result.drop(columns="protocol").to_csv(args.output_dir / "protocols.csv", index=False)
    print(f"finished: {len(result)} of {len(seeds)} protocols, {len(metadata['trial_errors'])} failed, "
          f"{metadata['skipped_completed']} already complete", flush=True)
    return 0 if not metadata["trial_errors"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=Path, required=True, help="JSON lines of registry records")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server", action="append", default=[], help="BASE_URL[=MAX_CONCURRENCY] for a new servers.json")
    parser.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--limit", type=int, help="Only the first N seeds (smoke tests)")
    sys.exit(main(parser.parse_args()))
