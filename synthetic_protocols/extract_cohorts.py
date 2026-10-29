"""Extract cohorts and eligibility criteria from a synthetic protocol batch with matchminer_ai.trials.extract_protocol_cohorts.

Reads a generate_batch.py output directory: every protocols/<NCT>/protocol.md
whose meta.json records a completed generation, plus the registry seeds in
seeds.jsonl. Each trial's decision is saved under <output-dir>/trials/ and
reused on rerun. The exports carry each cohort's provenance: the synthetic
protocol (path, SHA-256, generation prompt hash, layout, generating model and
endpoint), the ClinicalTrials.gov record it was seeded from (including the
original registry eligibility criteria and when they were retrieved), and the
extraction prompt hash, schema, model and endpoint. Servers come from
<output-dir>/servers.json, written from --server on first run and re-read while
running, as in generate_batch.py. See synthetic_protocols/README.md.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import httpx
import pandas as pd

import matchminer_ai
from matchminer_ai import load_default_preset
from matchminer_ai.trials import extract_protocol_cohorts


SEED_FIELDS = ("title", "brief_summary", "detailed_description", "eligibility_criteria")
REGISTRY_FIELDS = (*SEED_FIELDS, "last_update_post_date", "retrieved_at")


def served_model(url):
    response = httpx.get(url.rstrip("/") + "/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def git_revision(path):
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True,
                                        check=True).stdout.strip()
        return {"path": str(path), "commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(path), "commit": None, "dirty": None}


def text(value):
    return value if isinstance(value, str) else ""


def load_batch(batch_dir):
    """Completed synthetic protocols in seed order, with generation and registry provenance."""
    seeds = [json.loads(line) for line in (batch_dir / "seeds.jsonl").read_text().splitlines() if line.strip()]
    trials, not_generated = [], []
    for seed in seeds:
        trial_dir = batch_dir / "protocols" / seed["nct_id"]
        meta = json.loads((trial_dir / "meta.json").read_text()) if (trial_dir / "meta.json").exists() else {}
        if meta.get("finish_reason") != "stop" or not (trial_dir / "protocol.md").exists():
            not_generated.append(seed["nct_id"])
            continue
        # The protocol must have been written from this exact registry record.
        changed = [key for key in SEED_FIELDS if text(meta["seed"].get(key)) != text(seed.get(key))]
        if changed:
            raise ValueError(f"{seed['nct_id']}: meta.json seed differs from seeds.jsonl in {changed}")
        response = meta.get("response_metadata") or {}
        trials.append({
            "trial_id": seed["nct_id"], "protocol_path": str(trial_dir / "protocol.md"),
            "synthetic_prompt_sha256": meta["prompt_sha256"], "synthetic_layout_index": meta["layout_index"],
            "synthetic_attempt": meta["attempt"], "synthetic_completed_at": meta["completed_at"],
            "synthetic_endpoint": response.get("endpoint"), "synthetic_model": response.get("request_model"),
            **{f"registry_{key}": seed.get(key) for key in REGISTRY_FIELDS},
        })
    trials = pd.DataFrame(trials)
    if trials.empty:
        raise ValueError(f"No completed synthetic protocols in {batch_dir / 'protocols'}")
    if trials.synthetic_prompt_sha256.nunique() > 1:
        raise ValueError("The batch mixes synthetic protocol prompt versions; extract each version separately")
    return trials, not_generated


def trial_provenance(trials, metadata, decisions_dir):
    """One row per synthetic protocol: its sources plus the extraction outcome."""
    rows = []
    for trial_id in trials.trial_id:
        decision = metadata["trial_decisions"].get(trial_id)
        attempts = metadata["trial_attempts"].get(trial_id, [])
        response = (attempts[-1].get("response_metadata") or {}) if attempts else {}
        status = ("failed" if decision is None else "cohorts" if decision["cancer_treatment_trial"] == "yes"
                  else "not_cancer_treatment")
        rows.append({
            "trial_id": trial_id, "protocol_sha256": metadata["source_documents"][trial_id]["sha256"],
            "extraction_status": status,
            "cohort_count": (len(decision.get("cohorts", [])) or 1) if status == "cohorts" else 0,
            "extraction_error": metadata["trial_errors"].get(trial_id),
            "extraction_attempts": len(attempts), "extraction_endpoint": response.get("endpoint"),
            "extraction_model": response.get("request_model"),
            "extraction_prompt_sha256": metadata["prompt_sha256"],
            "extraction_schema_version": metadata["schema_version"],
            "extraction_decision_path": str(decisions_dir / trial_id / "extraction.json") if decision else None,
        })
    return trials.merge(pd.DataFrame(rows), on="trial_id", validate="one_to_one")


def main(args):
    output_dir = args.output_dir or args.batch_dir / "cohort_extraction"
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = output_dir / "servers.json"
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

    trials, not_generated = load_batch(args.batch_dir)
    if args.limit:
        trials = trials.head(args.limit)
    print(f"{len(trials)} synthetic protocols to extract; {len(not_generated)} seeds have no completed protocol",
          flush=True)
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=[], server_urls_file=str(registry), request_timeout=7200)
    config.remote["adaptive_concurrency"] = {**config.remote.get("adaptive_concurrency", {}),
                                             "enabled": True, "metrics_interval": 10}
    settings = config.protocol_cohort_extraction
    settings["chunk_size"] = args.chunk_size
    # The runtime model selects the sampling profile; each server's own model ID comes from the registry.
    settings["remote"]["model_name"] = args.model_name
    settings["remote"]["request_params"]["max_tokens"] = args.max_tokens
    decisions_dir = output_dir / "trials"
    # Record the code that runs, before either repository can change during a long run.
    repositories = {"training_repo": git_revision(Path(__file__).resolve().parents[1]),
                    "inference_repo": git_revision(Path(matchminer_ai.__file__).resolve().parents[2])}
    started_at, started = datetime.now(timezone.utc).isoformat(timespec="seconds"), time.time()
    cohorts, metadata = extract_protocol_cohorts(
        trials[["trial_id", "protocol_path"]], config=config, output_dir=decisions_dir, return_metadata=True,
        progress_callback=lambda stage, done, total, detail: None if stage == "read" and done < total else print(
            f"[{datetime.now(timezone.utc):%H:%M:%S} +{(time.time() - started) / 60:,.1f} min] {stage} {detail}",
            flush=True),
    )

    provenance = trial_provenance(trials, metadata, decisions_dir)
    cohorts = cohorts.assign(protocol_origin="synthetic").merge(provenance, on="trial_id", validate="many_to_one")
    for name, frame in (("cohorts", cohorts), ("trials", provenance)):
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    metadata.pop("config_snapshot", None)
    (output_dir / "trial_attempts.json").write_text(json.dumps(metadata["trial_attempts"], default=str))
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch_dir": str(args.batch_dir.resolve()), "servers": servers, **repositories,
        "settings": {"model_name": args.model_name, "max_tokens": args.max_tokens, "chunk_size": args.chunk_size,
                     "limit": args.limit},
        "synthetic_prompt_sha256": provenance.synthetic_prompt_sha256.iloc[0],
        "trials": len(provenance), "cohort_rows": len(cohorts),
        "status_counts": provenance.extraction_status.value_counts().to_dict(),
        "seeds_without_protocol": not_generated,
        **{key: metadata[key] for key in ("extraction_method", "schema_version", "representation_version",
                                          "prompt_sha256", "structured_outputs", "skipped_completed",
                                          "trial_errors", "model_metadata")},
        "trial_rejections": sorted(metadata["trial_rejections"]),
    }, indent=2, default=str))
    print(f"finished: {len(cohorts)} cohort rows from {len(provenance)} protocols; "
          f"{len(metadata['trial_errors'])} failed, {len(metadata['trial_rejections'])} not cancer treatment, "
          f"{metadata['skipped_completed']} reused", flush=True)
    return 0 if not metadata["trial_errors"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch-dir", type=Path, required=True, help="A generate_batch.py output directory")
    parser.add_argument("--output-dir", type=Path, help="Default: <batch-dir>/cohort_extraction")
    parser.add_argument("--server", action="append", default=[], help="BASE_URL[=MAX_CONCURRENCY] for a new servers.json")
    parser.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--limit", type=int, help="Only the first N protocols (smoke tests)")
    sys.exit(main(parser.parse_args()))
