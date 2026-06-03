#!/usr/bin/env python3
"""
Re-pull, re-spacify, and re-embed the trials the system already knows about.

This refreshes the pre-embedded trial-space index that simulated_oa_run.py reads
(data/phi/real_time/trial_space_embeddings.parquet). It:

  1. Pulls the distinct NCT IDs from the trial_spaces table in the database the
     other real-time scripts access (database_secrets.txt, auto-resolved to
     ../../../data/phi/database_secrets.txt relative to this script).
  2. Drives ../../pull_spacify_and_embed_ctgov in NCT-ID mode, which pulls those
     trials' current text from ClinicalTrials.gov, runs ../../0b_create_trial_spaces.py
     to extract trial spaces, and embeds them with embed_trial_spaces.py.
  3. Writes the embedded parquet (by default the canonical name read by
     simulated_oa_run.py) into ../../../data/phi/real_time.

Spacification needs a vLLM backend, so a real run requires --gpus (local mode)
or --server_urls / --server_urls_file (remote mode); these are forwarded to the
wrapper. Any other unrecognized flags are forwarded too (e.g. --spacify-model,
--gpu-mem-util, --skip-embed, --dry-run).

Examples:
    # Local GPUs
    python spacify_trials_freshly.py --gpus 0,1,2,3

    # Remote vLLM pool
    python spacify_trials_freshly.py --server_urls_file /path/to/servers.json

    # Quick test on the first 5 trials
    python spacify_trials_freshly.py --gpus 0 --limit 5

    # Inspect the command without pulling/spacifying/embedding
    python spacify_trials_freshly.py --gpus 0 --dry-run
"""

from __future__ import annotations

import argparse
import configparser
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]  # matchminer-ai-training
SECRETS_FILE = Path(__file__).resolve().parents[3] / "data" / "phi" / "database_secrets.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "phi" / "real_time"
DEFAULT_EMBEDDING_MODEL = Path(__file__).resolve().parents[3] / "models" / "trialspace"
WRAPPER = REPO_ROOT / "pull_spacify_and_embed_ctgov"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--table", default="trial_spaces",
                        help="Database table to read NCT IDs from (default: trial_spaces).")
    parser.add_argument("--nct-column", default="nct_id",
                        help="Column holding the NCT IDs (default: nct_id).")
    parser.add_argument("--nct-ids-file", default=None,
                        help="Use this file of NCT IDs instead of querying the database.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use the first N NCT IDs (handy for quick test runs).")
    parser.add_argument("--embedding-model", default=str(DEFAULT_EMBEDDING_MODEL),
                        help="Path/HF ID for the SentenceTransformer TrialSpace embedding model.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for generated artifacts (default: data/phi/real_time).")
    parser.add_argument("--output-name", default="trial_space_embeddings.parquet",
                        help="Filename for the embedded parquet inside --output-dir "
                             "(default: trial_space_embeddings.parquet, the name simulated_oa_run.py reads).")
    parser.add_argument("--browser-json-output", default=None,
                        help="Browser-loadable embedded JSON path "
                             "(default: <output-name stem>.browser.json inside --output-dir).")
    parser.add_argument("--secrets", default=str(SECRETS_FILE),
                        help="Path to database_secrets.txt.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for the wrapper and its child scripts.")

    # Backend pass-through for the spacify stage (forwarded to the wrapper).
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU IDs for local spacify mode.")
    parser.add_argument("--gpus-per-instance", "--gpus-per-server", dest="gpus_per_instance",
                        type=int, default=None, help="GPUs per local vLLM instance.")
    parser.add_argument("--server_urls", "--server-urls", dest="server_urls", default=None)
    parser.add_argument("--server_urls_file", "--server-urls-file", dest="server_urls_file", default=None)

    return parser.parse_known_args()


def get_db_connection(secrets_path: str):
    import psycopg2

    cfg = configparser.ConfigParser()
    cfg.read(secrets_path)

    def _strip(val: str) -> str:
        return val.strip('"')

    return psycopg2.connect(
        host=_strip(cfg["database"]["server"]),
        port=int(_strip(cfg["database"]["port"])),
        dbname=_strip(cfg["database"]["database"]),
        user=_strip(cfg["user"]["user"]),
        password=_strip(cfg["user"]["password"]),
    )


def load_nct_ids_from_db(args: argparse.Namespace) -> list[str]:
    conn = get_db_connection(args.secrets)
    cur = conn.cursor()
    # Table/column come from local CLI args, not untrusted input; quoted as identifiers.
    cur.execute(
        f'SELECT DISTINCT "{args.nct_column}" FROM "{args.table}" '
        f'WHERE "{args.nct_column}" IS NOT NULL ORDER BY "{args.nct_column}"'
    )
    nct_ids = [str(r[0]).strip() for r in cur.fetchall() if r[0] is not None and str(r[0]).strip()]
    cur.close()
    conn.close()
    print(f"Pulled {len(nct_ids)} distinct NCT IDs from {args.table}.{args.nct_column}")
    return nct_ids


def append_optional(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def main() -> None:
    args, extra_args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve the NCT-ID list ------------------------------------------
    if args.nct_ids_file:
        nct_ids_file = Path(args.nct_ids_file).expanduser().resolve()
        if not nct_ids_file.exists():
            raise SystemExit(f"--nct-ids-file does not exist: {nct_ids_file}")
        print(f"Using NCT IDs from {nct_ids_file}")
    else:
        nct_ids = load_nct_ids_from_db(args)
        if args.limit is not None:
            nct_ids = nct_ids[: args.limit]
            print(f"Limiting to first {len(nct_ids)} NCT IDs (--limit {args.limit})")
        if not nct_ids:
            raise SystemExit("No NCT IDs found to refresh.")
        nct_ids_file = output_dir / "fresh_nct_ids.txt"
        nct_ids_file.write_text("\n".join(nct_ids) + "\n")
        print(f"Wrote {len(nct_ids)} NCT IDs to {nct_ids_file}")

    # If a file was supplied directly, --limit still applies via re-write.
    if args.nct_ids_file and args.limit is not None:
        ids = [ln.split("#", 1)[0].strip() for ln in nct_ids_file.read_text().splitlines()]
        ids = [i for i in ids if i][: args.limit]
        nct_ids_file = output_dir / "fresh_nct_ids.txt"
        nct_ids_file.write_text("\n".join(ids) + "\n")
        print(f"Limiting to first {len(ids)} NCT IDs (--limit {args.limit}) -> {nct_ids_file}")

    # --- Resolve output paths ---------------------------------------------
    parquet_output = output_dir / args.output_name
    if args.browser_json_output:
        browser_json_output = Path(args.browser_json_output).expanduser().resolve()
    else:
        browser_json_output = output_dir / f"{parquet_output.stem}.browser.json"

    embedding_model = str(Path(args.embedding_model).expanduser().resolve())

    # --- Build and run the wrapper command --------------------------------
    cmd = [
        args.python,
        str(WRAPPER),
        "--nct-ids-file", str(nct_ids_file),
        "--embedding-model", embedding_model,
        "--output-dir", str(output_dir),
        "--parquet-output", str(parquet_output),
        "--browser-json-output", str(browser_json_output),
    ]
    append_optional(cmd, "--gpus", args.gpus)
    append_optional(cmd, "--gpus-per-instance", args.gpus_per_instance)
    append_optional(cmd, "--server_urls", args.server_urls)
    append_optional(cmd, "--server_urls_file", args.server_urls_file)
    cmd.extend(extra_args)

    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    print("\nDone.")
    print(f"  Embedded parquet: {parquet_output}")
    print(f"  Browser JSON:     {browser_json_output}")


if __name__ == "__main__":
    main()
