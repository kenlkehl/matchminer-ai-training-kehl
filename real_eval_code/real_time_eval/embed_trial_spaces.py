#!/usr/bin/env python3
"""
Pre-embed all trial spaces from the database and save to a parquet file.

Reads DB credentials from database_secrets.txt (auto-resolved relative to the
script location at ../../data/phi/database_secrets.txt).

Trial spaces come from the trial_spaces table (this_cohort column).
Embeddings are saved to a parquet file with an 'embedding' column (list of
floats per row), following the same format as hfspaces/preembed_trials.py.

Usage:
    python embed_trial_spaces.py /path/to/embedding_model

Examples:
    # Basic usage
    python embed_trial_spaces.py ../../../models/trialspace

    # Custom output path and GPU
    python embed_trial_spaces.py /path/to/model --output /data/trial_embeddings.parquet --gpu 1

    # Larger batch size for faster encoding
    python embed_trial_spaces.py /path/to/model --batch-size 24
"""

import argparse
import configparser
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from sentence_transformers import SentenceTransformer

SECRETS_FILE = Path(__file__).resolve().parents[3] / "data" / "phi" / "database_secrets.txt"

QUERY_PROMPT = (
    "Instruct: Given a cancer patient summary, retrieve clinical trial options "
    "that are reasonable for that patient; or, given a clinical trial option, "
    "retrieve cancer patients who are reasonable candidates for that trial."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", help="Folder containing the saved embedding model")
    default_output = str(
        Path(__file__).resolve().parents[3]
        / "data" / "phi" / "real_time" / "trial_space_embeddings.parquet"
    )
    parser.add_argument("--output", type=str, default=default_output,
                        help="Output parquet file path (default: data/phi/real_time/trial_space_embeddings.parquet)")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id (default: 0)")
    parser.add_argument("--max-seq-length", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Encoding batch size (default: 12)")
    parser.add_argument("--secrets", type=str, default=str(SECRETS_FILE),
                        help="Path to database_secrets.txt")
    return parser.parse_args()


def get_db_connection(secrets_path: str):
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


def main():
    args = parse_args()

    # --- Database ----------------------------------------------------------
    conn = get_db_connection(args.secrets)
    cur = conn.cursor()

    cur.execute(
        "SELECT id, nct_id, this_cohort, boilerplate_text FROM trial_spaces "
        "WHERE this_cohort IS NOT NULL"
    )
    spaces_rows = cur.fetchall()
    cur.close()
    conn.close()

    # De-duplicate trial spaces: keep only the first row per (nct_id, this_cohort)
    seen = set()
    deduped_rows = []
    for r in spaces_rows:
        key = (r[1], r[2])  # (nct_id, this_cohort)
        if key not in seen:
            seen.add(key)
            deduped_rows.append(r)

    print(f"Fetched {len(spaces_rows)} trial spaces, "
          f"{len(deduped_rows)} unique after de-duplicating by (nct_id, this_cohort).")
    spaces_rows = deduped_rows

    space_ids = [r[0] for r in spaces_rows]
    nct_ids = [r[1] for r in spaces_rows]
    space_texts = [r[2] for r in spaces_rows]
    boilerplate_texts = [r[3] for r in spaces_rows]

    # --- Embedding model ---------------------------------------------------
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model from {args.model_dir} on {device} ...")
    model = SentenceTransformer(args.model_dir, trust_remote_code=True, device=device)
    model.max_seq_length = args.max_seq_length
    model.prompts["query"] = QUERY_PROMPT

    # --- Encode ------------------------------------------------------------
    print(f"Encoding {len(space_texts)} trial spaces ...")
    t0 = time.time()
    with torch.no_grad():
        space_embs = model.encode(
            space_texts,
            batch_size=args.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            prompt="query",
            show_progress_bar=True,
        )
    elapsed = time.time() - t0
    print(f"Encoding completed in {elapsed:.1f}s")

    embeddings_np = space_embs.cpu().to(dtype=torch.float32).numpy()

    # --- Save to parquet ---------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "id": space_ids,
        "nct_id": nct_ids,
        "this_cohort": space_texts,
        "boilerplate_text": boilerplate_texts,
        "embedding": [row.tolist() for row in embeddings_np],
    })
    df.to_parquet(output_path, index=False)

    print(f"Saved {len(df)} trial space embeddings to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Embedding dimension: {embeddings_np.shape[1]}")

    # --- Save metadata JSON ------------------------------------------------
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedder_model": str(Path(args.model_dir).resolve()),
        "max_seq_length": args.max_seq_length,
        "num_trials": len(df),
        "embedding_dim": int(embeddings_np.shape[1]),
        "normalized": True,
        "batch_size": args.batch_size,
        "encoding_time_s": round(elapsed, 1),
        "query_prompt": QUERY_PROMPT,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
