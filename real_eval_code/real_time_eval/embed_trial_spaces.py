#!/usr/bin/env python3
"""
Pre-embed trial spaces and save to a parquet file.

Reads DB credentials from database_secrets.txt (auto-resolved relative to the
script location at ../../data/phi/database_secrets.txt) unless --input-spaces
is supplied.

Trial spaces come from the trial_spaces table (this_cohort column), or from a
CSV/JSON/JSONL/parquet file such as the output of 0b_create_trial_spaces.py.
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

    # Also write a browser-loadable JSON file for the Electron app
    python embed_trial_spaces.py /path/to/model --browser-json-output trial_space_embeddings.browser.json

    # Embed spaces extracted offline by 0b_create_trial_spaces.py
    python embed_trial_spaces.py /path/to/model \
      --input-spaces trial_space_lineitems.csv \
      --browser-json-output trial_space_embeddings.browser.json
"""

import argparse
import configparser
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

torch.backends.cuda.enable_cudnn_sdp(False)

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
    parser.add_argument("--browser-json-output", type=str, default=None,
                        help="Optional JSON output path for browser_app embedded trial index import")
    parser.add_argument("--input-spaces", type=str, default=None,
                        help="Optional CSV/JSON/JSONL/parquet file with trial spaces. "
                             "Accepts 0b_create_trial_spaces.py columns "
                             "(nct_id, this_space, trial_boilerplate_text) and "
                             "real-time/browser import columns "
                             "(id, nct_id, this_cohort, boilerplate_text). "
                             "If omitted, reads from the database.")
    parser.add_argument("--secrets", type=str, default=str(SECRETS_FILE),
                        help="Path to database_secrets.txt")
    return parser.parse_args()


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


def load_spaces_from_database(args) -> pd.DataFrame:
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
    return pd.DataFrame({
        "id": [r[0] for r in deduped_rows],
        "nct_id": [r[1] for r in deduped_rows],
        "this_cohort": [r[2] for r in deduped_rows],
        "boilerplate_text": [r[3] for r in deduped_rows],
    })


def load_spaces_from_file(input_spaces: str) -> pd.DataFrame:
    input_path = Path(input_spaces)
    if not input_path.exists():
        raise FileNotFoundError(f"Trial spaces file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        raw = pd.read_parquet(input_path)
    elif suffix == ".jsonl" or suffix == ".ndjson":
        raw = pd.read_json(input_path, lines=True)
    elif suffix == ".json":
        payload = json.loads(input_path.read_text())
        if isinstance(payload, list):
            raw = pd.DataFrame(payload)
        elif isinstance(payload, dict):
            rows = None
            for key in ("records", "trialSpaces", "trial_spaces", "data"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
            if rows is None:
                raise ValueError(f"{input_path} looks like metadata, not trial-space records")
            raw = pd.DataFrame(rows)
        else:
            raise ValueError(f"{input_path} must contain a JSON array or an object wrapping a record array")
    else:
        raw = pd.read_csv(input_path)

    if raw.empty:
        raise ValueError(f"Trial spaces file is empty: {input_path}")

    nct_col = first_existing_column(raw, ["nct_id", "nctId", "nct"])
    text_col = first_existing_column(raw, ["this_cohort", "this_space", "trialSpaceText", "trial_space_text", "cohort", "text"])
    if nct_col is None:
        raise ValueError(f"{input_path} must contain nct_id, nctId, or nct")
    if text_col is None:
        raise ValueError(
            f"{input_path} must contain this_space, this_cohort, trialSpaceText, trial_space_text, cohort, or text"
        )

    boiler_col = first_existing_column(raw, ["boilerplate_text", "trial_boilerplate_text", "boilerplateText", "trial_boilerplate"])
    id_col = first_existing_column(raw, ["id", "spaceId", "space_id", "space_index"])
    title_col = first_existing_column(raw, ["title", "briefTitle", "brief_title", "officialTitle", "official_title"])

    df = pd.DataFrame({
        "nct_id": raw[nct_col].map(clean_text),
        "this_cohort": raw[text_col].map(clean_text),
        "boilerplate_text": raw[boiler_col].map(clean_text) if boiler_col else "",
    })
    if id_col:
        df["id"] = raw[id_col].map(clean_text)
    else:
        df["id"] = ""
    if title_col:
        df["title"] = raw[title_col].map(clean_text)

    if "space_number" in raw.columns:
        df["space_number"] = raw["space_number"].map(clean_text)

    df = df[(df["nct_id"] != "") & (df["this_cohort"] != "")].copy()
    df = df.drop_duplicates(subset=["nct_id", "this_cohort"], keep="first").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{input_path} has no non-empty trial spaces after filtering")

    df["id"] = fill_missing_space_ids(df)
    df["id"] = ensure_unique_ids(df["id"].tolist())
    keep_cols = ["id", "nct_id", "this_cohort", "boilerplate_text"]
    if "title" in df.columns:
        keep_cols.append("title")
    print(f"Loaded {len(df)} trial spaces from {input_path}")
    return df[keep_cols]


def first_existing_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def fill_missing_space_ids(df: pd.DataFrame) -> pd.Series:
    ids = df["id"].copy()
    for idx, row in df.iterrows():
        if ids.iloc[idx]:
            continue
        space_number = row.get("space_number", "")
        if space_number:
            ids.iloc[idx] = f"{row['nct_id']}-space-{space_number}"
        else:
            ids.iloc[idx] = f"{row['nct_id']}-space-{idx + 1}"
    return ids


def ensure_unique_ids(ids: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw_id in ids:
        space_id = str(raw_id)
        count = seen.get(space_id, 0)
        if count:
            out.append(f"{space_id}-{count + 1}")
        else:
            out.append(space_id)
        seen[space_id] = count + 1
    return out


def main():
    args = parse_args()

    # --- Load spaces -------------------------------------------------------
    if args.input_spaces:
        df_spaces = load_spaces_from_file(args.input_spaces)
    else:
        df_spaces = load_spaces_from_database(args)

    if df_spaces.empty:
        raise ValueError("No trial spaces to embed")

    space_ids = df_spaces["id"].tolist()
    nct_ids = df_spaces["nct_id"].tolist()
    space_texts = df_spaces["this_cohort"].tolist()
    boilerplate_texts = df_spaces["boilerplate_text"].tolist()

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
    if "title" in df_spaces.columns:
        df["title"] = df_spaces["title"].tolist()
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

    if args.browser_json_output:
        browser_json_path = Path(args.browser_json_output)
        browser_json_path.parent.mkdir(parents=True, exist_ok=True)
        browser_records = [
            {
                "spaceId": str(space_id),
                "nctId": nct_id,
                "title": title or nct_id,
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
                "trialSpaceText": clean_text(space_text),
                "boilerplateText": clean_text(boilerplate_text),
                "embedding": embedding,
            }
            for space_id, nct_id, title, space_text, boilerplate_text, embedding in zip(
                space_ids,
                nct_ids,
                df_spaces["title"].tolist() if "title" in df_spaces.columns else [""] * len(df_spaces),
                space_texts,
                boilerplate_texts,
                embeddings_np.tolist()
            )
        ]
        with open(browser_json_path, "w") as f:
            json.dump(
                {
                    "createdAt": metadata["created_at"],
                    "embeddingModel": metadata["embedder_model"],
                    "embeddingDim": metadata["embedding_dim"],
                    "trialSpaces": len(browser_records),
                    "records": browser_records,
                },
                f,
            )
        print(f"Saved browser JSON trial index to {browser_json_path}")


def clean_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


if __name__ == "__main__":
    main()
