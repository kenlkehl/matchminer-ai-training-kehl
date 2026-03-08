#!/usr/bin/env python3
"""
Pick a random patient summary from the database, embed it, embed all trial
spaces, and print the top 10 most cosine-similar trial spaces.

Reads DB credentials from database_secrets.txt (auto-resolved relative to the
script location at ../../data/phi/database_secrets.txt).

Patient summaries come from the activate_info table (patient_summary column).
Trial spaces come from the trial_spaces table (this_cohort column).

Usage:
    python eval_random_patient.py /path/to/embedding_model

Examples:
    # Basic usage with the trialspace model
    python eval_random_patient.py ../../models/trialspace

    # Specify GPU and max sequence length
    python eval_random_patient.py /path/to/model --gpu 1 --max-seq-length 2500

    # Use an alternate secrets file
    python eval_random_patient.py /path/to/model --secrets /alt/path/to/database_secrets.txt
"""

import argparse
import configparser
from pathlib import Path

import numpy as np
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Folder containing the saved embedding model")
    parser.add_argument("--max-seq-length", type=int, default=2500)
    parser.add_argument("--gpu", type=str, default="0", help="GPU id (default: 0)")
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

    # Fetch a random patient summary
    cur.execute(
        "SELECT id, mrn, patient_summary FROM activate_info "
        "WHERE patient_summary IS NOT NULL "
        "ORDER BY random() LIMIT 1"
    )
    patient_id, mrn, patient_summary = cur.fetchone()

    # Fetch all trial spaces
    cur.execute(
        "SELECT id, nct_id, this_cohort FROM trial_spaces "
        "WHERE this_cohort IS NOT NULL"
    )
    spaces_rows = cur.fetchall()
    cur.close()
    conn.close()

    space_ids = [r[0] for r in spaces_rows]
    nct_ids = [r[1] for r in spaces_rows]
    space_texts = [r[2] for r in spaces_rows]

    print(f"Loaded 1 patient (id={patient_id}, mrn={mrn}) and {len(space_texts)} trial spaces.\n")

    # --- Embedding model ---------------------------------------------------
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model from {args.model_dir} on {device} ...")
    model = SentenceTransformer(args.model_dir, trust_remote_code=True, device=device)
    model.max_seq_length = args.max_seq_length
    model.prompts["query"] = QUERY_PROMPT

    # --- Encode ------------------------------------------------------------
    print("Encoding patient summary ...")
    with torch.no_grad():
        patient_emb = model.encode(
            [patient_summary],
            convert_to_tensor=True,
            normalize_embeddings=True,
            prompt="query",
        )

    print(f"Encoding {len(space_texts)} trial spaces ...")
    with torch.no_grad():
        space_embs = model.encode(
            space_texts,
            batch_size=128,
            convert_to_tensor=True,
            normalize_embeddings=True,
            prompt="query",
            show_progress_bar=True,
        )

    # --- Cosine similarity (embeddings are already L2-normalised) ----------
    similarities = (patient_emb @ space_embs.T).squeeze(0).cpu().numpy()
    top_indices = np.argsort(similarities)[::-1][:10]

    # --- Print results -----------------------------------------------------
    print("\n" + "=" * 80)
    print("PATIENT SUMMARY")
    print("=" * 80)
    print(patient_summary)

    print("\n" + "=" * 80)
    print("TOP 10 MOST SIMILAR TRIAL SPACES")
    print("=" * 80)
    for rank, idx in enumerate(top_indices, 1):
        print(f"\n--- Rank {rank} | similarity={similarities[idx]:.4f} | "
              f"nct_id={nct_ids[idx]} | space_id={space_ids[idx]} ---")
        print(space_texts[idx])


if __name__ == "__main__":
    main()
