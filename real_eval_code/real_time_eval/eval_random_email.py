#!/usr/bin/env python3
"""
Pull a random email from the database, look up the patient summary by MRN,
retrieve top trial spaces from a pre-embedded parquet file, optionally
re-rank with a TrialChecker model, and print the results.

The pre-embedded parquet file is produced by embed_trial_spaces.py.

Usage:
    python eval_random_email.py /path/to/embedding_model --embeddings trial_space_embeddings.parquet

Examples:
    # Basic retrieval
    python eval_random_email.py ../../../models/trialspace --embeddings trial_space_embeddings.parquet

    # With trial checker re-ranking
    python eval_random_email.py /path/to/model --embeddings trial_space_embeddings.parquet \
        --trial-checker /path/to/trial_checker_model

    # Specify GPU
    python eval_random_email.py /path/to/model --embeddings trial_space_embeddings.parquet --gpu 1
"""

import argparse
import configparser
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
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to pre-embedded trial spaces parquet file "
                             "(produced by embed_trial_spaces.py)")
    parser.add_argument("--trial-checker", type=str, default=None,
                        help="Path to a trained trial checker model (ModernBERT). "
                             "If provided, re-scores top-10 trial spaces.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id (default: 0)")
    parser.add_argument("--max-seq-length", type=int, default=2500)
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

    # Fetch a random email
    cur.execute(
        "SELECT id, mrn, subject, body, cc, email_from, email_not_send, "
        "provider_email_not_send_reason, other_email_not_send_reason, "
        "email_sent, date_sent, date_created, redcap_record_id, "
        "execution_timestamp, hipaa_logged, recipient_npi "
        "FROM activate_emails "
        "WHERE body IS NOT NULL AND body != '' "
        "ORDER BY random() LIMIT 1"
    )
    email_row = cur.fetchone()

    if email_row is None:
        print("No emails found in activate_emails.")
        cur.close()
        conn.close()
        return

    email_columns = [
        "id", "mrn", "subject", "body", "cc", "email_from", "email_not_send",
        "provider_email_not_send_reason", "other_email_not_send_reason",
        "email_sent", "date_sent", "date_created", "redcap_record_id",
        "execution_timestamp", "hipaa_logged", "recipient_npi",
    ]
    email_data = dict(zip(email_columns, email_row))
    email_mrn = email_data["mrn"]

    # Look up patient summary by MRN
    cur.execute(
        "SELECT id, mrn, patient_summary, patient_boilerplate FROM activate_info "
        "WHERE mrn = %s AND patient_summary IS NOT NULL "
        "LIMIT 1",
        (email_mrn,)
    )
    patient_row = cur.fetchone()
    cur.close()
    conn.close()

    if patient_row is None:
        print(f"No patient summary found in activate_info for mrn={email_mrn}.")
        print("Cannot proceed with trial matching without a patient summary.")
        return

    patient_id, mrn, patient_summary, patient_boilerplate = patient_row

    # --- Load pre-embedded trial spaces ------------------------------------
    print(f"Loading pre-embedded trial spaces from {args.embeddings} ...")
    df = pd.read_parquet(args.embeddings)

    if len(df) == 0:
        print("Pre-embedded trial spaces file is empty.")
        return

    space_ids = df["id"].tolist()
    nct_ids = df["nct_id"].tolist()
    space_texts = df["this_cohort"].tolist()
    trial_boilerplates = df["boilerplate_text"].tolist()
    space_embs_np = np.array(df["embedding"].tolist(), dtype=np.float32)

    print(f"Loaded {len(df)} pre-embedded trial spaces (dim={space_embs_np.shape[1]}).")

    # --- Embedding model ---------------------------------------------------
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model from {args.model_dir} on {device} ...")
    model = SentenceTransformer(args.model_dir, trust_remote_code=True, device=device)
    model.max_seq_length = args.max_seq_length
    model.prompts["query"] = QUERY_PROMPT

    # --- Encode patient summary --------------------------------------------
    print("Encoding patient summary ...")
    with torch.no_grad():
        patient_emb = model.encode(
            [patient_summary],
            convert_to_tensor=False,
            normalize_embeddings=True,
            prompt="query",
        )

    assert patient_emb.shape[1] == space_embs_np.shape[1], (
        f"Embedding dimension mismatch: patient={patient_emb.shape[1]}, "
        f"trials={space_embs_np.shape[1]}"
    )

    # --- Cosine similarity (embeddings are already L2-normalised) ----------
    similarities = (patient_emb @ space_embs_np.T).squeeze(0)
    top_indices = np.argsort(similarities)[::-1][:10]

    # --- Trial checker re-scoring (optional) ------------------------------
    tc_scores = None
    if args.trial_checker:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"Loading trial checker model from {args.trial_checker} on {device} ...")
        tc_tokenizer = AutoTokenizer.from_pretrained(args.trial_checker)
        tc_model = AutoModelForSequenceClassification.from_pretrained(
            args.trial_checker
        ).to(device)
        tc_model.eval()

        print("Running trial checker on top 10 trial spaces ...")
        tc_texts = [
            space_texts[idx] + "\nNow here is the patient summary:" + patient_summary
            for idx in top_indices
        ]
        inputs = tc_tokenizer(
            tc_texts, truncation=True, padding=True,
            max_length=4096, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = tc_model(**inputs)
            logits = outputs.logits.squeeze(-1)
            tc_scores = torch.sigmoid(logits).cpu().numpy()

        # Re-rank top indices by trial checker score (highest first)
        rerank_order = np.argsort(tc_scores)[::-1]
        top_indices = top_indices[rerank_order]
        tc_scores = tc_scores[rerank_order]

    # --- Print results -----------------------------------------------------
    print("\n" + "=" * 80)
    print("EMAIL FROM activate_emails")
    print("=" * 80)
    for col in ["id", "mrn", "subject", "email_from", "cc",
                "date_sent", "email_sent", "date_created"]:
        print(f"{col}: {email_data[col]}")

    print("\n" + "=" * 80)
    print(f"PATIENT SUMMARY (id={patient_id}, mrn={mrn})")
    print("=" * 80)
    print(patient_summary)

    print("\n" + "=" * 80)
    if tc_scores is not None:
        print("TOP 10 TRIAL SPACES (re-ranked by trial checker)")
    else:
        print("TOP 10 MOST SIMILAR TRIAL SPACES")
    print("=" * 80)
    for rank, idx in enumerate(top_indices, 1):
        line = (f"\n--- Rank {rank} | similarity={similarities[idx]:.4f} | "
                f"nct_id={nct_ids[idx]} | space_id={space_ids[idx]}")
        if tc_scores is not None:
            line += f" | tc_score={tc_scores[rank - 1]:.4f}"
        line += " ---"
        print(line)
        print(space_texts[idx])


if __name__ == "__main__":
    main()
