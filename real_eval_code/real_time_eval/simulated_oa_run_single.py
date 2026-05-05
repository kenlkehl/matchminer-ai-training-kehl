#!/usr/bin/env python3
"""
Run the trial-matching pipeline for a single patient identified by DFCI MRN.

This is a single-patient variant of simulated_oa_run.py.  It pulls raw notes
from the database, summarizes the patient, embeds the summary, retrieves the
top trial spaces by cosine similarity, optionally re-ranks with TrialChecker
and BoilerplateChecker models, and prints everything to the console.

Usage:
    python simulated_oa_run_single.py <MRN> /path/to/embedding_model

Examples:
    # Cosine similarity only
    python simulated_oa_run_single.py 12345678 ../../../models/trialspace \
        --embeddings trial_space_embeddings.parquet

    # With trial checker and boilerplate checker
    python simulated_oa_run_single.py 12345678 ../../../models/trialspace \
        --embeddings trial_space_embeddings.parquet \
        --trial-checker ../../../models/trialchecker \
        --boilerplate-checker ../../../models/boilerplatechecker \
        --gpu 2,3
"""

import argparse
import configparser
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import psycopg2
import torch
from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer

SECRETS_FILE = Path(__file__).resolve().parents[3] / "data" / "phi" / "database_secrets.txt"

DEFAULT_EMBEDDINGS = (
    Path(__file__).resolve().parents[3]
    / "data" / "phi" / "real_time" / "trial_space_embeddings.parquet"
)

DEFAULT_SUMMARIZE_SCRIPT = Path(__file__).resolve().parents[2] / "6_summarize_patients.py"

DEFAULT_SUMMARIZATION_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "data" / "phi" / "real_time" / "simulated_oa_run_single_summary"
)

QUERY_PROMPT = (
    "Instruct: Given a cancer patient summary, retrieve clinical trial options "
    "that are reasonable for that patient; or, given a clinical trial option, "
    "retrieve cancer patients who are reasonable candidates for that trial."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mrn", type=int, help="DFCI MRN of the patient to evaluate")
    parser.add_argument("model_dir", help="Folder containing the saved embedding model")
    parser.add_argument(
        "--embeddings", type=str, default=str(DEFAULT_EMBEDDINGS),
        help="Path to pre-embedded trial spaces parquet file "
             "(produced by embed_trial_spaces.py)",
    )
    parser.add_argument(
        "--trial-checker", type=str, default=None,
        help="Path to a trained trial checker model (ModernBERT). "
             "If provided, re-scores top-20 trial spaces.",
    )
    parser.add_argument(
        "--boilerplate-checker", type=str, default=None,
        help="Path to a trained boilerplate checker model (ModernBERT). "
             "If provided, scores exclusion probability for retrieved trials.",
    )
    parser.add_argument("--gpu", type=str, default="0",
                        help="Comma-separated GPU ids, e.g. '0,1,2' (default: 0)")
    parser.add_argument("--max-seq-length", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Embedding batch size (default: 12)")
    parser.add_argument("--checker-batch-size", type=int, default=32,
                        help="Batch size for trial/boilerplate checker (default: 32)")
    parser.add_argument("--secrets", type=str, default=str(SECRETS_FILE),
                        help="Path to database_secrets.txt")
    parser.add_argument(
        "--summarize-script", type=str, default=str(DEFAULT_SUMMARIZE_SCRIPT),
        help="Path to ../../6_summarize_patients.py",
    )
    parser.add_argument(
        "--summarization-model", type=str, default="openai/gpt-oss-120b",
        help="Model passed to 6_summarize_patients.py",
    )
    parser.add_argument(
        "--summarization-download-dir", type=str, default="/data1/ken/models",
        help="Download/cache directory passed to 6_summarize_patients.py",
    )
    parser.add_argument(
        "--summarization-gpu-ids", type=str, default=None,
        help="Comma-separated GPU IDs for summarization. Defaults to --gpu.",
    )
    parser.add_argument(
        "--summarization-gpus-per-server", type=int, default=1,
        help="GPUs per vLLM server for summarization (default: 1)",
    )
    parser.add_argument(
        "--summarization-server-urls", type=str, default=None,
        help="Comma-separated existing vLLM server URLs for summarization",
    )
    parser.add_argument(
        "--summarization-chunk-size", type=int, default=50000,
        help="Chunk size passed to 6_summarize_patients.py (default: 50000)",
    )
    parser.add_argument(
        "--summarization-chunk-overlap", type=int, default=500,
        help="Chunk overlap passed to 6_summarize_patients.py (default: 500)",
    )
    parser.add_argument(
        "--summarization-base-port", type=int, default=8000,
        help="Base port for summarization vLLM servers (default: 8000)",
    )
    parser.add_argument(
        "--summarization-max-concurrent-requests", type=int, default=8,
        help="Max concurrent summarization requests (default: 8)",
    )
    parser.add_argument(
        "--summarization-max-retries", type=int, default=10,
        help="Max retries per failed summarization request (default: 10)",
    )
    parser.add_argument(
        "--summarization-request-timeout", type=float, default=600.0,
        help="Per-request timeout for summarization (default: 600)",
    )
    parser.add_argument(
        "--summarization-gpu-memory-utilization",
        "--summarization-max-memory-utilization",
        dest="summarization_gpu_memory_utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization passed to 6_summarize_patients.py "
             "(default: 0.90)",
    )
    parser.add_argument(
        "--summarization-server-timeout", type=int, default=600,
        help="Server startup timeout for summarization (default: 600)",
    )
    parser.add_argument(
        "--summarization-run-deterministic", action="store_true",
        help="Pass --run_deterministic to 6_summarize_patients.py",
    )
    parser.add_argument(
        "--summarization-artifact-root", type=str,
        default=str(DEFAULT_SUMMARIZATION_ARTIFACT_ROOT),
        help="Directory where fresh summarization artifacts are written",
    )
    parser.add_argument(
        "--resume-summarization-dir", type=str, default=None,
        help="Reuse or resume a prior summarization run directory. "
             "If patient_summaries.parquet already exists there, it is reused. "
             "Otherwise the summarizer is rerun against that directory's "
             "input/shards.",
    )
    parser.add_argument(
        "--top-k-retrieve", type=int, default=20,
        help="Number of trial spaces to retrieve before re-ranking (default: 20)",
    )
    parser.add_argument(
        "--top-k-output", type=int, default=10,
        help="Number of unique trials to display after re-ranking (default: 10)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Database helpers (reused from simulated_oa_run.py)
# ---------------------------------------------------------------------------

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


def fetch_patient_id(conn, mrn):
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM activate_info WHERE mrn = %s ORDER BY id DESC LIMIT 1",
        (mrn,),
    )
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def fetch_oncologist_name(conn, mrn):
    cur = conn.cursor()
    cur.execute(
        "SELECT r.full_name "
        "FROM email_assignments ea "
        "JOIN recipients r ON r.npi = ea.recipient_npi "
        "WHERE ea.mrn = %s "
        "ORDER BY ea.enabled DESC, ea.date_assigned DESC NULLS LAST "
        "LIMIT 1",
        (mrn,),
    )
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def fetch_note_level_input(conn, mrn):
    cur = conn.cursor()
    cur.execute(
        "SELECT mrn, event_date AS date, "
        "       COALESCE(NULLIF(btrim(rpt_text), ''), NULLIF(btrim(narrative_text), '')) AS text "
        "FROM patient_notes "
        "WHERE mrn = %s "
        "  AND COALESCE(NULLIF(btrim(rpt_text), ''), NULLIF(btrim(narrative_text), '')) IS NOT NULL "
        "ORDER BY event_date, id",
        (mrn,),
    )
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=["mrn", "date", "text"])


# ---------------------------------------------------------------------------
# Model inference helpers (reused from simulated_oa_run.py)
# ---------------------------------------------------------------------------

def run_checker_batched(texts, tokenizer, model, device, batch_size, max_length,
                        output_mode="sigmoid"):
    all_scores = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            if output_mode == "sigmoid":
                scores = torch.sigmoid(logits).squeeze(-1)
            else:
                scores = torch.softmax(logits, dim=1)[:, 1]
            all_scores.append(scores.cpu().numpy())
    return np.concatenate(all_scores)


def chunk_ranges(n_items: int, n_chunks: int):
    base = n_items // n_chunks
    rem = n_items % n_chunks
    ranges, start = [], 0
    for i in range(n_chunks):
        end = start + base + (1 if i < rem else 0)
        ranges.append((start, end))
        start = end
    return ranges


def _encode_worker(texts, device_str, model_path, encode_batch_size,
                   max_seq_length, query_prompt):
    torch.cuda.set_device(int(device_str.split(":")[-1]))
    model = SentenceTransformer(model_path, trust_remote_code=True,
                                device=device_str)
    model.max_seq_length = max_seq_length
    model.prompts["query"] = query_prompt
    with torch.no_grad():
        embs = model.encode(
            texts,
            batch_size=encode_batch_size,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
            prompt="query",
        )
    return embs


def parallel_encode(all_texts, gpu_ids, model_path, encode_batch_size,
                    max_seq_length, query_prompt):
    if len(all_texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if len(gpu_ids) == 1:
        return _encode_worker(all_texts, f"cuda:{gpu_ids[0]}", model_path,
                              encode_batch_size, max_seq_length, query_prompt)
    n = len(all_texts)
    shards = chunk_ranges(n, len(gpu_ids))
    outputs = [None] * len(shards)
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futures = []
        for wi, (s, e) in enumerate(shards):
            if s == e:
                outputs[wi] = np.zeros((0, 0), dtype=np.float32)
                continue
            fut = ex.submit(_encode_worker, all_texts[s:e],
                            f"cuda:{gpu_ids[wi]}", model_path,
                            encode_batch_size, max_seq_length, query_prompt)
            futures.append((wi, fut))
        for wi, fut in futures:
            outputs[wi] = fut.result()
    embs = [arr for arr in outputs if arr.size > 0]
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), dtype=np.float32)


def _checker_worker(texts, device_str, model_path, batch_size, max_length,
                    output_mode):
    torch.cuda.set_device(int(device_str.split(":")[-1]))
    device = torch.device(device_str)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return run_checker_batched(texts, tokenizer, model, device, batch_size,
                               max_length, output_mode)


def parallel_checker(all_texts, gpu_ids, model_path, batch_size, max_length,
                     output_mode):
    if len(all_texts) == 0:
        return np.array([], dtype=np.float32)
    if len(gpu_ids) == 1:
        return _checker_worker(all_texts, f"cuda:{gpu_ids[0]}", model_path,
                               batch_size, max_length, output_mode)
    n = len(all_texts)
    shards = chunk_ranges(n, len(gpu_ids))
    outputs = [None] * len(shards)
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futures = []
        for wi, (s, e) in enumerate(shards):
            if s == e:
                outputs[wi] = np.array([], dtype=np.float32)
                continue
            fut = ex.submit(_checker_worker, all_texts[s:e],
                            f"cuda:{gpu_ids[wi]}", model_path,
                            batch_size, max_length, output_mode)
            futures.append((wi, fut))
        for wi, fut in futures:
            outputs[wi] = fut.result()
    return np.concatenate([arr for arr in outputs if arr.size > 0])


# ---------------------------------------------------------------------------
# Summarization helpers (reused from simulated_oa_run.py)
# ---------------------------------------------------------------------------

def summarization_artifact_paths(run_dir: Path):
    return {
        "run_dir": run_dir,
        "shard_dir": run_dir / "summary_shards",
        "input_parquet": run_dir / "input_notes.parquet",
        "full_output": run_dir / "patient_summaries_full.parquet",
        "patient_output": run_dir / "patient_summaries.parquet",
    }


def load_summarization_output(patient_output: Path):
    refreshed_df = pd.read_parquet(patient_output)
    if "mrn" not in refreshed_df.columns:
        raise ValueError(
            f"Expected 'mrn' column in {patient_output}, found {refreshed_df.columns.tolist()}"
        )
    refreshed_df["mrn"] = pd.to_numeric(refreshed_df["mrn"], errors="raise").astype("int64")
    refreshed_df["patient_summary"] = refreshed_df["patient_summary"].fillna("").astype(str)
    refreshed_df["patient_boilerplate_text"] = (
        refreshed_df["patient_boilerplate_text"].fillna("").astype(str)
    )
    refreshed_df = refreshed_df[
        refreshed_df["patient_summary"].str.strip() != ""
    ].drop_duplicates(subset=["mrn"], keep="last")
    return refreshed_df


def run_or_resume_summarization(notes_df, args):
    summarize_script = Path(args.summarize_script).resolve()
    if not summarize_script.exists():
        raise FileNotFoundError(f"Summarization script not found: {summarize_script}")

    if args.resume_summarization_dir:
        run_dir = Path(args.resume_summarization_dir).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume summarization dir not found: {run_dir}")
        if not run_dir.is_dir():
            raise NotADirectoryError(f"Resume summarization dir is not a directory: {run_dir}")
    else:
        artifact_root = Path(args.summarization_artifact_root)
        run_dir = artifact_root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
        run_dir.mkdir(parents=True, exist_ok=False)

    paths = summarization_artifact_paths(run_dir)
    shard_dir = paths["shard_dir"]
    input_parquet = paths["input_parquet"]
    full_output = paths["full_output"]
    patient_output = paths["patient_output"]

    if patient_output.exists():
        print(f"Reusing completed summarization output from {patient_output}")
        return load_summarization_output(patient_output), run_dir

    if input_parquet.exists():
        print(f"Resuming summarization run from existing artifacts in {run_dir}")
    else:
        if notes_df is None:
            raise FileNotFoundError(
                f"{input_parquet} does not exist and no note data was provided to restage it."
            )
        notes_df = notes_df.copy()
        notes_df["date"] = pd.to_datetime(notes_df["date"])
        notes_df.to_parquet(input_parquet, index=False)
        print(f"Staged {len(notes_df)} notes for summarization at {input_parquet}")

    summarization_gpu_ids = args.summarization_gpu_ids or args.gpu
    cmd = [
        sys.executable,
        str(summarize_script),
        "--input_parquet", str(input_parquet),
        "--output_parquet", str(full_output),
        "--patient_summaries_parquet", str(patient_output),
        "--shard_dir", str(shard_dir),
        "--model", args.summarization_model,
        "--download_dir", args.summarization_download_dir,
        "--gpu_ids", summarization_gpu_ids,
        "--gpus_per_server", str(args.summarization_gpus_per_server),
        "--patient_id_col", "mrn",
        "--date_col", "date",
        "--text_col", "text",
        "--chunk_size", str(args.summarization_chunk_size),
        "--chunk_overlap", str(args.summarization_chunk_overlap),
        "--base_port", str(args.summarization_base_port),
        "--max_concurrent_requests", str(args.summarization_max_concurrent_requests),
        "--request_timeout", str(args.summarization_request_timeout),
        "--gpu_memory_utilization", str(args.summarization_gpu_memory_utilization),
        "--server_timeout", str(args.summarization_server_timeout),
        "--max_retries", str(args.summarization_max_retries),
    ]
    if args.summarization_server_urls:
        cmd.extend(["--server_urls", args.summarization_server_urls])
    if args.summarization_run_deterministic:
        cmd.append("--run_deterministic")

    print("Running patient summarization pipeline:")
    print(f"  {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)

    if not patient_output.exists():
        raise FileNotFoundError(
            f"Summarization completed but {patient_output} was not created."
        )

    return load_summarization_output(patient_output), run_dir


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 80

def print_patient_header(mrn, patient_id, oncologist_name):
    print(f"\n{SEPARATOR}")
    print(f"  MRN:        {mrn}")
    print(f"  Patient ID: {patient_id}")
    if oncologist_name:
        print(f"  Oncologist: {oncologist_name}")
    print(SEPARATOR)


def print_patient_summary(summary, boilerplate=None):
    print(f"\n{'─' * 80}")
    print("PATIENT SUMMARY")
    print(f"{'─' * 80}")
    print(summary)
    if boilerplate:
        print(f"\n{'─' * 80}")
        print("PATIENT BOILERPLATE TEXT")
        print(f"{'─' * 80}")
        print(boilerplate)


def print_trial_results(trials):
    print(f"\n{'─' * 80}")
    print(f"RETRIEVED TRIAL SPACES  ({len(trials)} results)")
    print(f"{'─' * 80}")
    for i, t in enumerate(trials, 1):
        print(f"\n  ┌─ Rank {i} {'─' * 66}")
        print(f"  │ NCT ID:           {t['nct_id']}")
        print(f"  │ Space ID:         {t['space_id']}")
        print(f"  │ Cosine Similarity: {t['cosine_similarity']:.4f}")
        if not np.isnan(t["trialchecker_score"]):
            print(f"  │ TrialChecker:     {t['trialchecker_score']:.4f}")
        if not np.isnan(t["boilerplate_score"]):
            print(f"  │ Boilerplate:      {t['boilerplate_score']:.4f}")
        print(f"  │")
        wrapped = textwrap.fill(t["trial_space_text"], width=74)
        for line in wrapped.splitlines():
            print(f"  │ {line}")
        if t.get("trial_boilerplate_text"):
            print(f"  │")
            print(f"  │ TRIAL BOILERPLATE TEXT:")
            wrapped_bp = textwrap.fill(t["trial_boilerplate_text"], width=74)
            for line in wrapped_bp.splitlines():
                print(f"  │ {line}")
        print(f"  └{'─' * 77}")

    print(f"\n{SEPARATOR}\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    mrn = args.mrn
    gpu_ids = [int(x.strip()) for x in args.gpu.split(",") if x.strip()]
    top_k_retrieve = args.top_k_retrieve
    top_k_output = args.top_k_output

    print(f"Looking up MRN {mrn} ...")
    conn = get_db_connection(args.secrets)

    patient_id = fetch_patient_id(conn, mrn)
    if patient_id is None:
        conn.close()
        print(f"Error: MRN {mrn} not found in activate_info.")
        sys.exit(1)

    oncologist_name = fetch_oncologist_name(conn, mrn)

    notes_df = None
    resume_paths = None
    if args.resume_summarization_dir:
        resume_dir = Path(args.resume_summarization_dir).expanduser().resolve()
        resume_paths = summarization_artifact_paths(resume_dir)

    need_notes_df = (
        resume_paths is None
        or (
            not resume_paths["patient_output"].exists()
            and not resume_paths["input_parquet"].exists()
        )
    )
    if need_notes_df:
        notes_df = fetch_note_level_input(conn, mrn)
    conn.close()

    if need_notes_df and len(notes_df) == 0:
        print(f"Error: No patient_notes found for MRN {mrn}.")
        sys.exit(1)

    refreshed_summaries, summarization_run_dir = run_or_resume_summarization(
        notes_df, args
    )
    if mrn not in refreshed_summaries["mrn"].values:
        print(f"Error: Summarization produced no output for MRN {mrn}.")
        sys.exit(1)

    row = refreshed_summaries[refreshed_summaries["mrn"] == mrn].iloc[0]
    patient_summary = row["patient_summary"]
    patient_boilerplate = row["patient_boilerplate_text"]
    print(f"Summarization complete (artifacts in {summarization_run_dir})")

    # --- Load pre-embedded trial spaces -----------------------------------
    print(f"Loading pre-embedded trial spaces from {args.embeddings} ...")
    df_trials = pd.read_parquet(args.embeddings)
    if len(df_trials) == 0:
        print("Pre-embedded trial spaces file is empty.")
        sys.exit(1)

    # --- Exclude trials that should never be ranked -----------------------
    oncore_path = (
        Path(__file__).resolve().parents[3]
        / "data" / "phi" / "real_time" / "oncore_data.csv"
    )
    excluded_ncts = {"NCT04301765", "NCT04049331"}
    if oncore_path.exists():
        df_oncore = pd.read_csv(oncore_path)
        excluded_groups = df_oncore.loc[
            df_oncore["groupName"].str.contains(
                "Supportive Oncology|Radiation Oncology", na=False
            ),
            "nctId",
        ].unique()
        excluded_ncts.update(excluded_groups)
        print(f"Excluding {len(excluded_ncts)} NCT IDs "
              f"(Supportive/Radiation Oncology groups + hardcoded).")
    else:
        print(f"Warning: {oncore_path} not found; only hardcoded exclusions applied.")

    df_trials = df_trials[~df_trials["nct_id"].isin(excluded_ncts)].reset_index(drop=True)

    space_ids = df_trials["id"].tolist()
    nct_ids = df_trials["nct_id"].tolist()
    space_texts = df_trials["this_cohort"].tolist()
    trial_boilerplates = df_trials["boilerplate_text"].tolist()
    space_embs_np = np.array(df_trials["embedding"].tolist(), dtype=np.float32)
    print(f"Loaded {len(df_trials)} trial spaces (dim={space_embs_np.shape[1]}).")

    # --- Encode patient summary -------------------------------------------
    print(f"Encoding patient summary across {len(gpu_ids)} GPU(s) ...")
    patient_emb = parallel_encode(
        [patient_summary], gpu_ids, args.model_dir,
        args.batch_size, args.max_seq_length, QUERY_PROMPT,
    )

    # --- Cosine similarity ------------------------------------------------
    sim_scores = (patient_emb @ space_embs_np.T).squeeze(0)
    top_indices = np.argsort(sim_scores)[::-1][:top_k_retrieve]

    # --- Trial checker scoring --------------------------------------------
    tc_scores = np.full(top_k_retrieve, np.nan)

    if args.trial_checker:
        print(f"Running trial checker on top {top_k_retrieve} trial spaces ...")
        tc_texts = [
            space_texts[idx]
            + "\nNow here is the patient summary:"
            + patient_summary
            for idx in top_indices
        ]
        tc_scores = parallel_checker(
            tc_texts, gpu_ids, args.trial_checker,
            args.checker_batch_size, 4096, "sigmoid",
        )
        print("Trial checker scoring complete.")

    # --- Re-rank ----------------------------------------------------------
    if args.trial_checker:
        rerank_order = np.argsort(tc_scores)[::-1]
        top_indices = top_indices[rerank_order]
        tc_scores = tc_scores[rerank_order]

    # --- Deduplicate by nct_id, keep top_k_output -------------------------
    seen_ncts = set()
    selected = []
    for j, idx in enumerate(top_indices):
        nct = nct_ids[idx]
        if nct in seen_ncts:
            continue
        seen_ncts.add(nct)
        selected.append((idx, j))
        if len(selected) >= top_k_output:
            break

    # --- Boilerplate checker scoring --------------------------------------
    bp_scores = np.full(len(selected), np.nan)

    if args.boilerplate_checker:
        print(f"Running boilerplate checker on {len(selected)} selected trials ...")
        bp_texts = [
            f"Patient history: {patient_boilerplate or ''}"
            f"\nTrial exclusions:{trial_boilerplates[idx] or ''}"
            for idx, _ in selected
        ]
        bp_scores = parallel_checker(
            bp_texts, gpu_ids, args.boilerplate_checker,
            args.checker_batch_size, 3192, "softmax",
        )
        print("Boilerplate checker scoring complete.")

    # --- Build result list ------------------------------------------------
    trials = []
    for rank, ((idx, j), bp) in enumerate(zip(selected, bp_scores)):
        trials.append({
            "space_id": space_ids[idx],
            "nct_id": nct_ids[idx],
            "trial_space_text": space_texts[idx],
            "trial_boilerplate_text": trial_boilerplates[idx] or "",
            "cosine_similarity": float(sim_scores[idx]),
            "trialchecker_score": float(tc_scores[j]),
            "boilerplate_score": float(bp),
            "rank": rank + 1,
        })

    # --- Print to console -------------------------------------------------
    print_patient_header(mrn, patient_id, oncologist_name)
    print_patient_summary(patient_summary, patient_boilerplate)
    print_trial_results(trials)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
