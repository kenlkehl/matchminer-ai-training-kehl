#!/usr/bin/env python3
"""
Run the real-time trial-matching pipeline on freshly regenerated summaries for
patients who generated outbound email drafts.

The script first identifies the MRNs associated with drafted outbound emails,
pulls their raw notes from the database, runs ../../6_summarize_patients.py to
rebuild patient summaries from scratch, then retrieves the top 20 trial spaces
by cosine similarity, optionally re-ranks with a TrialChecker model,
optionally scores with a BoilerplateChecker model, and keeps the top 10 per
patient.

Outputs a parquet file with one row per patient per retrieved trial space.

Usage:
    python simulated_oa_run.py /path/to/embedding_model --embeddings trial_space_embeddings.parquet

Examples:
    # Cosine similarity only
    python simulated_oa_run.py ../../../models/trialspace \
        --embeddings trial_space_embeddings.parquet

    # With trial checker and boilerplate checker
    python simulated_oa_run.py ../../../models/trialspace \
        --embeddings trial_space_embeddings.parquet \
        --trial-checker ../../../models/trialchecker \
        --boilerplate-checker ../../../models/boilerplatechecker \
        --batch-size 64 --checker-batch-size 64 \
        --gpu 2,3
"""

import argparse
import configparser
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import psycopg2
import torch
from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer

SECRETS_FILE = Path(__file__).resolve().parents[3] / "data" / "phi" / "database_secrets.txt"

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "data" / "phi" / "real_time" / "simulated_oa_run.parquet"
)

DEFAULT_EMBEDDINGS = (
    Path(__file__).resolve().parents[3]
    / "data" / "phi" / "real_time" / "trial_space_embeddings.parquet"
)

DEFAULT_SUMMARIZE_SCRIPT = Path(__file__).resolve().parents[2] / "6_summarize_patients.py"

DEFAULT_SUMMARIZATION_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "data" / "phi" / "real_time" / "simulated_oa_run_summary_refresh"
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
    parser.add_argument("model_dir", help="Folder containing the saved embedding model")
    parser.add_argument(
        "--embeddings", type=str, default=str(DEFAULT_EMBEDDINGS),
        help="Path to pre-embedded trial spaces parquet file "
             "(produced by embed_trial_spaces.py)",
    )
    parser.add_argument(
        "--trial-checker", type=str, default=None,
        help="Path to a trained trial checker model (ModernBERT). "
             "If provided, re-scores top-20 trial spaces per patient.",
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
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output parquet file path")
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
        "--execution-timestamp", type=str, default=None,
        help="If provided, only consider outbound email drafts from "
             "activate_emails with this exact execution_timestamp value.",
    )
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


def run_checker_batched(texts, tokenizer, model, device, batch_size, max_length,
                        output_mode="sigmoid"):
    """Run a ModernBERT classifier on a list of texts in batches.

    output_mode:
        "sigmoid"  → torch.sigmoid(logits).squeeze(-1)   (trial checker)
        "softmax"  → torch.softmax(logits, dim=1)[:, 1]  (boilerplate checker)
    """
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


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------

def chunk_ranges(n_items: int, n_chunks: int):
    """Return [(start, end), ...] covering range(n_items) in ~equal chunks."""
    base = n_items // n_chunks
    rem = n_items % n_chunks
    ranges = []
    start = 0
    for i in range(n_chunks):
        add = base + (1 if i < rem else 0)
        end = start + add
        ranges.append((start, end))
        start = end
    return ranges


def _encode_worker(texts, device_str, model_path, encode_batch_size,
                   max_seq_length, query_prompt):
    """Runs in a subprocess on one GPU; returns float32 numpy (N, D)."""
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
    """Encode texts across GPUs; returns (N, D) float32 numpy."""
    if len(all_texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    n = len(all_texts)
    if len(gpu_ids) == 1:
        return _encode_worker(all_texts, f"cuda:{gpu_ids[0]}", model_path,
                              encode_batch_size, max_seq_length, query_prompt)
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
    """Load a checker model on one GPU and score texts."""
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
    """Score texts with a checker model across GPUs; returns 1-D numpy."""
    if len(all_texts) == 0:
        return np.array([], dtype=np.float32)
    n = len(all_texts)
    if len(gpu_ids) == 1:
        return _checker_worker(all_texts, f"cuda:{gpu_ids[0]}", model_path,
                               batch_size, max_length, output_mode)
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


def fetch_email_triggered_patients(conn, execution_timestamp=None):
    """Return latest activate_info IDs for MRNs that generated email drafts.

    If *execution_timestamp* is given, only emails with that exact
    execution_timestamp value are considered.
    """
    cur = conn.cursor()
    ts_clause = ""
    params = []
    if execution_timestamp is not None:
        ts_clause = "      AND ae.execution_timestamp = %s "
        params.append(execution_timestamp)
    cur.execute(
        "SELECT DISTINCT ON (ai.mrn) ai.id, ai.mrn "
        "FROM activate_info ai "
        "WHERE EXISTS ("
        "    SELECT 1 FROM activate_emails ae "
        "    WHERE ae.mrn = ai.mrn "
        "      AND ae.body IS NOT NULL "
        "      AND btrim(ae.body) <> '' "
        + ts_clause +
        ") "
        "ORDER BY ai.mrn, ai.id DESC",
        params or None,
    )
    rows = cur.fetchall()
    cur.close()
    return rows


def fetch_oncologist_names(conn, patient_mrns):
    """Return a dict mapping mrn -> oncologist full_name via email_assignments + recipients."""
    if not patient_mrns:
        return {}
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT ON (ea.mrn) ea.mrn, r.full_name "
        "FROM email_assignments ea "
        "JOIN recipients r ON r.npi = ea.recipient_npi "
        "WHERE ea.mrn = ANY(%s) "
        "ORDER BY ea.mrn, ea.enabled DESC, ea.date_assigned DESC NULLS LAST",
        (patient_mrns,),
    )
    result = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()
    return result


def fetch_note_level_input(conn, patient_mrns):
    """Load raw note text for the selected MRNs in summarizer-ready format."""
    if not patient_mrns:
        return pd.DataFrame(columns=["mrn", "date", "text"])

    cur = conn.cursor()
    cur.execute(
        "SELECT mrn, event_date AS date, "
        "       COALESCE(NULLIF(btrim(rpt_text), ''), NULLIF(btrim(narrative_text), '')) AS text "
        "FROM patient_notes "
        "WHERE mrn = ANY(%s) "
        "  AND COALESCE(NULLIF(btrim(rpt_text), ''), NULLIF(btrim(narrative_text), '')) IS NOT NULL "
        "ORDER BY mrn, event_date, id",
        (patient_mrns,),
    )
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=["mrn", "date", "text"])


def summarization_artifact_paths(run_dir: Path):
    """Return standard artifact paths for a summarization run directory."""
    return {
        "run_dir": run_dir,
        "shard_dir": run_dir / "summary_shards",
        "input_parquet": run_dir / "input_notes.parquet",
        "full_output": run_dir / "patient_summaries_full.parquet",
        "patient_output": run_dir / "patient_summaries.parquet",
    }


def load_summarization_output(patient_output: Path):
    """Load and validate patient_summaries.parquet."""
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
    """Stage, reuse, or resume ../../6_summarize_patients.py artifacts."""
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


def main():
    args = parse_args()
    gpu_ids = [int(x.strip()) for x in args.gpu.split(",") if x.strip()]

    # --- Database: identify patients that need fresh summaries ------------
    print("Connecting to database ...")
    conn = get_db_connection(args.secrets)
    patient_rows = fetch_email_triggered_patients(conn, args.execution_timestamp)
    if not patient_rows:
        conn.close()
        print("No patients generated outbound email drafts. Nothing to do.")
        return

    patient_ids = [r[0] for r in patient_rows]
    patient_mrns = [r[1] for r in patient_rows]
    print(f"Identified {len(patient_rows)} patients with outbound email drafts.")

    oncologist_by_mrn = fetch_oncologist_names(conn, patient_mrns)

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
        notes_df = fetch_note_level_input(conn, patient_mrns)
    conn.close()
    if need_notes_df and len(notes_df) == 0:
        print("No patient_notes rows found for the selected MRNs. Nothing to do.")
        return

    if need_notes_df:
        mrns_with_notes = set(pd.to_numeric(notes_df["mrn"], errors="raise").astype("int64"))
        if len(mrns_with_notes) < len(patient_mrns):
            missing_mrns = [mrn for mrn in patient_mrns if mrn not in mrns_with_notes]
            print(f"Warning: {len(missing_mrns)} selected MRNs had no note text and will be skipped.")
            patient_rows = [row for row in patient_rows if row[1] in mrns_with_notes]
            patient_ids = [r[0] for r in patient_rows]
            patient_mrns = [r[1] for r in patient_rows]

    if not patient_rows:
        print("No selected patients had note text available for summarization.")
        return

    if notes_df is not None:
        notes_df = notes_df[notes_df["mrn"].isin(patient_mrns)].copy()
    refreshed_summaries, summarization_run_dir = run_or_resume_summarization(notes_df, args)
    refreshed_by_mrn = (
        refreshed_summaries.set_index("mrn")[
            ["patient_summary", "patient_boilerplate_text"]
        ].to_dict("index")
    )

    missing_summary_mrns = [mrn for mrn in patient_mrns if mrn not in refreshed_by_mrn]
    if missing_summary_mrns:
        print(f"Warning: {len(missing_summary_mrns)} MRNs were not summarized successfully and will be skipped.")
        patient_rows = [row for row in patient_rows if row[1] in refreshed_by_mrn]

    if not patient_rows:
        print("Fresh summarization produced no usable patient summaries. Nothing to do.")
        return

    patient_ids = [r[0] for r in patient_rows]
    patient_mrns = [r[1] for r in patient_rows]
    patient_summaries = [refreshed_by_mrn[mrn]["patient_summary"] for mrn in patient_mrns]
    patient_boilerplates = [
        refreshed_by_mrn[mrn]["patient_boilerplate_text"] for mrn in patient_mrns
    ]
    print(f"Loaded {len(patient_rows)} freshly summarized patients from {summarization_run_dir}")

    # --- Load pre-embedded trial spaces -----------------------------------
    print(f"Loading pre-embedded trial spaces from {args.embeddings} ...")
    df_trials = pd.read_parquet(args.embeddings)
    if len(df_trials) == 0:
        print("Pre-embedded trial spaces file is empty.")
        return

    space_ids = df_trials["id"].tolist()
    nct_ids = df_trials["nct_id"].tolist()
    space_texts = df_trials["this_cohort"].tolist()
    trial_boilerplates = df_trials["boilerplate_text"].tolist()
    space_embs_np = np.array(df_trials["embedding"].tolist(), dtype=np.float32)
    print(f"Loaded {len(df_trials)} trial spaces (dim={space_embs_np.shape[1]}).")

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

    keep_mask = ~df_trials["nct_id"].isin(excluded_ncts)
    n_before = len(df_trials)
    df_trials = df_trials[keep_mask].reset_index(drop=True)
    space_ids = df_trials["id"].tolist()
    nct_ids = df_trials["nct_id"].tolist()
    space_texts = df_trials["this_cohort"].tolist()
    trial_boilerplates = df_trials["boilerplate_text"].tolist()
    space_embs_np = np.array(df_trials["embedding"].tolist(), dtype=np.float32)
    print(f"Kept {len(df_trials)} of {n_before} trial spaces after exclusions.")

    # --- Batch encode all patient summaries -------------------------------
    print(f"Encoding {len(patient_summaries)} patient summaries "
          f"across {len(gpu_ids)} GPU(s) ...")
    patient_embs = parallel_encode(
        patient_summaries, gpu_ids, args.model_dir,
        args.batch_size, args.max_seq_length, QUERY_PROMPT,
    )
    print(f"Patient embeddings shape: {patient_embs.shape}")

    # --- Cosine similarity matrix -----------------------------------------
    print("Computing cosine similarity matrix ...")
    sim_matrix = patient_embs @ space_embs_np.T  # (N_patients, N_trials)

    # Top 20 per patient
    top_k_retrieve = 20
    top_k_output = 10
    top20_indices = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :top_k_retrieve]

    # --- Trial checker scoring (top 20 per patient) -----------------------
    # Flatten all (patient, trial) pairs for batched inference
    n_patients = len(patient_ids)
    tc_scores_flat = np.full(n_patients * top_k_retrieve, np.nan)

    if args.trial_checker:
        print(f"Running trial checker on {n_patients} patients x {top_k_retrieve} "
              f"trials across {len(gpu_ids)} GPU(s) ...")
        tc_texts = []
        for i in range(n_patients):
            for j in range(top_k_retrieve):
                idx = top20_indices[i, j]
                tc_texts.append(
                    space_texts[idx]
                    + "\nNow here is the patient summary:"
                    + patient_summaries[i]
                )
        tc_scores_flat = parallel_checker(
            tc_texts, gpu_ids, args.trial_checker,
            args.checker_batch_size, 4096, "sigmoid",
        )
        print("Trial checker scoring complete.")

    tc_scores_matrix = tc_scores_flat.reshape(n_patients, top_k_retrieve)

    # --- Re-rank per patient ----------------------------------------------
    if args.trial_checker:
        # Re-rank top 20 by trial checker score (descending)
        rerank_orders = np.argsort(tc_scores_matrix, axis=1)[:, ::-1]
        top20_indices_reranked = np.take_along_axis(top20_indices, rerank_orders, axis=1)
        tc_scores_matrix = np.take_along_axis(tc_scores_matrix, rerank_orders, axis=1)
    else:
        top20_indices_reranked = top20_indices

    # Trim to top 10 unique trials per patient (deduplicate by nct_id).
    # A trial may have multiple spaces; keep only the highest-ranked space
    # for each trial, then continue down the list to fill up to top_k_output.
    top_selected_indices = []   # per-patient list of trial-space indices
    top_selected_tc_scores = []
    top_selected_cos_sims = []
    for i in range(n_patients):
        seen_ncts = set()
        sel_indices = []
        sel_tc = []
        sel_cos = []
        for j in range(top_k_retrieve):
            idx = top20_indices_reranked[i, j]
            nct = nct_ids[idx]
            if nct in seen_ncts:
                continue
            seen_ncts.add(nct)
            sel_indices.append(idx)
            sel_tc.append(float(tc_scores_matrix[i, j]))
            sel_cos.append(float(sim_matrix[i, idx]))
            if len(sel_indices) >= top_k_output:
                break
        top_selected_indices.append(sel_indices)
        top_selected_tc_scores.append(sel_tc)
        top_selected_cos_sims.append(sel_cos)

    # --- Boilerplate checker scoring (deduplicated top trials per patient) -
    bp_scores_per_patient = [
        [np.nan] * len(top_selected_indices[i]) for i in range(n_patients)
    ]

    if args.boilerplate_checker:
        total_pairs = sum(len(s) for s in top_selected_indices)
        print(f"Running boilerplate checker on {total_pairs} patient-trial pairs "
              f"across {len(gpu_ids)} GPU(s) ...")
        bp_texts = []
        bp_mapping = []  # (patient_idx, position_in_selection)
        for i in range(n_patients):
            for j, idx in enumerate(top_selected_indices[i]):
                bp_texts.append(
                    f"Patient history: {patient_boilerplates[i] or ''}"
                    f"\nTrial exclusions:{trial_boilerplates[idx] or ''}"
                )
                bp_mapping.append((i, j))
        bp_scores_flat = parallel_checker(
            bp_texts, gpu_ids, args.boilerplate_checker,
            args.checker_batch_size, 3192, "softmax",
        )
        for flat_idx, (i, j) in enumerate(bp_mapping):
            bp_scores_per_patient[i][j] = float(bp_scores_flat[flat_idx])
        print("Boilerplate checker scoring complete.")

    # --- Build output dataframe -------------------------------------------
    print("Building output dataframe ...")
    rows = []
    for i in range(n_patients):
        for j, idx in enumerate(top_selected_indices[i]):
            rows.append({
                "patient_id": patient_ids[i],
                "mrn": patient_mrns[i],
                "oncologist_name": oncologist_by_mrn.get(patient_mrns[i]),
                "patient_summary": patient_summaries[i],
                "space_id": space_ids[idx],
                "nct_id": nct_ids[idx],
                "trial_space_text": space_texts[idx],
                "cosine_similarity": top_selected_cos_sims[i][j],
                "trialchecker_score": top_selected_tc_scores[i][j],
                "boilerplate_score": bp_scores_per_patient[i][j],
                "rank": j + 1,
            })

    df_out = pd.DataFrame(rows)

    # --- Save -------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)

    csv_path = output_path.with_suffix(".csv")
    df_out.to_csv(csv_path, index=False)

    excel_path = output_path.with_suffix(".xlsx")
    df_out.to_excel(excel_path, index=False)

    trials_per_patient = df_out.groupby("patient_id").size()
    print(f"\nSaved {len(df_out)} rows to {output_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  Excel: {excel_path}")
    print(f"  Patients: {n_patients}")
    print(f"  Trials per patient: median={trials_per_patient.median():.0f}, "
          f"min={trials_per_patient.min()}, max={trials_per_patient.max()}")
    print(f"  Columns: {list(df_out.columns)}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # --- Summary statistics PDF (requires trial checker) ------------------
    if args.trial_checker:
        pdf_path = output_path.with_suffix(".pdf")
        print(f"\nGenerating summary statistics PDF at {pdf_path} ...")

        tc_vals = df_out["trialchecker_score"].dropna().values
        thresholds = np.arange(0.0, 1.0, 0.1)

        with PdfPages(pdf_path) as pdf:
            # Page 1: histogram of trialchecker scores
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(tc_vals, bins=50, edgecolor="black", linewidth=0.5)
            ax.set_xlabel("TrialChecker Score (sigmoid)")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of TrialChecker Scores (all patient–trial pairs)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: proportion of patients with >= 1 trial above threshold
            proportions = []
            for thresh in thresholds:
                patients_with_any = (
                    df_out[df_out["trialchecker_score"] >= thresh]
                    ["patient_id"].nunique()
                )
                proportions.append(patients_with_any / n_patients)

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(
                [f"{t:.1f}" for t in thresholds], proportions,
                edgecolor="black", linewidth=0.5,
            )
            for bar, prop in zip(bars, proportions):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{prop:.0%}", ha="center", va="bottom", fontsize=9,
                )
            ax.set_xlabel("TrialChecker Score Threshold")
            ax.set_ylabel("Proportion of Patients with >= 1 Trial Above Threshold")
            ax.set_title("Patient Coverage by TrialChecker Threshold")
            ax.set_ylim(0, min(1.15, max(proportions) + 0.15))
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"  Saved summary PDF to {pdf_path}")


if __name__ == "__main__":
    # Use 'spawn' to avoid CUDA re-init issues in subprocesses
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
