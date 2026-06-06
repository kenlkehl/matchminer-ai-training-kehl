#!/usr/bin/env python3
"""One-off: re-rank trials for existing patient summaries by TrialSpace only.

Takes the patient summaries already present in an existing simulated_oa_run
output parquet and, for each patient, fetches the top-K unique trials
(deduplicated by nct_id) purely by TrialSpace cosine similarity -- no
TrialChecker / BoilerplateChecker / LLM filtering.

The output reproduces the exact column schema of simulated_oa_run.parquet; the
checker/LLM columns are left as NaN (numeric) or "" (text).

Usage:
    python oneoff_trialspace_top20.py \
        --input  /ksg/.../data/phi/real_time/simulated_oa_run.parquet \
        --embeddings /ksg/.../data/phi/real_time/trial_space_embeddings.parquet \
        --model /ksg/kehl_mm_data/mmai/v22/models/trialspace \
        --gpu 1 --top-k 20
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulated_oa_run import QUERY_PROMPT, parallel_encode

REAL_TIME_DIR = Path("/ksg/kehl_mm_data/mmai/v22/data/phi/real_time")

DEFAULT_INPUT = REAL_TIME_DIR / "simulated_oa_run.parquet"
DEFAULT_EMBEDDINGS = REAL_TIME_DIR / "trial_space_embeddings.parquet"
DEFAULT_MODEL = Path("/ksg/kehl_mm_data/mmai/v22/models/trialspace")
DEFAULT_OUTPUT = REAL_TIME_DIR / "simulated_oa_run_trialspace_top20.parquet"
DEFAULT_ONCORE_CSV = REAL_TIME_DIR / "oncore_data.csv"

HARDCODED_EXCLUDED_NCTS = {"NCT04301765", "NCT04049331"}

# Per-patient identity columns carried over verbatim from the input file.
PATIENT_COLS = [
    "patient_id", "mrn", "oncologist_name",
    "patient_summary", "patient_boilerplate_text",
]

OUTPUT_COLS = [
    "patient_id", "mrn", "oncologist_name", "patient_summary",
    "patient_boilerplate_text", "space_id", "nct_id", "trial_space_text",
    "trial_boilerplate_text", "cosine_similarity", "trialchecker_score",
    "boilerplate_score", "llm_trialcheck_score", "llm_trialcheck_reasoning",
    "llm_boilerplate_excluded", "llm_boilerplate_reasoning", "rank",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help="Existing simulated_oa_run parquet to take summaries from.")
    p.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS),
                   help="Pre-embedded trial-space parquet.")
    p.add_argument("--model", default=str(DEFAULT_MODEL),
                   help="Folder with the saved TrialSpace SentenceTransformer.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Output parquet (also writes .csv and .xlsx).")
    p.add_argument("--oncore-csv", default=str(DEFAULT_ONCORE_CSV),
                   help="oncore_data.csv for Supportive/Radiation exclusion "
                        "(skipped if missing).")
    p.add_argument("--gpu", default="1",
                   help="Comma-separated GPU ids for embedding (default: 1).")
    p.add_argument("--top-k", type=int, default=20,
                   help="Unique trials kept per patient (default: 20).")
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--max-seq-length", type=int, default=100000)
    return p.parse_args()


def load_trial_embeddings(args):
    df = pd.read_parquet(args.embeddings)
    if len(df) == 0:
        raise ValueError(f"Trial embeddings file is empty: {args.embeddings}")

    excluded = set(HARDCODED_EXCLUDED_NCTS)
    oncore_path = Path(args.oncore_csv)
    if oncore_path.exists():
        df_oncore = pd.read_csv(oncore_path)
        extra = df_oncore.loc[
            df_oncore["groupName"].str.contains(
                "Supportive Oncology|Radiation Oncology", na=False
            ),
            "nctId",
        ].unique()
        excluded.update(extra)
        print(f"Excluding {len(excluded)} NCT IDs "
              f"(Supportive/Radiation Oncology groups + hardcoded).")
    else:
        print(f"Warning: {oncore_path} not found; only hardcoded "
              f"exclusions applied.")

    n_before = len(df)
    df = df[~df["nct_id"].isin(excluded)].reset_index(drop=True)
    print(f"Kept {len(df)} of {n_before} trial spaces after exclusions.")
    return df


def main():
    args = parse_args()
    gpu_ids = [int(x) for x in args.gpu.split(",") if x.strip()]

    # --- Patient summaries (one row per patient) from the existing output ---
    print(f"Loading patient summaries from {args.input} ...")
    df_in = pd.read_parquet(args.input)
    df_pat = (
        df_in[PATIENT_COLS]
        .drop_duplicates(subset=["mrn"], keep="first")
        .reset_index(drop=True)
    )
    n_patients = len(df_pat)
    patient_summaries = df_pat["patient_summary"].fillna("").astype(str).tolist()
    print(f"Loaded {n_patients} unique patients.")

    # --- Trial spaces ------------------------------------------------------
    print(f"Loading trial-space embeddings from {args.embeddings} ...")
    df_trials = load_trial_embeddings(args)
    space_ids = df_trials["id"].tolist()
    nct_ids = df_trials["nct_id"].tolist()
    space_texts = df_trials["this_cohort"].tolist()
    trial_boilerplates = df_trials["boilerplate_text"].tolist()
    space_embs_np = np.array(df_trials["embedding"].tolist(), dtype=np.float32)
    print(f"Loaded {len(df_trials)} trial spaces "
          f"({df_trials['nct_id'].nunique()} unique NCTs, "
          f"dim={space_embs_np.shape[1]}).")

    top_k = args.top_k
    n_unique = df_trials["nct_id"].nunique()
    if top_k > n_unique:
        print(f"Warning: --top-k {top_k} exceeds {n_unique} unique trials; "
              f"clamping to {n_unique}.")
        top_k = n_unique

    # --- Encode + cosine similarity ---------------------------------------
    print(f"Encoding {n_patients} patient summaries on GPU(s) {gpu_ids} ...")
    patient_embs = parallel_encode(
        patient_summaries, gpu_ids, args.model,
        args.batch_size, args.max_seq_length, QUERY_PROMPT,
    )
    print(f"Patient embeddings shape: {patient_embs.shape}")

    print("Computing cosine similarity matrix ...")
    sim_matrix = patient_embs @ space_embs_np.T  # (N_patients, N_spaces)
    # Rank every space by cosine (descending), then dedup by nct_id below.
    ranked_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]

    # --- Build output: top-K unique trials per patient by cosine ----------
    print(f"Selecting top {top_k} unique trials per patient ...")
    rows = []
    for i in range(n_patients):
        meta = df_pat.iloc[i]
        seen_ncts = set()
        rank = 0
        for idx in ranked_indices[i]:
            nct = nct_ids[idx]
            if nct in seen_ncts:
                continue
            seen_ncts.add(nct)
            rank += 1
            rows.append({
                "patient_id": meta["patient_id"],
                "mrn": meta["mrn"],
                "oncologist_name": meta["oncologist_name"],
                "patient_summary": meta["patient_summary"],
                "patient_boilerplate_text": meta["patient_boilerplate_text"],
                "space_id": space_ids[idx],
                "nct_id": nct,
                "trial_space_text": space_texts[idx],
                "trial_boilerplate_text": trial_boilerplates[idx] or "",
                "cosine_similarity": float(sim_matrix[i, idx]),
                "trialchecker_score": np.nan,
                "boilerplate_score": np.nan,
                "llm_trialcheck_score": np.nan,
                "llm_trialcheck_reasoning": "",
                "llm_boilerplate_excluded": np.nan,
                "llm_boilerplate_reasoning": "",
                "rank": rank,
            })
            if rank >= top_k:
                break

    df_out = pd.DataFrame(rows)[OUTPUT_COLS]

    # --- Save --------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    df_out.to_csv(out_path.with_suffix(".csv"), index=False)
    df_out.to_excel(out_path.with_suffix(".xlsx"), index=False)

    per_pat = df_out.groupby("mrn").size()
    print(f"\nSaved {len(df_out)} rows to {out_path}")
    print(f"  CSV:   {out_path.with_suffix('.csv')}")
    print(f"  Excel: {out_path.with_suffix('.xlsx')}")
    print(f"  Patients: {n_patients}  Trials/patient: "
          f"min={per_pat.min()}, median={per_pat.median():.0f}, max={per_pat.max()}")
    print(f"  Columns: {list(df_out.columns)}")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
