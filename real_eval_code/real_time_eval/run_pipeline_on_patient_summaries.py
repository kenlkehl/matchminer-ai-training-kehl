#!/usr/bin/env python3
"""Score an existing patient-summary parquet against the trial bank.

Unlike simulated_oa_run.py, this script does NOT touch the database and does
NOT re-summarize patients. It starts from a pre-existing
patient_summaries.parquet (the schema produced by summarize_patients.py)
and runs: patient embedding -> top-K trial-space retrieval -> TrialChecker ->
BoilerplateChecker. (The optional LLM boilerplate check, which ran the removed
14_check_boilerplate.py, was removed on 2026-11-05, user instruction.)

Each row of the output parquet is one (patient, retrieved trial space) pair.
No NCT-level deduplication is applied. Trial spaces are retrieved by TrialSpace
cosine similarity into a pool of size --top-k-retrieve and the top
--top-k-output per patient are kept, so the output contains exactly
n_patients * --top-k-output rows. (The legacy --top-k flag sets both.)
Ranking is always by TrialSpace cosine similarity; the checker stages only
populate score columns and never re-order or filter the output.

Default behavior runs both scoring stages. Use --stages to subset.

Examples:
    # Both stages
    python run_pipeline_on_patient_summaries.py \
        --input-summaries patient_summaries.parquet \
        --embedding-model ../../../models/trialspace \
        --embeddings trial_space_embeddings.parquet \
        --trial-checker ../../../models/trialchecker \
        --boilerplate-checker ../../../models/boilerplatechecker \
        --gpu 0,1

    # Only the two ModernBERT checkers
    python run_pipeline_on_patient_summaries.py \
        --input-summaries patient_summaries.parquet \
        --embedding-model ../../../models/trialspace \
        --embeddings trial_space_embeddings.parquet \
        --trial-checker ../../../models/trialchecker \
        --boilerplate-checker ../../../models/boilerplatechecker \
        --stages trial,boilerplate --gpu 0
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from simulated_oa_run import (
    QUERY_PROMPT,
    parallel_encode,
    parallel_checker,
)


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "data" / "phi" / "real_time" / "run_pipeline_on_patient_summaries.parquet"
)

DEFAULT_EMBEDDINGS = (
    Path(__file__).resolve().parents[3]
    / "data" / "phi" / "real_time" / "trial_space_embeddings.parquet"
)

DEFAULT_ONCORE_CSV = (
    Path(__file__).resolve().parents[3]
    / "data" / "phi" / "real_time" / "oncore_data.csv"
)

HARDCODED_EXCLUDED_NCTS = {"NCT04301765", "NCT04049331"}

# The legacy single-score LLM trial check (llm_trial) was removed on 2026-11-04 (user instruction).
ALL_STAGES = ("trial", "boilerplate")


def parse_stages(raw: str):
    tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("--stages cannot be empty")
    if tokens == ["all"]:
        return set(ALL_STAGES)
    expanded = set()
    for tok in tokens:
        if tok == "all":
            expanded.update(ALL_STAGES)
        elif tok in ALL_STAGES:
            expanded.add(tok)
        else:
            raise argparse.ArgumentTypeError(
                f"Unknown stage '{tok}'. Valid: {','.join(ALL_STAGES)},all"
            )
    return expanded


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input-summaries", required=True,
                        help="Path to a patient-summary parquet (same schema "
                             "as summarize_patients.py output).")
    parser.add_argument("--embedding-model", required=True,
                        help="Folder containing the saved SentenceTransformer "
                             "embedding model (the 'TrialSpace' model).")
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS),
                        help="Pre-embedded trial-space parquet "
                             "(produced by embed_trial_spaces.py).")

    parser.add_argument("--patient-id-col", default="mrn",
                        help="Column in --input-summaries holding the patient id.")
    parser.add_argument("--summary-col", default="patient_summary",
                        help="Column in --input-summaries holding the summary text.")
    parser.add_argument("--boilerplate-col", default="patient_boilerplate_text",
                        help="Column in --input-summaries holding the boilerplate text.")

    parser.add_argument("--stages", type=parse_stages,
                        default=set(ALL_STAGES),
                        help="Comma-separated subset of "
                             f"{{{','.join(ALL_STAGES)}}}; or 'all'. "
                             "Default: all four stages.")

    parser.add_argument("--trial-checker", default=None,
                        help="Path to TrialChecker ModernBERT folder. "
                             "Required if 'trial' is in --stages.")
    parser.add_argument("--boilerplate-checker", default=None,
                        help="Path to BoilerplateChecker ModernBERT folder. "
                             "Required if 'boilerplate' is in --stages.")

    parser.add_argument("--gpu", default="0",
                        help="Comma-separated GPU ids for embedding + ModernBERT "
                             "checkers (default: '0').")
    parser.add_argument("--max-seq-length", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Embedding batch size (default: 12).")
    parser.add_argument("--checker-batch-size", type=int, default=32,
                        help="Batch size for TrialChecker / BoilerplateChecker "
                             "(default: 32).")

    parser.add_argument("--top-k-retrieve", type=int, default=None,
                        help="Trial spaces retrieved per patient by TrialSpace "
                             "cosine similarity before trimming (default: 20).")
    parser.add_argument("--top-k-output", type=int, default=None,
                        help="Trial spaces kept per patient in the output "
                             "(default: 20). The output has exactly "
                             "n_patients * top_k_output rows.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="DEPRECATED alias: sets both --top-k-retrieve and "
                             "--top-k-output to this value (back-compat).")
    parser.add_argument("--trialspace-only-ranking", action="store_true",
                        help="Accepted for parity with simulated_oa_run.py. "
                             "This script always ranks output by TrialSpace "
                             "cosine similarity, so the flag has no extra effect.")

    parser.add_argument("--no-oncore-exclusion", action="store_true",
                        help="Skip the Supportive/Radiation Oncology NCT filter.")
    parser.add_argument("--oncore-csv", default=str(DEFAULT_ONCORE_CSV),
                        help="Path to oncore_data.csv used for NCT exclusion.")

    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output parquet path (also writes .csv and .xlsx).")

    return parser.parse_args()


def load_patient_summaries(args):
    path = Path(args.input_summaries)
    df = pd.read_parquet(path)
    for col in (args.patient_id_col, args.summary_col, args.boilerplate_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in {path}. "
                f"Available columns: {df.columns.tolist()}"
            )
    df = df[[args.patient_id_col, args.summary_col, args.boilerplate_col]].copy()
    df.columns = ["patient_id", "patient_summary", "patient_boilerplate_text"]
    df["patient_summary"] = df["patient_summary"].fillna("").astype(str)
    df["patient_boilerplate_text"] = (
        df["patient_boilerplate_text"].fillna("").astype(str)
    )
    df = df[df["patient_summary"].str.strip() != ""]
    df = df.drop_duplicates(subset=["patient_id"], keep="last").reset_index(drop=True)
    return df


def load_trial_embeddings(args):
    df = pd.read_parquet(args.embeddings)
    if len(df) == 0:
        raise ValueError(f"Trial embeddings file is empty: {args.embeddings}")

    excluded_ncts = set(HARDCODED_EXCLUDED_NCTS)
    if not args.no_oncore_exclusion:
        oncore_path = Path(args.oncore_csv)
        if oncore_path.exists():
            df_oncore = pd.read_csv(oncore_path)
            extra = df_oncore.loc[
                df_oncore["groupName"].str.contains(
                    "Supportive Oncology|Radiation Oncology", na=False
                ),
                "nctId",
            ].unique()
            excluded_ncts.update(extra)
            print(f"Excluding {len(excluded_ncts)} NCT IDs "
                  f"(Supportive/Radiation Oncology groups + hardcoded).")
        else:
            print(f"Warning: {oncore_path} not found; only hardcoded "
                  f"exclusions applied.")
    else:
        print("--no-oncore-exclusion set; only hardcoded exclusions applied.")

    keep_mask = ~df["nct_id"].isin(excluded_ncts)
    n_before = len(df)
    df = df[keep_mask].reset_index(drop=True)
    print(f"Kept {len(df)} of {n_before} trial spaces after exclusions.")
    return df


def main():
    args = parse_args()
    stages = args.stages
    gpu_ids = [int(x.strip()) for x in args.gpu.split(",") if x.strip()]

    # Validate stage prerequisites up front.
    if "trial" in stages and not args.trial_checker:
        sys.exit("--trial-checker is required when 'trial' is in --stages "
                 "(or drop 'trial' from --stages).")
    if "boilerplate" in stages and not args.boilerplate_checker:
        sys.exit("--boilerplate-checker is required when 'boilerplate' is in "
                 "--stages (or drop 'boilerplate' from --stages).")

    print(f"Stages enabled: {sorted(stages)}")

    # --- Load patient summaries ------------------------------------------
    print(f"Loading patient summaries from {args.input_summaries} ...")
    df_pat = load_patient_summaries(args)
    if len(df_pat) == 0:
        sys.exit("No usable patient summaries in input. Nothing to do.")
    patient_ids = df_pat["patient_id"].tolist()
    patient_summaries = df_pat["patient_summary"].tolist()
    patient_boilerplates = df_pat["patient_boilerplate_text"].tolist()
    n_patients = len(df_pat)
    print(f"Loaded {n_patients} patient summaries.")

    # --- Load pre-embedded trial spaces ----------------------------------
    print(f"Loading pre-embedded trial spaces from {args.embeddings} ...")
    df_trials = load_trial_embeddings(args)
    space_ids = df_trials["id"].tolist()
    nct_ids = df_trials["nct_id"].tolist()
    space_texts = df_trials["this_cohort"].tolist()
    trial_boilerplates = df_trials["boilerplate_text"].tolist()
    space_embs_np = np.array(df_trials["embedding"].tolist(), dtype=np.float32)
    print(f"Loaded {len(df_trials)} trial spaces (dim={space_embs_np.shape[1]}).")

    # --- Resolve retrieve / output counts --------------------------------
    # Legacy --top-k sets both retrieve and output for any value not set
    # explicitly; otherwise each defaults to 20.
    DEFAULT_TOPK = 20
    top_k_retrieve = args.top_k_retrieve
    top_k_output = args.top_k_output
    if args.top_k is not None:
        if top_k_retrieve is None:
            top_k_retrieve = args.top_k
        if top_k_output is None:
            top_k_output = args.top_k
    if top_k_retrieve is None:
        top_k_retrieve = DEFAULT_TOPK
    if top_k_output is None:
        top_k_output = DEFAULT_TOPK

    if args.trialspace_only_ranking:
        print("--trialspace-only-ranking: run_pipeline always ranks output by "
              "TrialSpace cosine similarity; flag has no additional effect.")

    n_trials_total = len(df_trials)
    if top_k_retrieve > n_trials_total:
        print(f"Warning: --top-k-retrieve {top_k_retrieve} exceeds "
              f"{n_trials_total} available trial spaces; clamping to "
              f"{n_trials_total}.")
        top_k_retrieve = n_trials_total
    if top_k_output > top_k_retrieve:
        print(f"Warning: --top-k-output {top_k_output} exceeds the retrieve "
              f"pool {top_k_retrieve}; clamping to {top_k_retrieve}.")
        top_k_output = top_k_retrieve

    # Downstream loops/reshapes operate on the final output count.
    top_k = top_k_output

    # --- Encode patient summaries ----------------------------------------
    print(f"Encoding {n_patients} patient summaries across "
          f"{len(gpu_ids)} GPU(s) ...")
    patient_embs = parallel_encode(
        patient_summaries, gpu_ids, args.embedding_model,
        args.batch_size, args.max_seq_length, QUERY_PROMPT,
    )
    print(f"Patient embeddings shape: {patient_embs.shape}")

    # --- Cosine similarity + top-K ---------------------------------------
    # Retrieve the top_k_retrieve pool by cosine, then (since ranking is purely
    # by cosine) keep the first top_k_output columns for scoring + output.
    print("Computing cosine similarity matrix ...")
    sim_matrix = patient_embs @ space_embs_np.T  # (N_patients, N_trials)
    pool_indices = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :top_k_retrieve]
    top_indices = pool_indices[:, :top_k_output]

    # --- TrialChecker ----------------------------------------------------
    tc_scores_matrix = np.full((n_patients, top_k), np.nan, dtype=np.float64)
    if "trial" in stages:
        print(f"Running TrialChecker on {n_patients} x {top_k} pairs "
              f"across {len(gpu_ids)} GPU(s) ...")
        tc_texts = []
        for i in range(n_patients):
            for j in range(top_k):
                idx = top_indices[i, j]
                tc_texts.append(
                    space_texts[idx]
                    + "\nNow here is the patient summary:"
                    + patient_summaries[i]
                )
        tc_flat = parallel_checker(
            tc_texts, gpu_ids, args.trial_checker,
            args.checker_batch_size, 4096, "sigmoid",
        )
        tc_scores_matrix = tc_flat.reshape(n_patients, top_k).astype(np.float64)
        print("TrialChecker scoring complete.")

    # --- BoilerplateChecker ----------------------------------------------
    bp_scores_matrix = np.full((n_patients, top_k), np.nan, dtype=np.float64)
    if "boilerplate" in stages:
        print(f"Running BoilerplateChecker on {n_patients} x {top_k} pairs "
              f"across {len(gpu_ids)} GPU(s) ...")
        bp_texts = []
        for i in range(n_patients):
            for j in range(top_k):
                idx = top_indices[i, j]
                bp_texts.append(
                    f"Patient history: {patient_boilerplates[i] or ''}"
                    f"\nTrial exclusions:{trial_boilerplates[idx] or ''}"
                )
        bp_flat = parallel_checker(
            bp_texts, gpu_ids, args.boilerplate_checker,
            args.checker_batch_size, 3192, "softmax",
        )
        bp_scores_matrix = bp_flat.reshape(n_patients, top_k).astype(np.float64)
        print("BoilerplateChecker scoring complete.")

    # --- Build output ----------------------------------------------------
    print("Building output dataframe ...")
    rows = []
    for i in range(n_patients):
        for j in range(top_k):
            idx = top_indices[i, j]
            rows.append({
                args.patient_id_col: patient_ids[i],
                "patient_summary": patient_summaries[i],
                "patient_boilerplate_text": patient_boilerplates[i],
                "space_id": space_ids[idx],
                "nct_id": nct_ids[idx],
                "trial_space_text": space_texts[idx],
                "trial_boilerplate_text": trial_boilerplates[idx] or "",
                "cosine_similarity": float(sim_matrix[i, idx]),
                "trialchecker_score": float(tc_scores_matrix[i, j]),
                "boilerplate_score": float(bp_scores_matrix[i, j]),
                "rank": j + 1,
            })

    df_out = pd.DataFrame(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)

    csv_path = output_path.with_suffix(".csv")
    df_out.to_csv(csv_path, index=False)

    excel_path = output_path.with_suffix(".xlsx")
    df_out.to_excel(excel_path, index=False)

    print(f"\nSaved {len(df_out)} rows to {output_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  Excel: {excel_path}")
    print(f"  Patients: {n_patients}  Trials/patient: {top_k}")
    print(f"  Stages run: {sorted(stages)}")
    print(f"  Columns: {list(df_out.columns)}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
