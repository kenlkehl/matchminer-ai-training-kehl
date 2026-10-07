#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel retrieval (patients->trials and trials->patients) with SentenceTransformers
- Each GPU loads the model and processes a shard
- Samples trials at the trial level, ranks at the space level, outputs spaces
- For patients->trials: samples N trials per patient, ranks their spaces, returns top-K spaces
- For trials->patients: samples N patients per trial space, ranks them, returns top-K patients
- Rejoins shards to produce the same output files (now configurable via CLI):
    - default: top_cohorts_tocheck_round1.parquet
    - default: top_patients_tocheck_round1.parquet

Example:
python make_top_matches.py \
  --parquet trial_specific_eligibility_checks.parquet \
  --model pt_trial_summary_pertrial_finetuned.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 \
  --top_k_patients 40 \
  --encode_batch_size 128 \
  --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet top_cohorts_tocheck_round1.parquet \
  --out_patients_parquet top_patients_tocheck_round1.parquet
"""

import os
import argparse
import multiprocessing as mp
import hashlib
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor
from finetune_embedder import load_text_model

# -------------------------
# Helpers
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default="trial_specific_eligibility_checks.parquet",
        help="Input parquet containing columns: patient_summary, patient_boilerplate_text, this_space, nct_id, eligibility_result",
    )
    ap.add_argument("--patients-parquet", help="Independent synthetic patient summary table, including IDs and splits")
    ap.add_argument("--trials-file", help="Independent trial-space CSV or parquet")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    ap.add_argument("--model", default="google/embeddinggemma-2")
    ap.add_argument(
        "--gpus", default="0", help="Comma-separated CUDA device indices, e.g. '0,1,2,3'"
    )
    ap.add_argument(
        "--sample_trials_per_patient",
        type=int,
        required=True,
        help="Number of trials to randomly sample for consideration per patient",
    )
    ap.add_argument(
        "--sample_patients_per_trial",
        type=int,
        required=True,
        help="Number of patients to randomly sample for consideration per trial space",
    )
    ap.add_argument(
        "--top_k_spaces",
        type=int,
        default=10,
        help="Top spaces per patient for patients->trials ranking (from sampled trials)",
    )
    ap.add_argument(
        "--top_k_patients",
        type=int,
        default=20,
        help="Top patients per trial space for trials->patients ranking (from sampled patients)",
    )
    ap.add_argument(
        "--encode_batch_size", type=int, default=256, help="Batch size for encoding texts"
    )
    ap.add_argument(
        "--score_batch_size",
        type=int,
        default=2048,
        help="Batch size for similarity/topk scoring per worker",
    )
    ap.add_argument("--max_seq_length", type=int, default=1500)
    ap.add_argument(
        "--query_prompt",
        default=None,
        help="Optional literal prefix override; otherwise use the model's saved query prompt.",
    )
    ap.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility",
    )
    # Output filenames
    ap.add_argument(
        "--out_cohorts_parquet",
        default="top_cohorts_tocheck_round1.parquet",
        help="Output parquet for patients->trials results",
    )
    ap.add_argument(
        "--out_patients_parquet",
        default="top_patients_tocheck_round1.parquet",
        help="Output parquet for trials->patients results",
    )
    return ap.parse_args()


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


def _encode_worker(texts, device_str, model_path, encode_batch_size, max_seq_length, query_prompt):
    """Runs in a subprocess on one GPU; returns a float32 numpy (N, D)."""
    torch.cuda.set_device(int(device_str.split(":")[-1]))
    model = load_text_model(model_path, device_str)
    # Fine-tuning persists the same prefix for both matching directions.
    if query_prompt is not None:
        model.prompts["query"] = query_prompt
    model.max_seq_length = max_seq_length

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
            enabled=torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        embs = model.encode(
            texts,
            batch_size=encode_batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,  # makes dot product == cosine similarity
            show_progress_bar=False,
            prompt_name="query",
        ).cpu().to(dtype=torch.float32).numpy()
    return embs


def parallel_encode(all_texts, gpu_ids, model_path, encode_batch_size, max_seq_length, query_prompt):
    """Encode a list of texts across GPUs; returns (N,D) float32 numpy aligned to original order."""
    if len(all_texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    n = len(all_texts)
    shards = chunk_ranges(n, len(gpu_ids))
    futures = []
    outputs = [None] * len(shards)
    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=mp.get_context("spawn")) as ex:
        for wi, (s, e) in enumerate(shards):
            if s == e:
                outputs[wi] = np.zeros((0, 0), dtype=np.float32)
                continue
            texts_slice = all_texts[s:e]
            fut = ex.submit(
                _encode_worker,
                texts_slice,
                f"cuda:{gpu_ids[wi]}",
                model_path,
                encode_batch_size,
                max_seq_length,
                query_prompt,
            )
            futures.append((wi, s, e, fut))
        for wi, s, e, fut in futures:
            outputs[wi] = fut.result()

    # Concatenate in-order
    embs = []
    for (s, e), arr in zip(shards, outputs):
        if s == e:
            continue
        embs.append(arr)
    embs = np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), dtype=np.float32)
    assert embs.shape[0] == n, f"Encoded rows {embs.shape[0]} != expected {n}"
    return embs


def _topk_with_sampling_worker(query_emb_batch, all_ref_embs, sample_indices_batch, topk, device_str):
    """
    query_emb_batch: (B, D) float32 numpy
    all_ref_embs: (M, D) float32 numpy
    sample_indices_batch: list of B arrays, each containing indices to sample from all_ref_embs
    topk: int
    
    Returns: list of B arrays, each containing topk indices into the ORIGINAL all_ref_embs space
    """
    torch.cuda.set_device(int(device_str.split(":")[-1]))
    device = torch.device(device_str)
    
    results = []
    with torch.no_grad():
        q_tensor = torch.from_numpy(query_emb_batch).to(device, non_blocking=True)
        all_ref_tensor = torch.from_numpy(all_ref_embs).to(device, non_blocking=True)
        
        for i, sample_indices in enumerate(sample_indices_batch):
            # Get the sampled reference embeddings
            if len(sample_indices) == 0:
                results.append(np.array([], dtype=np.int64))
                continue
                
            sampled_refs = all_ref_tensor[sample_indices]  # (n_sampled, D)
            query_vec = q_tensor[i:i+1]  # (1, D)
            
            # Compute similarities
            sims = query_vec @ sampled_refs.t()  # (1, n_sampled)
            
            # Get topk from sampled set
            k_actual = min(topk, len(sample_indices))
            _, topk_in_sample = torch.topk(sims, k=k_actual, dim=1, largest=True, sorted=True)
            
            # Map back to original indices
            topk_in_sample_np = topk_in_sample.cpu().numpy()[0]
            original_indices = sample_indices[topk_in_sample_np]
            results.append(original_indices)
    
    return results


def parallel_topk_with_sampling(all_query_embs, all_ref_embs, sample_indices_per_query, 
                                 gpu_ids, topk, score_batch_size):
    """
    Splits queries across GPUs with per-query sampling.
    
    all_query_embs: (N_queries, D) float32 numpy
    all_ref_embs: (M, D) float32 numpy
    sample_indices_per_query: list of N_queries arrays, each containing indices to sample
    topk: int
    
    Returns: list of N_queries arrays, each containing topk indices into all_ref_embs
    """
    n = all_query_embs.shape[0]
    if n == 0:
        return []

    ranges = chunk_ranges(n, len(gpu_ids))
    results = [None] * n

    def batches_for_range(start, end, bsz):
        for s in range(start, end, bsz):
            e = min(s + bsz, end)
            yield s, e

    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=mp.get_context("spawn")) as ex:
        futs = []
        for wi, (rg_s, rg_e) in enumerate(ranges):
            if rg_s == rg_e:
                continue
            device_str = f"cuda:{gpu_ids[wi]}"
            for bs, be in batches_for_range(rg_s, rg_e, score_batch_size):
                fut = ex.submit(
                    _topk_with_sampling_worker,
                    all_query_embs[bs:be, :],
                    all_ref_embs,
                    sample_indices_per_query[bs:be],
                    topk,
                    device_str,
                )
                futs.append((bs, be, fut))

        for bs, be, fut in futs:
            batch_results = fut.result()
            for i, result in enumerate(batch_results):
                results[bs + i] = result

    return results


# -------------------------
# Main
# -------------------------

def _normalized_split(series):
    result = series.fillna("train").astype(str).str.lower().replace({"validation": "val", "valid": "val", "training": "train"})
    if not result.isin(["train", "val", "test"]).all():
        raise ValueError("Unknown source split")
    return result


def load_retrieval_tables(args):
    """Keep identity and source splits without needing old checker labels."""
    def read_table(path, wanted):
        if str(path).endswith('.csv'):
            return pd.read_csv(path, usecols=lambda name: name in wanted)
        import pyarrow.parquet as pq
        available = pq.read_schema(path).names
        return pd.read_parquet(path, columns=[name for name in wanted if name in available])
    patient_columns = ["pseudo_mrn", "patient_summary", "patient_boilerplate_text", "split"]
    trial_columns = ["nct_id", "this_space", "trial_boilerplate_text", "split"]
    if bool(args.patients_parquet) != bool(args.trials_file):
        raise ValueError("Supply both --patients-parquet and --trials-file")
    if args.patients_parquet:
        patient_source = read_table(args.patients_parquet, patient_columns)
        trial_source = read_table(args.trials_file, trial_columns)
    else:
        patient_source = trial_source = read_table(args.parquet, list(dict.fromkeys(patient_columns + trial_columns)))
    patients = patient_source.dropna(subset=["patient_summary"]).copy()
    trials = trial_source.dropna(subset=["nct_id", "this_space"]).copy()
    for frame in (patients, trials):
        frame["split"] = _normalized_split(frame["split"]) if "split" in frame else "train"
    if "pseudo_mrn" not in patients:
        patients["pseudo_mrn"] = patients.patient_summary.map(lambda text: hashlib.sha256(str(text).encode()).hexdigest())
    if patients.pseudo_mrn.isna().any():
        raise ValueError("Missing patient identifiers")
    if pd.api.types.is_float_dtype(patients.pseudo_mrn):
        if not patients.pseudo_mrn.mod(1).eq(0).all():
            raise ValueError("Noninteger numeric patient identifiers")
        patients["pseudo_mrn"] = patients.pseudo_mrn.astype('int64')
    patients["pseudo_mrn"] = patients.pseudo_mrn.astype(str)
    for frame, identity in ((patients, "pseudo_mrn"), (trials, "nct_id")):
        if frame.groupby(identity).split.nunique().gt(1).any():
            raise ValueError("Conflicting source split assignments")
    if patients.groupby("pseudo_mrn").patient_summary.nunique().gt(1).any():
        raise ValueError("Multiple summaries per patient; select the final summary first")
    for frame, column in ((patients, "patient_boilerplate_text"), (trials, "trial_boilerplate_text")):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("")
    patients = patients[patients.split.isin(args.splits)].drop_duplicates("pseudo_mrn")
    trials = trials[trials.split.isin(args.splits)].copy()
    trials["this_space"] = trials.this_space.str.replace(r'^\s*\d+\.', '', regex=True).str.strip()
    trials = trials.drop_duplicates(["nct_id", "this_space"])
    if patients.empty or trials.empty:
        raise ValueError("No patients or trial spaces in the requested splits")
    return (patients[["pseudo_mrn", "patient_summary", "patient_boilerplate_text", "split"]].reset_index(drop=True),
            trials[["nct_id", "this_space", "trial_boilerplate_text", "split"]].reset_index(drop=True))


def sample_candidates(patients, spaces, trials_per_patient, patients_per_space, seed):
    """Sample distinct NCT IDs, then expand their spaces; preserve co-splits."""
    rng = np.random.default_rng(seed)
    by_trial = {key: group.index.to_numpy() for key, group in spaces.groupby(["split", "nct_id"], sort=True)}
    trial_ids = {split: group.nct_id.unique() for split, group in spaces.groupby("split")}
    patient_indices = {split: group.index.to_numpy() for split, group in patients.groupby("split")}
    p_to_s, s_to_p = [], []
    for split in patients.split:
        available = trial_ids.get(split, np.array([], dtype=str))
        selected = rng.choice(available, size=min(trials_per_patient, len(available)), replace=False)
        p_to_s.append(np.concatenate([by_trial[(split, trial)] for trial in selected]) if len(selected) else np.array([], dtype=int))
    for split in spaces.split:
        available = patient_indices.get(split, np.array([], dtype=int))
        s_to_p.append(rng.choice(available, size=min(patients_per_space, len(available)), replace=False))
    return p_to_s, s_to_p


def main():
    args = parse_args()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    if len(gpu_ids) == 0:
        raise ValueError("No GPUs specified. Use --gpus like '0,1'.")

    # Set random seed
    np.random.seed(args.random_seed)

    # Ensure output directories exist
    for out_path in [args.out_cohorts_parquet, args.out_patients_parquet]:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    patients, trials_spaces = load_retrieval_tables(args)
    sample_space_indices_per_patient, sample_patient_indices_per_space = sample_candidates(
        patients, trials_spaces, args.sample_trials_per_patient,
        args.sample_patients_per_trial, args.random_seed,
    )

    # Encode patients and trial spaces across GPUs
    patient_texts = patients["patient_summary"].astype(str).tolist()
    this_spaces = trials_spaces["this_space"].astype(str).tolist()

    print(f"Encoding {len(patient_texts)} patients...")
    patient_embs = parallel_encode(
        patient_texts,
        gpu_ids=gpu_ids,
        model_path=args.model,
        encode_batch_size=args.encode_batch_size,
        max_seq_length=args.max_seq_length,
        query_prompt=args.query_prompt,
    )  # (P, D)

    print(f"Encoding {len(this_spaces)} trial spaces...")
    trial_space_embs = parallel_encode(
        this_spaces,
        gpu_ids=gpu_ids,
        model_path=args.model,
        encode_batch_size=args.encode_batch_size,
        max_seq_length=args.max_seq_length,
        query_prompt=args.query_prompt,
    )  # (S, D)

    # Run parallel topk with sampling
    pts_to_spaces_results = parallel_topk_with_sampling(
        all_query_embs=patient_embs,
        all_ref_embs=trial_space_embs,
        sample_indices_per_query=sample_space_indices_per_patient,
        gpu_ids=gpu_ids,
        topk=args.top_k_spaces,
        score_batch_size=args.score_batch_size,
    )

    # Build output dataframe (patient + space pairs)
    out_rows = []
    for p_idx, top_space_indices in enumerate(pts_to_spaces_results):
        if len(top_space_indices) == 0:
            continue
            
        patient_row = patients.iloc[p_idx]
        
        # Get the top-ranked spaces
        for space_idx in top_space_indices:
            space_row = trials_spaces.iloc[space_idx]
            out_rows.append({
                "pseudo_mrn": patient_row["pseudo_mrn"],
                "split": patient_row["split"],
                "trial_split": space_row["split"],
                "patient_summary": patient_row["patient_summary"],
                "patient_boilerplate_text": patient_row["patient_boilerplate_text"],
                "nct_id": space_row["nct_id"],
                "this_space": space_row["this_space"],
                "trial_boilerplate_text": space_row["trial_boilerplate_text"]
            })

    pts_to_spaces_df = pd.DataFrame(out_rows) if out_rows else pd.DataFrame()
    if not pts_to_spaces_df.empty:
        pts_to_spaces_df["patient_summary"] = pts_to_spaces_df["patient_summary"].str.strip()
    pts_to_spaces_df.to_parquet(args.out_cohorts_parquet, index=False)

    # -------------------------
    # Trials -> Patients (sample patients, rank them, return top patients)
    # -------------------------
    print(f"\nProcessing trial spaces -> patients...")
    print(f"  Sampling {args.sample_patients_per_trial} patients per trial space")
    print(f"  Ranking them and returning top {args.top_k_patients} patients")
    
    # Run parallel topk with sampling
    spaces_to_pts_results = parallel_topk_with_sampling(
        all_query_embs=trial_space_embs,
        all_ref_embs=patient_embs,
        sample_indices_per_query=sample_patient_indices_per_space,
        gpu_ids=gpu_ids,
        topk=args.top_k_patients,
        score_batch_size=args.score_batch_size,
    )

    # Build output dataframe (space + patient pairs)
    out_rows = []
    for s_idx, top_patient_indices in enumerate(spaces_to_pts_results):
        if len(top_patient_indices) == 0:
            continue
            
        space_row = trials_spaces.iloc[s_idx]
        
        # Get the top-ranked patients
        for p_idx in top_patient_indices:
            patient_row = patients.iloc[p_idx]
            out_rows.append({
                "pseudo_mrn": patient_row["pseudo_mrn"],
                "split": patient_row["split"],
                "trial_split": space_row["split"],
                "patient_summary": patient_row["patient_summary"],
                "patient_boilerplate_text": patient_row["patient_boilerplate_text"],
                "nct_id": space_row["nct_id"],
                "this_space": space_row["this_space"],
                "trial_boilerplate_text": space_row["trial_boilerplate_text"]
            })

    spaces_to_pts_df = pd.DataFrame(out_rows) if out_rows else pd.DataFrame()
    if not spaces_to_pts_df.empty:
        spaces_to_pts_df["patient_summary"] = spaces_to_pts_df["patient_summary"].str.strip()
    spaces_to_pts_df.to_parquet(args.out_patients_parquet, index=False)

    print("\nWrote:")
    print(f"  {args.out_cohorts_parquet} ({len(pts_to_spaces_df)} rows)")
    print(f"  {args.out_patients_parquet} ({len(spaces_to_pts_df)} rows)")


if __name__ == "__main__":
    # Use 'spawn' to avoid CUDA re-init issues in subprocesses
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
