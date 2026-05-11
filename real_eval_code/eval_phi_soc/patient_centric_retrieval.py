#!/usr/bin/env python3
"""
Patient-centric trial retrieval for SOC evaluation.
For each patient, finds the top N matching trial spaces.

Uses embedding similarity to match patient summaries to trial space descriptions.

Usage:
    python patient_centric_retrieval.py --gpu 0 [--top-k 20]
"""

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parents[3]  # matchminer-ai-training


def parse_args():
    parser = argparse.ArgumentParser(description="Patient-centric trial retrieval")
    parser.add_argument("--gpu", type=str, required=True,
                        help="GPU ID to use (e.g., '0')")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top matching spaces per patient")
    parser.add_argument("--patient-summaries", type=str, default=None,
                        help="Input parquet with patient summaries (default: data/phi_soc/patient_summaries.parquet)")
    parser.add_argument("--trial-spaces", type=str, default=None,
                        help="Input CSV with trial spaces (default: data/phi_soc/sample_spaces.csv)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output CSV file (default: data/phi_soc/patient_centric_candidates.csv)")
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Path to sentence transformer model (default: models/trialspace)")
    parser.add_argument("--shard-dir", type=str, default=None,
                        help="Directory for checkpoint shards (default: data/phi_soc/shards_patient_centric)")
    parser.add_argument("--shard-size", type=int, default=500,
                        help="Number of patients per shard")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard ID (0-indexed) when sharding work across multiple GPUs")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards (1 = no sharding)")
    parser.add_argument("--skip-consolidate", action="store_true",
                        help="Skip the final consolidation (caller will merge after all shards finish)")
    return parser.parse_args()


def get_completed_shards(shard_dir):
    """Get list of completed shard indices."""
    pattern = os.path.join(shard_dir, "shard_*.csv")
    files = glob.glob(pattern)
    indices = set()
    for f in files:
        try:
            idx = int(os.path.basename(f).split("_")[1].replace(".csv", ""))
            indices.add(idx)
        except:
            pass
    return indices


def consolidate_shards(shard_dir, output_file):
    """Consolidate all shards into a single file.

    Looks for both the legacy flat layout (shard_dir/shard_*.csv) and the sharded
    layout written when --num-shards > 1 (shard_dir/shard_s*/shard_*.csv).
    """
    flat_pattern = os.path.join(shard_dir, "shard_*.csv")
    nested_pattern = os.path.join(shard_dir, "shard_s*", "shard_*.csv")
    files = sorted(glob.glob(flat_pattern) + glob.glob(nested_pattern))
    if not files:
        print(f"No shards found")
        return None

    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, axis=0).reset_index(drop=True)
    combined.to_csv(output_file, index=False)
    print(f"Consolidated {len(files)} shards into {output_file}")
    return combined


def main():
    args = parse_args()

    # Set defaults relative to repo root
    if args.patient_summaries is None:
        args.patient_summaries = str(REPO_ROOT.parent / "data/phi_soc/patient_summaries.parquet")
    if args.trial_spaces is None:
        args.trial_spaces = str(REPO_ROOT.parent / "data/phi_soc/sample_spaces.csv")
    if args.output_file is None:
        args.output_file = str(REPO_ROOT.parent / "data/phi_soc/patient_centric_candidates.csv")
    if args.embedding_model is None:
        args.embedding_model = str(REPO_ROOT.parent / "models/trialspace")
    if args.shard_dir is None:
        args.shard_dir = str(REPO_ROOT.parent / "data/phi_soc/shards_patient_centric")

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    from sentence_transformers import SentenceTransformer

    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_root = Path(args.shard_dir)
    shard_root.mkdir(parents=True, exist_ok=True)
    # When sharding across multiple GPUs, each parallel process writes to its own
    # subdirectory so resume state and shard filenames don't collide.
    if args.num_shards > 1:
        shard_dir = shard_root / f"shard_s{args.shard_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)
    else:
        shard_dir = shard_root

    # Load patient summaries
    print("Loading patient summaries...")
    patients = pd.read_parquet(args.patient_summaries)
    patients = patients[~patients.patient_summary.isnull()]
    print(f"Loaded {patients.shape[0]} patient records")

    # Load trial spaces
    print("Loading trial spaces...")
    spaces = pd.read_csv(args.trial_spaces)
    print(f"Loaded {spaces.shape[0]} trial space records")

    # Prepare patient summaries dataframe - deduplicate
    patient_summaries = patients.groupby('patient_summary').first().reset_index()[
        ['dfci_mrn', 'patient_summary', 'patient_boilerplate_text', 'split']
    ]

    # If running as one of multiple parallel shards, slice the patient list to this
    # shard. Each shard then computes embeddings only for its own patients (and all
    # spaces, which are needed for matching).
    if args.num_shards > 1:
        total_patients_full = patient_summaries.shape[0]
        chunk = (total_patients_full + args.num_shards - 1) // args.num_shards
        slice_start = args.shard_id * chunk
        slice_end = min(slice_start + chunk, total_patients_full)
        print(f"Shard {args.shard_id}/{args.num_shards}: patients [{slice_start}, {slice_end}) of {total_patients_full}")
        patient_summaries = patient_summaries.iloc[slice_start:slice_end].reset_index(drop=True)

    # Prepare spaces dataframe - deduplicate
    spaces = spaces.groupby(['nct_id', 'this_space']).first().reset_index()
    if 'trial_boilerplate_text' not in spaces.columns:
        spaces['trial_boilerplate_text'] = None
    spaces = spaces[['nct_id', 'this_space', 'trial_boilerplate_text']]
    print(f"Unique spaces: {spaces.shape[0]}")

    # Load embedding model
    print(f"Loading embedding model from {args.embedding_model}...")
    embedding_model = SentenceTransformer(
        args.embedding_model, trust_remote_code=True, device='cuda'
    )
    embedding_model.prompts['query'] = (
        "Instruct: Given a cancer patient summary, retrieve clinical trial options "
        "that are reasonable for that patient; or, given a clinical trial option, "
        "retrieve cancer patients who are reasonable candidates for that trial. "
    )
    embedding_model.max_seq_length = 2500

    # Compute embeddings
    print("Computing patient embeddings...")
    with torch.no_grad():
        patient_embeddings = embedding_model.encode(
            patient_summaries.patient_summary.tolist(),
            convert_to_tensor=True,
            prompt='query'
        )

    print("Computing space embeddings...")
    with torch.no_grad():
        space_embeddings = embedding_model.encode(
            spaces.this_space.tolist(),
            convert_to_tensor=True,
            prompt='query'
        )

    # Process patients with sharding
    print("Matching patients to spaces...")
    completed_shards = get_completed_shards(str(shard_dir))
    total_patients = patient_summaries.shape[0]

    for shard_idx in range(0, total_patients, args.shard_size):
        shard_num = shard_idx // args.shard_size
        if shard_num in completed_shards:
            print(f"Shard {shard_num} already complete, skipping...")
            continue

        end_idx = min(shard_idx + args.shard_size, total_patients)
        print(f"Processing shard {shard_num} (patients {shard_idx} to {end_idx})...")

        output_list = []
        for i in range(shard_idx, end_idx):
            patient_embedding = patient_embeddings[i, :]

            # Compute similarities to all spaces
            similarities = F.cosine_similarity(patient_embedding.unsqueeze(0), space_embeddings)
            sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)

            top_k = min(args.top_k, len(spaces))
            top_indices = sorted_indices[0:top_k].cpu().numpy()

            relevant_spaces = spaces.iloc[top_indices].this_space.tolist()
            trial_boilerplate = spaces.iloc[top_indices].trial_boilerplate_text.tolist()

            output = pd.DataFrame({
                'dfci_mrn': patient_summaries.iloc[i].dfci_mrn,
                'split': patient_summaries.iloc[i].split,
                'patient_summary': patient_summaries.iloc[i].patient_summary,
                'patient_boilerplate_text': patient_summaries.iloc[i].patient_boilerplate_text,
                'this_space': relevant_spaces,
                'trial_boilerplate_text': trial_boilerplate
            })
            output_list.append(output)

        if output_list:
            shard_df = pd.concat(output_list, axis=0).reset_index(drop=True)
            shard_df['patient_summary'] = shard_df.patient_summary.str.strip()
            shard_file = shard_dir / f"shard_{shard_num}.csv"
            shard_df.to_csv(str(shard_file), index=False)
            print(f"Saved shard {shard_num} ({len(shard_df)} records)")

    # Consolidate shards (skip when caller will merge across multiple shards)
    if args.skip_consolidate:
        print("--skip-consolidate set; leaving consolidation to caller.")
    else:
        print("Consolidating shards...")
        consolidate_shards(str(shard_root), args.output_file)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
