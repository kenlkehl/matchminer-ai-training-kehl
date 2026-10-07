#!/usr/bin/env python3
import argparse
import os
import glob
import json
import pandas as pd
import torch

# Google's symmetric similarity format: either side can be the retrieval query.
# https://huggingface.co/google/embeddinggemma-2#best-practices
PROMPT_PREFIX = "task: sentence similarity | query: "
DEFAULT_BASE_MODEL = "google/embeddinggemma-2"
DEFAULT_INPUT = "space_specific_eligibility_checks.parquet"
MAX_TOKENS = 2500


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a text embedding model for trial–patient retrieval."
    )
    # Allow multiple -i flags; also supports glob patterns
    parser.add_argument(
        "-i", "--input-parquet",
        action="append",
        help="Path(s) to input parquet file(s). May be provided multiple times "
             "and may include glob patterns. If omitted, defaults to "
             f"'{DEFAULT_INPUT}'."
    )
    parser.add_argument(
        "-c", "--ckpt-dir",
        default="./initial_embeddinggemma2_training",
        help="Checkpoint/output directory for the SentenceTransformer trainer."
    )
    parser.add_argument(
        "-o", "--output-model",
        default="pt_trial_summary_pertrial_embeddinggemma2_finetuned.model",
        help="Directory name to save the final fine-tuned model (directory will be created)."
    )
    parser.add_argument(
        "-m", "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="EmbeddingGemma 2 model ID or a local TrialSpace checkpoint from this model family."
    )
    return parser.parse_args()


def expand_inputs(input_args):
    """Expand a list of input paths (which may include globs) to a concrete file list."""
    if not input_args:
        input_args = [DEFAULT_INPUT]
    files = []
    for pattern in input_args:
        expanded = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[]") else [pattern]
        files.extend(expanded)
    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    if not unique_files:
        raise FileNotFoundError("No input parquet files found after expanding patterns.")
    return unique_files


def load_and_concat(parquet_files):
    """Load multiple parquet files and concatenate them."""
    frames = []
    for p in parquet_files:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input parquet not found: {p}")
        df = pd.read_parquet(p)
        print(f"[INFO] Loaded {p} with shape {df.shape}")
        frames.append(df)
    if len(frames) == 1:
        return frames[0]
    concat_df = pd.concat(frames, axis=0, ignore_index=True)
    print(f"[INFO] Concatenated {len(frames)} files -> shape {concat_df.shape}")
    return concat_df


def load_text_model(base_model, device):
    """Keep Google's mean pooling/normalization, loading only the text encoder."""
    from sentence_transformers import SentenceTransformer
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_model)
    if config.model_type != "embedding_gemma2":
        raise ValueError(
            "TrialSpace now requires google/embeddinggemma-2 or a checkpoint "
            "fine-tuned from it. Use fresh model/checkpoint directories for this run."
        )
    model = SentenceTransformer(
        base_model,
        device=device,
        config_kwargs={"vision_config": None, "audio_config": None},
        # FP32 master weights; the trainer uses BF16 autocast on supported GPUs.
        # EmbeddingGemma 2 does not support FP16.
        model_kwargs={"dtype": torch.float32},
    )
    model.max_seq_length = MAX_TOKENS
    model.prompts.update({
        "SentenceSimilarity": PROMPT_PREFIX,
        "query": PROMPT_PREFIX,
        "document": PROMPT_PREFIX,
    })
    model.default_prompt_name = "SentenceSimilarity"
    model.set_pooling_include_prompt(True)
    return model


def validate_resume_checkpoint(checkpoint):
    """Do not silently resume old Qwen or differently formatted training runs."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint)
    with open(os.path.join(checkpoint, "config_sentence_transformers.json")) as handle:
        saved = json.load(handle)
    if (
        config.model_type != "embedding_gemma2"
        or config.vision_config is not None
        or config.audio_config is not None
        or saved.get("default_prompt_name") != "SentenceSimilarity"
        or any(saved.get("prompts", {}).get(name) != PROMPT_PREFIX
               for name in ("SentenceSimilarity", "query", "document"))
    ):
        raise ValueError(
            f"Incompatible TrialSpace checkpoint: {checkpoint}. "
            "Choose a fresh --ckpt-dir for text-only EmbeddingGemma 2 training."
        )


def main():
    args = parse_args()
    input_files = expand_inputs(args.input_parquet)
    print(f"[INFO] Using {len(input_files)} input file(s):")
    for f in input_files:
        print(f"       - {f}")
    print(f"[INFO] Checkpoint dir: {args.ckpt_dir}")
    print(f"[INFO] Output model dir: {args.output_model}")
    print(f"[INFO] Base model: {args.base_model}")

    # Deferred imports
    from sentence_transformers import (
        losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    )
    from datasets import Dataset
    from transformers.trainer_utils import get_last_checkpoint

    checkpoint = get_last_checkpoint(args.ckpt_dir) if os.path.isdir(args.ckpt_dir) else None
    if checkpoint:
        validate_resume_checkpoint(checkpoint)

    # --- Load & filter data ---
    trial_checks = load_and_concat(input_files)

    # Optional filter if present
    if "patient_long_text" in trial_checks.columns:
        trial_checks = trial_checks[trial_checks.patient_long_text.fillna("") != ""]

    req_cols = {"patient_summary", "this_space", "eligibility_result"}
    missing = req_cols - set(trial_checks.columns)
    if missing:
        raise ValueError(f"Input parquet missing required columns: {missing}")

    trial_checks = trial_checks[
        ~trial_checks.patient_summary.fillna("").str.contains(
            "No cancer|no cancer|No primary|No evidence of malignancy", na=False
        )
    ]
    trial_checks = trial_checks[
        ~trial_checks.patient_summary.fillna("").str.startswith("No information")
    ]

    # remove leading digit and period and space from trial space if present
    trial_checks["this_space"] = trial_checks['this_space'].str.replace(r'^\s*\d+\.', '', regex=True)

    # Drop rows where the LLM could not parse a score
    trial_checks = trial_checks[trial_checks.eligibility_result >= 0].copy()

    # Normalize eligibility_result (raw score 0–5) to [-1, 1] for cosine-similarity-scale labels
    MAX_SCORE = 5
    trial_checks["eligibility_label"] = (trial_checks["eligibility_result"] / MAX_SCORE) * 2 - 1

    print("\n[INFO] Dataframe info after filtering:")
    trial_checks.info()
    print("\n[INFO] eligibility_result (raw score) counts:")
    print(trial_checks.eligibility_result.value_counts(dropna=False))
    print("\n[INFO] eligibility_label (normalized) distribution:")
    print(trial_checks.eligibility_label.describe())

    # Keep raw text in the datasets. The trainer applies the prefix once, then
    # the model tokenizer truncates the complete input, including special tokens.
    for column in ("patient_summary", "this_space"):
        trial_checks[column] = trial_checks[column].fillna("").astype(str)

    eligible = trial_checks[trial_checks.eligibility_result >= 1]
    print("\n[INFO] Eligible subset info:")
    eligible.info()

    # --- Model (from the base model) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")
    model = load_text_model(args.base_model, device)

    # --- Datasets ---
    mnri_dataset = Dataset.from_pandas(
        eligible[["patient_summary", "this_space"]],
        preserve_index=False
    )
    contrastive_dataset = Dataset.from_pandas(
        trial_checks[["patient_summary", "this_space", "eligibility_label"]]
        .rename(columns={"eligibility_label": "label"}),
        preserve_index=False
    )
    train_dataset = {
        "mnri_dataset": mnri_dataset,
        "contrastive_dataset": contrastive_dataset,
    }

    # --- Losses ---
    mll_train_loss = losses.MultipleNegativesRankingLoss(model=model)
    contrastive_train_loss = losses.CoSENTLoss(model=model)
    losses_map = {
        "mnri_dataset": mll_train_loss,
        "contrastive_dataset": contrastive_train_loss,
    }

    # --- Trainer ---
    args_st = SentenceTransformerTrainingArguments(
        output_dir=args.ckpt_dir,
        per_device_train_batch_size=10,
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.01,
        save_strategy="steps",
        save_steps=2500,
        save_total_limit=2,
        logging_steps=100,
        num_train_epochs=3,
        prompts=PROMPT_PREFIX,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args_st,
        train_dataset=train_dataset,
        loss=losses_map,
    )

    print("\n[INFO] Starting training...")
    if checkpoint:
        print(f"[INFO] Resuming from {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print("[INFO] Starting fresh training")
        trainer.train()
    print("\n[INFO] Training complete.")

    # --- Save final model ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
    model.save(args.output_model)
    print(f"[INFO] Saved fine-tuned model to: {args.output_model}")


if __name__ == "__main__":
    main()
