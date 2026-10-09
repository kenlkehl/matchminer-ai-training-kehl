"""Snapshot completed synthetic labels and distill on the original co-splits.

Run `prepare` once, then `train` with torchrun. No teacher requests are made.

Deprecated with the GoodOptionChecker it trains; GoodOption scoring uses the
LLM rubric over drug and class evidence. Kept to reproduce earlier checkers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import pandas as pd

from matchminer_ai.matching import RUBRIC_CRITERIA
from matchminer_ai.trials import load_good_option_catalog

import train_good_option_checker as core


def split_lookup(frame: pd.DataFrame, key: str) -> pd.Series:
    frame = frame[[key, "split"]].copy()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    if frame.groupby(key)["split"].nunique().gt(1).any():
        raise ValueError(f"Conflicting source split assignments for {key}")
    return frame.drop_duplicates(key).set_index(key)["split"]


def assign_source_splits(labels, patients, trials):
    patients = patients.copy()
    patients["patient_group_id"] = patients.patient_summary.map(
        lambda text: core._hash_text(b"GoodOption patient group", text)
    )
    result = labels.copy()
    result["patient_split"] = result.patient_group_id.map(
        split_lookup(patients, "patient_group_id")
    )
    result["trial_split"] = result.nct_id.map(split_lookup(trials, "nct_id"))
    if result[["patient_split", "trial_split"]].isna().any().any():
        raise ValueError("Some labels have no authoritative source split")
    result["partition"] = "excluded"
    for split, partition in (("train", "train"), ("val", "validation")):
        result.loc[
            result.patient_split.eq(split) & result.trial_split.eq(split),
            "partition",
        ] = partition
    return result


def prepare(args):
    run = args.run_dir
    for path in (args.shards, args.catalog):
        if not core._inside_no_phi(path):
            raise ValueError("Training sources must be inside data/no_phi")
    run.mkdir(parents=True, exist_ok=False)
    shards = sorted(args.shards.glob("labels_*.parquet"))
    columns = ["candidate_id", "patient_id", "patient_group_id", "patient_summary",
               "nct_id", "good_option_status", "drug_assessments_json",
               "catalog_compatibility_id"]
    inventory = []
    frames = []
    for path in shards:
        if not core._inside_no_phi(path):
            raise ValueError("Label shard resolves outside data/no_phi")
        before = path.stat()
        frames.append(pd.read_parquet(path, columns=columns))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Shard changed during snapshot: {path.name}")
        inventory.append({"path": str(path), "bytes": after.st_size,
                          "mtime_ns": after.st_mtime_ns})
    labels = pd.concat(frames, ignore_index=True).drop_duplicates("candidate_id", keep="last")
    if args.all_labels:
        labels["patient_split"] = "all"
        labels["trial_split"] = "all"
        labels["partition"] = "train"
    else:
        patients = pd.read_parquet(args.patients, columns=["patient_summary", "split"])
        trials = pd.read_csv(args.trials, usecols=["nct_id", "split"])
        labels = assign_source_splits(labels, patients, trials)
    labels.to_parquet(run / "labels_snapshot.parquet", index=False)
    metadata = {
        "shards": inventory, "catalog": str(args.catalog),
        "patient_split_source": None if args.all_labels else str(args.patients),
        "trial_split_source": None if args.all_labels else str(args.trials),
        "all_labels": args.all_labels,
        "label_counts": labels.groupby(["patient_split", "trial_split", "good_option_status"]).size().to_dict(),
    }
    metadata["label_counts"] = {"/".join(k): int(v) for k,v in metadata["label_counts"].items()}
    (run / "snapshot_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata["label_counts"], indent=2), flush=True)
    finalize(args)


def finalize(args):
    run = args.run_dir
    labels = pd.read_parquet(run / "labels_snapshot.parquet")
    metadata = json.loads((run / "snapshot_manifest.json").read_text())
    catalog = load_good_option_catalog(args.catalog)
    parts = []
    audits = {}
    all_labels = metadata.get("all_labels", False)
    for partition in (("train",) if all_labels else ("train", "validation")):
        selected = labels.loc[labels.partition.eq(partition)]
        print(f"Flattening {partition}: {len(selected)} trial labels", flush=True)
        part = core.flatten_patient_drug_labels(selected, catalog, conflict_policy="drop")
        audits[partition] = part.attrs.pop("conflict_audit")
        print(f"{partition}: retained {len(part)}; excluded {audits[partition]['excluded_patient_drug_pairs']} conflicting pairs", flush=True)
        part["partition"] = partition
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    if not all_labels and set(parts[0].patient_group_id) & set(parts[1].patient_group_id):
        raise ValueError("Patient leakage across co-splits")
    if set(labels.loc[labels.partition.eq("train"), "nct_id"]) & set(labels.loc[labels.partition.eq("validation"), "nct_id"]):
        raise ValueError("Trial leakage across co-splits")
    data.to_parquet(run / "prepared.parquet", index=False)
    manifest = core.build_split_manifest(data, strategy="none" if all_labels else "original_patient_trial_cosplit", seed=42,
        patient_validation_fraction=0, drug_validation_fraction=0, catalog=catalog)
    manifest["source_label_counts"] = metadata["label_counts"]
    manifest["prepared_sha256"] = hashlib.sha256((run / "prepared.parquet").read_bytes()).hexdigest()
    manifest["shared_drugs"] = 0 if all_labels else len(set(parts[0].drug_id) & set(parts[1].drug_id))
    manifest["conflict_exclusions"] = {key: {k:v for k,v in audit.items() if k != "excluded_patient_drug_ids"} for key,audit in audits.items()}
    (run / "conflict_audit.json").write_text(json.dumps(audits, indent=2))
    (run / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


def train(args):
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
    set_seed(42)
    run = args.run_dir
    manifest = json.loads((run / "split_manifest.json").read_text())
    if hashlib.sha256((run / "prepared.parquet").read_bytes()).hexdigest() != manifest["prepared_sha256"]:
        raise ValueError("Prepared training snapshot checksum mismatch")
    frame = pd.read_parquet(run / "prepared.parquet")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model,
        local_files_only=True, num_labels=4, problem_type="multi_label_classification",
        id2label=dict(enumerate(RUBRIC_CRITERIA)),
        label2id={v:k for k,v in enumerate(RUBRIC_CRITERIA)},
        attn_implementation="sdpa")
    # ModernBERT returns mean BCE and ignores num_items_in_batch despite **kwargs.
    # Tell Trainer to normalize it across gradient accumulation steps.
    core.configure_mean_bce_accumulation(model)
    for key, value in {
        "matchminer_catalog_compatibility_id": manifest["catalog_compatibility_id"],
        "matchminer_split_fingerprint_sha256": manifest["split_fingerprint_sha256"],
    }.items():
        setattr(model.config, key, value)
    def dataset(partition):
        ds = Dataset.from_pandas(frame.loc[frame.partition.eq(partition), ["checker_text", "labels"]], preserve_index=False)
        return ds.map(lambda batch: tokenizer(batch["checker_text"], truncation=True, max_length=8192),
                      batched=True, remove_columns=["checker_text"])
    training = TrainingArguments(output_dir=str(run / "checkpoints"),
        learning_rate=2e-5, per_device_train_batch_size=8, per_device_eval_batch_size=8,
        gradient_accumulation_steps=4, num_train_epochs=args.num_train_epochs, weight_decay=0.01,
        warmup_ratio=0.05, logging_steps=20, save_strategy="epoch", save_total_limit=2,
        eval_strategy="no", bf16=True, report_to="none", seed=42,
        gradient_checkpointing=False, ddp_find_unused_parameters=False,
        dataloader_num_workers=2)
    trainer = Trainer(model=model, args=training, train_dataset=dataset("train"),
        processing_class=tokenizer, compute_metrics=core.compute_metrics)
    result = trainer.train()
    trainer.save_model(str(run / "model"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(run / "model")
        metrics = dict(result.metrics, train_rows=len(trainer.train_dataset),
                       num_train_epochs=args.num_train_epochs,
                       validation_rows=int(frame.partition.eq("validation").sum()))
        (run / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    if not frame.partition.eq("validation").any():
        return
    validation = dataset("validation")
    prediction = trainer.predict(validation, metric_key_prefix="val_val")
    trainer.save_model(str(run / "model"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(run / "model")
        metrics = prediction.metrics
        from sklearn.metrics import average_precision_score, precision_recall_fscore_support
        probabilities = core._sigmoid(prediction.predictions)
        for index, criterion in enumerate(RUBRIC_CRITERIA):
            truth = prediction.label_ids[:, index]
            pred = probabilities[:, index] >= 0.5
            precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
            metrics[f"val_val_{criterion}_prevalence"] = float(truth.mean())
            metrics[f"val_val_{criterion}_auprc"] = float(average_precision_score(truth, probabilities[:,index]))
            for name, value in (("precision", precision), ("recall", recall), ("f1", f1)):
                metrics[f"val_val_{criterion}_{name}"] = float(value)
        metrics["train_rows"] = int(frame.partition.eq("train").sum())
        metrics["val_val_rows"] = len(validation)
        metrics["checkpoint_policy"] = f"Final epoch after {args.num_train_epochs} fixed epochs; Val/val used only for final evaluation"
        (run / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
        result = frame.loc[frame.partition.eq("validation"), ["patient_drug_id", "patient_group_id", "drug_id"]].reset_index(drop=True)
        for i, criterion in enumerate(RUBRIC_CRITERIA):
            result[f"label_{criterion}"] = prediction.label_ids[:,i]
            result[f"probability_{criterion}"] = probabilities[:,i]
        result.to_parquet(run / "validation_predictions.parquet", index=False)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    warnings.warn(
        "train_good_option_cosplit.py trains the deprecated GoodOptionChecker; "
        "GoodOption scoring now uses the LLM rubric over drug and class evidence.",
        FutureWarning,
        stacklevel=1,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "finalize", "train"])
    parser.add_argument("--all-labels", action="store_true", help="Use all valid labels without holdouts")
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shards", type=Path, default=core.DEFAULT_LABEL_SHARDS)
    parser.add_argument("--catalog", type=Path, default=core.DEFAULT_CATALOG_DIR)
    parser.add_argument("--patients", type=Path, default=core.DEFAULT_DATA_DIR / "patient_summaries_with_spaces.parquet")
    parser.add_argument("--trials", type=Path, default=core.DEFAULT_DATA_DIR / "trial_space_lineitems.csv")
    parser.add_argument("--base-model", default=str(core.DEFAULT_MODEL_DIR / "hf/answerdotai_ModernBERT-large"))
    args = parser.parse_args()
    {"prepare": prepare, "finalize": finalize, "train": train}[args.stage](args)
