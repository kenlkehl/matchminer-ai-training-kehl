#!/usr/bin/env python3
"""Build a drug evidence catalog, label patient-drug pairs, and train v2.

Public research is completed and validated before any patient-bearing LLM call.
The student receives one patient summary and one clean drug summary and predicts
the four unchanged GoodOption criteria as independent logits.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from matchminer_ai.config import MMAIConfig, load_config, load_default_preset
from matchminer_ai.good_options import (
    GOOD_OPTION_INPUT_VERSION,
    GOOD_OPTION_LABEL_SCHEMA_VERSION,
    GOOD_OPTION_PROMPT_VERSION,
    RUBRIC_CRITERIA,
    GoodOptionCatalog,
    ResearchSettings,
    build_good_option_catalog,
    build_good_option_checker_text,
    load_good_option_catalog,
    score_good_options_with_llm,
    validate_good_option_catalog,
)


REPOSITORY_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = REPOSITORY_DIR.parent
DEFAULT_DATA_DIR = WORKSPACE_DIR / "data" / "no_phi"
DEFAULT_MODEL_DIR = WORKSPACE_DIR / "models"
DEFAULT_CATALOG_DIR = DEFAULT_DATA_DIR / "good_option_catalog_v2"
DEFAULT_LABEL_OUTPUT = DEFAULT_DATA_DIR / "good_option_four_point_labels_v2.parquet"
DEFAULT_LABEL_SHARDS = DEFAULT_DATA_DIR / "good_option_four_point_label_shards_v2"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "goodoptionchecker_four_point_v2"
DEFAULT_CHECKPOINT_DIR = (
    DEFAULT_MODEL_DIR / "goodoptionchecker_four_point_v2_checkpoints"
)
DEFAULT_CANDIDATE_FILES = tuple(
    DEFAULT_DATA_DIR / f"top_{direction}_tocheck_round{round_index}.parquet"
    for round_index in (1, 2, 3)
    for direction in ("cohorts", "patients")
)
SPLIT_STRATEGY_VERSION = "patient-and-canonical-drug-holdout-v2"


def _hash_text(prefix: bytes, *values: Any) -> str:
    digest = hashlib.sha256(prefix)
    for value in values:
        digest.update(b"\0")
        digest.update(str(value or "").strip().encode("utf-8", errors="replace"))
    return digest.hexdigest()


def _inside_no_phi(path: Path) -> bool:
    try:
        path.resolve().relative_to(DEFAULT_DATA_DIR.resolve())
    except ValueError:
        return False
    return True


def _validate_candidate_paths(
    paths: Sequence[Path], *, confirm_inputs_are_non_phi: bool
) -> None:
    custom = [path for path in paths if not _inside_no_phi(path)]
    if custom and not confirm_inputs_are_non_phi:
        raise ValueError(
            "Custom candidate inputs require --confirm-inputs-are-non-phi because "
            "patient summaries are sent to the configured LLM endpoint: "
            + ", ".join(str(path) for path in custom)
        )


def load_candidates(
    paths: Sequence[str | Path],
    *,
    confirm_inputs_are_non_phi: bool = False,
    max_candidates: int | None = None,
) -> pd.DataFrame:
    """Load and deduplicate mining tables at patient-trial granularity."""

    resolved = [Path(path).expanduser().resolve() for path in paths]
    _validate_candidate_paths(
        resolved, confirm_inputs_are_non_phi=confirm_inputs_are_non_phi
    )
    missing_files = [str(path) for path in resolved if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Candidate files not found: " + ", ".join(missing_files))
    frames: list[pd.DataFrame] = []
    for path in resolved:
        frame = pd.read_parquet(path)
        required = {"patient_summary", "nct_id"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        keep = [
            column
            for column in ("pseudo_mrn", "patient_summary", "nct_id", "split")
            if column in frame.columns
        ]
        frame = frame[keep].copy()
        if "pseudo_mrn" not in frame:
            frame["pseudo_mrn"] = ""
        if "split" not in frame:
            frame["split"] = ""
        frames.append(frame)
    candidates = pd.concat(frames, ignore_index=True)
    candidates["patient_summary"] = (
        candidates["patient_summary"].fillna("").astype(str).str.strip()
    )
    candidates["nct_id"] = (
        candidates["nct_id"].fillna("").astype(str).str.strip().str.upper()
    )
    candidates = candidates.loc[
        candidates["patient_summary"].ne("")
        & candidates["nct_id"].str.fullmatch(r"NCT\d{8}")
    ].copy()
    candidates["patient_id"] = [
        str(raw).strip()
        or _hash_text(b"GoodOption synthetic patient", summary)[:24]
        for raw, summary in zip(
            candidates["pseudo_mrn"], candidates["patient_summary"], strict=True
        )
    ]
    candidates["patient_group_id"] = candidates["patient_summary"].map(
        lambda value: _hash_text(b"GoodOption patient group", value)
    )
    candidates["candidate_id"] = [
        _hash_text(b"GoodOption candidate v2", patient_group, nct_id)
        for patient_group, nct_id in zip(
            candidates["patient_group_id"], candidates["nct_id"], strict=True
        )
    ]
    candidates = candidates.sort_values("candidate_id", kind="stable").drop_duplicates(
        "candidate_id", keep="first"
    )
    if max_candidates is not None:
        candidates = candidates.head(max(0, int(max_candidates)))
    return candidates.reset_index(drop=True)


def _read_nct_ids_file(path: str | Path) -> list[str]:
    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() == ".parquet":
        frame = pd.read_parquet(source)
        column = "nct_id" if "nct_id" in frame else "trial_id"
        if column not in frame:
            raise ValueError("NCT ID Parquet must contain nct_id or trial_id.")
        return frame[column].dropna().astype(str).tolist()
    if source.suffix.casefold() == ".csv":
        frame = pd.read_csv(source)
        column = "nct_id" if "nct_id" in frame else "trial_id"
        if column not in frame:
            raise ValueError("NCT ID CSV must contain nct_id or trial_id.")
        return frame[column].dropna().astype(str).tolist()
    return [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _read_candidate_nct_ids(paths: Sequence[str | Path]) -> list[str]:
    """Read only public trial identifiers from candidate Parquets."""

    resolved = [Path(path).expanduser().resolve() for path in paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError("Candidate files not found: " + ", ".join(missing))
    values: list[str] = []
    for path in resolved:
        try:
            frame = pd.read_parquet(path, columns=["nct_id"])
        except Exception as error:  # noqa: BLE001 - add path-specific context.
            raise ValueError(
                f"Catalog candidate input {path} must contain nct_id."
            ) from error
        values.extend(frame["nct_id"].dropna().astype(str))
    return values


def resolve_nct_ids(args: argparse.Namespace) -> tuple[str, ...]:
    ids = list(args.nct_id or [])
    if args.nct_ids_file:
        ids.extend(_read_nct_ids_file(args.nct_ids_file))
    if not ids:
        ids.extend(_read_candidate_nct_ids(args.candidate_files))
    normalized = tuple(
        dict.fromkeys(str(value).strip().upper() for value in ids if str(value).strip())
    )
    invalid = [value for value in normalized if not re.fullmatch(r"NCT\d{8}", value)]
    if invalid:
        raise ValueError(f"Invalid NCT IDs: {invalid[:10]}")
    return normalized


def _server_urls_from_file(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return [
        str(item.get("url") or "").strip()
        for item in payload.get("servers", [])
        if isinstance(item, Mapping) and str(item.get("url") or "").strip()
    ]


def configure_teacher(args: argparse.Namespace) -> MMAIConfig:
    """Apply shared model/endpoint defaults, allowing per-config stage overrides."""

    config = load_config(args.config) if args.config else load_default_preset()
    urls = [
        value.strip()
        for value in str(getattr(args, "server_urls", "") or "").split(",")
        if value.strip()
    ]
    if getattr(args, "server_urls_file", ""):
        urls.extend(_server_urls_from_file(args.server_urls_file))
    urls = list(dict.fromkeys(urls))
    model = str(getattr(args, "model", "") or "").strip()
    if urls:
        config.remote["enabled"] = True
        config.remote["server_urls"] = urls
        config.remote["request_timeout"] = float(args.llm_request_timeout)
        config.remote["max_concurrent_requests"] = int(args.max_concurrent_requests)
        if model:
            config.llm_good_option.setdefault("remote", {})["model_name"] = model
    else:
        config.remote["enabled"] = False
        if model:
            config.llm_good_option.setdefault("local", {})["model_name"] = model
        config.llm_good_option.setdefault("local", {}).setdefault("engine", {})[
            "tensor_parallel_size"
        ] = int(args.tensor_parallel_size)
    return config


def _research_settings(args: argparse.Namespace) -> ResearchSettings:
    return ResearchSettings(
        request_timeout=args.research_request_timeout,
        max_attempts=args.max_source_attempts,
        registry_max_attempts=args.registry_max_attempts,
        initial_backoff=args.initial_backoff,
        maximum_backoff=args.maximum_backoff,
        max_concurrency=args.research_concurrency,
        web_results_per_query=args.web_results_per_query,
        max_web_results_per_drug=args.max_web_results_per_drug,
        max_web_documents_per_drug=args.max_web_documents_per_drug,
        max_pubmed_records=args.max_pubmed_records,
        max_registry_studies=args.max_registry_studies,
        max_civic_records=args.max_civic_records,
        max_europe_pmc_records=args.max_europe_pmc_records,
        max_regulatory_records=args.max_regulatory_records,
        max_passage_chars=args.max_passage_chars,
    )


def run_catalog(args: argparse.Namespace) -> None:
    nct_ids = resolve_nct_ids(args)
    config = configure_teacher(args)

    def progress(stage: str, completed: int, total: int, label: str) -> None:
        print(f"[{stage}] {completed}/{total}: {label}", flush=True)

    catalog = asyncio.run(
        build_good_option_catalog(
            nct_ids,
            args.catalog,
            config=config,
            settings=_research_settings(args),
            overwrite=args.overwrite,
            progress_callback=progress,
        )
    )
    print(
        f"Saved catalog {catalog.path} ({len(catalog.trial_registry)} trials, "
        f"{len(catalog.drug_summaries)} unique drugs)."
    )


def run_validate_catalog(args: argparse.Namespace) -> None:
    manifest = validate_good_option_catalog(args.catalog)
    print(
        f"Catalog valid: {args.catalog} compatibility={manifest['compatibility_id']} "
        f"counts={manifest['counts']}"
    )


def _existing_label_ids(
    shards_dir: Path, *, catalog: GoodOptionCatalog
) -> set[str]:
    ids: set[str] = set()
    for shard in sorted(shards_dir.glob("labels_*.parquet")):
        frame = pd.read_parquet(shard)
        required = {
            "candidate_id",
            "catalog_compatibility_id",
            "prompt_version",
            "label_schema_version",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Existing label shard {shard} is missing {missing}.")
        expected = {
            "catalog_compatibility_id": catalog.compatibility_id,
            "prompt_version": GOOD_OPTION_PROMPT_VERSION,
            "label_schema_version": GOOD_OPTION_LABEL_SCHEMA_VERSION,
        }
        for column, value in expected.items():
            if not frame[column].astype(str).eq(value).all():
                raise ValueError(
                    f"Existing label shard {shard} has incompatible {column}; "
                    "use a fresh v2 shard directory."
                )
        ids.update(frame["candidate_id"].astype(str))
    return ids


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _serialize_label_frame(
    source: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    catalog: GoodOptionCatalog,
) -> pd.DataFrame:
    by_key = {
        (str(row.patient_id), str(row.trial_id)): row._asdict()
        for row in scored.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    for row in source.to_dict(orient="records"):
        result = by_key[(str(row["patient_id"]), str(row["nct_id"]))]
        records.append(
            {
                "candidate_id": row["candidate_id"],
                "patient_id": row["patient_id"],
                "patient_group_id": row["patient_group_id"],
                "patient_summary": row["patient_summary"],
                "nct_id": row["nct_id"],
                "split": row.get("split", ""),
                "good_option_score": result.get("good_option_score"),
                "good_option_points": result.get("good_option_points"),
                "good_option_max_points": result.get("good_option_max_points"),
                "good_option_drug_count": result.get("good_option_drug_count"),
                "good_option_status": result.get("good_option_status"),
                "patient_disease_type": result.get(
                    "good_option_patient_disease_type", ""
                ),
                "drug_assessments_json": json.dumps(
                    result.get("good_option_drug_assessments") or [],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "uncertainties_json": json.dumps(
                    result.get("good_option_uncertainties") or [],
                    ensure_ascii=False,
                ),
                "llm_response": result.get("good_option_answer_text", ""),
                "parse_error": result.get("good_option_parse_error", ""),
                "catalog_compatibility_id": catalog.compatibility_id,
                "prompt_version": GOOD_OPTION_PROMPT_VERSION,
                "label_schema_version": GOOD_OPTION_LABEL_SCHEMA_VERSION,
            }
        )
    return pd.DataFrame(records)


def _label_batch_with_retries(
    batch: pd.DataFrame,
    *,
    catalog: GoodOptionCatalog,
    config: MMAIConfig,
    max_attempts: int,
) -> pd.DataFrame:
    pending = batch.copy()
    resolved: list[pd.DataFrame] = []
    last: pd.DataFrame | None = None
    terminal_statuses = {
        "ok",
        "no_scoreable_drug",
        "drug_research_blocked",
        "trial_registry_blocked",
        "missing_catalog_trial",
        "missing_drug_summary",
    }
    for _attempt in range(1, max(1, max_attempts) + 1):
        pairs = pending.rename(
            columns={
                "nct_id": "trial_id",
                "patient_summary": "cancer_history_summary",
            }
        )[["patient_id", "trial_id", "cancer_history_summary"]]
        scored = score_good_options_with_llm(pairs, catalog=catalog, config=config)
        last = scored
        valid = scored["good_option_status"].isin(terminal_statuses)
        if valid.any():
            resolved.append(scored.loc[valid].copy())
        invalid_keys = {
            (str(row.patient_id), str(row.trial_id))
            for row in scored.loc[~valid].itertuples(index=False)
        }
        if not invalid_keys:
            break
        pending = pending.loc[
            [
                (str(row.patient_id), str(row.nct_id)) in invalid_keys
                for row in pending.itertuples(index=False)
            ]
        ].copy()
    if last is not None:
        resolved_keys = {
            (str(row.patient_id), str(row.trial_id))
            for frame in resolved
            for row in frame.itertuples(index=False)
        }
        final_unresolved = last.loc[
            [
                (str(row.patient_id), str(row.trial_id)) not in resolved_keys
                for row in last.itertuples(index=False)
            ]
        ]
        if not final_unresolved.empty:
            resolved.append(final_unresolved)
    return pd.concat(resolved, ignore_index=True) if resolved else pd.DataFrame()


def run_label(args: argparse.Namespace) -> None:
    catalog = load_good_option_catalog(args.catalog)
    candidates = load_candidates(
        args.candidate_files,
        confirm_inputs_are_non_phi=args.confirm_inputs_are_non_phi,
        max_candidates=args.max_candidates,
    )
    shards_dir = Path(args.label_shards_dir).expanduser().resolve()
    shards_dir.mkdir(parents=True, exist_ok=True)
    completed = _existing_label_ids(shards_dir, catalog=catalog)
    pending = candidates.loc[~candidates["candidate_id"].isin(completed)].copy()
    config = configure_teacher(args)
    config.debug_mode = True
    next_index = len(list(shards_dir.glob("labels_*.parquet")))
    for start in range(0, len(pending), args.submission_batch_size):
        batch = pending.iloc[start : start + args.submission_batch_size].copy()
        scored = _label_batch_with_retries(
            batch,
            catalog=catalog,
            config=config,
            max_attempts=args.label_parse_attempts,
        )
        serialized = _serialize_label_frame(batch, scored, catalog=catalog)
        shard = shards_dir / f"labels_{next_index:06d}.parquet"
        _write_parquet_atomic(serialized, shard)
        next_index += 1
        print(
            f"[label] {min(start + len(batch), len(pending))}/{len(pending)}",
            flush=True,
        )
    shards = sorted(shards_dir.glob("labels_*.parquet"))
    if not shards:
        raise ValueError("No label shards were produced.")
    aggregate = pd.concat((pd.read_parquet(path) for path in shards), ignore_index=True)
    aggregate = aggregate.sort_values("candidate_id", kind="stable").drop_duplicates(
        "candidate_id", keep="last"
    )
    _write_parquet_atomic(aggregate, Path(args.label_output).expanduser().resolve())
    print(f"Saved {len(aggregate)} labels to {args.label_output}.")


def flatten_patient_drug_labels(
    labels: pd.DataFrame, catalog: GoodOptionCatalog
) -> pd.DataFrame:
    """Explode valid trial labels to unique patient-drug four-target rows."""

    required = {
        "candidate_id",
        "patient_id",
        "patient_group_id",
        "patient_summary",
        "nct_id",
        "good_option_status",
        "drug_assessments_json",
        "catalog_compatibility_id",
        "prompt_version",
        "label_schema_version",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Label data is missing columns: {missing}")
    incompatible = labels["catalog_compatibility_id"].astype(str).ne(
        catalog.compatibility_id
    )
    if incompatible.any():
        raise ValueError("Labels and catalog have different compatibility IDs.")
    if not labels["prompt_version"].astype(str).eq(GOOD_OPTION_PROMPT_VERSION).all():
        raise ValueError("Labels use an incompatible GoodOption prompt version.")
    if not labels["label_schema_version"].astype(str).eq(
        GOOD_OPTION_LABEL_SCHEMA_VERSION
    ).all():
        raise ValueError("Labels use an incompatible GoodOption label schema.")
    records: list[dict[str, Any]] = []
    for row in labels.loc[labels["good_option_status"].eq("ok")].to_dict(
        orient="records"
    ):
        assignments = catalog.assignments_for_trial(
            str(row["nct_id"]), scoreable_only=True
        )
        by_name = {item.preferred_name.casefold(): item for item in assignments}
        assessments = json.loads(str(row["drug_assessments_json"] or "[]"))
        if len(assessments) != len(assignments):
            raise ValueError(
                f"{row['candidate_id']}: assessment count does not match catalog drugs."
            )
        for assessment in assessments:
            drug_name = str(assessment.get("drug_name") or "")
            assignment = by_name.get(drug_name.casefold())
            if assignment is None:
                raise ValueError(
                    f"{row['candidate_id']}: unknown scoreable drug {drug_name!r}."
                )
            summary = catalog.summary_for_drug(assignment.drug_id)
            if summary is None or summary.synthesis_status != "ok":
                raise ValueError(f"Missing completed summary for {assignment.drug_id}.")
            target = []
            rationales: dict[str, str] = {}
            for criterion in RUBRIC_CRITERIA:
                value = assessment.get(criterion)
                if not isinstance(value, Mapping) or value.get("point") not in {0, 1}:
                    raise ValueError(
                        f"{row['candidate_id']} {drug_name}: invalid {criterion}."
                    )
                target.append(int(value["point"]))
                rationales[criterion] = str(value.get("rationale") or "")
            patient_drug_id = _hash_text(
                b"GoodOption patient drug v2",
                row["patient_group_id"],
                assignment.drug_id,
            )
            records.append(
                {
                    "patient_drug_id": patient_drug_id,
                    "patient_id": str(row["patient_id"]),
                    "patient_group_id": str(row["patient_group_id"]),
                    "patient_summary": str(row["patient_summary"]),
                    "drug_id": assignment.drug_id,
                    "drug_name": summary.preferred_name,
                    "drug_summary": summary.good_option_summary,
                    "labels": target,
                    "labels_json": json.dumps(target),
                    "rationales_json": json.dumps(rationales, ensure_ascii=False),
                    "checker_text": build_good_option_checker_text(
                        str(row["patient_summary"]), summary
                    ),
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No valid patient-drug labels remain.")
    conflicts = frame.groupby("patient_drug_id", sort=False)["labels_json"].nunique().gt(1)
    if conflicts.any():
        ids = conflicts[conflicts].index.tolist()
        raise ValueError(f"Conflicting duplicate patient-drug labels: {ids[:10]}")
    return (
        frame.sort_values("patient_drug_id", kind="stable")
        .drop_duplicates("patient_drug_id", keep="first")
        .reset_index(drop=True)
    )


def _held_out(value: str, *, seed: int, fraction: float, namespace: bytes) -> bool:
    if fraction <= 0:
        return False
    integer = int(_hash_text(namespace, seed, value)[:16], 16)
    return integer / float(16**16) < fraction


def assign_partitions(
    frame: pd.DataFrame,
    *,
    strategy: str,
    patient_validation_fraction: float,
    drug_validation_fraction: float,
    seed: int,
) -> pd.Series:
    """Create train/unseen-patient/unseen-drug/strict partitions."""

    if strategy == "none":
        return pd.Series("train", index=frame.index, dtype="string")
    if strategy != "patient_drug":
        raise ValueError("split strategy must be patient_drug or none.")
    if not 0 <= patient_validation_fraction < 1 or not 0 <= drug_validation_fraction < 1:
        raise ValueError("Validation fractions must be in [0, 1).")
    patient_holdout = frame["patient_group_id"].map(
        lambda value: _held_out(
            str(value),
            seed=seed,
            fraction=patient_validation_fraction,
            namespace=b"GoodOption patient split",
        )
    )
    drug_holdout = frame["drug_id"].map(
        lambda value: _held_out(
            str(value),
            seed=seed,
            fraction=drug_validation_fraction,
            namespace=b"GoodOption drug split",
        )
    )
    return pd.Series(
        np.select(
            [patient_holdout & drug_holdout, patient_holdout, drug_holdout],
            ["strict_validation", "unseen_patient", "unseen_drug"],
            default="train",
        ),
        index=frame.index,
        dtype="string",
    )


def build_split_manifest(
    frame: pd.DataFrame,
    *,
    strategy: str,
    seed: int,
    patient_validation_fraction: float,
    drug_validation_fraction: float,
    catalog: GoodOptionCatalog,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for row in frame[["patient_drug_id", "partition"]].sort_values(
        "patient_drug_id", kind="stable"
    ).itertuples(index=False):
        digest.update(f"{row.patient_drug_id}\0{row.partition}\n".encode())
    return {
        "split_strategy": strategy,
        "split_strategy_version": SPLIT_STRATEGY_VERSION,
        "split_fingerprint_sha256": digest.hexdigest(),
        "seed": seed,
        "patient_validation_fraction": patient_validation_fraction,
        "drug_validation_fraction": drug_validation_fraction,
        "catalog_compatibility_id": catalog.compatibility_id,
        "counts": {
            str(key): int(value)
            for key, value in frame["partition"].value_counts().items()
        },
        "patients": int(frame["patient_group_id"].nunique()),
        "drugs": int(frame["drug_id"].nunique()),
        "rows": len(frame),
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def compute_metrics(evaluation: Any) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    logits = np.asarray(evaluation.predictions)
    labels = np.asarray(evaluation.label_ids)
    probabilities = _sigmoid(logits)
    metrics: dict[str, float] = {
        "binary_accuracy": float(np.mean((probabilities >= 0.5) == labels)),
        "brier": float(np.mean(np.square(probabilities - labels))),
    }
    aucs: list[float] = []
    for index, criterion in enumerate(RUBRIC_CRITERIA):
        if np.unique(labels[:, index]).size == 2:
            auc = float(roc_auc_score(labels[:, index], probabilities[:, index]))
            metrics[f"auroc_{criterion}"] = auc
            aucs.append(auc)
        else:
            metrics[f"auroc_{criterion}"] = float("nan")
    metrics["auroc_macro"] = float(np.mean(aucs)) if aucs else float("nan")
    return metrics


def validate_resume_checkpoint(
    checkpoint: str | Path,
    *,
    catalog: GoodOptionCatalog,
    split_manifest: Mapping[str, Any],
) -> None:
    """Reject checkpoints produced for a different catalog or data contract."""

    checkpoint_dir = Path(checkpoint).expanduser().resolve()
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(
            f"Resume checkpoint is missing its model config: {config_path}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "matchminer_catalog_compatibility_id": catalog.compatibility_id,
        "matchminer_checker_input_version": GOOD_OPTION_INPUT_VERSION,
        "matchminer_label_schema_version": GOOD_OPTION_LABEL_SCHEMA_VERSION,
        "matchminer_split_fingerprint_sha256": split_manifest[
            "split_fingerprint_sha256"
        ],
    }
    mismatches = {
        key: {"checkpoint": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Resume checkpoint is incompatible with this GoodOption run: "
            + json.dumps(mismatches, sort_keys=True)
        )


def run_train(args: argparse.Namespace) -> None:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    catalog = load_good_option_catalog(args.catalog)
    labels = pd.read_parquet(args.label_output)
    frame = flatten_patient_drug_labels(labels, catalog)
    frame["partition"] = assign_partitions(
        frame,
        strategy=args.split_strategy,
        patient_validation_fraction=args.patient_validation_fraction,
        drug_validation_fraction=args.drug_validation_fraction,
        seed=args.seed,
    )
    if not frame["partition"].eq("train").any():
        raise ValueError("The configured holdouts left no training rows.")
    manifest = build_split_manifest(
        frame,
        strategy=args.split_strategy,
        seed=args.seed,
        patient_validation_fraction=args.patient_validation_fraction,
        drug_validation_fraction=args.drug_validation_fraction,
        catalog=catalog,
    )
    if args.resume_from_checkpoint:
        validate_resume_checkpoint(
            args.resume_from_checkpoint,
            catalog=catalog,
            split_manifest=manifest,
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=4,
        problem_type="multi_label_classification",
        id2label={index: criterion for index, criterion in enumerate(RUBRIC_CRITERIA)},
        label2id={criterion: index for index, criterion in enumerate(RUBRIC_CRITERIA)},
        trust_remote_code=True,
    )
    model.config.matchminer_catalog_compatibility_id = catalog.compatibility_id
    model.config.matchminer_checker_input_version = GOOD_OPTION_INPUT_VERSION
    model.config.matchminer_label_schema_version = GOOD_OPTION_LABEL_SCHEMA_VERSION
    model.config.matchminer_split_fingerprint_sha256 = manifest[
        "split_fingerprint_sha256"
    ]

    def make_dataset(partition: str) -> Dataset:
        selected = frame.loc[
            frame["partition"].eq(partition), ["checker_text", "labels"]
        ]
        dataset = Dataset.from_pandas(selected, preserve_index=False)

        def tokenize(batch: Mapping[str, Sequence[Any]]) -> Mapping[str, Any]:
            encoded = tokenizer(
                list(batch["checker_text"]),
                truncation=True,
                max_length=args.max_length,
            )
            encoded["labels"] = [list(map(float, value)) for value in batch["labels"]]
            return encoded

        return dataset.map(tokenize, batched=True, remove_columns=["checker_text"])

    train_dataset = make_dataset("train")
    evaluation_partitions = [
        partition
        for partition in ("strict_validation", "unseen_patient", "unseen_drug")
        if frame["partition"].eq(partition).any()
    ]
    eval_dataset = make_dataset(evaluation_partitions[0]) if evaluation_partitions else None
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="brier",
        greater_is_better=False,
        bf16=args.bf16,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    evaluation_metrics: dict[str, Any] = {}
    for partition in evaluation_partitions:
        evaluation_metrics[partition] = trainer.evaluate(
            make_dataset(partition), metric_key_prefix=partition
        )
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(evaluation_metrics, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Saved four-logit GoodOptionChecker to {output_dir}.")


def _teacher_cli_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--config",
        str(args.config),
        "--model",
        str(args.model),
        "--server-urls",
        str(args.server_urls),
        "--server-urls-file",
        str(args.server_urls_file),
        "--llm-request-timeout",
        str(args.llm_request_timeout),
        "--max-concurrent-requests",
        str(args.max_concurrent_requests),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
    ]


def _build_all_stage_commands(
    args: argparse.Namespace,
) -> list[tuple[str, list[str]]]:
    """Build isolated catalog, validation, labeling, and training commands."""

    python = sys.executable
    script = str(Path(__file__).resolve())

    catalog = [
        python,
        script,
        "catalog",
        "--catalog",
        str(args.catalog),
        "--candidate-files",
        *(str(path) for path in args.candidate_files),
    ]
    for nct_id in args.nct_id:
        catalog.extend(("--nct-id", str(nct_id)))
    if args.nct_ids_file:
        catalog.extend(("--nct-ids-file", str(args.nct_ids_file)))
    if args.overwrite:
        catalog.append("--overwrite")
    catalog.extend(_teacher_cli_arguments(args))
    for flag, value in (
        ("--research-request-timeout", args.research_request_timeout),
        ("--max-source-attempts", args.max_source_attempts),
        ("--registry-max-attempts", args.registry_max_attempts),
        ("--initial-backoff", args.initial_backoff),
        ("--maximum-backoff", args.maximum_backoff),
        ("--research-concurrency", args.research_concurrency),
        ("--web-results-per-query", args.web_results_per_query),
        ("--max-web-results-per-drug", args.max_web_results_per_drug),
        ("--max-web-documents-per-drug", args.max_web_documents_per_drug),
        ("--max-pubmed-records", args.max_pubmed_records),
        ("--max-registry-studies", args.max_registry_studies),
        ("--max-civic-records", args.max_civic_records),
        ("--max-europe-pmc-records", args.max_europe_pmc_records),
        ("--max-regulatory-records", args.max_regulatory_records),
        ("--max-passage-chars", args.max_passage_chars),
    ):
        catalog.extend((flag, str(value)))

    validate = [
        python,
        script,
        "validate-catalog",
        "--catalog",
        str(args.catalog),
    ]

    label = [
        python,
        script,
        "label",
        "--catalog",
        str(args.catalog),
        "--label-output",
        str(args.label_output),
        "--label-shards-dir",
        str(args.label_shards_dir),
        "--submission-batch-size",
        str(args.submission_batch_size),
        "--label-parse-attempts",
        str(args.label_parse_attempts),
        "--candidate-files",
        *(str(path) for path in args.candidate_files),
    ]
    if args.max_candidates is not None:
        label.extend(("--max-candidates", str(args.max_candidates)))
    if args.confirm_inputs_are_non_phi:
        label.append("--confirm-inputs-are-non-phi")
    label.extend(_teacher_cli_arguments(args))

    train = [
        python,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(args.num_processes),
        script,
        "train",
        "--catalog",
        str(args.catalog),
        "--label-output",
        str(args.label_output),
        "--base-model",
        str(args.base_model),
        "--output-dir",
        str(args.output_dir),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--split-strategy",
        str(args.split_strategy),
        "--patient-validation-fraction",
        str(args.patient_validation_fraction),
        "--drug-validation-fraction",
        str(args.drug_validation_fraction),
        "--seed",
        str(args.seed),
        "--max-length",
        str(args.max_length),
        "--learning-rate",
        str(args.learning_rate),
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
        "--per-device-eval-batch-size",
        str(args.per_device_eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--num-train-epochs",
        str(args.num_train_epochs),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--logging-steps",
        str(args.logging_steps),
        "--bf16" if args.bf16 else "--no-bf16",
    ]
    if args.resume_from_checkpoint:
        train.extend(("--resume-from-checkpoint", str(args.resume_from_checkpoint)))

    return [
        ("catalog", catalog),
        ("validate-catalog", validate),
        ("label", label),
        ("train", train),
    ]


def run_all(args: argparse.Namespace) -> None:
    """Run the complete workflow in isolated, fail-fast subprocesses."""

    commands = _build_all_stage_commands(args)
    for index, (stage, command) in enumerate(commands, start=1):
        print(f"[all] {index}/{len(commands)}: {stage}", flush=True)
        subprocess.run(command, check=True)
    print("[all] GoodOptionChecker workflow complete.", flush=True)


def add_candidate_files_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidate-files",
        nargs="+",
        default=[str(path) for path in DEFAULT_CANDIDATE_FILES],
    )


def add_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    add_candidate_files_argument(parser)
    parser.add_argument("--confirm-inputs-are-non-phi", action="store_true")


def add_teacher_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="")
    parser.add_argument("--model", default="nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--server-urls", default="")
    parser.add_argument("--server-urls-file", default="")
    parser.add_argument("--llm-request-timeout", type=float, default=7200.0)
    parser.add_argument("--max-concurrent-requests", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)


def add_research_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--research-request-timeout", type=float, default=30.0)
    parser.add_argument("--max-source-attempts", type=int, default=5)
    parser.add_argument("--registry-max-attempts", type=int, default=10)
    parser.add_argument("--initial-backoff", type=float, default=1.0)
    parser.add_argument("--maximum-backoff", type=float, default=60.0)
    parser.add_argument("--research-concurrency", type=int, default=6)
    parser.add_argument("--web-results-per-query", type=int, default=10)
    parser.add_argument("--max-web-results-per-drug", type=int, default=60)
    parser.add_argument("--max-web-documents-per-drug", type=int, default=24)
    parser.add_argument("--max-pubmed-records", type=int, default=40)
    parser.add_argument("--max-registry-studies", type=int, default=25)
    parser.add_argument("--max-civic-records", type=int, default=40)
    parser.add_argument("--max-europe-pmc-records", type=int, default=12)
    parser.add_argument("--max-regulatory-records", type=int, default=8)
    parser.add_argument("--max-passage-chars", type=int, default=5000)


def add_catalog_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_DIR))
    parser.add_argument("--nct-id", action="append", default=[])
    parser.add_argument("--nct-ids-file", default="")
    parser.add_argument("--overwrite", action="store_true")
    add_candidate_files_argument(parser)
    add_teacher_arguments(parser)
    add_research_arguments(parser)


def add_label_arguments(
    parser: argparse.ArgumentParser, *, shared_catalog_arguments: bool = False
) -> None:
    if not shared_catalog_arguments:
        parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_DIR))
    parser.add_argument("--label-output", default=str(DEFAULT_LABEL_OUTPUT))
    parser.add_argument("--label-shards-dir", default=str(DEFAULT_LABEL_SHARDS))
    parser.add_argument("--submission-batch-size", type=int, default=256)
    parser.add_argument("--label-parse-attempts", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=None)
    if shared_catalog_arguments:
        parser.add_argument("--confirm-inputs-are-non-phi", action="store_true")
    else:
        add_candidate_arguments(parser)
        add_teacher_arguments(parser)


def add_train_arguments(
    parser: argparse.ArgumentParser, *, shared_catalog_arguments: bool = False
) -> None:
    if not shared_catalog_arguments:
        parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_DIR))
        parser.add_argument("--label-output", default=str(DEFAULT_LABEL_OUTPUT))
    parser.add_argument("--base-model", default="answerdotai/ModernBERT-large")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument(
        "--split-strategy", choices=("patient_drug", "none"), default="patient_drug"
    )
    parser.add_argument("--patient-validation-fraction", type=float, default=0.20)
    parser.add_argument("--drug-validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="Build all patient-free research.")
    add_catalog_arguments(catalog)
    catalog.set_defaults(handler=run_catalog)

    validate = subparsers.add_parser("validate-catalog")
    validate.add_argument("--catalog", default=str(DEFAULT_CATALOG_DIR))
    validate.set_defaults(handler=run_validate_catalog)

    label = subparsers.add_parser("label")
    add_label_arguments(label)
    label.set_defaults(handler=run_label)

    train = subparsers.add_parser("train")
    add_train_arguments(train)
    train.set_defaults(handler=run_train)

    all_steps = subparsers.add_parser(
        "all", help="Run catalog, validation, labeling, and training in sequence."
    )
    add_catalog_arguments(all_steps)
    add_label_arguments(all_steps, shared_catalog_arguments=True)
    add_train_arguments(all_steps, shared_catalog_arguments=True)
    all_steps.add_argument("--num-processes", type=positive_int, default=8)
    all_steps.set_defaults(handler=run_all)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
