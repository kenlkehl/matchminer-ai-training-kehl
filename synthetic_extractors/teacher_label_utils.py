from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise SystemExit(f"Unsupported data file extension for {path}")


def normalize_split_values(frame: pd.DataFrame) -> pd.DataFrame:
    if "split" not in frame.columns:
        raise SystemExit("Training data must contain a split column.")
    frame = frame.copy()
    frame["split"] = frame["split"].replace(
        {
            "val": "validation",
            "valid": "validation",
            "dev": "validation",
        }
    )
    return frame


def ensure_record_id(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "record_id" in frame.columns:
        frame["record_id"] = frame["record_id"].astype(str)
        return frame
    missing = [col for col in ("pseudo_mrn", "row_id") if col not in frame.columns]
    if missing:
        raise SystemExit(
            "Expected record_id or pseudo_mrn + row_id column(s); missing "
            + ", ".join(missing)
        )
    frame["record_id"] = frame["pseudo_mrn"].astype(str) + "-" + frame["row_id"].astype(str)
    return frame


def merge_labels_with_notes(labels: pd.DataFrame, notes_path: str | Path) -> pd.DataFrame:
    notes = ensure_record_id(read_table(notes_path))
    labels = ensure_record_id(labels)
    merged = notes.merge(labels, on="record_id", how="inner", suffixes=("", "_label"))
    if merged.empty and not labels.empty:
        raise SystemExit(
            f"No label rows matched notes from {notes_path}. Check record_id / pseudo_mrn-row_id alignment."
        )
    return merged


def parse_labels_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False
    if isinstance(is_missing, bool) and is_missing:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def field_object(labels: dict[str, Any], field: str) -> dict[str, Any]:
    value = labels.get(field)
    return value if isinstance(value, dict) else {}


def field_code(labels: dict[str, Any], field: str) -> Any:
    value = labels.get(field)
    if isinstance(value, dict):
        value = value.get("code")
    return coerce_int(value)


def field_value(labels: dict[str, Any], field: str) -> Any:
    value = labels.get(field)
    if isinstance(value, dict):
        return value.get("value")
    return value


def labels_are_usable(labels: dict[str, Any]) -> bool:
    should_curate = field_value(labels, "should_curate")
    return should_curate is not False


def coerce_int(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
    return value


def cancer_presence_label(code: Any) -> float:
    if code == 1:
        return 1.0
    if code in {2, 4}:
        return 0.0
    return float("nan")


def response_label(any_cancer: float, status_code: Any) -> float:
    if pd.isna(any_cancer):
        return float("nan")
    if any_cancer == 0:
        return 0.0
    if status_code in {1, 2, 3, 4, 5}:
        return 1.0 if status_code == 1 else 0.0
    return float("nan")


def progression_label(any_cancer: float, status_code: Any) -> float:
    if pd.isna(any_cancer):
        return float("nan")
    if any_cancer == 0:
        return 0.0
    if status_code in {1, 2, 3, 4, 5}:
        return 1.0 if status_code in {3, 4} else 0.0
    return float("nan")


def split_train_val_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = normalize_split_values(frame)
    return (
        frame[frame["split"] == "train"].copy(),
        frame[frame["split"] == "validation"].copy(),
        frame[frame["split"] == "test"].copy(),
    )
