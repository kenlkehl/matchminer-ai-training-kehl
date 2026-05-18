#!/usr/bin/env python
"""Prepare synthetic extractor inputs from the combined synthetic notes file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent.parent / "data" / "no_phi"
DEFAULT_INPUT = DATA_DIR / "all_synthetic_notes.parquet"
DEFAULT_IMAGING_OUTPUT = DATA_DIR / "synthetic_imaging.parquet"
DEFAULT_CLINICAL_OUTPUT = DATA_DIR / "synthetic_clinical.parquet"

REQUIRED_COLUMNS = {"event_type", "pseudo_mrn", "row_id", "synthetic_note", "split"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter all_synthetic_notes.parquet into imaging and clinical-note extractor inputs."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Combined synthetic notes parquet.")
    parser.add_argument(
        "--imaging-output",
        default=str(DEFAULT_IMAGING_OUTPUT),
        help="Output parquet for synthetic imaging reports.",
    )
    parser.add_argument(
        "--clinical-output",
        default=str(DEFAULT_CLINICAL_OUTPUT),
        help="Output parquet for synthetic clinical oncology notes.",
    )
    return parser.parse_args()


def validate_input(frame: pd.DataFrame, input_path: Path) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise SystemExit(f"{input_path} is missing required column(s): {', '.join(missing)}")


def write_subset(frame: pd.DataFrame, event_type: str, output_path: Path) -> pd.DataFrame:
    subset = frame[frame["event_type"].eq(event_type)].reset_index(drop=True)
    if subset.empty:
        raise SystemExit(f"No rows found with event_type={event_type!r}; refusing to write {output_path}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output_path, index=False)
    return subset


def print_summary(name: str, frame: pd.DataFrame, path: Path) -> None:
    print(f"{name}: wrote {len(frame)} rows to {path}")
    if "split" in frame.columns:
        counts = frame["split"].value_counts(dropna=False).sort_index()
        for split, count in counts.items():
            print(f"  {split}: {count}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    imaging_output = Path(args.imaging_output)
    clinical_output = Path(args.clinical_output)

    if not input_path.exists():
        raise SystemExit(f"Input parquet not found: {input_path}")

    frame = pd.read_parquet(input_path)
    validate_input(frame, input_path)

    imaging = write_subset(frame, "imaging_report", imaging_output)
    clinical = write_subset(frame, "clinical_note", clinical_output)

    print_summary("imaging_report", imaging, imaging_output)
    print_summary("clinical_note", clinical, clinical_output)


if __name__ == "__main__":
    main()
