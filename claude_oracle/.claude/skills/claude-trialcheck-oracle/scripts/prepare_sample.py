#!/usr/bin/env python3
"""Sample rows from a parquet or CSV of patient-trial candidate matches.

Mirrors the preprocessing in llm_check_trials.py:
  * Drop rows where patient_summary is null.
  * Strip a leading "N." numbering from this_space.
Then randomly samples n rows with the given seed and writes:
  * <work_dir>/staging.parquet  — full sampled dataframe + __row_id__
  * <work_dir>/staging.jsonl    — one JSON object per sampled row
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SKILL_DIR = Path(__file__).resolve().parent.parent


def reject_if_inside_skill(path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(SKILL_DIR)
    except ValueError:
        return
    sys.exit(
        f"ERROR: {label} ({path}) is inside the skill directory ({SKILL_DIR}). "
        "Choose a location outside the skill folder."
    )


def load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    sys.exit(f"ERROR: unsupported input extension {ext!r}; expected .parquet or .csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input parquet or CSV")
    ap.add_argument("--n", type=int, required=True, help="Sample size")
    ap.add_argument("--seed", type=int, required=True, help="Random seed")
    ap.add_argument("--work_dir", required=True, help="Directory for staging files")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        sys.exit(f"ERROR: input file not found: {input_path}")

    work_dir = Path(args.work_dir).resolve()
    reject_if_inside_skill(work_dir, "work_dir")
    work_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(input_path)

    for col in ("patient_summary", "this_space"):
        if col not in df.columns:
            sys.exit(f"ERROR: input is missing required column {col!r}")

    df = df[~df["patient_summary"].isnull()].reset_index(drop=True)
    if len(df) == 0:
        sys.exit("ERROR: input has no rows after filtering null patient_summary.")

    df["this_space"] = df["this_space"].astype(str).str.replace(
        r"^\s*\d+\.", "", regex=True
    )

    n = min(args.n, len(df))
    sampled = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    sampled.insert(0, "__row_id__", range(len(sampled)))

    staging_parquet = work_dir / "staging.parquet"
    staging_jsonl = work_dir / "staging.jsonl"

    sampled.to_parquet(staging_parquet, index=False)

    with staging_jsonl.open("w") as f:
        for rec in sampled.to_dict(orient="records"):
            clean = {
                k: (None if isinstance(v, float) and pd.isna(v) else v)
                for k, v in rec.items()
            }
            f.write(json.dumps(clean, default=str) + "\n")

    print(f"wrote {len(sampled)} rows")
    print(f"  {staging_parquet}")
    print(f"  {staging_jsonl}")


if __name__ == "__main__":
    main()
