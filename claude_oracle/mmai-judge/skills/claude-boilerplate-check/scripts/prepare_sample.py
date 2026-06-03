#!/usr/bin/env python3
"""Sample rows from a parquet or CSV of patient/trial boilerplate pairs.

Mirrors the preprocessing in 14_check_boilerplate.py:
  * Drop rows where patient_boilerplate_text or trial_boilerplate_text is
    null or empty string.
Then randomly samples n rows with the given seed and writes:
  * <work_dir>/staging.parquet     — full sampled dataframe + __row_id__
  * <work_dir>/staging.jsonl       — one JSON object per sampled row
  * <work_dir>/rows/row_NN.txt     — slim per-row text file with just the
                                     patient_boilerplate_text and
                                     trial_boilerplate_text blocks the
                                     model needs to score the row. Keeping
                                     these inside work_dir (rather than
                                     /tmp) lets existing Read rules that
                                     cover work_dir auto-grant the per-row
                                     reads with no extra approvals.
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

    for col in ("patient_boilerplate_text", "trial_boilerplate_text"):
        if col not in df.columns:
            sys.exit(f"ERROR: input is missing required column {col!r}")

    df = df[~df["patient_boilerplate_text"].isnull()]
    df = df[df["patient_boilerplate_text"].astype(str).str.strip() != ""]
    df = df[~df["trial_boilerplate_text"].isnull()]
    df = df[df["trial_boilerplate_text"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    if len(df) == 0:
        sys.exit("ERROR: input has no rows after filtering null/empty boilerplate columns.")

    n = min(args.n, len(df))
    sampled = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    sampled.insert(0, "__row_id__", range(len(sampled)))

    staging_parquet = work_dir / "staging.parquet"
    staging_jsonl = work_dir / "staging.jsonl"
    rows_dir = work_dir / "rows"

    sampled.to_parquet(staging_parquet, index=False)

    with staging_jsonl.open("w") as f:
        for rec in sampled.to_dict(orient="records"):
            clean = {
                k: (None if isinstance(v, float) and pd.isna(v) else v)
                for k, v in rec.items()
            }
            f.write(json.dumps(clean, default=str) + "\n")

    # Fresh per-row slim text files for the model to read one at a time.
    if rows_dir.exists():
        for old in rows_dir.glob("row_*.txt"):
            old.unlink()
    rows_dir.mkdir(parents=True, exist_ok=True)

    width = max(2, len(str(max(len(sampled) - 1, 0))))
    for rec in sampled.to_dict(orient="records"):
        rid = int(rec["__row_id__"])
        patient = rec.get("patient_boilerplate_text") or ""
        trial = rec.get("trial_boilerplate_text") or ""
        slim_path = rows_dir / f"row_{rid:0{width}d}.txt"
        slim_path.write_text(
            "=== PATIENT BOILERPLATE EXTRACT ===\n"
            f"{patient}\n\n"
            "=== TRIAL BOILERPLATE EXCLUSIONS ===\n"
            f"{trial}\n"
        )

    print(f"wrote {len(sampled)} rows")
    print(f"  {staging_parquet}")
    print(f"  {staging_jsonl}")
    print(f"  {rows_dir}/row_*.txt  ({len(sampled)} files)")


if __name__ == "__main__":
    main()
