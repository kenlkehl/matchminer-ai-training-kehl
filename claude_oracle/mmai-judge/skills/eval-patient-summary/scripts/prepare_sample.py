#!/usr/bin/env python3
"""Sample patient summaries and build per-row judge context from raw notes.

Takes two inputs — a summaries file (one row per patient summary) and a notes
file (one row per clinical note) — samples N summary rows by seed, sorts each
patient's notes by date, concatenates them, and writes slim per-row text files
(source notes + MM-AI summary) under work_dir/rows/.

Sorts the notes dataframe by (patient_id, date) with a stable mergesort before
grouping, so tie-date notes retain their incoming order.

Truncates a per-patient note concatenation to the last ~1M tokens (approximated
as the last 4_000_000 characters).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


SKILL_DIR = Path(__file__).resolve().parent.parent
CHAR_BUDGET = 4_000_000  # rough 1M-token tail window


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
    sys.exit(f"ERROR: unsupported extension {ext!r} for {path}; expected .parquet or .csv")


def require_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {label} is missing required column(s): {missing}")


def concat_notes_per_patient(notes: pd.DataFrame, note_col: str) -> pd.DataFrame:
    notes = notes.sort_values(["patient_id", "date"], kind="mergesort").reset_index(drop=True)

    rows = []
    for patient_id, group in notes.groupby("patient_id", sort=False):
        parts = []
        for _, row in group.iterrows():
            text = row[note_col]
            if pd.isna(text):
                continue
            parts.append(f"--- NOTE {row['date']} ---\n{text}")
        rows.append({"patient_id": patient_id, "source_notes": "\n\n".join(parts)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summaries", required=True, help="Parquet/CSV with patient_id + patient_summary")
    ap.add_argument("--notes", required=True, help="Parquet/CSV with patient_id + date + note text")
    ap.add_argument("--note_col", default="note_text", help="Column name holding the note text in --notes")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--work_dir", required=True)
    args = ap.parse_args()

    summaries_path = Path(args.summaries).resolve()
    notes_path = Path(args.notes).resolve()
    if not summaries_path.is_file():
        sys.exit(f"ERROR: summaries file not found: {summaries_path}")
    if not notes_path.is_file():
        sys.exit(f"ERROR: notes file not found: {notes_path}")

    work_dir = Path(args.work_dir).resolve()
    reject_if_inside_skill(work_dir, "work_dir")
    work_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_table(summaries_path)
    notes = load_table(notes_path)

    require_cols(summaries, ["patient_id", "patient_summary", "last_date"], "summaries")
    require_cols(notes, ["patient_id", "date", args.note_col], "notes")
    has_boilerplate = "patient_boilerplate_text" in summaries.columns

    # Collapse multi-chunk summaries to the temporally last summary per patient.
    n_before = len(summaries)
    summaries["last_date"] = pd.to_datetime(summaries["last_date"], errors="coerce")
    summaries = (
        summaries.sort_values(["patient_id", "last_date"], kind="mergesort")
        .groupby("patient_id", sort=False, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    print(
        f"deduped {n_before} chunk rows -> {len(summaries)} patients "
        "(last summary by last_date)"
    )

    summaries = summaries[~summaries["patient_summary"].isnull()].reset_index(drop=True)
    if len(summaries) == 0:
        sys.exit("ERROR: summaries has no rows after filtering null patient_summary.")

    n = min(args.n, len(summaries))
    sampled = summaries.sample(n=n, random_state=args.seed).reset_index(drop=True)
    sampled.insert(0, "__row_id__", range(len(sampled)))

    needed_ids = set(sampled["patient_id"].unique())
    notes_subset = notes[notes["patient_id"].isin(needed_ids)]
    if len(notes_subset) == 0:
        sys.exit("ERROR: no notes found for any sampled patient_id.")

    per_patient = concat_notes_per_patient(notes_subset, args.note_col)
    merged = sampled.merge(per_patient, on="patient_id", how="left")

    missing_notes = merged[merged["source_notes"].isnull()]["patient_id"].tolist()
    if missing_notes:
        sys.exit(
            f"ERROR: sampled patient_id(s) have no notes: {missing_notes[:10]}"
            f"{' ...' if len(missing_notes) > 10 else ''}"
        )

    # Tail-truncate to the last ~1M-token window.
    merged["source_notes"] = merged["source_notes"].apply(
        lambda s: s[-CHAR_BUDGET:] if isinstance(s, str) and len(s) > CHAR_BUDGET else s
    )

    staging_parquet = work_dir / "staging.parquet"
    rows_dir = work_dir / "rows"
    merged.to_parquet(staging_parquet, index=False)

    if rows_dir.exists():
        for old in rows_dir.glob("row_*.txt"):
            old.unlink()
    rows_dir.mkdir(parents=True, exist_ok=True)

    width = max(2, len(str(max(len(merged) - 1, 0))))
    boilerplate_merged = 0
    for rec in merged.to_dict(orient="records"):
        rid = int(rec["__row_id__"])
        src = rec.get("source_notes") or ""
        summ = rec.get("patient_summary") or ""
        if has_boilerplate:
            bp = rec.get("patient_boilerplate_text")
            if isinstance(bp, str) and bp.strip():
                summ = f"{summ}\n\nBoilerplate:\n{bp}"
                boilerplate_merged += 1
        slim = rows_dir / f"row_{rid:0{width}d}.txt"
        slim.write_text(
            "=== SOURCE NOTES (may be truncated to last ~1M-token window) ===\n"
            f"{src}\n\n"
            "=== MM-AI SUMMARY ===\n"
            f"{summ}\n"
        )

    print(f"wrote {len(merged)} rows")
    print(f"  {staging_parquet}")
    print(f"  {rows_dir}/row_*.txt  ({len(merged)} files)")
    if has_boilerplate:
        print(f"  boilerplate column: present, merged into {boilerplate_merged}/{len(merged)} rows")
    else:
        print("  boilerplate column: absent")


if __name__ == "__main__":
    main()
