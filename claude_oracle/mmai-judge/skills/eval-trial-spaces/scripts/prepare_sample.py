#!/usr/bin/env python3
"""Sample trials and build per-row judge context for trial-space extraction.

Accepts two shapes:
  (a) one row per trial, with nct_id + trial_text + extracted_spaces (JSON list,
      newline-separated, or semicolon-separated string), OR
  (b) one row per space (lineitems shape) with nct_id + trial_text + this_space,
      in which case the skill aggregates to one row per trial.

Always normalizes the extracted-spaces list into a numbered block for the slim
row file.
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
    sys.exit(f"ERROR: unsupported extension {ext!r}; expected .parquet or .csv")


def split_spaces(raw) -> list[str]:
    """Normalize extracted_spaces to a list of non-empty strings.

    Accepts a list, a JSON string, a newline-separated string, or a
    semicolon-separated string.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    if not isinstance(raw, str):
        raw = str(raw)
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except json.JSONDecodeError:
            pass
    if "\n" in raw:
        parts = [p.strip() for p in raw.split("\n")]
    else:
        parts = [p.strip() for p in raw.split(";")]
    return [p for p in parts if p]


def aggregate_lineitems(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for nct_id, group in df.groupby("nct_id", sort=False):
        if "space_number" in group.columns:
            group = group.sort_values("space_number", kind="mergesort")
        trial_text = (
            group["trial_text"].dropna().iloc[0] if group["trial_text"].notna().any() else ""
        )
        spaces = [str(s).strip() for s in group["this_space"].tolist() if str(s).strip()]
        rows.append({"nct_id": nct_id, "trial_text": trial_text, "extracted_spaces": spaces})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--work_dir", required=True)
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        sys.exit(f"ERROR: input file not found: {input_path}")

    work_dir = Path(args.work_dir).resolve()
    reject_if_inside_skill(work_dir, "work_dir")
    work_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(input_path)

    if "nct_id" not in df.columns:
        sys.exit("ERROR: input is missing required column 'nct_id'.")
    if "trial_text" not in df.columns:
        sys.exit("ERROR: input is missing required column 'trial_text'.")

    if "extracted_spaces" in df.columns:
        df["extracted_spaces"] = df["extracted_spaces"].apply(split_spaces)
    elif "this_space" in df.columns:
        df = aggregate_lineitems(df)
    else:
        sys.exit(
            "ERROR: input must have either 'extracted_spaces' (pre-aggregated) "
            "or 'this_space' (one row per space, will be aggregated by nct_id)."
        )

    df = df[df["extracted_spaces"].map(len) > 0].reset_index(drop=True)
    df = df[~df["trial_text"].isnull()].reset_index(drop=True)
    if len(df) == 0:
        sys.exit("ERROR: no trials with non-empty trial_text and spaces.")

    n = min(args.n, len(df))
    sampled = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    sampled.insert(0, "__row_id__", range(len(sampled)))

    # Store the numbered block as a string column on staging for the final output.
    def _block(spaces: list[str]) -> str:
        return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(spaces))

    sampled["spaces_numbered"] = sampled["extracted_spaces"].apply(_block)

    staging_parquet = work_dir / "staging.parquet"
    # Parquet doesn't love list columns under all engines — serialize to JSON
    # for durable storage and keep spaces_numbered for display.
    sampled_to_save = sampled.copy()
    sampled_to_save["extracted_spaces"] = sampled_to_save["extracted_spaces"].apply(
        lambda xs: json.dumps(list(xs), ensure_ascii=False)
    )
    sampled_to_save.to_parquet(staging_parquet, index=False)

    rows_dir = work_dir / "rows"
    if rows_dir.exists():
        for old in rows_dir.glob("row_*.txt"):
            old.unlink()
    rows_dir.mkdir(parents=True, exist_ok=True)

    width = max(2, len(str(max(len(sampled) - 1, 0))))
    for rec in sampled.to_dict(orient="records"):
        rid = int(rec["__row_id__"])
        trial_text = rec.get("trial_text") or ""
        block = rec.get("spaces_numbered") or ""
        slim = rows_dir / f"row_{rid:0{width}d}.txt"
        slim.write_text(
            "=== TRIAL TEXT ===\n"
            f"{trial_text}\n\n"
            "=== MM-AI EXTRACTED SPACES ===\n"
            f"{block}\n"
        )

    print(f"wrote {len(sampled)} rows")
    print(f"  {staging_parquet}")
    print(f"  {rows_dir}/row_*.txt  ({len(sampled)} files)")


if __name__ == "__main__":
    main()
