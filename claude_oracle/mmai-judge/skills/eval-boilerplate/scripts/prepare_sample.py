#!/usr/bin/env python3
"""Sample MM-AI boilerplate-check outputs and build per-row judge context.

Required input columns:
  - patient_boilerplate_text
  - trial_boilerplate_text
  - an MM-AI reasoning column (auto-detected; user may override)
  - an MM-AI verdict column (auto-detected; user may override)

Verdict column may be a string ("Yes!"/"No!") or a numeric 0.0/1.0. The skill
normalizes it to a human-readable string in the slim row file.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


SKILL_DIR = Path(__file__).resolve().parent.parent
REASONING_ALIASES = (
    "mm_ai_reasoning",
    "boilerplate_check_llm_response",
    "llm_boilerplate_response",
    "llama_response",
    "llm_response",
)
VERDICT_ALIASES = (
    "mm_ai_verdict",
    "exclusion_result",
    "boilerplate_verdict",
)


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


def pick_column(df: pd.DataFrame, explicit: str | None, aliases: tuple[str, ...], label: str) -> str:
    if explicit:
        if explicit not in df.columns:
            sys.exit(f"ERROR: {label} column {explicit!r} not present in input.")
        return explicit
    for name in aliases:
        if name in df.columns:
            return name
    sys.exit(
        f"ERROR: could not find a {label} column. Tried {list(aliases)}. "
        f"Pass --{label}_col <name> explicitly."
    )


def render_verdict(v) -> str:
    if v is None:
        return "(missing)"
    if isinstance(v, (int, float)):
        if pd.isna(v):
            return "(missing)"
        return "Yes!" if float(v) >= 0.5 else "No!"
    s = str(v).strip()
    if s.lower() in ("nan", "none", ""):
        return "(missing)"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--reasoning_col", default=None)
    ap.add_argument("--verdict_col", default=None)
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

    reasoning_col = pick_column(df, args.reasoning_col, REASONING_ALIASES, "reasoning")
    verdict_col = pick_column(df, args.verdict_col, VERDICT_ALIASES, "verdict")

    df = df[~df["patient_boilerplate_text"].isnull()].reset_index(drop=True)
    df = df[~df["trial_boilerplate_text"].isnull()].reset_index(drop=True)
    df = df[~df[reasoning_col].isnull()].reset_index(drop=True)
    if len(df) == 0:
        sys.exit("ERROR: no rows remain after filtering null required columns.")

    n = min(args.n, len(df))
    sampled = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    sampled.insert(0, "__row_id__", range(len(sampled)))

    staging_parquet = work_dir / "staging.parquet"
    sampled.to_parquet(staging_parquet, index=False)
    (work_dir / "cols.txt").write_text(
        f"reasoning_col={reasoning_col}\nverdict_col={verdict_col}\n"
    )

    rows_dir = work_dir / "rows"
    if rows_dir.exists():
        for old in rows_dir.glob("row_*.txt"):
            old.unlink()
    rows_dir.mkdir(parents=True, exist_ok=True)

    width = max(2, len(str(max(len(sampled) - 1, 0))))
    for rec in sampled.to_dict(orient="records"):
        rid = int(rec["__row_id__"])
        slim = rows_dir / f"row_{rid:0{width}d}.txt"
        slim.write_text(
            "=== PATIENT BOILERPLATE EXTRACT ===\n"
            f"{rec.get('patient_boilerplate_text') or ''}\n\n"
            "=== TRIAL BOILERPLATE EXCLUSIONS ===\n"
            f"{rec.get('trial_boilerplate_text') or ''}\n\n"
            "=== MM-AI REASONING ===\n"
            f"{rec.get(reasoning_col) or ''}\n\n"
            "=== MM-AI VERDICT ===\n"
            f"{render_verdict(rec.get(verdict_col))}\n"
        )

    print(f"wrote {len(sampled)} rows")
    print(f"  reasoning column: {reasoning_col}")
    print(f"  verdict column: {verdict_col}")
    print(f"  {staging_parquet}")
    print(f"  {rows_dir}/row_*.txt  ({len(sampled)} files)")


if __name__ == "__main__":
    main()
