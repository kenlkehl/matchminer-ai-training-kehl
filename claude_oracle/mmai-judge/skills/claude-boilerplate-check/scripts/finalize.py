#!/usr/bin/env python3
"""Merge Claude's per-row responses into the sampled dataframe and parse verdicts.

Ports the parsing logic from 14_check_boilerplate.py (parse_yes_no_at_end)
verbatim so outputs match the format produced by that script: appends
boilerplate_check_llm_response, exclusion_result (float 0.0/1.0 matching
production), and exclusion_verdict (readable "Yes!"/"No!"/"PARSE_FAILED").
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


def parse_one(txt: str) -> tuple[float, str]:
    # Same window the production parser uses, made a little wider so a
    # trailing newline or whitespace doesn't push the token out of view.
    tail = txt[-16:].upper()
    if "YES!" in tail:
        return 1.0, "Yes!"
    if "NO!" in tail:
        return 0.0, "No!"
    # Match production fallback: bare YES / NO without the bang.
    if "YES" in tail:
        return 1.0, "Yes!"
    if "NO" in tail:
        return 0.0, "No!"
    return -1.0, "PARSE_FAILED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True, help="Dir holding staging.parquet and responses.jsonl")
    ap.add_argument("--output", required=True, help="Output parquet or CSV path")
    args = ap.parse_args()

    work_dir = Path(args.work_dir).resolve()
    reject_if_inside_skill(work_dir, "work_dir")
    output_path = Path(args.output).resolve()
    reject_if_inside_skill(output_path, "output")

    staging = work_dir / "staging.parquet"
    responses = work_dir / "responses.jsonl"
    if not staging.is_file():
        sys.exit(f"ERROR: {staging} not found.")
    if not responses.is_file():
        sys.exit(f"ERROR: {responses} not found.")

    df = pd.read_parquet(staging)

    resp_map: dict[int, str] = {}
    with responses.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            resp_map[int(rec["__row_id__"])] = rec["boilerplate_check_llm_response"]

    missing = [rid for rid in df["__row_id__"] if rid not in resp_map]
    if missing:
        sys.exit(f"ERROR: missing responses for __row_id__ values: {missing[:10]}"
                 f"{' ...' if len(missing) > 10 else ''}")

    texts = [resp_map[int(rid)] for rid in df["__row_id__"]]
    parsed = [parse_one(t) for t in texts]
    df["boilerplate_check_llm_response"] = texts
    df["exclusion_result"] = [p[0] for p in parsed]
    df["exclusion_verdict"] = [p[1] for p in parsed]

    df = df.drop(columns=["__row_id__"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext == ".parquet":
        df.to_parquet(output_path, index=False)
    elif ext == ".csv":
        df.to_csv(output_path, index=False)
    else:
        sys.exit(f"ERROR: unsupported output extension {ext!r}; expected .parquet or .csv")

    total = len(df)
    failed = int((df["exclusion_verdict"] == "PARSE_FAILED").sum())
    parsed_n = total - failed
    print(f"{total} scored ({parsed_n} parsed, {failed} failed) -> {output_path}")
    print(f"verdict histogram: {df['exclusion_verdict'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
