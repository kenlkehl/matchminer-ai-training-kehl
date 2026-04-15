#!/usr/bin/env python3
"""Merge Claude's per-row responses into the sampled dataframe and parse scores.

Ports the parsing logic from llm_check_trials.py (lines 128-154) verbatim so
outputs match the format produced by that script: three new columns appended —
trialcheck_llm_response, eligibility_result, eligibility_verdict.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


SKILL_DIR = Path(__file__).resolve().parent.parent
SCORE_PATTERN = re.compile(r"[Ff]inal\s+[Ss]core\s*:\s*(\d)")


def reject_if_inside_skill(path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(SKILL_DIR)
    except ValueError:
        return
    sys.exit(
        f"ERROR: {label} ({path}) is inside the skill directory ({SKILL_DIR}). "
        "Choose a location outside the skill folder."
    )


def parse_one(txt: str) -> tuple[int, str]:
    tail = txt[-60:].replace("*", "").replace("\u202f", " ")
    m = SCORE_PATTERN.search(tail)
    if m:
        score = min(int(m.group(1)), 5)
        return score, f"Score:{score}"
    tail_upper = tail.upper()
    fallback = re.search(r"SCORE\s*[:\-=]\s*(\d)", tail_upper)
    if fallback:
        score = min(int(fallback.group(1)), 5)
        return score, f"Score:{score}"
    if "NOT REASONABLE" in tail_upper or "NOT A REASONABLE" in tail_upper:
        return 0, "Score:0"
    return -1, "PARSE_FAILED"


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
            resp_map[int(rec["__row_id__"])] = rec["trialcheck_llm_response"]

    missing = [rid for rid in df["__row_id__"] if rid not in resp_map]
    if missing:
        sys.exit(f"ERROR: missing responses for __row_id__ values: {missing[:10]}"
                 f"{' ...' if len(missing) > 10 else ''}")

    texts = [resp_map[int(rid)] for rid in df["__row_id__"]]
    parsed = [parse_one(t) for t in texts]
    df["trialcheck_llm_response"] = texts
    df["eligibility_result"] = [p[0] for p in parsed]
    df["eligibility_verdict"] = [p[1] for p in parsed]

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
    failed = int((df["eligibility_result"] == -1).sum())
    parsed_n = total - failed
    print(f"{total} scored ({parsed_n} parsed, {failed} failed) -> {output_path}")
    hist = df["eligibility_result"].value_counts().sort_index().to_dict()
    print(f"score histogram: {hist}")


if __name__ == "__main__":
    main()
