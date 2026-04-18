#!/usr/bin/env python3
"""Merge judge responses and parse the two verdict lines.

Expects responses to end with:
  Mutually exclusive: <YES|PARTIAL|NO>
  Exhaustive: <YES|PARTIAL|NO>
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


SKILL_DIR = Path(__file__).resolve().parent.parent
ME_RE = re.compile(r"[Mm]utually\s+[Ee]xclusive\s*:\s*(YES|PARTIAL|NO)\b")
EXH_RE = re.compile(r"[Ee]xhaustive\s*:\s*(YES|PARTIAL|NO)\b")


def reject_if_inside_skill(path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(SKILL_DIR)
    except ValueError:
        return
    sys.exit(
        f"ERROR: {label} ({path}) is inside the skill directory ({SKILL_DIR}). "
        "Choose a location outside the skill folder."
    )


def parse_verdicts(text: str) -> tuple[str, str]:
    tail = text[-600:].replace("*", "").replace("\u202f", " ")
    me = ME_RE.search(tail)
    exh = EXH_RE.search(tail)
    return (me.group(1) if me else "PARSE_FAILED",
            exh.group(1) if exh else "PARSE_FAILED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--output", required=True)
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
            resp_map[int(rec["__row_id__"])] = rec["spaces_judge_response"]

    missing = [rid for rid in df["__row_id__"] if rid not in resp_map]
    if missing:
        sys.exit(
            f"ERROR: missing responses for __row_id__ values: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )

    texts = [resp_map[int(rid)] for rid in df["__row_id__"]]
    parsed = [parse_verdicts(t) for t in texts]
    df["spaces_judge_response"] = texts
    df["spaces_judge_me"] = [p[0] for p in parsed]
    df["spaces_judge_exh"] = [p[1] for p in parsed]
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
    failed = int(((df["spaces_judge_me"] == "PARSE_FAILED") | (df["spaces_judge_exh"] == "PARSE_FAILED")).sum())
    parsed_n = total - failed
    print(f"{total} judged ({parsed_n} fully parsed, {failed} with parse failure) -> {output_path}")
    print(f"mutually_exclusive histogram: {df['spaces_judge_me'].value_counts().to_dict()}")
    print(f"exhaustive histogram: {df['spaces_judge_exh'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
