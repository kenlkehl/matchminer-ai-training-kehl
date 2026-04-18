#!/usr/bin/env python3
"""Merge judge responses; parse Likert + disagreement category."""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


SKILL_DIR = Path(__file__).resolve().parent.parent
LIKERT_RE = re.compile(
    r"[Aa]greement\s*:\s*(STRONGLY_AGREE|STRONGLY_DISAGREE|AGREE|DISAGREE|NEUTRAL)\b"
)
CATEGORY_RE = re.compile(
    r"[Dd]isagreement\s+[Cc]ategory\s*:\s*"
    r"(false_exclusion|missed_exclusion|reasoning_flaw|other|n/a)\b",
    re.IGNORECASE,
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


def parse_verdicts(text: str) -> tuple[str, str]:
    tail = text[-600:].replace("*", "").replace("\u202f", " ")
    lk = LIKERT_RE.search(tail)
    cat = CATEGORY_RE.search(tail)
    return (
        lk.group(1) if lk else "PARSE_FAILED",
        (cat.group(1).lower() if cat else "PARSE_FAILED"),
    )


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
            resp_map[int(rec["__row_id__"])] = rec["boilerplate_judge_response"]

    missing = [rid for rid in df["__row_id__"] if rid not in resp_map]
    if missing:
        sys.exit(
            f"ERROR: missing responses for __row_id__ values: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )

    texts = [resp_map[int(rid)] for rid in df["__row_id__"]]
    parsed = [parse_verdicts(t) for t in texts]
    df["boilerplate_judge_response"] = texts
    df["boilerplate_judge_agreement"] = [p[0] for p in parsed]
    df["boilerplate_judge_category"] = [p[1] for p in parsed]
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
    failed = int((df["boilerplate_judge_agreement"] == "PARSE_FAILED").sum())
    parsed_n = total - failed
    print(f"{total} judged ({parsed_n} parsed, {failed} failed) -> {output_path}")
    print(f"agreement histogram: {df['boilerplate_judge_agreement'].value_counts().to_dict()}")
    print(f"category histogram: {df['boilerplate_judge_category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
