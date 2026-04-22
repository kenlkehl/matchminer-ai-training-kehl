#!/usr/bin/env python3
"""Append a single row-level judge response to <work_dir>/responses.jsonl.

Reads the response text from stdin and writes one JSON object per invocation:
  {"__row_id__": <int>, "summary_judge_response": <str>}

Stable command prefix so the user grants one "always allow" for
`python .../append_response.py *` across all rows.
"""
import argparse
import fcntl
import json
import sys
from pathlib import Path


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--row_id", required=True, type=int)
    p.add_argument("--response_file", default=None)
    args = p.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    reject_if_inside_skill(work_dir, "work_dir")
    if not work_dir.is_dir():
        sys.exit(f"ERROR: work_dir does not exist: {work_dir}")

    if args.response_file:
        resp_path = Path(args.response_file).expanduser().resolve()
        reject_if_inside_skill(resp_path, "response_file")
        if not resp_path.is_file():
            sys.exit(f"ERROR: response_file does not exist: {resp_path}")
        text = resp_path.read_text()
    else:
        text = sys.stdin.read()

    if not text.strip():
        sys.exit("ERROR: empty response text")

    out_path = work_dir / "responses.jsonl"
    line = json.dumps(
        {"__row_id__": args.row_id, "summary_judge_response": text},
        ensure_ascii=False,
    )
    # LOCK_EX serializes concurrent appenders so parallel subagents do
    # not interleave bytes within a single JSON line. The OS releases
    # the lock when the file handle closes.
    with out_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(line + "\n")
        f.flush()
    print(f"appended row_id={args.row_id} ({len(text)} chars) -> {out_path}")


if __name__ == "__main__":
    main()
