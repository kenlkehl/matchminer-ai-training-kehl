#!/usr/bin/env python3
"""
One-off backfill for the ModernBERT boilerplate *prediction* files.

The modernbert boilerplate eval, when run WITHOUT --run-inference, loads a
precomputed `boilerplate_<mode>_with_predictions[_soc].csv` that already contains
the classifier's `prediction_score` alongside the gold `exclusion_result`. Those
files were written before the sentinel fix, so their `exclusion_result` still
carries the old silent-0.0 parse failures.

The classifier predictions do not depend on the gold label, so we can correct
just the `exclusion_result` column in place (re-deriving it from the stored
`llm_boilerplate_response`) and then re-run the eval in metrics-only mode
(no GPU / no ModernBERT inference).

Processes the files in chunks (they are 0.6-2.5 GB). Reads every field as a raw
string (na_filter=False) so all other columns are round-tripped byte-for-byte;
only `exclusion_result` is rewritten. Dry-run by default; pass --apply to write.

Usage:
    python backfill_boilerplate_prediction_files.py            # dry-run report
    python backfill_boilerplate_prediction_files.py --apply     # overwrite in place
"""

import argparse
import os
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]           # matchminer-ai-training
DATA_ROOT = REPO_ROOT.parent / "data"
BP_DIR = "evaluation/modernbert-boilerplate-checker"

FILES = [
    DATA_ROOT / f"phi/enrollments/{BP_DIR}/boilerplate_patient_centric_with_predictions.csv",
    DATA_ROOT / f"phi/enrollments/{BP_DIR}/boilerplate_trial_centric_with_predictions.csv",
    DATA_ROOT / f"phi/soc/{BP_DIR}/boilerplate_patient_centric_with_predictions_soc.csv",
    DATA_ROOT / f"phi/soc/{BP_DIR}/boilerplate_trial_centric_with_predictions_soc.csv",
]

RESP_COL = "llm_boilerplate_response"
LABEL_COL = "exclusion_result"
CHUNK_ROWS = 20_000


def parse_exclusion(text) -> float:
    """Canonical lenient parser with -1.0 PARSE_FAILED sentinel (matches generators)."""
    if not isinstance(text, str):
        return -1.0
    tail = text.rstrip()[-10:].upper()
    if "YES!" in tail:
        return 1.0
    if "NO!" in tail:
        return 0.0
    if "YES" in tail:      # loose near-miss (missing '!')
        return 1.0
    if "NO" in tail:       # loose near-miss (missing '!')
        return 0.0
    return -1.0            # PARSE_FAILED sentinel


def backfill(path: Path, apply: bool) -> None:
    print(f"\n=== {path} ===")
    if not path.exists():
        print("  MISSING - skipped")
        return

    old_counts: Counter = Counter()
    new_counts: Counter = Counter()
    trans_counts: Counter = Counter()

    tmp = None
    if apply:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".csv.tmp")
        os.close(fd)

    header_written = False
    reader = pd.read_csv(path, chunksize=CHUNK_ROWS, dtype=str, na_filter=False)
    for chunk in reader:
        if RESP_COL not in chunk.columns or LABEL_COL not in chunk.columns:
            print("  required columns missing - skipped")
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
            return
        old_vals = [float(v) if v not in ("", "nan") else -999.0 for v in chunk[LABEL_COL]]
        new_vals = [parse_exclusion(t) for t in chunk[RESP_COL]]
        for o, n in zip(old_vals, new_vals):
            old_counts[o] += 1
            new_counts[n] += 1
            if o != n:
                trans_counts[(o, n)] += 1
        if apply:
            chunk[LABEL_COL] = [repr(v) for v in new_vals]  # 1.0 / 0.0 / -1.0
            chunk.to_csv(tmp, mode="a", index=False, header=not header_written)
            header_written = True

    print(f"  old dist: {dict(sorted(old_counts.items()))}")
    print(f"  new dist: {dict(sorted(new_counts.items()))}")
    print(f"  new -1 sentinels: {new_counts.get(-1.0, 0)}")
    if trans_counts:
        print("  label transitions (old -> new : count):")
        for (o, n), c in sorted(trans_counts.items()):
            print(f"    {o:+.0f} -> {n:+.0f} : {c}")
    else:
        print("  no label changes")

    if not apply:
        print("  [dry-run] not written")
        return
    if not trans_counts:
        print("  no changes - file left untouched")
        os.remove(tmp)
        return
    os.replace(tmp, path)
    print(f"  WROTE {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Overwrite files in place (default: dry-run report only)")
    args = ap.parse_args()

    mode = "APPLY (overwrite in place)" if args.apply else "DRY-RUN (no writes)"
    print(f"Boilerplate PREDICTION-file sentinel backfill - mode: {mode}")
    for f in FILES:
        backfill(f, args.apply)
    print("\nDone.")


if __name__ == "__main__":
    main()
