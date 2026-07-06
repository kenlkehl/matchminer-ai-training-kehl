#!/usr/bin/env python3
"""
One-off backfill: re-derive boilerplate `exclusion_result` labels from the
already-stored LLM responses, introducing the -1.0 PARSE_FAILED sentinel.

Historically the Yes!/No! parser forced any unparseable response to 0.0
("not excluded"), silently mislabeling parse failures as true negatives. The
generators now emit a -1.0 sentinel instead (matching the inference repo's
`_parse_exclusion_result`). This script applies the same corrected, lenient
parser to the raw responses already saved in the data files, so we do NOT need
to re-run any LLM inference.

Targets (overwritten in place; only `exclusion_result` changes):
  - Training parquet:  data/no_phi/boilerplate_checks/final_boilerplate_checks.parquet
                       (response col: boilerplate_check_llm_response)
  - Eval gold CSVs:    data/phi/{enrollments,soc}/consolidated_boilerplate_{patient,trial}_centric.csv
                       (response col: llm_boilerplate_response)

NOT touched: oncoreasoning_boilerplate_* prediction files (regenerated separately).

Runs as a dry-run by default (reports the diff without writing). Pass --apply to
overwrite in place. Writes go to a temp file in the same directory and are then
atomically os.replace()'d over the original; the raw response columns are
retained, so re-running this script reproduces the labels (reversible).

Usage:
    python backfill_boilerplate_sentinel.py            # dry-run report
    python backfill_boilerplate_sentinel.py --apply     # overwrite in place
"""

import argparse
import os
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Repo root = parent of real_eval_code; data lives at <root>/../data
REPO_ROOT = Path(__file__).resolve().parents[1]           # matchminer-ai-training
DATA_ROOT = REPO_ROOT.parent / "data"

PARQUET_PATH = DATA_ROOT / "no_phi/boilerplate_checks/final_boilerplate_checks.parquet"
CSV_PATHS = [
    DATA_ROOT / "phi/enrollments/consolidated_boilerplate_patient_centric.csv",
    DATA_ROOT / "phi/enrollments/consolidated_boilerplate_trial_centric.csv",
    DATA_ROOT / "phi/soc/consolidated_boilerplate_patient_centric.csv",
    DATA_ROOT / "phi/soc/consolidated_boilerplate_trial_centric.csv",
]

BATCH_ROWS = 100_000


def parse_exclusion(text) -> float:
    """Canonical lenient parser with -1.0 PARSE_FAILED sentinel.

    Mirrors the corrected generator parsers and the inference repo's
    `_parse_exclusion_result` (which returns False/True/None for
    excluded/not-excluded/parse-failed). rstrip() guards against trailing
    whitespace/newlines that survive a CSV/parquet round-trip.
    """
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


def _report(name: str, old: pd.Series, new: pd.Series) -> None:
    """Print old/new distribution and the transition breakdown."""
    old = old.astype(float)
    new = new.astype(float)
    trans = Counter(zip(old.tolist(), new.tolist()))
    changed = {k: v for k, v in trans.items() if k[0] != k[1]}
    print(f"  old dist: {old.value_counts(dropna=False).sort_index().to_dict()}")
    print(f"  new dist: {new.value_counts(dropna=False).sort_index().to_dict()}")
    print(f"  new -1 sentinels: {int((new < 0).sum())}")
    if changed:
        print("  label transitions (old -> new : count):")
        for (o, n), c in sorted(changed.items()):
            print(f"    {o:+.0f} -> {n:+.0f} : {c}")
    else:
        print("  no label changes")


def backfill_csv(path: Path, apply: bool) -> None:
    print(f"\n=== CSV: {path} ===")
    if not path.exists():
        print("  MISSING - skipped")
        return
    df = pd.read_csv(path)
    if "llm_boilerplate_response" not in df.columns or "exclusion_result" not in df.columns:
        print("  required columns missing - skipped")
        return
    old = df["exclusion_result"].copy()
    new = df["llm_boilerplate_response"].map(parse_exclusion)
    _report(path.name, old, new)
    if not apply:
        print("  [dry-run] not written")
        return
    if old.astype(float).equals(new.astype(float)):
        print("  no changes - file left untouched")
        return
    df["exclusion_result"] = new.values
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".csv.tmp")
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    print(f"  WROTE {path.name} ({len(df)} rows)")


def backfill_parquet(path: Path, apply: bool) -> None:
    print(f"\n=== PARQUET: {path} ===")
    if not path.exists():
        print("  MISSING - skipped")
        return
    pf = pq.ParquetFile(str(path))
    schema = pf.schema_arrow
    names = set(schema.names)
    if "boilerplate_check_llm_response" not in names or "exclusion_result" not in names:
        print("  required columns missing - skipped")
        return
    excl_idx = schema.get_field_index("exclusion_result")
    excl_field = schema.field(excl_idx)

    # Pass 1: detect changes (no writing).
    old_counts: Counter = Counter()
    new_counts: Counter = Counter()
    trans_counts: Counter = Counter()
    for batch in pf.iter_batches(batch_size=BATCH_ROWS):
        table = pa.Table.from_batches([batch], schema=schema)
        responses = table.column("boilerplate_check_llm_response").to_pylist()
        old_vals = table.column("exclusion_result").to_pylist()
        new_vals = [parse_exclusion(r) for r in responses]
        for o, n in zip(old_vals, new_vals):
            o_key = -999.0 if o is None else float(o)
            old_counts[o_key] += 1
            new_counts[n] += 1
            if o_key != n:
                trans_counts[(o_key, n)] += 1

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
        return

    # Pass 2: rewrite with corrected column (only reached when changes exist).
    try:
        comp = pf.metadata.row_group(0).column(0).compression.lower()
        if comp == "uncompressed":
            comp = None
    except Exception:
        comp = "snappy"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".parquet.tmp")
    os.close(fd)
    writer = pq.ParquetWriter(tmp, schema, compression=comp)
    try:
        for batch in pf.iter_batches(batch_size=BATCH_ROWS):
            table = pa.Table.from_batches([batch], schema=schema)
            responses = table.column("boilerplate_check_llm_response").to_pylist()
            new_arr = pa.array([parse_exclusion(r) for r in responses], type=excl_field.type)
            table = table.set_column(excl_idx, excl_field, new_arr)
            writer.write_table(table)
    finally:
        writer.close()
    os.replace(tmp, path)
    print(f"  WROTE {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Overwrite files in place (default: dry-run report only)")
    args = ap.parse_args()

    mode = "APPLY (overwrite in place)" if args.apply else "DRY-RUN (no writes)"
    print(f"Boilerplate sentinel backfill - mode: {mode}")

    backfill_parquet(PARQUET_PATH, args.apply)
    for csv_path in CSV_PATHS:
        backfill_csv(csv_path, args.apply)

    print("\nDone.")


if __name__ == "__main__":
    main()
