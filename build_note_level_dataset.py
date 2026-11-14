#!/usr/bin/env python3
"""Build v22's note-level dataset (report text before each index date) for one denominator.

The second half of v22 prepare_data.py (create_note_level_dataset), kept as its own step
because it copies note text: run it only when summarization is planned, and write only to
a PHI directory. Reads processed_trial_enrollments.csv or processed_soc_treatments.csv from
pull_denominators.py and writes a parquet of (pseudo_mrn, dfci_mrn, trial_start_dt, date,
note_type, RPT_ID, text), mode 600. See mmai_real_eval/notes.py for differences from v22.

The cutoff is explicit: ``--days-offset`` (no default; negative = before the anchor) from ``--anchor``:
  index        the index date (v22: index + 5 days, with its consent-date enrollment index and plan-start SOC index)
  note_anchor  note_anchor_dt, the earliest known trial registration (research-plan trial cohort; default when the
               column is present). Registration and screening notes name the trial, so cut before them.

Example (research-plan trial cohort, notes dated before registration minus 7 days):
  python build_note_level_dataset.py --anchor note_anchor --days-offset -7 \
      --denominator-csv ../data/phi/real_eval_denominators_20261114/processed_trial_enrollments.csv \
      --output-parquet ../data/phi/real_eval_denominators_20261114/enrollments_note_level_dataset.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from mmai_real_eval.notes import DEFAULT_SOURCES, REPORT_FILES, build_note_level_dataset


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--denominator-csv", type=Path, required=True)
    p.add_argument("--output-parquet", type=Path, required=True)
    p.add_argument("--derived-data-dir", type=Path, default=Path("/data1/ken/pan_dfci_2026/derived_data"))
    p.add_argument("--days-offset", type=int, required=True,
                   help="Cutoff = anchor + this many days, exclusive; negative = before the anchor (v22: +5 from index)")
    p.add_argument("--anchor", choices=["index", "note_anchor"], default=None,
                   help="index = trial_start_dt; note_anchor = note_anchor_dt (default when the CSV has it)")
    p.add_argument("--report-sources", default=",".join(DEFAULT_SOURCES), help=", ".join(REPORT_FILES))
    p.add_argument("--require-prior-reports", action="store_true",
                   help="Skip index events whose has_prior_reports is False (they produce no rows anyway)")
    args = p.parse_args(argv)
    os.umask(0o077)
    if "/data/phi/" not in str(args.output_parquet.resolve()) and "/data1/" not in str(args.output_parquet.resolve()):
        raise SystemExit("Refusing to write note text outside a PHI directory (data/phi/ or /data1/)")
    events = pd.read_csv(args.denominator_csv, usecols=lambda c: c in {
        "pseudo_mrn", "dfci_mrn", "trial_start_dt", "note_anchor_dt", "has_prior_reports"})
    events["trial_start_dt"] = pd.to_datetime(events.trial_start_dt)
    anchor = args.anchor or ("note_anchor" if "note_anchor_dt" in events else "index")
    if anchor == "note_anchor" and "note_anchor_dt" not in events:
        raise SystemExit("--anchor note_anchor needs a note_anchor_dt column (research-plan trial cohort)")
    anchor_col = "note_anchor_dt" if anchor == "note_anchor" else "trial_start_dt"
    events[anchor_col] = pd.to_datetime(events[anchor_col])
    if args.require_prior_reports and "has_prior_reports" in events:
        events = events[events.has_prior_reports.astype(str).str.lower().eq("true")]
    stats = build_note_level_dataset(events, args.derived_data_dir, args.output_parquet, args.days_offset,
                                     [s.strip() for s in args.report_sources.split(",") if s.strip()],
                                     anchor_col=anchor_col)
    print(f"Wrote {stats['rows']} report rows for {stats['index_events']} index events "
          f"({stats['rows_by_source']}; cutoff = {anchor_col} {args.days_offset:+d} days) to {args.output_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
