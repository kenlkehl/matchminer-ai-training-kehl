#!/usr/bin/env python3
"""
Build the DFCI enrollment evaluation's trial line items (trial_space_lineitems.csv) from cohorts.

One row per enrolled trial (nct_id, title, brief_summary and eligibility_criteria from the
enrollment table) goes through ../eval_cohort_lineitems.py, training's ctgov cohort
extraction (0d_extract_ctgov_cohorts.py). That writes one row per cohort with its key criteria
in this_space and its boilerplate criteria in trial_boilerplate_text, the columns the
retrieval and evaluation stages read. Until 2026-11-06 this ran the legacy pre-cohort
trial-space extractor (removed).

Extraction is remote only. Unrecognized options (--server_urls / --server_urls_file, --model,
--tokenizer-model, --allow-failures, ...) are forwarded to it.

Usage:
    python spacify_dfci_trials.py --server_urls_file servers.json --model <served model id>
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # matchminer-ai-training-kehl
EXTRACTION_SCRIPT = REPO_ROOT / "real_eval_code" / "eval_cohort_lineitems.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-file", type=str, default=None,
                        help="Enrollment table, parquet or CSV (default: data/phi/phi_enrollments_longnotes.parquet)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/phi)")
    parser.add_argument("--python", default=sys.executable, help="Python for the extraction")
    return parser.parse_known_args()


def main():
    args, forwarded = parse_args()
    input_file = Path(args.input_file or REPO_ROOT.parent / "data/phi/phi_enrollments_longnotes.parquet")
    output_dir = Path(args.output_dir or REPO_ROOT.parent / "data/phi")
    if not input_file.exists():
        sys.exit(f"Error: input file not found: {input_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python, str(EXTRACTION_SCRIPT),
        "--trials", str(input_file),
        "--output-spaces", str(output_dir / "trial_space_lineitems.csv"),
        "--work-dir", str(output_dir / "trial_cohort_extraction"),
        *forwarded,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Error: cohort extraction failed with code {result.returncode}")


if __name__ == "__main__":
    main()
