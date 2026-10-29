"""Turn a synthetic protocol batch's extracted cohorts into the patient-generation cohort table.

Reads <batch-dir>/cohort_extraction/{cohorts,trials}.parquet from extract_cohorts.py
and writes a CSV that generate_synthetic_data.py (run_pipeline.py --sampled-cohorts)
accepts: the accepted cancer-treatment cohorts in seed order with a contiguous
cohort_index, each trial's source split copied from the registry sample it was
seeded from, and every provenance column kept. Cohort criteria come from the
synthetic protocol; the original registry criteria stay in
registry_eligibility_criteria and are never used as cohort criteria. A
<output>.provenance.json receipt records the sources. See synthetic_protocols/README.md.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trial_cohort_format import (  # noqa: E402
    REPRESENTATION, file_stamp, read_clinical_csv, representation_for, training_rows,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "no_phi"


def build(cohorts, trials, splits):
    """Generation-ready rows, validated by the same adapter the generator uses."""
    if "eligibility_criteria" in cohorts:
        raise ValueError("eligibility_criteria would replace protocol criteria for single cohorts")
    if not cohorts.cohort_criteria_source.eq("protocol").all():
        raise ValueError("Every cohort must come from protocol extraction")
    order = {trial: rank for rank, trial in enumerate(trials.trial_id)}
    frame = cohorts.assign(_rank=cohorts.trial_id.map(order))
    if frame._rank.isna().any():
        raise ValueError("Cohorts include trials missing from the trial table")
    frame = frame.sort_values(["_rank", "cohort_number"]).drop(columns="_rank").reset_index(drop=True)
    split = splits.drop_duplicates()
    if split.nct_id.duplicated().any():
        raise ValueError("A seed trial has more than one source split")
    frame = frame.merge(split, left_on="trial_id", right_on="nct_id", how="left", validate="many_to_one")
    if frame.split.isna().any():
        raise ValueError(f"No source split for {sorted(frame.loc[frame.split.isna(), 'trial_id'].unique())[:5]}")
    frame["cohort_index"] = range(len(frame))
    frame["trial_representation"] = representation_for(frame)
    if frame.trial_representation.iloc[0] != REPRESENTATION:
        raise ValueError("Protocol cohorts must use the named-cohort representation")
    # The generator rebuilds criteria from these rows; protocol text must survive unchanged.
    rebuilt = training_rows(frame)
    if not rebuilt.cohort_eligibility_criteria.eq(frame.cohort_eligibility_criteria).all():
        raise ValueError("Rebuilt cohort criteria differ from the extracted protocol criteria")
    return frame


def main(args):
    extraction = args.batch_dir / "cohort_extraction"
    metadata = json.loads((extraction / "run_metadata.json").read_text())
    if metadata["trial_errors"]:
        print(f"note: {len(metadata['trial_errors'])} trials failed extraction and have no cohorts", flush=True)
    cohorts = pd.read_parquet(extraction / "cohorts.parquet")
    trials = pd.read_parquet(extraction / "trials.parquet")
    splits = read_clinical_csv(args.splits_from, usecols=["nct_id", "split"])
    frame = build(cohorts, trials, splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(args.output)
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or None
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": str(args.output.resolve()), "output_stamp": file_stamp(args.output),
        "batch_dir": str(args.batch_dir.resolve()),
        "sources": {name: {"path": str((extraction / name).resolve()), "stamp": file_stamp(extraction / name)}
                    for name in ("cohorts.parquet", "trials.parquet", "run_metadata.json")},
        "splits_from": {"path": str(args.splits_from.resolve()), "stamp": file_stamp(args.splits_from)},
        "training_commit": commit, "trial_representation": REPRESENTATION,
        "cohort_criteria_source": "protocol",
        "trials": int(frame.trial_id.nunique()), "cohorts": len(frame),
        "multiple_cohort_trials": int(frame.loc[frame.multiple_cohorts.eq("yes"), "trial_id"].nunique()),
        "split_counts": frame.drop_duplicates("trial_id").split.value_counts().to_dict(),
        "extraction_status_counts": metadata["status_counts"],
        "extraction_failures": sorted(metadata["trial_errors"]),
        "not_cancer_treatment": metadata["trial_rejections"],
        "synthetic_prompt_sha256": metadata["synthetic_prompt_sha256"],
        "extraction_prompt_sha256": metadata["prompt_sha256"], "extraction_schema_version": metadata["schema_version"],
    }
    Path(f"{args.output}.provenance.json").write_text(json.dumps(receipt, indent=2, default=str))
    print(f"wrote {len(frame)} cohorts from {receipt['trials']} trials to {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch-dir", type=Path, required=True, help="A generate_batch.py output directory")
    parser.add_argument("--output", type=Path, required=True, help="Generation input CSV to write")
    parser.add_argument("--splits-from", type=Path, default=DATA / "sample_trial_cohort_lineitems.csv",
                        help="Registry cohort sample whose nct_id/split columns give each trial's source split")
    main(parser.parse_args())
