"""Combine each sampled trial's protocol and registry cohorts into the retrieval cohort table.

Users may hold some trials' full protocols and only the ClinicalTrials.gov text
of others, so retrieval and labeling see both representations of a trial: the
cohorts extracted from its synthetic protocol (extract_cohorts.py, via
patient_cohorts.py) and the cohorts extracted from the registry record that
protocol was written from (0b_extract_trial_cohorts.py --trials-jsonl on the
batch's seeds.jsonl). Both use the same prompt rules and response contract.
Patients are generated from the protocol cohorts only.

The output keeps every protocol row unchanged (cohort_criteria_source
"protocol", IDs <NCT>-cohort-<n>) and adds the registry rows (cohort_criteria_source
"registry", IDs <NCT>-registry-cohort-<n>, registry fields renamed registry_*,
cohort_index continuing after the protocol rows), each trial's split taken
from the registry sample. A trial accepted in only one version keeps that one.
Mining then represents each trial by one version, chosen at random per
patient-trial pair and per trial for cohort queries
(colbert_experiments/retrieval.py). Writes <output> and <output>.provenance.json.
See synthetic_protocols/README.md.
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
from trial_cohort_format import file_stamp, read_clinical_csv, training_rows  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "no_phi"
REGISTRY_FIELDS = ("title", "brief_summary", "detailed_description", "eligibility_criteria",
                   "last_update_post_date", "retrieved_at")


def build(protocol, registry, splits):
    """Protocol rows unchanged plus source-qualified registry rows, validated together."""
    if not protocol.cohort_criteria_source.eq("protocol").all():
        raise ValueError("Every protocol-table row must come from protocol extraction")
    if not registry.cohort_criteria_source.eq("registry").all():
        raise ValueError("Every registry row must come from current registry extraction (trial-registry-cohorts-v1)")
    split = splits.drop_duplicates()
    if split.nct_id.duplicated().any():
        raise ValueError("A sampled trial has more than one source split")
    split = split.set_index("nct_id").split
    known = protocol.drop_duplicates("trial_id").set_index("trial_id").split
    if not known.eq(split.reindex(known.index)).all():
        raise ValueError("Protocol cohort splits disagree with the registry sample")
    registry = registry.rename(columns={key: f"registry_{key}" for key in REGISTRY_FIELDS if key in registry})
    registry = registry.drop(columns=["split"], errors="ignore").assign(
        cohort_trial_id=registry.trial_id + "-registry-cohort-" + registry.cohort_number.astype(int).astype(str),
        nct_id=registry.trial_id, split=registry.trial_id.map(split))
    if registry.split.isna().any():
        raise ValueError(f"No source split for {sorted(registry.loc[registry.split.isna(), 'trial_id'].unique())[:5]}")
    registry = registry.sort_values(["trial_id", "cohort_number"]).reset_index(drop=True)
    registry["cohort_index"] = range(int(protocol.cohort_index.max()) + 1, int(protocol.cohort_index.max()) + 1 + len(registry))
    frame = pd.concat([protocol, registry], ignore_index=True)
    rebuilt = training_rows(frame)
    if not rebuilt.cohort_eligibility_criteria.eq(frame.cohort_eligibility_criteria).all():
        raise ValueError("Rebuilt cohort criteria differ from the extracted criteria")
    return frame


def main(args):
    protocol = read_clinical_csv(args.protocol_cohorts)
    registry = read_clinical_csv(args.registry_cohorts)
    audit = read_clinical_csv(args.registry_trials)
    splits = read_clinical_csv(args.splits_from, usecols=["nct_id", "split"])
    frame = build(protocol, registry, splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(args.output)
    protocol_trials, registry_trials = set(protocol.trial_id), set(registry.trial_id)
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or None
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": str(args.output.resolve()), "output_stamp": file_stamp(args.output), "training_commit": commit,
        "sources": {name: {"path": str(path.resolve()), "stamp": file_stamp(path)} for name, path in (
            ("protocol_cohorts", args.protocol_cohorts), ("registry_cohorts", args.registry_cohorts),
            ("registry_trials", args.registry_trials), ("splits_from", args.splits_from))},
        "cohorts": {"protocol": len(protocol), "registry": len(registry), "total": len(frame)},
        "trials": {"both_versions": len(protocol_trials & registry_trials),
                   "protocol_only": len(protocol_trials - registry_trials),
                   "registry_only": len(registry_trials - protocol_trials),
                   "total": len(protocol_trials | registry_trials)},
        "registry_extraction": {
            "records": len(audit), "not_cancer_treatment": int(audit.cancer_treatment_trial.eq("no").sum()),
            "failed": sorted(audit.loc[audit.extraction_error.notna(), "nct_id"])},
        "retrieval_policy": "one version per trial: random per patient-trial pair, and per trial for cohort queries",
    }
    Path(f"{args.output}.provenance.json").write_text(json.dumps(receipt, indent=2, default=str))
    print(json.dumps({key: receipt[key] for key in ("cohorts", "trials")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--protocol-cohorts", type=Path, required=True,
                        help="patient_cohorts.py output: the protocol cohorts patients were generated from")
    parser.add_argument("--registry-cohorts", type=Path, required=True,
                        help="0b_extract_trial_cohorts.py --trials-jsonl cohort rows for the same batch's seeds")
    parser.add_argument("--registry-trials", type=Path, required=True,
                        help="The same run's per-trial audit table (--output-trials)")
    parser.add_argument("--output", type=Path, required=True, help="Retrieval cohort CSV to write")
    parser.add_argument("--splits-from", type=Path, default=DATA / "sample_trial_cohort_lineitems.csv",
                        help="Registry cohort sample whose nct_id/split columns give each trial's source split")
    main(parser.parse_args())
