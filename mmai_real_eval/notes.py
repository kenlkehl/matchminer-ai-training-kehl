"""Reports available before each index date, and v22's note-level dataset.

v22 prepare_data.py (both cohorts) built ``note_level_dataset.parquet``: for each index
event, every imaging report, clinical note and pathology report of the patient dated
before ``trial_start_dt + days_buffer`` (default 5 days). Index events with no such report
produced no rows, so they never reached summarization or evaluation: an implicit
exclusion. ``count_prior_reports`` reproduces that cutoff on dates alone (no note text is
read) so the denominators can carry ``n_prior_reports``; ``build_note_level_dataset``
writes the note text itself and is a separate, explicit step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPORT_FILES = {
    "imaging": "all_imaging_reports.parquet",
    "clinical": "all_clinical_notes.parquet",
    "path": "all_path_reports.parquet",
    # New in pan_dfci_2026; not used by v22 and off by default.
    "radonc": "all_rad_completion.parquet",
}
DEFAULT_SOURCES = ("imaging", "clinical", "path")

_EPOCH = np.datetime64("1800-01-01", "D")
_DAY_SPAN = 100_000  # days after _EPOCH representable in a key (through 2073)


def _day_numbers(dates) -> np.ndarray:
    days = np.asarray(dates, dtype="datetime64[ns]").astype("datetime64[D]")
    return (days - _EPOCH).astype(np.int64)


def load_report_dates(derived_data_dir, sources=DEFAULT_SOURCES, patients=None) -> tuple[pd.DataFrame, dict]:
    """(dfci_mrn, report day) for every report in ``sources``; reads two columns only."""
    frames, stats = [], {}
    for source in sources:
        path = Path(derived_data_dir) / REPORT_FILES[source]
        table = pq.read_table(path, columns=["dfci_mrn", "date"])
        frame = table.to_pandas()
        n = len(frame)
        if patients is not None:
            frame = frame[frame.dfci_mrn.isin(patients)]
        frame = frame[frame.date.notna()]
        stats[source] = {"file": str(path), "reports": int(n), "reports_for_cohort_patients": int(len(frame))}
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), stats


def count_prior_reports(events: pd.DataFrame, report_dates: pd.DataFrame, days_buffer: int = 5) -> np.ndarray:
    """Reports per event dated before ``trial_start_dt + days_buffer`` (v22's cutoff, exclusive).

    v22 compared report timestamps with the midnight cutoff (``date < trial_start + 5 days``);
    comparing calendar days is equivalent. Reports dated outside 1800-2073 are ignored.
    """
    report_day = _day_numbers(report_dates.date.values)
    ok = (report_day >= 0) & (report_day < _DAY_SPAN)
    keys = np.sort(report_dates.dfci_mrn.values[ok].astype(np.int64) * _DAY_SPAN + report_day[ok])
    cutoff = _day_numbers((events.trial_start_dt + pd.Timedelta(days=days_buffer)).values)
    if (cutoff < 0).any() or (cutoff >= _DAY_SPAN).any():
        raise ValueError("Index date outside the supported range")
    base = events.dfci_mrn.values.astype(np.int64) * _DAY_SPAN
    return np.searchsorted(keys, base + cutoff, side="left") - np.searchsorted(keys, base, side="left")


def build_note_level_dataset(events: pd.DataFrame, derived_data_dir, output_parquet, days_buffer: int = 5,
                             sources=DEFAULT_SOURCES, batch_rows: int = 200_000) -> dict:
    """Write v22's note-level dataset: one row per (index event, report before its cutoff).

    ``events`` needs pseudo_mrn, dfci_mrn and trial_start_dt (one row per pseudo_mrn is
    enough). Differences from v22, which looped over every enrollment row: each index event
    gets one copy of its reports (v22 repeated them for each row sharing a pseudo_mrn), trial
    or plan details are not copied onto every note (join on pseudo_mrn instead), only the
    columns shared by all report files are kept, and rows are not globally sorted (the
    summarizer sorts by patient and date). Streams each file once.
    """
    events = events[["pseudo_mrn", "dfci_mrn", "trial_start_dt"]].drop_duplicates("pseudo_mrn").copy()
    events["cutoff"] = events.trial_start_dt + pd.Timedelta(days=days_buffer)
    wanted = set(events.dfci_mrn.astype("int64"))
    keep = ["dfci_mrn", "date", "note_type", "RPT_ID", "text"]
    schema = pa.schema([("pseudo_mrn", pa.int64()), ("dfci_mrn", pa.int64()), ("trial_start_dt", pa.timestamp("ns")),
                        ("date", pa.timestamp("ns")), ("note_type", pa.string()), ("RPT_ID", pa.string()),
                        ("text", pa.string())])
    output_parquet = Path(output_parquet)
    tmp = output_parquet.with_name(output_parquet.name + ".tmp")
    written, per_source = 0, {}
    with pq.ParquetWriter(tmp, schema) as writer:
        for source in sources:
            path = Path(derived_data_dir) / REPORT_FILES[source]
            n_source = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=keep):
                frame = batch.to_pandas()
                frame = frame[frame.dfci_mrn.isin(wanted) & frame.date.notna()]
                if frame.empty:
                    continue
                merged = frame.merge(events, on="dfci_mrn", how="inner")
                merged = merged[merged.date < merged.cutoff]
                if merged.empty:
                    continue
                out = merged[["pseudo_mrn", "dfci_mrn", "trial_start_dt", "date", "note_type", "RPT_ID", "text"]]
                writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))
                n_source += len(out)
            per_source[source] = int(n_source)
            written += n_source
    tmp.chmod(0o600)
    tmp.replace(output_parquet)
    return {"rows": int(written), "rows_by_source": per_source, "index_events": int(len(events))}
