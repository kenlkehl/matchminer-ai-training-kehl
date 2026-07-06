#!/usr/bin/env python3
"""
Demographic helpers for stratified evaluation of trial-matching models.

Loads patient demographics (age, race, ethnicity, sex) from the pan-DFCI
structured-data registration tables and attaches them to evaluation candidate
frames so the existing metrics (AUC, MAP@K) can be broken down by demographic
group.

Age is computed *at treatment start* -- the point in time each candidate's
``patient_summary`` simulates trial matching (clinical notes are grabbed and
summarized through that treatment-start date). The treatment-start date for each
candidate is recovered from ``patient_summaries.parquet`` by joining on
``(dfci_mrn, patient_summary)`` (one ``pseudo_mrn`` per (dfci_mrn, treatment
start), so the same patient can appear with several distinct summaries/dates).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Default location of the pan-DFCI structured-data registration tables.
DEFAULT_STRUCTURED_DATA_DIR = "/data1/ken/pan_dfci_2025/structured_data"
PT_INFO_FILE = "REQ_KK71_192437_F1_PT_INFO_STATUS_REGISTRATION.csv"
DEMOGRAPHICS_FILE = "REQ_KK71_192437_F1_DEMOGRAPHICS_REGISTRATION.csv"

# Decade age categories (age at treatment start). Anything missing -> 'Unknown'.
AGE_BIN_EDGES = [-np.inf, 40, 50, 60, 70, 80, np.inf]
AGE_LABELS = ['<40', '40-49', '50-59', '60-69', '70-79', '80+']
UNKNOWN = 'Unknown'

# The four demographic dimensions every breakdown iterates over.
DEMOGRAPHIC_COLS = ['age_category', 'race', 'ethnicity', 'sex']


def _coerce_mrn(series: pd.Series) -> pd.Series:
    """Coerce a dfci_mrn column to a nullable integer for stable joins.

    dfci_mrn is stored as int64 in some files and float64 in others; normalize
    to pandas nullable Int64 so merges line up regardless of source dtype.
    """
    return pd.to_numeric(series, errors='coerce').round().astype('Int64')


def load_demographics(structured_data_dir: str = DEFAULT_STRUCTURED_DATA_DIR) -> pd.DataFrame:
    """Load per-patient demographics keyed by ``dfci_mrn``.

    Returns a DataFrame with columns: ``dfci_mrn`` (Int64), ``birth_dt``
    (datetime), ``sex``, ``race``, ``ethnicity``. One row per dfci_mrn.

    Sex uses ``GENDER_NM`` (well populated) and falls back to
    ``SEX_AT_BIRTH_NM`` (mostly missing). Race uses the consolidated
    ``IDM_RACE_NM``; ethnicity uses the ``HISPANIC_IND`` flag.
    """
    structured_data_dir = Path(structured_data_dir)

    pt_info = pd.read_csv(
        structured_data_dir / PT_INFO_FILE,
        usecols=lambda c: c in {'DFCI_MRN', 'BIRTH_DT', 'GENDER_NM', 'SEX_AT_BIRTH_NM'},
        low_memory=False,
    )
    demo = pd.read_csv(
        structured_data_dir / DEMOGRAPHICS_FILE,
        usecols=lambda c: c in {'DFCI_MRN', 'IDM_RACE_NM', 'HISPANIC_IND'},
        low_memory=False,
    )

    pt_info['dfci_mrn'] = _coerce_mrn(pt_info['DFCI_MRN'])
    demo['dfci_mrn'] = _coerce_mrn(demo['DFCI_MRN'])

    pt_info['birth_dt'] = pd.to_datetime(
        pt_info['BIRTH_DT'], format='%d-%b-%Y', errors='coerce'
    )

    # Sex: prefer GENDER_NM, fall back to SEX_AT_BIRTH_NM, normalize casing.
    gender = pt_info['GENDER_NM'].astype('string').str.strip().str.title()
    if 'SEX_AT_BIRTH_NM' in pt_info.columns:
        sab = pt_info['SEX_AT_BIRTH_NM'].astype('string').str.strip().str.title()
        gender = gender.fillna(sab)
    pt_info['sex'] = _clean_category(gender, valid={'Male', 'Female'})

    # Race: consolidated name; collapse "Unknown ..." variants and the many small
    # multi-race ("A/B") combinations into single buckets to keep strata stable.
    race = demo['IDM_RACE_NM'].astype('string').str.strip()
    race = race.mask(race.str.startswith('Unknown', na=False), UNKNOWN)
    race = race.mask(race.str.contains('/', na=False), 'Multiple')
    demo['race'] = _clean_category(race)

    # Ethnicity from Hispanic indicator.
    hispanic = demo['HISPANIC_IND'].astype('string').str.strip().str.upper()
    demo['ethnicity'] = np.select(
        [hispanic.eq('Y'), hispanic.eq('N')],
        ['Hispanic', 'Non-Hispanic'],
        default=UNKNOWN,
    )

    pt_info = pt_info.dropna(subset=['dfci_mrn']).drop_duplicates(subset=['dfci_mrn'])
    demo = demo.dropna(subset=['dfci_mrn']).drop_duplicates(subset=['dfci_mrn'])

    merged = pt_info[['dfci_mrn', 'birth_dt', 'sex']].merge(
        demo[['dfci_mrn', 'race', 'ethnicity']], on='dfci_mrn', how='outer'
    )
    return merged


def _clean_category(series: pd.Series, valid: Optional[set] = None) -> pd.Series:
    """Fill missing/blank category values with 'Unknown'; optionally restrict."""
    out = series.astype('string').str.strip()
    out = out.mask(out.isna() | out.eq(''), UNKNOWN)
    if valid is not None:
        out = out.where(out.isin(valid | {UNKNOWN}), UNKNOWN)
    return out.astype(object)


def load_summary_dates(parquet_path: str) -> pd.DataFrame:
    """Recover each candidate's treatment-start date from patient_summaries.parquet.

    Returns columns ``dfci_mrn`` (Int64), ``patient_summary``, ``trial_start_dt``
    (datetime), deduplicated to one row per (dfci_mrn, patient_summary).
    """
    df = pd.read_parquet(parquet_path, columns=['dfci_mrn', 'patient_summary', 'trial_start_dt'])
    df['dfci_mrn'] = _coerce_mrn(df['dfci_mrn'])
    df['trial_start_dt'] = pd.to_datetime(df['trial_start_dt'], errors='coerce')
    df = df.dropna(subset=['patient_summary'])
    df = df.drop_duplicates(subset=['dfci_mrn', 'patient_summary'])
    return df[['dfci_mrn', 'patient_summary', 'trial_start_dt']]


def assign_demographics(df: pd.DataFrame,
                        demo: pd.DataFrame,
                        summary_dates: pd.DataFrame) -> pd.DataFrame:
    """Attach age_category, race, ethnicity, sex to an eval candidate frame.

    ``df`` must contain ``dfci_mrn`` and ``patient_summary``. The candidate's
    treatment-start date is joined from ``summary_dates`` on
    (dfci_mrn, patient_summary); demographics are joined from ``demo`` on
    dfci_mrn. Age at treatment start is binned into decade categories. Missing
    values land in the 'Unknown' bucket for every dimension.
    """
    out = df.copy()
    out['dfci_mrn'] = _coerce_mrn(out['dfci_mrn'])

    out = out.merge(summary_dates, on=['dfci_mrn', 'patient_summary'], how='left')
    out = out.merge(demo, on='dfci_mrn', how='left')

    age_years = (out['trial_start_dt'] - out['birth_dt']).dt.days / 365.25
    age_cat = pd.cut(age_years, bins=AGE_BIN_EDGES, labels=AGE_LABELS, right=False)
    out['age_category'] = age_cat.astype(object)
    out['age_category'] = out['age_category'].where(out['age_category'].notna(), UNKNOWN)

    for col in ('race', 'ethnicity', 'sex'):
        out[col] = out[col].where(out[col].notna(), UNKNOWN)

    return out
