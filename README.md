# matchminer-ai-real-eval

Real-patient **denominator pulls** for the MatchMiner-AI retrospective evaluations, ported
from v22 to the 2026 pan-DFCI data (`/data1/ken/pan_dfci_2026`). Two cohorts:

- **Enrollments**: real patients registered to, and starting treatment on, a DFCI
  therapeutic trial that has a ClinicalTrials.gov ("ctgov") listing.
- **SOC**: real patients starting a standard-of-care oncology chemotherapy plan.

This repository holds **code only**. Every output is real-patient data (PHI): write it only
under `../data/phi/` (mode 700 directories, mode 600 files; the scripts set both). Never
commit, print or paste row-level output. The pulls need no GPU, no LLM and no network.

It is deliberately separate from `matchminer-ai-training-kehl`. It is a research tool: the
denominators feed retrospective evaluations of a human-in-the-loop ranking aid, not
eligibility decisions.

## Quick start

```bash
cd /ksg/kehl_mm_data/mmai/v23/matchminer-ai-real-eval
PY=/data1/ken/envs/gptoss3/bin/python          # pandas >= 2.2, pyarrow
$PY -m pytest -q tests/                        # fabricated data only

OUT=../data/phi/real_eval_denominators_$(date +%Y%m%d)
$PY pull_denominators.py --output-dir $OUT     # ~3 min, CPU; prints counts only
$PY sample_soc_trials.py --output-file $OUT/soc_trial_sample.csv   # no patient data

# Optional, large: copy report text before each index date (v22's note_level_dataset)
$PY build_note_level_dataset.py --denominator-csv $OUT/processed_trial_enrollments.csv \
    --output-parquet $OUT/enrollments_note_level_dataset.parquet
```

Main options of `pull_denominators.py` (defaults in brackets):
`--start-date` [2016-01-01], `--end-date` [2025-12-31] (both inclusive, applied to the index
date of both cohorts), `--index-date-rule` [earliest], `--soc-split-filter` [test; '' keeps
all splits], `--days-buffer` [5], `--report-sources` [imaging,clinical,path],
`--no-summary-nct-fill`, `--no-require-ctgov-listing`, `--cohorts` [enrollments,soc], and
every input path (defaults below).

## Files

| File | Role |
| --- | --- |
| `pull_denominators.py` | CLI: both pulls, prior-report counts, outputs and `denominator_summary.json` |
| `mmai_real_eval/enrollments.py` | Enrollment cohort, per-trial table, window flags |
| `mmai_real_eval/soc.py` | SOC cohort |
| `mmai_real_eval/trials.py` | OnCore protocol map, DFCI trial dates, ctgov listings, holdout/training flags, trial sampling |
| `mmai_real_eval/notes.py` | Report counts before the index date; note-level dataset builder |
| `mmai_real_eval/io_utils.py` | Dates, protocol keys, file stamps, private writes, pseudo_mrn, counts |
| `build_note_level_dataset.py` | CLI: v22's note-level dataset (not run by default) |
| `sample_soc_trials.py` | CLI: v22's SOC trial-corpus sample (non-PHI) |
| `tests/test_denominators.py` | Rule tests on fabricated tables |

## What v22 did

v22 code: `/ksg/kehl_mm_data/mmai/v22/matchminer-ai-training/real_eval_code/`. Only the
`prepare` stage of each `run_pipeline.py` defines the patients; everything after it is model
evaluation.

### Enrollments (`eval_phi_enrollments/`)

1. **Upstream notebook** `pan_dfci_2024/code/1.5_pull_eligibility_criteria.ipynb` wrote
   `useful_trial_enrollments.csv`:
   - `PROTOCOL_ENROLLMENT_DFHCC.txt` rows with `TREATMENT_TYPE_CD == 'Tre'`, a
     `TREATMENT_START_DT` and a `STUDY_NBR` ("this gets rid of unspecified active protocols
     that are being masked");
   - joined on the numeric protocol number (`'17-123'` -> 17123) to OnCore protocols with
     `protocol_type == 'Treatment'` and an NCT ID (`oncore_all.clean.csv`), keeping only
     the first protocol per NCT ID (`groupby('nct_id').first()`);
   - joined to title / brief summary / detailed summary / criteria fetched live from the
     ctgov API, dropping trials with an empty brief summary.
2. **`prepare_data.py`**:
   - `parse_enrollments`: OnCore subject records from `forken.json` (an OnCore API export
     keyed by protocol), DFCI-site MRNs only (`studySiteOriginalMrn` starting with "DFCI");
     index date `trial_start_dt` = **`consentDate`**.
   - `load_and_process_trial_enrollments`: one row per (dfci_mrn, protocol) from step 1,
     inner-joined to those consents, `trial_start_dt >= 2016-01-01`, no upper bound (the
     data ended 2024-03-07), one `pseudo_mrn` per unique (dfci_mrn, trial_start_dt).
   - `create_note_level_dataset`: every imaging, clinical and pathology report dated before
     `trial_start_dt + 5 days`. Index events with no such report got no rows, so they never
     reached summarization (an implicit exclusion).
3. **Trial windows.** `trials.parquet` held each trial's `open_date` / `close_date`, which
   retrieval used to search only trials open at the patient's index date. These were not
   DFCI open/close dates. On v22's own files they equal, for 99% of trials, the earliest and
   latest `TREATMENT_START_DT` among the trial's `useful_trial_enrollments` rows on or after
   2016-01-01.

### SOC (`eval_phi_soc/`)

`prepare_data.py` (`load_soc_treatments`, `create_pseudo_mrn`). `pull_soc_treatments.py` is
the same steps without pseudo_mrn, and `run_pipeline.py` does not call it.

1. `TREATMENT_PLAN.txt` rows with a standard or research chemo plan name.
   `plan` = standard name, else research name; `is_trial` = no standard name;
   `dx` = first 3 characters of `TPLAN_ICD_DX_CODES`.
2. `TPLAN_GOAL` present; `is_palliative` = goal contains PALLIATIVE or CONTROL.
3. `TREATMENT_PLAN_CATEGORY == 'ONCOLOGY STANDARD CHEMO PLAN'`.
4. `start_date <= TPLAN_START_DT <= end_date`. `run_pipeline.py` passed 2016-01-01 and
   **2024-01-01**, so SOC follow-up ended with 2023 (`prepare_data.py`'s own default end
   was 2023-01-01).
5. Inner join to the patient split `split_5-2024.csv`; keep `split` containing 'test'.
6. Index date `trial_start_dt` = `TPLAN_START_DT`; one pseudo_mrn per (dfci_mrn, date). A
   patient can contribute several index events (no washout, no line-of-therapy rule).
7. Note-level dataset as for enrollments.

The SOC trial corpus (`prepare_trials.py`; `pull_trials.py` is an older copy): all trials
in v20's spacified ctgov corpus minus the synthetic-training trials, 500 unique NCT IDs
sampled with seed 42.

## What changed and why

### Data source mapping

| v22 input | v23 input | Notes |
| --- | --- | --- |
| `pan_dfci_2024/structured_data/PROTOCOL_ENROLLMENT_DFHCC.txt` (pipe, latin-1; April 2024 extract) | `pan_dfci_2026/structured_data/PROTOCOL_ENROLLMENT.csv` (comma, UTF-8; extract 2026-08-31) | Same core columns. Dropped: `PATIENT_ID`, `D_*`/`AGE90_*` dates, `*_DESCR` (now `*_DSC`). New: `REG_DT`, `PROTOCOL_ID`, `SUMMARY_TYPE_DSC`. |
| `forken.json` (OnCore subject export, consent dates) | **none** | No consent date exists in pan_dfci_2026. See "Index date". |
| `pan_dfci_2024/structured_data/oncore_all.clean.csv` | `data/no_phi/oncore_nct_status_10-17-26.csv`, filled from `data/no_phi/dfci_trials_summary_10-18-26.csv` | Same OnCore fields (`protocol_no`, `nct_id`, `protocol_type`). Every v22 'Treatment' protocol with an NCT ID is in the v23 file with the same NCT ID. |
| live ctgov API (2024) | `data/no_phi/ctgov_trials.csv` (10-15-26 export, 148,056 trials) | No network calls. The export has no detailed summary. |
| `trials.parquet` dates (derived from enrollments) | same derivation + `dfci_trials_summary_10-18-26.csv` + OnCore accrual dates | See "Trial windows". |
| `pan_dfci_2024/structured_data/TREATMENT_PLAN.txt` (pipe, latin-1; April 2024) | `pan_dfci_2026/structured_data/TREATMENT_PLAN.csv` (comma, UTF-8; 2026-08-31) | Dropped: `TPLAN_ID`, `PATIENT_ID`, `TPLAN_SOURCE_INST`, `D_*`/`AGE90_*`. Renamed: `TPLAN_DX_NM`→`TPLAN_DX_NAME`. `TPLAN_DC_REASON` values gained letter prefixes. |
| `pan_dfci_2024/derived_data/split_5-2024.csv` | `pan_dfci_2026/derived_data/split_8-2026_with_text_but_not_structured_data_yet.csv` | Carries forward every earlier assignment: all 608,099 MRNs of the 2024 split keep their split. Despite its name it covers every patient in both cohorts. |
| `pan_dfci_2024/derived_data/all_{imaging_reports,clinical_notes,path_reports}.parquet` | same names in `pan_dfci_2026/derived_data/` | Same columns plus `RPT_ID`. Reports run to July 2026. The new `all_rad_completion.parquet` is available (`--report-sources radonc`) but off by default. |

### Follow-up

`--end-date` (default **2025-12-31**, inclusive) now bounds both cohorts, and `--start-date`
stays 2016-01-01. v22 did not end "through 2024": SOC ended 2024-01-01, and enrollments had
no bound but the 2024 data ended 2024-03-07 (16 enrollments in 2024).

### Enrollment definition changes

- **Index date (forced change).** v22 used the OnCore consent date from `forken.json`;
  pan_dfci_2026 has none. The default `--index-date-rule earliest` uses the earliest of
  `REG_DT` (OnCore registration timestamp, floored to the day), `PROTOCOL_ON_DT` and
  `TREATMENT_START_DT`. Consent precedes all three. `REG_DT` gives the index for 97.9% of
  rows; `PROTOCOL_ON_DT` is earlier for the rest.
  - Validation against v22's consent dates: 10,443 registrations appear in both sources.
    For the 8,144 with consent from 2016, the v23 index falls on the consent day for 46.7%,
    0-5 days after consent for 77.7%, more than 30 days after for 2.1% and before consent
    for 0.5% (median 1 day after).
  - For v22's 8,033 output rows (7,977 found in v23 by patient and protocol), the median v23
    index is 1 day after v22's, the 95th percentile 20 days after.
  - By contrast, `PROTOCOL_ON_DT` (OnCore on-study date) is a median 11 days after consent,
    and `TREATMENT_START_DT` 14 days. The other rules are kept as options.
  - Every row keeps `reg_dt`, `protocol_on_dt` and `treatment_start_dt`.
  - With the 5-day buffer, a note cutoff of index + 5 days falls a median 6 days after
    consent, against 5 in v22.
- **No consent-file restriction (forced change).** `forken.json` covered only 977 of v22's
  1,894 protocols; only 8,078 of 16,124 eligible 2016+ (patient, protocol) pairs had a consent
  record. v22's enrollment denominator was therefore about half of all qualifying treatment
  registrations, selected by protocol coverage, not by any clinical rule. v23 keeps every
  qualifying registration. That alone roughly doubles 2016-2023 counts relative to v22.
- **No DFCI-site restriction (forced change).** v22 dropped consents recorded at DFCI
  satellites (about 3% of forken records). PROTOCOL_ENROLLMENT has no site column.
- **Masked active protocols (unchanged, but larger).** The extract replaces the protocol of
  every registration on a protocol still active at extraction with `STUDY_NBR '0000'`
  ("Unspecified Active Therapeutic Protocol"). Those registrations cannot be tied to a trial
  and are excluded, as in v22. Because recent trials are mostly still active, this removes a
  growing share of recent registrations (see below). Recovering them would need an OnCore
  subject export for active protocols, like v22's `forken.json`.
- **Trial map.** OnCore 'Treatment' protocols with an NCT ID, as in v22, with two changes:
  - **NCT IDs filled in (judgment call).** A protocol absent from the status file (mostly
    older protocols, also absent from v22's list), or present without an NCT ID, takes its
    NCT ID from the DFCI trial summary (the full OnCore dump). Its therapeutic type then
    rests on `TREATMENT_TYPE_CD == 'Tre'`: every 'Tre' protocol that is in the status file
    (2,431) is OnCore 'Treatment' there. This adds 289 rows (`nct_source`). `--no-summary-nct-fill` restores
    the status-file-only map.
  - **No first-protocol-per-NCT dedup (judgment call).** v22's rule would drop 113 rows on a
    second protocol sharing an NCT ID. They are kept, and `first_protocol_for_nct` marks
    them.
- **ctgov listing requirement** kept, against the local export instead of the API (31
  in-window rows dropped). `--no-require-ctgov-listing` turns it off.
- Exact duplicate (patient, protocol, index date) rows are dropped (none in this extract).
  Distinct registrations of one patient to one protocol on different dates are kept, as v22
  kept every consent.
- `pseudo_mrn` is numbered in (index date, dfci_mrn) order so reruns are reproducible.
  v22 numbered by order of appearance. Enrollment and SOC pseudo_mrns are separate number
  spaces, as in v22.

### Trial windows (enrollments)

`enrolled_trials.csv` carries three windows per trial, and `processed_trial_enrollments.csv`
flags whether each index date falls in each:

- `first/last_dfci_treatment_start` (`index_in_v22_style_window`): v22's `trials.parquet`
  definition, over qualifying registrations with a treatment start in the follow-up window.
  - Because the index is consent-like but the window is built from treatment starts, early
    enrollees fall before their own trial's "open" date: 2,611 of 22,306 v23 rows.
  - The same happened in v22: 1,065 of 8,027 v22 rows had their consent before their trial's
    `open_date`. v22's patient-centric retrieval therefore could not return the enrolled
    trial for about 13% of patients. This is reported, not fixed (retrieval is out of scope).
- `dfci_accrual_open_dt` / `dfci_accrual_close_dt` (`index_in_dfci_accrual_window`): OnCore
  accrual open/close dates from PROTOCOL_ENROLLMENT, i.e. actual DFCI open/close dates.
  22,301 of 22,306 index dates fall inside. **Recommended** for future retrieval windows.
- `ctgov_start_date` / `ctgov_completion_date` (`index_in_ctgov_window`): from
  `dfci_trials_summary_10-18-26.csv`.
  - Despite the file's purpose, its `open_date`/`close_date` are the ctgov **study**
    start/completion dates (99% / 97.5% identical to the ctgov export), not DFCI site dates.
  - They have month or day precision, so month dates are widened to the whole month.
  - Missing for 16% of trials.

Trial flags: `in_v23_training_sample` (in the 5,000 synthetic-training trials:
`training_run_screened_sample5000_20261104/registry_sample_trials.csv`) and
`v23_dfci_holdout_2025_2026` (the training path's "open at DFCI 2025-2026" rule from
`0c_sample_trials.py`, re-implemented here). v22 did not exclude training trials from
enrollments, and neither does v23. The flags let an evaluation do so.

### SOC definition changes

- Rules 1-6 unchanged. Split = the 8-2026 file (same assignments for every earlier patient);
  the filter is exact `split == 'test'` (v22 used `str.contains('test')`, equivalent here).
- `TPLAN_ID` no longer exists; `epic_tplan_id` (unique per plan) is kept instead. No v22
  downstream code read `tplan_id`.
- v22 kept plans from every source institution (23% of its SOC rows came from MGH, BWH and
  others via `TPLAN_SOURCE_INST`). The new extract has no institution column, so v23 keeps
  them all as well. Every 2016-2025 SOC row is now an Epic plan.
- Provider and department names are no longer copied into the output (no downstream use).

### Prior reports (both cohorts)

`n_prior_reports` / `has_prior_reports` count imaging, clinical and pathology reports dated
before index + `--days-buffer` days: exactly v22's note-level cutoff, computed from dates
only. Rows with `has_prior_reports == False` would get no notes and would not have reached
v22's summaries. They are kept and flagged, not dropped.

## Outputs

All under `--output-dir`. CSV dates are `YYYY-MM-DD`.

**`processed_trial_enrollments.csv`**: one row per qualifying registration.
`pseudo_mrn`, `dfci_mrn`, `protocol_number` (OnCore, `17-123`), `nct_id`,
`trial_start_dt` (index date), `index_date_rule`, `reg_dt`, `protocol_on_dt`,
`treatment_start_dt`, `patient_study_seq` (OnCore row key), `split` (information only),
`dfci_accrual_open_dt`, `dfci_accrual_close_dt`, `index_in_dfci_accrual_window`,
`index_in_v22_style_window`, `index_in_ctgov_window`, `nct_source`,
`first_protocol_for_nct`, `n_prior_reports`, `has_prior_reports`.

**`enrolled_trials.csv`**: one row per NCT ID in the enrollments.
- Identity and OnCore: `nct_id`, `protocol_numbers` (`;`-joined), `oncore_status`,
  `oncore_phase`.
- Counts: `n_rows`, `n_index_events`, `n_patients`.
- Windows: `first_dfci_treatment_start`, `last_dfci_treatment_start`,
  `dfci_accrual_open_dt`, `dfci_accrual_close_dt`, `ctgov_start_date`,
  `ctgov_start_date_type`, `ctgov_completion_date`, `ctgov_completion_date_type`.
- Flags: `in_v23_training_sample`, `v23_dfci_holdout_2025_2026`.
- Listing text: `title`, `brief_summary`, `eligibility_criteria`, `trial_text`.

The listing columns are what `matchminer-ai-training-kehl/real_eval_code/eval_cohort_lineitems.py
--trials` reads.

**`processed_soc_treatments.csv`**: one row per qualifying plan.
`pseudo_mrn`, `dfci_mrn`, `split`, `trial_start_dt` (= `tplan_start_dt`), `tplan_start_dt`,
`epic_tplan_id`, `plan`, `is_trial` (always 0), `treatment_plan_category`, `tplan_goal`,
`is_palliative`, `dx`, `tplan_icd_dx_codes`, `tplan_dx_name`, `protocol_nbr`,
`first_cycle_dt`, `tplan_dc_reason`, `tplan_source_nm`, `n_prior_reports`,
`has_prior_reports`.

**`denominator_summary.json`** contains:
- run parameters and input file stamps (size and mtime, no hashing);
- code state;
- attrition at each step;
- masked-protocol counts by year;
- window-flag counts;
- SOC counts for every split before the split filter;
- per-year rows, index events and patients, with and without `has_prior_reports`.

**`soc_trial_sample.csv`** (+ `.sampling.json`), from `sample_soc_trials.py`: 500 NCT IDs
with listing text. No patient data.
- Pool: the v23 analog of v22's, i.e. trials the v23 training path screened as cancer
  treatment (`screened_trials.csv`, 21,789), minus the 5,000 training trials.
- That pool differs from v22's (v23 trials were first posted 2020 or later and passed an LLM
  screen when the training run built them; no LLM runs here).

**`build_note_level_dataset.py`** writes `pseudo_mrn, dfci_mrn, trial_start_dt, date,
note_type, RPT_ID, text`. It differs from v22 in four ways:
- one copy of the notes per index event, where v22 repeated them per enrollment row;
- no trial or plan columns copied onto every note (join on `pseudo_mrn` instead);
- only the report columns all sources share;
- rows not globally sorted.

It refuses output paths outside `data/phi/` or `/data1/`, and was not run on 2026-11-14.

## Run of 2026-11-14

Output: `/ksg/kehl_mm_data/mmai/v23/data/phi/real_eval_denominators_20261114/`.
Index events = unique (patient, index date).

| Index year | Enrollment rows | Enrollment index events | Enrollment patients | Masked registrations excluded | SOC rows | SOC index events | SOC patients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2016 | 2,703 | 2,703 | 2,537 | 32 | 1,265 | 1,251 | 901 |
| 2017 | 3,001 | 2,999 | 2,789 | 104 | 1,276 | 1,246 | 893 |
| 2018 | 3,217 | 3,215 | 2,958 | 131 | 1,384 | 1,341 | 968 |
| 2019 | 3,027 | 3,026 | 2,812 | 193 | 1,445 | 1,412 | 1,037 |
| 2020 | 2,079 | 2,079 | 1,969 | 212 | 1,623 | 1,588 | 1,108 |
| 2021 | 2,196 | 2,196 | 2,086 | 438 | 1,643 | 1,609 | 1,133 |
| 2022 | 1,956 | 1,956 | 1,866 | 562 | 1,711 | 1,663 | 1,173 |
| 2023 | 1,829 | 1,828 | 1,750 | 983 | 1,799 | 1,761 | 1,243 |
| 2024 | 1,471 | 1,471 | 1,433 | 1,398 | 1,863 | 1,825 | 1,288 |
| 2025 | 827 | 826 | 818 | 2,077 | 1,980 | 1,919 | 1,348 |
| **All** | **22,306** | **22,299** | **18,392** | **6,130** | **15,989** | **15,615** | **7,680** |

Totals:
- Enrollments: 2,074 trials. Of 28,545 in-window therapeutic registrations with a treatment
  start, 22,306 are kept, 6,130 are masked and 109 lack an NCT ID or ctgov listing. With
  `has_prior_reports`: 21,716 index events, 17,887 patients.
- SOC (test split): with `has_prior_reports`: 15,568 index events, 7,653 patients. All
  splits: 159,076 rows, 76,697 patients.

Concordance with v22 over the period they share:
- SOC: 10,794 of v22's 11,554 index events (2016-01-01..2024-01-01) are reproduced. 748 of
  the 760 misses have no plan row on that date in the 2026 extract.
- Enrollments: 7,977 of v22's 8,033 rows reappear by patient and protocol.

## Limitations and open questions

- **2024-2025 enrollments are incomplete.** Only 827 of 2,918 in-window 2025 therapeutic
  registrations, and 1,471 of 2,869 in 2024, are on protocols identifiable in this extract.
  The rest are on protocols still active in August 2026 and masked. The identifiable recent
  ones are biased toward trials that closed quickly. A fresh OnCore subject export would fix
  this.
- The enrollment index date is a consent proxy (validated above), not the consent date.
- v22's `forken.json` coverage and site restriction cannot be reproduced. That is the main
  reason the 2016-2023 enrollment counts are about 2.5 times v22's.
- The ctgov-listing and NCT-fill rules use local files dated October 2026. v22 used the
  2024 API.
- `dfci_trials_summary_10-18-26.csv` does not hold DFCI site dates (see "Trial windows").
- The split file's name says structured data was not yet included. It nevertheless covers
  every patient in both cohorts. When the next split file is written it should keep the
  same assignments.
- SOC keeps v22's breadth: every new standard chemo plan, including amended re-plans,
  plans later deleted (`tplan_dc_reason`), plans with no recorded first cycle, and plans
  possibly from non-DFCI sites.

## Not ported (model evaluation, out of scope)

From `eval_phi_enrollments/` and `eval_phi_soc/` (each directory has its own copy unless
noted):

- `run_pipeline.py`: orchestrator; only its `prepare` stage is ported.
- `patient_centric_retrieval.py`, `trial_centric_retrieval.py`: TrialSpace retrieval.
- `check_eligibility.py`, `check_boilerplate.py`: LLM checks.
- `vllm_parallel_trialcheck.py`, `vllm_parallel_boilerplate.py`: vLLM checker runs.
- `eval_llm_trial_checker.py`, `eval_llm_boilerplate_checker.py`: LLM checker evaluation.
- `eval_modernbert_trial_checker.py`, `eval_modernbert_boilerplate_checker.py`: ModernBERT
  checker evaluation.
- `eval_baseline.py`: embedding baseline.
- `eval_demographics.py`, `demographics_utils.py`: demographic stratification of metrics.
- `eval_utils.py`, `utils_102023.py` (SOC only): metric and plot helpers.
- `run_summarization_experiments.py`, `review_summarization_experiments.ipynb`
  (enrollments only): summarization experiments.
- `spacify_dfci_trials.py` (enrollments only): LLM trial-space extraction.
  `enrolled_trials.csv` is the input for v23's cohort extraction instead.
- `pull_soc_treatments.py`, `pull_trials.py` (SOC only): duplicates of the ported
  `prepare_data.py` / `prepare_trials.py` logic.
