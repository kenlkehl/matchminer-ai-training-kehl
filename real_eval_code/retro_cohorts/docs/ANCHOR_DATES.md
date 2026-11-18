# Anchor dates: index dates and note cutoffs

This file records every date-anchor decision for the two real-patient cohorts of the MatchMiner-AI retrospective
evaluation:

- the **trial cohort**: patients starting research treatment on a DFCI therapeutic trial;
- the **standard-of-care (SOC) cohort**: patients starting a standard chemotherapy plan.

For each cohort it says which date a patient is matched at (the index date), which notes may be summarized for that
match (the note cutoff), and why. It holds no patient-level data. Every figure is a count or an aggregate over the
2026-11-14 outputs in `data/phi/real_eval_denominators_20261114/`. Implementation details (regular expressions, flags
not listed here, output files) are in `README.md`.

Decisions current as of 2026-11-14.

## Terms

- **Index date** (`trial_start_dt`): the date the patient is matched at. In both current cohorts it is the start of
  treatment; in v22's trial cohort it was the consent date. One index event is one `pseudo_mrn`, which is one
  (patient, index date).
- **Note anchor**: the date the note cutoff is counted from. v22 and the SOC cohort use the index date. The trial
  cohort uses `note_anchor_dt`, the patient's earliest known registration to the trial (section 2.2).
- **Note cutoff** (`note_cutoff_dt`) = note anchor + offset in days (negative = before the anchor). A report is kept
  for summarization only if it is dated strictly before the cutoff (`date < note_cutoff_dt`). Later reports never reach
  the patient summary.
- **Dates are calendar days.** Index, plan and registration dates are floored to the day (time of day dropped). The
  cutoff is midnight at the start of the cutoff day, so a report written at any time on the cutoff day is excluded.
  The last day kept is the cutoff minus 1 day.
- Reports are imaging reports, clinical notes and pathology reports (`all_imaging_reports`, `all_clinical_notes`,
  `all_path_reports` in `/data1/ken/pan_dfci_2026/derived_data/`). There is no lower date limit: every report before
  the cutoff is kept, including reports from before the 2016 start of the index window.

## At a glance

| | v22 trial (enrollments) | v22 SOC | Trial cohort now | SOC cohort now |
| --- | --- | --- | --- | --- |
| Index event | OnCore consent | standard chemo plan start | first research chemo plan start on the protocol | standard chemo plan start (unchanged) |
| Note anchor | index | index | `note_anchor_dt` (earliest known registration, else the index) | index (`trial_start_dt` = `tplan_start_dt`) |
| Offset | +5 days | +5 days | −14 days | −14 days |
| Last report day kept | index + 4 | plan start + 4 | anchor − 15 | plan start − 15 |
| Masking of note text | none | none | own protocol number, NCT ID and title drug codes | none |
| Cutoff to treatment start | about 9 days before (median, see 2.1) | 5 days after | median 28 days before (10th-90th percentile 16-49) | exactly 14 days before |

## 1. Index dates

### 1.1 v22 (for reference)

v22 code: `/ksg/kehl_mm_data/mmai/v22/matchminer-ai-training/real_eval_code/eval_phi_enrollments/prepare_data.py` and
`.../eval_phi_soc/prepare_data.py`.

- **Trial cohort (enrollments):** `trial_start_dt` = the OnCore **consent date** (`consentDate` in `forken.json`, an
  OnCore subject export), for DFCI-site MRNs only. Index dates from 2016-01-01, with no upper bound (the 2024 data
  ended 2024-03-07).
- **SOC:** `trial_start_dt` = `TPLAN_START_DT`, the start of an `ONCOLOGY STANDARD CHEMO PLAN`, from 2016-01-01
  through 2024-01-01 inclusive (the dates `run_pipeline.py` passed).

### 1.2 Trial cohort from the enrollment table (`--trial-source enrollment_table`; now a cross-check)

The 2026 extract (`PROTOCOL_ENROLLMENT.csv`, extracted 2026-08-31) has **no consent date** and there is no
`forken.json` equivalent. This pull therefore uses a consent proxy from the enrollment table itself:

- `trial_start_dt` = the **earliest of `REG_DT`** (OnCore registration timestamp, floored to the day),
  **`PROTOCOL_ON_DT`** and **`TREATMENT_START_DT`** (`--index-date-rule earliest`, the default; `reg_dt`,
  `protocol_on_dt` and `treatment_start_dt` are the other choices). Consent comes before all three, so the earliest is
  the closest available proxy. `REG_DT` gives the index for 97.9% of rows. Each row keeps `reg_dt`,
  `protocol_on_dt`, `treatment_start_dt` and `index_date_rule`.
- **Validation against v22's consent dates** (README, "Enrollment definition changes"):
  - 10,443 registrations appear in both sources. For the 8,144 with consent from 2016 on, the v23 index falls on the
    consent day for 46.7%, 0-5 days after consent for 77.7%, more than 30 days after for 2.1% and before consent for
    0.5% (median 1 day after).
  - For v22's 8,033 output rows (7,977 found again by patient and protocol), the median v23 index is 1 day after
    v22's, the 95th percentile 20 days after.
  - By contrast, `PROTOCOL_ON_DT` is a median 11 days after consent and `TREATMENT_START_DT` a median 14 days.
- **Why it is no longer the default.** The extract replaces the protocol of every registration on a protocol still
  active at extraction with `STUDY_NBR` `0000` ("Unspecified Active Therapeutic Protocol"). Those registrations
  cannot be tied to a trial and are dropped: 6,130 of 28,545 in-window therapeutic registrations with a treatment
  start (22%), rising from 32 in 2016 to 2,077 in 2025 (71% of 2025's).
- Its 2026-11-14 output (22,306 rows) is kept in `enrollment_table_v1/`.

### 1.3 Trial cohort from research chemo plans (`--trial-source research_plans`; default since 2026-11-14)

- **Why.** `PROTOCOL_ENROLLMENT` masks active protocols as `STUDY_NBR` `0000`. Research chemo plans in
  `TREATMENT_PLAN` keep the protocol number (`RESEARCH_CHEMO_PLAN_NBR`), currently open protocols included: only 56 of
  44,665 research plans have no number.
  - Checked on unmasked registrations: the same patient's research plan starting 14 days before to 90 days after the
    registration named the true protocol 99% of the time; 79% of registrations had such a plan.
  - 98% of research-plan starts map to an NCT ID in every year 2016-2025.
- **Definition** (`mmai_real_eval/research_plans.py`):
  - One index event per (patient, protocol): the patient's **first** `ONCOLOGY RESEARCH CHEMO PLAN` on that protocol,
    on any date.
  - `trial_start_dt` = that plan's `TPLAN_START_DT`, floored to the day (`index_date_rule` =
    `first_research_plan_start`).
  - The event counts only if that first start falls in the window, so a later re-plan is never a new start.
  - The protocol must be an OnCore `protocol_type` "Treatment" protocol with an NCT ID, and by default the NCT ID needs
    a local ClinicalTrials.gov listing with a brief summary.
- **What the date means.** The start of research treatment: the same kind of event as the SOC cohort's plan start. It
  comes typically days to weeks after registration (median 14 days, section 3.3).
- **2026-11-14 run:** 21,803 trial starts (each its own index event and row), 17,730 patients, 2,086 trials.
- **Missed by construction:** trials without a research chemo plan, such as oral-only, radiation or surgical
  protocols.

### 1.4 SOC cohort (unchanged from v22)

- `trial_start_dt` = `tplan_start_dt` = `TPLAN_START_DT` of an `ONCOLOGY STANDARD CHEMO PLAN` with a plan goal, for
  patients in the `test` split (`mmai_real_eval/soc.py`). `trial_start_dt` is only an alias, kept so code can be shared
  with the trial cohort.
- One `pseudo_mrn` per (patient, plan start date). Plans starting the same day share it.
- A patient can contribute several index events: there is no washout and no line-of-therapy rule, and amended
  re-plans count as new plans.
- **2026-11-14 run:** 15,989 plans (rows), 15,615 index events, 7,680 patients.

### 1.5 Index window

- `--start-date 2016-01-01` and `--end-date 2025-12-31`, **both inclusive**, applied to the index date of both cohorts
  (for the research-plan pull, to the first research plan start on the protocol).
- v22: enrollments from 2016-01-01 with no end; SOC 2016-01-01 through 2024-01-01.
- The window bounds index dates only, not report dates (see "Terms").

## 2. Note anchors and cutoffs

### 2.1 v22

Both cohorts kept every report dated before **index + 5 days** (`date < trial_start_dt + 5 days`, `days_buffer=5`):
the consent date for enrollments, the plan start for SOC.

- For enrollments this admits the consent-visit note, which names the trial.
- For SOC it admits the first days of treatment.
- So v22 was itself asymmetric. The SOC cutoff fell 5 days after treatment start. The enrollment cutoff fell roughly
  9 days before it, since treatment start (`TREATMENT_START_DT`) was a median 14 days after consent (from the medians
  in 1.2).
- An index event with no report before its cutoff got no rows and silently never reached summarization.

### 2.2 Trial cohort now: `note_anchor_dt` − 14 days, with masking

**Anchor.** `note_anchor_dt` is the earliest of three dates (`research_plans.py`):

1. `registration_dt`: the patient's earliest unmasked therapeutic registration on the **same protocol**. For each
   registration the date is the earliest of `REG_DT`, `PROTOCOL_ON_DT` and `TREATMENT_START_DT`, floored to the day.
2. `masked_registration_dt`: used **only when there is no same-protocol registration**. It is the patient's masked
   (`STUDY_NBR` `0000`) therapeutic registration nearest the plan start, among those where the plan starts 14 days
   before to 90 days after the registration (that is, registration 90 days before to 14 days after the index). The
   masked registration's protocol is unknown.
3. `trial_start_dt`: the index (first research plan start).

**Cutoff.** `note_cutoff_dt` = `note_anchor_dt` − 14 days; reports with `date < note_cutoff_dt` are kept.

**Why the registration and not the plan start.** Registration and screening notes name the trial (section 3.1). The
plan start comes a median 14 days after registration, and most of the notes that name the patient's own protocol fall
in that stretch.

**Masking (decided 2026-11-14; being implemented in a separate change; details in `README.md`).** In each index
event's kept notes, that event's own trial identifiers are replaced:

- its own protocol number(s) → `[TRIAL PROTOCOL]`;
- its own NCT ID → `[NCT ID]`;
- investigational drug codes from its own trial's ClinicalTrials.gov title (for example a code like ABC-1234) →
  `[STUDY DRUG CODE]`.

Nothing else is masked. Other NN-NNN numbers stay, including other protocols the patient was on (section 3.2). After a
masked build the leakage checks are re-run on the written text.

**Where each anchor comes from (2026-11-14 run, 21,803 trial starts).**

| `enrollment_table_match` | Starts | Anchor | Anchor = plan start because |
| --- | ---: | --- | --- |
| `same_protocol` | 17,830 | `registration_dt` | registration on the plan-start day: 590; registration after the plan start: 43 |
| `masked_registration_nearby` | 3,822 | `masked_registration_dt` | same day: 72; after the plan start (up to 14 days): 10 |
| `none` | 151 | `trial_start_dt` (fallback) | no registration found |

- `note_anchor_dt` falls before the plan start for 20,937 starts and on the plan start for 866: 151 true fallbacks
  with no registration, 662 registered the same day as the plan start and 53 registered after it.
- Anchor to plan start: median 14 days (10th-90th percentile 2-35; 99th percentile 170 days). 448 starts have their
  earliest same-protocol registration more than 90 days before the plan, 74 more than a year before.
- Cutoff to plan start: median 28 days (10th-90th percentile 16-49).

### 2.3 SOC cohort now: plan start − 14 days, no masking

- **User decision, 2026-11-14.** `--anchor index --days-offset -14`: `note_cutoff_dt` = `trial_start_dt` − 14 days =
  `tplan_start_dt` − 14 days. Reports with `date < note_cutoff_dt` are kept, so the last day kept is plan start − 15.
- **No masking** (reasons in section 3.4). The SOC denominator has no protocol number or NCT ID, so there is nothing
  event-specific to mask.
- **2026-11-14 run** (`soc_note_level_dataset_minus14.parquet` and its `.summary.json`, code commit `51e2542`):
  - 711,764 report rows: 356,367 imaging, 218,784 clinical, 136,613 pathology.
  - 15,615 index events: 14,808 with at least one report and 807 (5.2%) with none. Under v22's index + 5 days rule,
    47 events had no report.
  - Reports per index event: median 30 over all events (33 among events with reports), 10th-90th percentile 3-109.
  - Checks passed: every row has `date < note_cutoff_dt`; `note_cutoff_dt` = `trial_start_dt` − 14 days in every row;
    the latest report kept is dated plan start − 15 days; a separate recount from report dates alone matched every
    source; the file is mode 600.

### 2.4 Column reference

| Column | File | Meaning |
| --- | --- | --- |
| `trial_start_dt` | both denominators; note-level dataset | Index date. Trial: first research plan start. SOC: plan start. |
| `tplan_start_dt` | SOC denominator | `TPLAN_START_DT` floored to the day; equal to `trial_start_dt` in every row. |
| `registration_dt` | trial denominator | Earliest unmasked registration on the same protocol; empty if none. |
| `masked_registration_dt` | trial denominator | Nearest masked registration in the window of 2.2; filled only when `registration_dt` is empty. |
| `note_anchor_dt` | trial denominator | Earliest of `registration_dt`, `masked_registration_dt` and `trial_start_dt`; never empty. |
| `enrollment_table_match` | trial denominator | `same_protocol`, `masked_registration_nearby` or `none`. |
| `days_registration_to_index` | trial denominator | `trial_start_dt` − `registration_dt` in days (same-protocol registrations only). |
| `note_cutoff_dt` | note-level dataset | Note anchor + `--days-offset`; every row has `date < note_cutoff_dt`. |
| `date` | note-level dataset | The report's date (a timestamp; compared with the midnight cutoff). |
| `n_prior_reports`, `has_prior_reports` | both denominators | Reports before index + 5 days (v22's rule; information only, see section 4). |
| `reg_dt`, `protocol_on_dt`, `treatment_start_dt`, `index_date_rule` | enrollment-table pull only | The three registration dates and the rule that chose the index. |

The note-level dataset written by `build_note_level_dataset.py` has `pseudo_mrn, dfci_mrn, trial_start_dt,
note_cutoff_dt, date, note_type, RPT_ID, text` (plus per-note masking counts for the masked trial build, see README).
Join it to the denominator on `pseudo_mrn`.

## 3. Evidence

### 3.1 How often kept notes name the patient's own trial

From `measure_note_leakage.py` over the 21,803 research-plan trial starts (`note_leakage_summary.json`). A start
counts when at least one of its kept notes contains the identifier. Cutoffs are counted from `note_anchor_dt`; the
first row keeps every note before the plan start. Drug codes apply only to the 7,013 starts whose trial title
contains an investigational code.

| Cutoff | Notes kept (summed over starts) | Own protocol number | Own NCT ID | Own title drug code | Any of the three |
| --- | ---: | ---: | ---: | ---: | ---: |
| every note before the plan start | 1,170,477 | 71.9% | 0.56% | 16.3% | 73.3% |
| registration (anchor + 0) | 1,076,185 | 37.0% | 0.43% | 9.4% | 39.5% |
| anchor − 7 days | 1,036,824 | 19.8% | 0.33% | 5.8% | 22.1% |
| **anchor − 14 days (chosen)** | **1,011,642** | **13.1%** | **0.23%** | **3.9%** | **14.9%** |
| anchor − 30 days | 972,888 | 7.3% | 0.19% | 2.3% | 8.5% |
| anchor − 60 days | 927,972 | 4.4% | 0.13% | 1.5% | 5.3% |
| anchor − 90 days | 888,784 | 2.9% | 0.10% | 1.1% | 3.6% |

In round numbers, the share of trial starts whose kept notes name their own protocol number is 72% with every note
before the plan start, 37% at registration, 20% at −7 days, 13% at −14, 7% at −30 and 3% at −90.

The first row is not v22's rule: v22 kept notes through consent + 4 days, and its consent-based index comes earlier
than this plan start.

**Where the mentions are.** Of the 38,362 notes before the plan start that contain the patient's own protocol number,
25,702 (67%) are dated between registration and the plan start, 5,855 in the week before registration and 2,165 in
the week before that. The −14 cutoff removes 88% of them and leaves 4,640 notes; masking replaces the identifier in
those.

**What the cutoff costs.** At −14 days, 1,011,642 of the 1,170,477 pre-plan notes (86%) are still kept.

### 3.2 Why masking is per event and not a blanket NN-NNN mask

OnCore protocol numbers look like `17-123`. At the −14 cutoff, 276,440 of the 1,011,642 kept notes (27%) contain some
other NN-NNN number (lab ranges, doses and the like): about the same share at every cutoff (26% before the plan
start, 27% at registration). Masking every such pattern would alter more than a quarter of the notes. Masking is
therefore limited to each event's own identifiers.

### 3.3 Registration to first research plan

`days_registration_to_index` over the 17,830 starts with a same-protocol registration: median 14 days (10th-90th
percentile 2-36; 25th-75th 7-21). For the 3,822 starts matched to a masked registration: median 15 days (10th-90th
percentile 4-35).

### 3.4 Why SOC is not masked

- Drug and regimen names in a SOC patient's notes also record real prior history: earlier lines of therapy, the same
  drug or drug class given before, an earlier course of the same regimen. Masking them would remove clinical history
  the summary needs.
- SOC has no unique identifier like a protocol number or NCT ID. A standard regimen name is not specific to one index
  event.
- No leakage measurement has been made for SOC. The −14-day cutoff is its only protection.

## 4. Asymmetries and caveats

1. **Different gaps to treatment start.** Trial summaries end about 4 weeks before research treatment starts
   (registration − 14 days, with registration a median 14 days before the plan: median 28 days, 10th-90th percentile
   16-49). SOC summaries end exactly 2 weeks before plan start. Trial notes are also masked and SOC notes are not.
   Comparisons between the cohorts (for example, how recent the last imaging is, or how current disease status is)
   should keep this in mind.
2. **Fallback to the index.** For 151 trial starts with no registration, the note anchor falls back to the plan start,
   so the cutoff is plan start − 14 days, as for SOC. The anchor also equals the plan start for 715 starts registered
   on or after the plan-start day (866 in all; section 2.2).
3. **Masked registrations are of unknown protocol.** For the 3,822 starts anchored on a masked registration, that
   registration may belong to a different trial. It is used only when no same-protocol registration exists.
4. **Long registration-to-plan gaps.** `registration_dt` is the earliest registration on the protocol. For 448 starts it
   is more than 90 days before the first research plan (74 more than a year), so their cutoff is far before treatment.
5. **`has_prior_reports` / `n_prior_reports`** in both denominators still count reports before index + 5 days (v22's
   rule, `--days-buffer 5`), as information only. They do not describe the −14-day datasets. The counts of events
   with no kept note are in each note-level summary JSON.
6. **Day-level, exclusive.** Dates are floored to the day and the cutoff day itself is excluded (see "Terms").
7. **Several SOC events per patient.** A patient's later plan sees the notes about earlier plans as history. An amended
   re-plan of the same regimen can have notes naming that regimen before its cutoff. The trial cohort has one event
   per patient and protocol (re-plans are not new starts).
8. **Events with no kept report produce no rows** in the note-level dataset, so they are not summarized unless a later
   step handles them. With the −14 cutoffs this happens to 807 of 15,615 SOC events (5.2%) and to 1,985 of 21,803 trial
   starts (9.1%, from the masked trial build's 2026-11-14 summary JSON). The trial cutoff sits further before treatment,
   which may contribute to that difference.

## 5. Commands

From the repository root, with `/data1/ken/envs/gptoss3/bin/python` (CPU only):

```bash
OUT=../data/phi/real_eval_denominators_20261114

# Both denominators. Defaults shown: --trial-source research_plans --start-date 2016-01-01 --end-date 2025-12-31
# --soc-split-filter test --days-buffer 5 (prior-report counts only)
python pull_denominators.py --output-dir $OUT

# Trial cohort notes: registration-based anchor, 14 days before it, masked (see README for the masking options)
python build_note_level_dataset.py --anchor note_anchor --days-offset -14 \
    --denominator-csv $OUT/processed_trial_enrollments.csv \
    --output-parquet $OUT/enrollments_note_level_dataset_minus14_masked.parquet

# SOC notes: plan start minus 14 days, no masking (run 2026-11-14 at commit 51e2542)
python build_note_level_dataset.py --anchor index --days-offset -14 \
    --denominator-csv $OUT/processed_soc_treatments.csv \
    --output-parquet $OUT/soc_note_level_dataset_minus14.parquet
```

`build_note_level_dataset.py` has no default offset: `--days-offset` is required. `--anchor` is `index`
(`trial_start_dt`) or `note_anchor` (`note_anchor_dt`, the default when the CSV has that column).

## Sources

- Code: `mmai_real_eval/research_plans.py`, `enrollments.py`, `soc.py`, `notes.py`, `io_utils.py`,
  `build_note_level_dataset.py`, `pull_denominators.py` (commit `51e2542`); `README.md`.
- Counts: `data/phi/real_eval_denominators_20261114/denominator_summary.json`, `note_leakage_summary.json`,
  `soc_note_level_dataset_minus14.summary.json`, `enrollments_note_level_dataset_minus14_masked.summary.json`, and
  aggregates over `processed_trial_enrollments.csv` (anchor sources and gaps). These files are PHI-restricted; the figures above are counts only.
- v22: `/ksg/kehl_mm_data/mmai/v22/matchminer-ai-training/real_eval_code/eval_phi_enrollments/prepare_data.py` and
  `eval_phi_soc/prepare_data.py` (`days_buffer=5`).
