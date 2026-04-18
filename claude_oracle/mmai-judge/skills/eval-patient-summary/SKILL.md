---
name: eval-patient-summary
description: Judge MM-AI patient summaries against the underlying clinical notes. Classifies each summary as AGREE (no clinically material differences), OMISSION (missed information that would affect treatment/trial eligibility), INCORRECT (fabricated or wrong information), or OMISSION_AND_INCORRECT. Use when the user asks to evaluate MM-AI patient summaries with Claude as judge, or mentions 6_summarize_patients / patient_summary eval.
disable-model-invocation: true
argument-hint: [summaries_parquet_or_csv] [notes_parquet_or_csv]
allowed-tools: Read Write Bash(python *) Bash(python3 *) Bash(ls *) Bash(mkdir *) Bash(wc *) Bash(: *)
---

# eval-patient-summary

You are acting as an **Opus-4.7 judge** of MM-AI's patient summarization.
Use **ultrathink** on every row — the verdict hinges on whether an omission
or a fabrication in the summary would plausibly change a trial-eligibility or
treatment decision, and that needs careful reading.

## Workflow

### 1. Resolve the inputs

Two positional inputs come from `$ARGUMENTS`:

- Summaries file — parquet or CSV with columns `patient_id`,
  `patient_summary`, `last_date`, and (optionally) `patient_boilerplate_text`.
  `prepare_sample.py` parses `last_date` with `pd.to_datetime`, sorts by
  `(patient_id, last_date)`, and keeps only the temporally last summary per
  patient — the running summary built from the chunk covering the latest
  notes — so multi-chunk summary parquets are collapsed automatically.
  When `patient_boilerplate_text` is present and non-empty, `prepare_sample.py`
  appends it to the end of `patient_summary` under a `Boilerplate:`
  header before writing the per-row file, so the judge sees MMAI's
  Boilerplate section even though it now lives in a separate column.
- Notes file — parquet or CSV with columns `patient_id`, `date`, and a
  note-text column (defaulting to `note_text`).

If either is missing or does not exist, ask the user for it. Both must end
in `.parquet` or `.csv`.

### 2. Collect parameters from the user

Use AskUserQuestion to collect **all five** of:

- **sample_size** — integer; default 20.
- **random_seed** — integer; default 42.
- **note_col** — note-text column name in the notes file. Default `note_text`.
  If the notes file is the synthetic notes dump, it is typically
  `synthetic_note`; if it is another source, ask.
- **work_dir** — default `<parent-of-summaries>/patient_summary_judge_work/`.
- **output_path** — default same directory and basename as the summaries
  file with `_judged` appended before the extension.

**Invariant — never violate.** Neither `work_dir` nor `output_path` may be
inside this skill directory. If either is, refuse and ask again. The
bundled scripts also enforce this, but catch it in conversation first.

### 3. Prepare the sample

```bash
python "${CLAUDE_SKILL_DIR}/scripts/prepare_sample.py" \
  --summaries <summaries_path> \
  --notes <notes_path> \
  --note_col <note_col> \
  --n <sample_size> \
  --seed <random_seed> \
  --work_dir <work_dir>
```

The script sorts the notes df by `(patient_id, date)` with stable mergesort,
concatenates them per patient (separated by `--- NOTE <date> ---` blocks),
truncates to the last ~1M-token tail, and writes `staging.parquet` and
`<work_dir>/rows/row_NN.txt` (one per sampled row, containing the source
notes block followed by the MM-AI summary). Reading those per-row files is
how you load each row for judging — do NOT copy rows elsewhere.

### 4. Judge each row

Initialize `<work_dir>/responses.jsonl` as empty with a single command:

```bash
: > <work_dir>/responses.jsonl
```

For each row, Read `<work_dir>/rows/row_NN.txt`. That file contains the two
blocks `=== SOURCE NOTES ===` and `=== MM-AI SUMMARY ===`. Internally
assemble the exact two-message prompt below. Produce your own extended-
thinking reasoning, then end on a line `Final verdict: X` where X is one
of `AGREE`, `OMISSION`, `INCORRECT`, or `OMISSION_AND_INCORRECT`.

Append the response with a **stable command prefix** so one "always allow"
grant covers every row:

```bash
python "${CLAUDE_SKILL_DIR}/scripts/append_response.py" \
  --work_dir <work_dir> --row_id <int> <<'SUMMARY_JUDGE_EOF'
<your full response text, ending with "Final verdict: X" on its own final line>
SUMMARY_JUDGE_EOF
```

Notes:

- Always keep the exact same command prefix so permission grants are reused.
- The helper writes one line per call:
  `{"__row_id__": <int>, "summary_judge_response": <your full response>}`.
- Use a unique heredoc sentinel (`SUMMARY_JUDGE_EOF`) so the body can
  contain arbitrary characters.

#### System message (verbatim)

```
Reasoning: high
```

#### User message template (verbatim — fill in `{source_notes}` and `{mm_ai_summary}`)

```
You are a brilliant oncologist serving as a judge in an AI evaluation. You will be shown the raw clinical notes for a single patient, followed by an AI-generated summary of that patient written for downstream trial-matching and treatment-planning. Your job is to judge whether the summary faithfully represents the notes, scored ONLY against what MMAI was actually instructed to capture.

MMAI's directive (from 6_summarize_patients.py) tells it to produce exactly these eight sections, and ONLY these sections:
- Age — patient's most recent age
- Sex — patient's sex
- Cancer type — primary site (e.g. breast cancer, lung cancer). Localized basal-cell or squamous-cell skin cancers and colon polyps do NOT count as cancers for this purpose. When the patient has multiple cancers, the currently or most recently active cancer is listed first.
- Histology — e.g. adenocarcinoma, squamous carcinoma
- Current extent — localized / advanced / metastatic / etc., and tumor markers used to follow disease status over time (e.g. CEA, PSA) when relevant
- Biomarkers — genomic results, IHC, protein expression. MMAI was told to err on the side of including ALL biomarkers, including all IHC results, all positive genomic findings, and pertinent negative genomic findings.
- Treatment history — surgery, radiation, chemo / targeted / immunotherapy, etc., with start and stop dates and best response when documented, in chronological order
- Boilerplate — history of conditions that might meet common boilerplate trial-exclusion criteria: uncontrolled brain metastases, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, liver dysfunction, HIV or hepatitis infection, etc.

SCOPE — read carefully:
- Do NOT flag the summary for omitting categories outside MMAI's directive. That includes (non-exhaustive): smoking history, family history, social history, ECOG performance status as a standalone field, sites of metastasis as a standalone field, allergies, full medication list. MMAI was never asked to capture these.
- ECOG / performance status, organ dysfunction, and comorbidities are only in scope to the extent they would meet a common boilerplate exclusion (i.e. they belong inside the Boilerplate section). A summary that does not call them out separately is not deficient.
- Format-only deviations — markdown, Unicode, tables, ordering of sections, restating prior-summary information, line breaks within Treatment history — are NOT clinical errors and must NOT change the verdict. This judge is about clinical fidelity, not formatting.
- An OMISSION is only material when (a) MMAI was instructed to capture the information AND (b) the omission would plausibly change a trial-eligibility or treatment decision.
- An INCORRECT verdict is reserved for content the summary asserts that the notes do not support or actively contradict — within the eight sections MMAI was instructed to write.

Here are the source clinical notes (possibly truncated to the last ~1M-token window):
{source_notes}

Here is the AI-generated summary:
{mm_ai_summary}

Reason step by step. For each of MMAI's eight sections, ask:
1. Did MMAI capture what the notes say about this section, to the level of detail MMAI was instructed to provide?
2. If something is missing, is it information MMAI was told to capture AND is the omission clinically material to trial eligibility or treatment choice? If either is no, do NOT count it.
3. If something is asserted, is it supported by the notes? Paraphrasing, date imprecision that does not cross a washout threshold, reordering, harmless summarization should NOT count against the summary.

Resolve to exactly one of the following labels:
- AGREE — no clinically material differences that would impact treatment or trial eligibility.
- OMISSION — the summary omits information that would impact treatment/trial eligibility.
- INCORRECT — the summary includes incorrect information (fabricated or contradicted by the notes).
- OMISSION_AND_INCORRECT — both OMISSION and INCORRECT problems are present.

Your response MUST end with the following line and nothing else after it:
Final verdict: X
where X is one of the four labels above, written exactly (uppercase, with underscores where shown).
```

### 5. Finalize

```bash
python "${CLAUDE_SKILL_DIR}/scripts/finalize.py" \
  --work_dir <work_dir> \
  --output <output_path>
```

Parses the final line, appends `summary_judge_response` and
`summary_judge_verdict` columns, writes parquet or CSV by extension.

### 6. Report

Echo the script's summary line plus a one-sentence note of the sample size,
notes column used, and output location. Do not re-summarize every row.

## Notes

- Do not write any file inside this skill directory at any step.
- Parsing only inspects the last ~400 chars of the response, so the final
  line must be `Final verdict: X` with nothing after it.
- Re-running over the same `work_dir` overwrites `staging.parquet`;
  `responses.jsonl` is yours to recreate each run.
