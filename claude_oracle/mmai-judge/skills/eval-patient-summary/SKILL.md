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

The synthesis report path is derived automatically from `output_path`:
strip the extension and append `_report.md` (e.g.
`summaries_judged.parquet` → `summaries_judged_report.md`). Do not ask
the user for it.

**Invariant — never violate.** Neither `work_dir`, `output_path`, nor
the derived `report_path` may be inside this skill directory. If any is,
refuse and ask again. The bundled scripts also enforce this, but catch
it in conversation first.

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

### 4. Judge each row in a per-patient subagent

Initialize `<work_dir>/responses.jsonl` as empty with a single command:

```bash
: > <work_dir>/responses.jsonl
```

Then dispatch **one `general-purpose` subagent per sampled row** via the
Agent tool. Each subagent reads its row file, judges that single patient
with extended thinking, and appends its verdict via `append_response.py`.
Run subagents in parallel by issuing multiple Agent tool calls in a
single message — cap concurrency at ~10 in flight so the context fan-out
stays manageable. The orchestrator does NOT do the judging itself.

Each subagent has no view of this conversation, so its prompt must be
fully self-contained. For row `NN` with file
`<work_dir>/rows/row_NN.txt`, send the subagent the verbatim template
below, substituting `{row_path}`, `{row_id}`, `{work_dir}`, and
`{skill_dir}` (the absolute value of `${CLAUDE_SKILL_DIR}`).

When all subagents return, sanity-check that
`wc -l <work_dir>/responses.jsonl` equals the sample size before
proceeding to step 5; re-dispatch any missing row_ids.

Notes on the append command:

- Always keep the exact same command prefix
  (`python "<skill_dir>/scripts/append_response.py" *`) so one user
  "always allow" grant covers every subagent invocation.
- The helper writes one line per call:
  `{"__row_id__": <int>, "summary_judge_response": <full response>}`,
  and uses `fcntl.flock` to serialize concurrent appends.
- Use a unique heredoc sentinel (`SUMMARY_JUDGE_EOF`) so the body can
  contain arbitrary characters.

#### Subagent prompt template (verbatim — substitute the four placeholders)

```
You are a brilliant oncologist serving as a judge in an AI evaluation of MM-AI patient summarization. Use **ultrathink** while reading and reasoning — the verdict hinges on whether an omission or fabrication in the summary would plausibly change a trial-eligibility or treatment decision, and that needs careful reading.

You are judging exactly one patient (row_id={row_id}). The clinical material is in this file:

  {row_path}

Read that file in full. It contains two clearly marked blocks:
- `=== SOURCE NOTES ===` — the raw clinical notes for this patient (possibly truncated to the last ~1M-token tail window).
- `=== MM-AI SUMMARY ===` — the AI-generated summary you are judging.

Your job is to judge whether the summary faithfully represents the notes, scored ONLY against what MMAI was actually instructed to capture.

MMAI's directive (from 6_summarize_patients.py) tells it to produce exactly these eight sections, and ONLY these sections:
- Age — patient's most recent age
- Sex — patient's sex
- Cancer type — primary site (e.g. breast cancer, lung cancer). Localized basal-cell or squamous-cell skin cancers and colon polyps do NOT count as cancers for this purpose. When the patient has multiple ACTIVE cancers, the most active cancer is listed first, followed by any other active cancers. Inactive prior cancers belong in the Boilerplate section (with a note that they are inactive and a date of last known activity if available), NOT in the cancer-history sections.
- Histology — e.g. adenocarcinoma, squamous carcinoma
- Current extent — localized / advanced / metastatic / etc., and tumor markers used to follow disease status (e.g. CEA, PSA) when relevant. MMAI was told NOT to list every historical tumor-marker value; just the most recent value and trend if relevant. A summary that omits earlier marker values is therefore not deficient.
- Biomarkers — genomic results, IHC, protein expression. MMAI was told to err on the side of including ALL biomarkers, including all IHC results, all positive genomic findings, and pertinent negative genomic findings.
- Treatment history — surgery, radiation, chemo / targeted / immunotherapy, etc., with start and stop dates and best response when documented, in chronological order. MMAI was told to use generic drug names and to expand common regimen abbreviations (e.g. AC → doxorubicin + cyclophosphamide; FOLFOX → 5-FU + leucovorin + oxaliplatin; pembro → pembrolizumab). Reasonable expansions following that reference list are correct, not fabricated.
- Boilerplate — history of conditions that might meet common boilerplate trial-exclusion criteria: uncontrolled brain metastases, lack of measurable disease, poor performance status, congestive heart failure, pneumonitis, renal dysfunction, liver dysfunction, HIV or hepatitis infection, prior unrelated/inactive cancer diagnoses (with date of last known activity if available), etc.

SCOPE — read carefully:
- Do NOT flag the summary for omitting categories outside MMAI's directive. That includes (non-exhaustive): smoking history, family history, social history, ECOG performance status as a standalone field, sites of metastasis as a standalone field, allergies, full medication list. MMAI was never asked to capture these.
- ECOG / performance status, organ dysfunction, and comorbidities are only in scope to the extent they would meet a common boilerplate exclusion (i.e. they belong inside the Boilerplate section). A summary that does not call them out separately is not deficient.
- Inactive cancers in Boilerplate are correct, NOT an omission. MMAI was instructed to put inactive prior cancers in the Boilerplate section (noting inactivity and date of last activity if available), not in the cancer-history sections. Do NOT flag a faithful Boilerplate listing of an inactive cancer as an OMISSION from cancer-history sections, and do NOT flag MMAI for not creating a separate cancer-history block for an inactive cancer.
- Format-only deviations — markdown, Unicode, tables, ordering of sections, restating prior-summary information, line breaks within Treatment history — are NOT clinical errors and must NOT change the verdict. This judge is about clinical fidelity, not formatting.
- An OMISSION is only material when (a) MMAI was instructed to capture the information AND (b) the omission would plausibly change a trial-eligibility or treatment decision.
- An INCORRECT verdict is reserved for content the summary asserts that the notes do not support or actively contradict — within the eight sections MMAI was instructed to write.

Reason step by step. For each of MMAI's eight sections, ask:
1. Did MMAI capture what the notes say about this section, to the level of detail MMAI was instructed to provide?
2. If something is missing, is it information MMAI was told to capture AND is the omission clinically material to trial eligibility or treatment choice? If either is no, do NOT count it.
3. If something is asserted, is it supported by the notes? Paraphrasing, date imprecision that does not cross a washout threshold, reordering, harmless summarization should NOT count against the summary.

Resolve to exactly one of the following labels:
- AGREE — no clinically material differences that would impact treatment or trial eligibility.
- OMISSION — the summary omits information that would impact treatment/trial eligibility.
- INCORRECT — the summary includes incorrect information (fabricated or contradicted by the notes).
- OMISSION_AND_INCORRECT — both OMISSION and INCORRECT problems are present.

Your full reasoning response MUST end with the following line and nothing else after it:
Final verdict: X
where X is one of the four labels above, written exactly (uppercase, with underscores where shown).

After producing your full response, append it to the run log via:

  python "{skill_dir}/scripts/append_response.py" \
    --work_dir {work_dir} --row_id {row_id} <<'SUMMARY_JUDGE_EOF'
  <your full response text, ending with the Final verdict line>
  SUMMARY_JUDGE_EOF

Use the heredoc sentinel `SUMMARY_JUDGE_EOF` exactly so the response body can contain arbitrary characters. The helper appends one JSON line under a file lock, so parallel sibling subagents are safe.

Once the append succeeds, reply to the orchestrator with only the row_id and the verdict label (under 30 words). Do not echo your reasoning back — the full response is already captured by `append_response.py`.
```

### 5. Finalize

```bash
python "${CLAUDE_SKILL_DIR}/scripts/finalize.py" \
  --work_dir <work_dir> \
  --output <output_path>
```

Parses the final line, appends `summary_judge_response` and
`summary_judge_verdict` columns, writes parquet or CSV by extension.

### 6. Synthesize a markdown report

Always produce a synthesis report. Dispatch a **single `general-purpose`
subagent** via the Agent tool whose only job is to load the finalized
output, read every judge response, and write a markdown synthesis to
`report_path`. Keeping this in a subagent avoids loading 20+ long
responses into the orchestrator's context.

Send the synthesis subagent the verbatim template below, substituting
`{output_path}`, `{report_path}`, `{sample_size}`, `{random_seed}`,
`{note_col}`, `{summaries_path}`, and `{notes_path}`.

#### Synthesis subagent prompt template (verbatim — substitute the seven placeholders)

```
You are writing a synthesis report for an MM-AI patient-summary evaluation. The judging is already complete; your job is to read the finalized results and produce a single markdown report.

Inputs:
- Finalized output file: {output_path} (parquet or CSV, with columns including patient_id, summary_judge_response, summary_judge_verdict)
- Sample size: {sample_size}
- Random seed: {random_seed}
- Notes column used: {note_col}
- Source summaries file: {summaries_path}
- Source notes file: {notes_path}

Steps:
1. Load the finalized output with pandas (`read_parquet` or `read_csv` by suffix). Compute the verdict histogram across {AGREE, OMISSION, INCORRECT, OMISSION_AND_INCORRECT, PARSE_FAILED}.
2. Read each row's `summary_judge_response`. Look for recurring themes across the non-AGREE verdicts:
   - What kinds of information are most often omitted? (e.g. brain mets, recent biomarker, treatment dates, performance status when in scope)
   - What kinds of content are most often incorrect or fabricated? (e.g. wrong drug names, invented dates, mis-stated metastatic status, wrong cancer histology)
   - Are there structural patterns (e.g. omissions cluster in Treatment history, fabrications cluster in Biomarkers)?
   - Note any PARSE_FAILED rows and include the tail of those responses verbatim.
3. Write the report to `{report_path}` with the **Write** tool. Structure:

   # MM-AI Patient Summary Judge — Synthesis Report

   **Run inputs**
   - Summaries: `{summaries_path}`
   - Notes: `{notes_path}` (note column: `{note_col}`)
   - Sample size: {sample_size}, seed: {random_seed}
   - Finalized output: `{output_path}`

   ## Verdict histogram

   | Verdict | Count | % |
   |---|---:|---:|
   | AGREE | … | … |
   | OMISSION | … | … |
   | INCORRECT | … | … |
   | OMISSION_AND_INCORRECT | … | … |
   | PARSE_FAILED | … | … |

   ## Themes — Omissions
   - bullet points grounded in specific rows; cite `row_id`/`patient_id` in parentheses

   ## Themes — Incorrect content
   - bullet points grounded in specific rows; cite `row_id`/`patient_id` in parentheses

   ## Cross-cutting patterns
   - higher-level observations (which sections are weakest, repeated failure modes, anything noteworthy about AGREE rows)

   ## Per-row verdicts

   | row | patient_id | verdict | one-sentence summary |
   |---|---|---|---|
   | 0 | … | … | … |
   | … | … | … | … |

   ## Illustrative excerpts
   - 2–4 short blockquote excerpts from individual responses that best illustrate the themes; cite row_id and patient_id

   ## Parse failures (if any)
   - For each PARSE_FAILED row, include row_id, patient_id, and the last ~400 characters of the response so a human can re-grade it.

4. After Write succeeds, reply to the orchestrator with only the report path and a single-sentence summary of the verdict mix (under 30 words). Do not echo the whole report.

Constraints:
- Do not modify the finalized output file. Read-only.
- Do not write any file inside the skill directory. `report_path` lives outside it by construction; refuse and stop if it is not.
- Use generic clinical language; do not invent facts not present in the responses you read.
```

### 7. Report to the user

Echo `finalize.py`'s summary line, the synthesis subagent's one-sentence
verdict-mix summary, and the absolute paths of both `output_path` and
`report_path`. Do not re-summarize every row.

## Notes

- Do not write any file inside this skill directory at any step.
- Parsing only inspects the last ~400 chars of the response, so the final
  line must be `Final verdict: X` with nothing after it.
- Re-running over the same `work_dir` overwrites `staging.parquet`;
  `responses.jsonl` is yours to recreate each run. The synthesis report
  at `report_path` is overwritten on each finalize+synthesize pass.
