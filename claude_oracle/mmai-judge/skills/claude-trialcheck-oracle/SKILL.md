---
name: claude-trialcheck-oracle
description: Score sampled patient-trial candidate matches for clinical-trial reasonableness (0-5) using the same prompt and rubric as llm_check_trials.py, with Claude acting as the oracle model. Use when the user asks to run trialcheck / oracle scoring on a parquet or CSV of candidate matches, or mentions llm_check_trials / space_specific_eligibility_checks.
disable-model-invocation: true
argument-hint: [input_parquet_or_csv]
allowed-tools: Read Write Bash(python *) Bash(python3 *) Bash(ls *) Bash(mkdir *) Bash(wc *) Bash(: *)
---

# claude-trialcheck-oracle

You are running the oracle replica of `llm_check_trials.py`. Use **ultrathink**
on every row: the original script sets `Reasoning: high`, so extended thinking
must be active throughout the scoring loop.

## Workflow

### 1. Resolve the input file

Input path comes from `$ARGUMENTS`. If empty or the path does not exist, ask
the user for it. Must end in `.parquet` or `.csv`.

### 2. Collect parameters from the user

Use AskUserQuestion to collect **all four** of:

- **sample_size** — how many rows to score (integer; default 20).
- **random_seed** — random seed for sampling (integer; default 42).
- **work_dir** — directory for interim files `staging.parquet`, `staging.jsonl`,
  `responses.jsonl`. Default: `<parent-of-input>/trialcheck_oracle_work/`.
- **output_path** — final output file path (extension `.parquet` or `.csv`).
  Default: same directory and basename as the input with `_oracle_checked`
  appended before the extension.

**Invariant — never violate.** Neither `work_dir` nor `output_path` may be
inside this skill directory. If either is, refuse and ask the user again. The
bundled scripts also enforce this, but catch it earlier in conversation.

### 3. Prepare the sample

Run the preparation script. It filters null `patient_summary`, strips leading
numbering from `this_space`, samples with the given seed, and writes
`staging.parquet`, `staging.jsonl`, and slim per-row text files under
`<work_dir>/rows/row_NN.txt` (one per sampled row, containing only the
`patient_summary` and `this_space` blocks). Reading those per-row files is
how you load each row for scoring — do NOT copy rows to `/tmp` or any other
path, because doing so triggers a fresh Read/Bash approval round.

```bash
python "${CLAUDE_SKILL_DIR}/scripts/prepare_sample.py" \
  --input <input_path> \
  --n <sample_size> \
  --seed <random_seed> \
  --work_dir <work_dir>
```

### 4. Score each row

Initialize `<work_dir>/responses.jsonl` as empty (overwrite any prior file)
with a single command, e.g.

```bash
: > <work_dir>/responses.jsonl
```

For each row, Read `<work_dir>/rows/row_NN.txt` (where `NN` is the zero-padded
`__row_id__`; the pad width matches the sample size). That file contains the
`patient_summary` and `this_space` blocks. Treat `patient_summary` as
`{patient_summary}` and `this_space` as `{trial_summary}` and internally
assemble the exact two-message prompt below. Then produce a response with
your own reasoning that ends on a line `Final score: X` (X in 0-5).

Do not re-split the staging file to a temp directory; the per-row files in
`<work_dir>/rows/` exist precisely so the model doesn't have to create any
new paths that would require fresh Read approvals.

Append the row response by invoking the bundled helper with a **stable command
prefix** so the user only needs to approve the pattern once (one "always allow"
covers every row):

```bash
python "${CLAUDE_SKILL_DIR}/scripts/append_response.py" \
  --work_dir <work_dir> --row_id <int> <<'TRIALCHECK_RESPONSE_EOF'
<your full response text, ending with "Final score: X" on its own final line>
TRIALCHECK_RESPONSE_EOF
```

Notes on this step:

- Always keep the exact same command prefix
  `python "${CLAUDE_SKILL_DIR}/scripts/append_response.py" --work_dir ... --row_id ...`
  so permission grants are reused across rows. Do not inline ad-hoc `python` /
  `python3` heredocs that write to `responses.jsonl` directly — those will
  prompt per invocation.
- The helper writes exactly one line per call:
  `{"__row_id__": <int>, "trialcheck_llm_response": <your full response text>}`.
- The stored `trialcheck_llm_response` must be the full response text
  including the trailing `Final score: X` line, since downstream parsing
  inspects the tail of the string.
- Use a unique heredoc sentinel (e.g. `TRIALCHECK_RESPONSE_EOF`) so the
  response body can contain arbitrary characters without terminating early.

#### System message (verbatim)

```
Reasoning: high
```

#### User message template (verbatim — fill in `{trial_summary}` and `{patient_summary}`)

```
You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment. Your job is to evaluate whether a given clinical trial is a reasonable consideration for a patient, given a clinical trial summary and a patient summary, and then score how targeted the trial is for this specific patient.

Here is a summary of the clinical trial:
{trial_summary}
Here is a summary of the patient:
{patient_summary}
Base your judgment on whether the patient generally fits the age requirements if any, sex requirements if any, cancer type(s), cancer burden, prior treatment(s), and biomarker criteria specified for the trial.
You do not have to determine if the patient is actually eligible; instead please just evaluate whether it is reasonable for the trial to be considered further by the patient's oncologist.
Biomarker criteria have to be considered carefully. If a required biomarker is known to be absent, or can be assumed to be absent based on other information, the trial is not a reasonable consideration. For example, if a trial for lung cancer requires an EGFR mutation, documentation that there is no EGFR mutation indicates the trial is not a reasonable consideration. Similarly, documentation of a KRAS mutation in the patient indicates the trial is not a reasonable consideration, since, as you know, KRAS and EGFR driver mutations in lung cancer are mutually exclusive.
Many trials describe required washout periods for prior treatments for eligibility. For example, the eligibility criteria might state that patients may not have received radiation or chemotherapy in the last 14 days or 30 days. It is CRITICAL that you IGNORE these eligibility criteria when considering prior treatment requirements. Assume that patients could wait for the washout period to enroll. Also CRITICAL: Ignore your knowledge of today's current date. Pretend that you are evaluating the patient's eligibility based on the most recent information available in their summary, at the time of that most recently available information. Do not provide ethical judgments or comment on resource constraints with respect whether the trial is a reasonable clinical consideration; just evaluate whether it is, given the available information.

SCORING INSTRUCTIONS:
After reasoning step by step, compute a score from 0 to 5 using the following rubric:

Start with 0 points.
1) REASONABLENESS (0 or 1 point): If the trial is at least a reasonable consideration for this patient (i.e., the patient does not clearly meet an exclusion criterion such as wrong cancer type, wrong age group, wrong sex, having an excluded biomarker, etc.), award 1 point. If the trial is NOT reasonable, the final score is 0 — skip the remaining categories.
2) CANCER TYPE SPECIFICITY (+1 point): If the trial specifies the patient's cancer type (e.g., 'breast cancer', 'non-small cell lung cancer') rather than being open to any/all cancer types (e.g., 'solid tumors', 'advanced cancers'), award +1 point.
3) CANCER BURDEN/STAGE SPECIFICITY (+1 point): If the trial specifies a particular disease stage or burden (e.g., 'metastatic', 'locally advanced', 'stage III-IV') that matches the patient's disease status, award +1 point. If the trial has no stage/burden requirements or is open to any stage, do not award a point.
4) PRIOR TREATMENT SPECIFICITY (+1 point): If the trial has specific prior treatment requirements (e.g., 'must have progressed on platinum-based chemotherapy', 'prior immunotherapy required') and the patient's treatment history matches those requirements, award +1 point. If the trial has no specific prior treatment requirements, do not award a point.
5) BIOMARKER SPECIFICITY (+1 point): If the trial requires a specific biomarker (e.g., 'EGFR mutation', 'PD-L1 ≥ 50%', 'HER2-positive') AND the patient is known to have that biomarker, award +1 point. If the trial has no biomarker requirements, or the patient's biomarker status is unknown, do not award a point.

Your response MUST end with the following line and nothing else after it:
Final score: X
where X is the total score (an integer from 0 to 5).
```

### 5. Finalize

Once every row has a line in `responses.jsonl`, run:

```bash
python "${CLAUDE_SKILL_DIR}/scripts/finalize.py" \
  --work_dir <work_dir> \
  --output <output_path>
```

The script joins staging with responses, parses `eligibility_result` and
`eligibility_verdict` using the exact regex and fallback cascade from
`llm_check_trials.py`, and writes parquet or CSV based on extension.

### 6. Report

Echo the script's summary line plus a one-sentence note of the sample size
and the output location. Do not re-summarize every row.

## Notes

- Do not write any file inside this skill directory at any step.
- Parsing reuses `llm_check_trials.py`'s exact logic: it only looks at the
  last ~60 chars of the response, so it is important that the final line is
  `Final score: X` with nothing after it.
- If you are asked to re-run over the same `work_dir`, the scripts overwrite
  the staging and output files cleanly; `responses.jsonl` is yours to
  recreate from scratch each run.
