---
name: claude-boilerplate-check
description: Score sampled patient/trial boilerplate pairs for clinical-trial boilerplate exclusion (Yes!/No!) using the same prompt and parsing as 14_check_boilerplate.py, with Claude acting as the oracle model. Use when the user asks to run boilerplate-check / oracle scoring on a parquet or CSV of boilerplate pairs, or mentions 14_check_boilerplate / vllm_parallel_boilerplate / exclusion_result. Distinct from the MM-AI-judging eval-boilerplate skill — this re-decides from scratch instead of judging MM-AI's reasoning.
disable-model-invocation: true
argument-hint: [input_parquet_or_csv]
allowed-tools: Read Write Bash(python *) Bash(python3 *) Bash(ls *) Bash(mkdir *) Bash(wc *) Bash(: *)
---

# claude-boilerplate-check

You are running the oracle replica of `14_check_boilerplate.py`. Use
**ultrathink** on every row: the original script sets `Reasoning: high`, so
extended thinking must be active throughout the scoring loop.

## Workflow

### 1. Resolve the input file

Input path comes from `$ARGUMENTS`. If empty or the path does not exist, ask
the user for it. Must end in `.parquet` or `.csv`.

### 2. Collect parameters from the user

Use AskUserQuestion to collect **all four** of:

- **sample_size** — how many rows to score (integer; default 20).
- **random_seed** — random seed for sampling (integer; default 42).
- **work_dir** — directory for interim files `staging.parquet`,
  `staging.jsonl`, `responses.jsonl`. Default:
  `<parent-of-input>/boilerplate_oracle_work/`.
- **output_path** — final output file path (extension `.parquet` or `.csv`).
  Default: same directory and basename as the input with `_oracle_checked`
  appended before the extension.

**Invariant — never violate.** Neither `work_dir` nor `output_path` may be
inside this skill directory. If either is, refuse and ask the user again. The
bundled scripts also enforce this, but catch it earlier in conversation.

### 3. Prepare the sample

Run the preparation script. It filters null/empty `patient_boilerplate_text`
and `trial_boilerplate_text`, samples with the given seed, and writes
`staging.parquet`, `staging.jsonl`, and slim per-row text files under
`<work_dir>/rows/row_NN.txt` (one per sampled row, containing only the
`patient_boilerplate_text` and `trial_boilerplate_text` blocks). Reading
those per-row files is how you load each row for scoring — do NOT copy rows
to `/tmp` or any other path, because doing so triggers a fresh Read/Bash
approval round.

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
`patient_boilerplate_text` and `trial_boilerplate_text` blocks. Treat
`patient_boilerplate_text` as `{p_blp}` and `trial_boilerplate_text` as
`{t_blp}` and internally assemble the exact two-message prompt below. Then
produce a response with your own reasoning that ends on a final one-word
line of either `Yes!` or `No!` (verbatim, case-sensitive, with the
exclamation point, and nothing after it).

Do not re-split the staging file to a temp directory; the per-row files in
`<work_dir>/rows/` exist precisely so the model doesn't have to create any
new paths that would require fresh Read approvals.

Append the row response by invoking the bundled helper with a **stable command
prefix** so the user only needs to approve the pattern once (one "always allow"
covers every row):

```bash
python "${CLAUDE_SKILL_DIR}/scripts/append_response.py" \
  --work_dir <work_dir> --row_id <int> <<'BOILERPLATE_RESPONSE_EOF'
<your full response text, ending with "Yes!" or "No!" on its own final line>
BOILERPLATE_RESPONSE_EOF
```

Notes on this step:

- Always keep the exact same command prefix
  `python "${CLAUDE_SKILL_DIR}/scripts/append_response.py" --work_dir ... --row_id ...`
  so permission grants are reused across rows. Do not inline ad-hoc `python` /
  `python3` heredocs that write to `responses.jsonl` directly — those will
  prompt per invocation.
- The helper writes exactly one line per call:
  `{"__row_id__": <int>, "boilerplate_check_llm_response": <your full response text>}`.
- The stored `boilerplate_check_llm_response` must be the full response text
  including the trailing `Yes!` / `No!` token, since downstream parsing
  inspects the tail of the string.
- Use a unique heredoc sentinel (e.g. `BOILERPLATE_RESPONSE_EOF`) so the
  response body can contain arbitrary characters without terminating early.

#### System message (verbatim)

```
Reasoning: high
```

#### User message template (verbatim — fill in `{p_blp}` and `{t_blp}`)

```
You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment.
Your job is to evaluate whether a patient has any underlying medical conditions that would exclude him or her from a specific clinical trial.

Here is an extract of the patient's history:
{p_blp}
Here are the exclusion criteria for the trial:
{t_blp}
Note that the extract was generated by prompting an LLM to determine whether the patient meets specific common exclusion criteria, such as uncontrolled brain metastases, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, liver dysfunction, and HIV or hepatitis infection, and to present evidence for whether the patient met the criterion.
You should therefore not assume that mention of such condition means the patient has the condition; it may represent the LLM reasoning about whether the patient has the condition.
Based on the extract, you should determine whether the patient clearly meets one of the exclusion criteria for this specific trial.
Do not evaluate exclusion criteria other than those listed for this trial.
Reason through one exclusion criterion at a time. Generate a numbered list of the criteria as you go. For each one, decide whether the patient clearly meets the exclusion criteron. If it is not completely clear that the patient meets the exclusion criterion, give the patient the benefit of the doubt, and err on the side of deciding the patient is not excluded. A description in the patient extract that a condition is mild, low-grade, or resolved is even more of a reason not to exclude the patient based on that condition.
Once you have evaluated all exclusion criteria, answer the question "Is this patient clearly excluded from this trial?" with a one-word "Yes!" or "No!" answer, based on whether the patient clearly met any of the individual exclusion criteria. It is critical that your final word be either "Yes!" or "No!", verbatim, and case-sensitive.
Make sure to include the exclamation point in your final one-word answer.
No introductory text or concluding text after that final answer.
```

### 5. Finalize

Once every row has a line in `responses.jsonl`, run:

```bash
python "${CLAUDE_SKILL_DIR}/scripts/finalize.py" \
  --work_dir <work_dir> \
  --output <output_path>
```

The script joins staging with responses, parses `exclusion_result` (float
0.0/1.0, matching production) and `exclusion_verdict` (string "Yes!" /
"No!" / "PARSE_FAILED") using the exact tail-search logic from
`14_check_boilerplate.py`, and writes parquet or CSV based on extension.

### 6. Report

Echo the script's summary line plus a one-sentence note of the sample size
and the output location. Do not re-summarize every row.

## Notes

- Do not write any file inside this skill directory at any step.
- Parsing reuses `14_check_boilerplate.py`'s exact logic: it only looks at
  the last ~16 chars of the response, so it is important that the final
  token be `Yes!` or `No!` with nothing meaningful after it.
- If you are asked to re-run over the same `work_dir`, the scripts overwrite
  the staging and output files cleanly; `responses.jsonl` is yours to
  recreate from scratch each run.
- This skill is NOT the same as `eval-boilerplate`. This skill re-decides
  the exclusion verdict from scratch; `eval-boilerplate` judges MM-AI's own
  reasoning.
