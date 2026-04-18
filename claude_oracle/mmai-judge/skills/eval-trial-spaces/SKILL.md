---
name: eval-trial-spaces
description: Judge whether MM-AI's decomposition of a clinical trial into a list of inclusion/exclusion "spaces" is (a) mutually exclusive and (b) exhaustive of the real enrollment arms in the trial text. Each trial gets two forced-choice verdicts — mutually_exclusive and exhaustive — each YES / PARTIAL / NO. Use when the user asks to evaluate trial space extraction with Claude as judge, or mentions 0b_create_trial_spaces / trial_space_lineitems eval.
disable-model-invocation: true
argument-hint: [trials_with_spaces_parquet_or_csv]
allowed-tools: Read Write Bash(python *) Bash(python3 *) Bash(ls *) Bash(mkdir *) Bash(wc *) Bash(: *)
---

# eval-trial-spaces

You are acting as an **Opus-4.7 judge** of MM-AI's trial-space extraction.
Use **ultrathink** on every row — checking whether a list of extracted
spaces is exhaustive of the enrollment arms in a long eligibility-criteria
document requires careful, structured reading.

## Workflow

### 1. Resolve the input file

Input path comes from `$ARGUMENTS`. If empty or non-existent, ask the user.
Must end in `.parquet` or `.csv`.

The input can be in either of two shapes; the preparation script detects:

- **Aggregated**: one row per trial, with columns `nct_id`, `trial_text`,
  `extracted_spaces` (JSON list / newline-separated / semicolon-separated).
- **Lineitems**: one row per space, with columns `nct_id`, `trial_text`,
  `this_space` (and optionally `space_number` for ordering). The script
  groups by `nct_id` and builds the list itself.

### 2. Collect parameters from the user

Use AskUserQuestion to collect **all four** of:

- **sample_size** — integer; default 20.
- **random_seed** — integer; default 42.
- **work_dir** — default `<parent-of-input>/trial_spaces_judge_work/`.
- **output_path** — default same directory and basename as the input with
  `_judged` appended before the extension.

**Invariant — never violate.** Neither `work_dir` nor `output_path` may be
inside this skill directory.

### 3. Prepare the sample

```bash
python "${CLAUDE_SKILL_DIR}/scripts/prepare_sample.py" \
  --input <input_path> \
  --n <sample_size> \
  --seed <random_seed> \
  --work_dir <work_dir>
```

Writes `staging.parquet` and `<work_dir>/rows/row_NN.txt`, each containing
the trial text and a numbered `MM-AI EXTRACTED SPACES` block.

### 4. Judge each row

Initialize responses:

```bash
: > <work_dir>/responses.jsonl
```

For each row, Read `<work_dir>/rows/row_NN.txt`. Treat the `TRIAL TEXT`
block as `{trial_text}` and the `MM-AI EXTRACTED SPACES` block as
`{spaces_block}`. Assemble the prompt below verbatim. End your response
with exactly two lines — `Mutually exclusive: <LABEL>` and
`Exhaustive: <LABEL>` — each label in `YES` / `PARTIAL` / `NO`.

Append with the stable helper:

```bash
python "${CLAUDE_SKILL_DIR}/scripts/append_response.py" \
  --work_dir <work_dir> --row_id <int> <<'SPACES_JUDGE_EOF'
<your full response text, ending with the two verdict lines>
SPACES_JUDGE_EOF
```

Keep the command prefix identical across rows so one "always allow" covers
every invocation.

#### System message (verbatim)

```
Reasoning: high
```

#### User message template (verbatim — fill in `{trial_text}` and `{spaces_block}`)

```
You are a brilliant oncologist serving as a judge in an AI evaluation. You will be shown the raw text of a clinical trial (typically eligibility criteria and protocol details) followed by a list of inclusion/exclusion "spaces" that MM-AI has extracted from it. A "space" is meant to represent a single coherent combination of eligibility requirements that the trial is enrolling to — age, sex, cancer type, histology, cancer burden/stage, required prior treatments, excluded prior treatments, required biomarkers, and excluded biomarkers. A trial with multiple cohorts or sub-arms should yield multiple spaces; a single-cohort trial should yield exactly one.

MMAI's directive (from 0b_create_trial_spaces.py) — judge against these rules, not a generic ideal:
- MMAI was told to IGNORE washout periods (e.g. "no chemo within 14 days", "no radiation within 30 days"). Do NOT mark a list as non-exhaustive because it lacks washout-only sub-cohorts or because spaces fail to encode washout-conditional criteria. A "missing arm" must be a real enrollment cohort, not a washout sub-case.
- MMAI was told to assume sex criteria for organ-specific cancers even when the trial text is silent: ovarian / uterine / vulvar / vaginal / fallopian-tube → female; testicular / penile / prostate → male. Do NOT mark a list non-exhaustive for missing the opposite-sex cohort in those cases. Breast cancer defaults to both sexes unless the trial text says otherwise.
- MMAI was told to produce a SEPARATE space for each cancer-type / histology combination. A trial that enrolls multiple cancer types in a single arm should yield multiple spaces — that is the correct behavior, not a mutual-exclusivity violation.
- MMAI was told each space must be SELF-CONTAINED with all nine fields restated; "same as #1"-style references are forbidden. Spaces that fully restate criteria are intended, not redundant.
- MMAI was told to spell out cancer types (e.g. "non-small cell lung cancer", not "NSCLC"); do NOT penalize verbose phrasing.
- The trailing `Boilerplate exclusions:` block lists generic exclusions (pneumonitis, CHF, brain mets, etc.) that intentionally are NOT encoded as spaces; do not count those as missing arms.

Your job is to judge the quality of the space list on two axes:

1. MUTUALLY EXCLUSIVE — no two spaces describe the same (or heavily overlapping) eligibility combination. Two spaces that differ only on an immaterial criterion (e.g. one lists a biomarker as "unknown" vs. the other omits it entirely, or a cohort numbering differs) are NOT mutually exclusive. Two spaces that differ on cancer type, histology, biomarker, or prior-line status ARE mutually exclusive.
2. EXHAUSTIVE — every distinct enrollment arm or cohort actually targeted by the trial is represented by at least one space in the list. If the trial has multiple cohorts (e.g. by cancer type, by prior-line status, by biomarker) and the list misses one, it is not exhaustive. Washout sub-cases and items belonging in the boilerplate exclusion list do NOT count as missing arms.

Here is the trial text:
{trial_text}

Here is the MM-AI extracted list of spaces:
{spaces_block}

Reason step by step:
- First, internally enumerate the actual enrollment arms/cohorts described by the trial text, applying the MMAI directive rules above (ignoring washouts, applying organ-specific sex assumptions, splitting per cancer type / histology).
- Then compare that enumeration to the MM-AI list: are there real arms missing? Are any listed spaces duplicative of each other on a substantive criterion?
- Ignore cosmetic differences (phrasing, ordering, numbering, verbose cancer-type names, fully restated criteria). Focus on substantive overlap and substantive gaps in real enrollment arms.

Resolve each axis to one of:
- YES — the list satisfies this property.
- PARTIAL — mostly satisfied, with one or two material exceptions.
- NO — the list clearly fails this property.

Your response MUST end with exactly the following two lines, in this order, and nothing after them:
Mutually exclusive: X
Exhaustive: Y
where X and Y are each one of YES, PARTIAL, or NO (uppercase).
```

### 5. Finalize

```bash
python "${CLAUDE_SKILL_DIR}/scripts/finalize.py" \
  --work_dir <work_dir> \
  --output <output_path>
```

Appends `spaces_judge_response`, `spaces_judge_me`, `spaces_judge_exh`.

### 6. Report

Echo the script's summary lines plus a one-sentence note of the sample
size and output path.

## Notes

- Do not write any file inside this skill directory at any step.
- Parsing inspects the last ~600 chars of the response. The two verdict
  lines must be the final non-whitespace content.
- The `extracted_spaces` column, when present, is durably serialized on
  staging as JSON to avoid parquet list-column issues.
