---
name: eval-boilerplate
description: Judge whether MM-AI agrees with itself on boilerplate exclusion decisions (Yes! / No!). Reads MM-AI's reasoning and verdict and produces a 5-point Likert agreement plus a disagreement category. Use when the user asks to evaluate MM-AI boilerplate checks with Claude as judge, or mentions 14_check_boilerplate / vllm_parallel_boilerplate / exclusion_result eval.
disable-model-invocation: true
argument-hint: [boilerplate_output_parquet_or_csv]
allowed-tools: Read Write Bash(python *) Bash(python3 *) Bash(ls *) Bash(mkdir *) Bash(wc *) Bash(: *)
---

# eval-boilerplate

You are acting as an **Opus-4.7 judge** of MM-AI's boilerplate-exclusion
decisions. Use **ultrathink** on every row — boilerplate exclusions hinge
on close reading of wording (e.g. "uncontrolled brain metastases" vs
"history of brain metastases") and benefit-of-the-doubt conventions.

## Workflow

### 1. Resolve the input file

Input path comes from `$ARGUMENTS`. If empty or non-existent, ask the user.
Must end in `.parquet` or `.csv`.

Required columns:
- `patient_boilerplate_text`
- `trial_boilerplate_text`
- MM-AI reasoning column — auto-detected from `mm_ai_reasoning`,
  `boilerplate_check_llm_response`, `llm_boilerplate_response`,
  `llama_response`, `llm_response`.
- MM-AI verdict column — auto-detected from `mm_ai_verdict`,
  `exclusion_result`, `boilerplate_verdict`. Accepts either string
  (`Yes!` / `No!`) or numeric (`1.0` / `0.0`); the skill renders
  numeric values as `Yes!` or `No!` in the slim row file.

### 2. Collect parameters from the user

Use AskUserQuestion for:

- **sample_size** — default 20.
- **random_seed** — default 42.
- **reasoning_col** — optional override.
- **verdict_col** — optional override.
- **work_dir** — default `<parent-of-input>/boilerplate_judge_work/`.
- **output_path** — default `<input_stem>_judged.<ext>`.

**Invariant — never violate.** Neither `work_dir` nor `output_path` may be
inside this skill directory.

### 3. Prepare the sample

```bash
python "${CLAUDE_SKILL_DIR}/scripts/prepare_sample.py" \
  --input <input_path> \
  --n <sample_size> \
  --seed <random_seed> \
  --work_dir <work_dir> \
  [--reasoning_col <name>] [--verdict_col <name>]
```

### 4. Judge each row

```bash
: > <work_dir>/responses.jsonl
```

For each row, Read `<work_dir>/rows/row_NN.txt`. Assemble the prompt below.
End your response with exactly two lines:

```
Agreement: <STRONGLY_AGREE|AGREE|NEUTRAL|DISAGREE|STRONGLY_DISAGREE>
Disagreement category: <false_exclusion|missed_exclusion|reasoning_flaw|other|n/a>
```

Use `n/a` when Agreement is STRONGLY_AGREE or AGREE.

Append with the stable helper:

```bash
python "${CLAUDE_SKILL_DIR}/scripts/append_response.py" \
  --work_dir <work_dir> --row_id <int> <<'BOILERPLATE_JUDGE_EOF'
<your full response, ending with the two verdict lines>
BOILERPLATE_JUDGE_EOF
```

#### System message (verbatim)

```
Reasoning: high
```

#### User message template (verbatim)

```
You are a brilliant oncologist serving as a judge in an AI evaluation. Another AI system (MM-AI) has been asked to decide whether a patient is clearly excluded from a clinical trial based on the trial's "boilerplate" exclusion criteria (e.g. uncontrolled brain metastases, severe organ dysfunction, active infection, prior cancers, etc.), using a structured extract of the patient's conditions as context. MM-AI answers "Yes!" (clearly excluded) or "No!" (not clearly excluded), giving the benefit of the doubt when a condition is absent or not documented.

You will be shown the patient's boilerplate extract, the trial's boilerplate exclusion list, MM-AI's reasoning, and MM-AI's final verdict. Your job is to judge whether MM-AI's verdict is defensible given the inputs and the directive MMAI was operating under — catching cases where MM-AI fabricated, missed, or misread an exclusion criterion.

MMAI's directive (from vllm_parallel_boilerplate.py) specifically told it to:
- Evaluate ONLY the exclusion criteria listed for THIS specific trial. Do NOT penalize MMAI for failing to consider boilerplate conditions that are not present in {trial_boilerplate_text}, even if those conditions appear in the patient extract.
- Give the patient the benefit of the doubt: if it is not COMPLETELY clear the patient meets an exclusion criterion, err on the side of "not excluded". A close-call No! is the intended behavior.
- Treat descriptions of conditions as "mild", "low-grade", "controlled", or "resolved" as EXTRA reason NOT to exclude. A No! verdict that turns on mild / low-grade / resolved language is the intended behavior, not a reasoning_flaw.
- Treat the patient boilerplate extract as itself an LLM reasoning trace — the extract was produced by another LLM reasoning about whether the patient meets common boilerplate criteria, so mention of a condition in the extract is NOT proof the patient actually has it. Do NOT penalize MMAI for treating ambiguous mentions in the extract as absent.
- Reason through one criterion at a time and end with exactly "Yes!" or "No!".

Here is the patient's boilerplate extract:
{patient_boilerplate_text}

Here is the trial's boilerplate exclusion list:
{trial_boilerplate_text}

Here is MM-AI's reasoning:
{mm_ai_reasoning}

Here is MM-AI's final verdict:
{mm_ai_verdict}

Reason step by step:
- For each criterion ON THE TRIAL'S LIST (and only those), is the patient clearly excluded? (Apply benefit of the doubt — absence of documentation, mild / low-grade / resolved language, and ambiguous mentions in the LLM-generated extract should not be treated as presence.)
- Did MM-AI correctly read each patient condition, with appropriate skepticism that the extract is itself LLM reasoning?
- Did MM-AI correctly read each trial exclusion, and confine itself to the trial's listed criteria?
- Did MM-AI arrive at the right Yes!/No! given its own reasoning and the benefit-of-doubt convention?

Resolve agreement on a 5-point Likert scale:
- STRONGLY_AGREE — reasoning and verdict both correct, no material problems.
- AGREE — essentially right, at most minor nits.
- NEUTRAL — reasoning has problems but verdict is defensible, or vice versa.
- DISAGREE — the verdict is wrong in a meaningful way.
- STRONGLY_DISAGREE — the verdict is wrong AND the reasoning is demonstrably flawed on a key point (fabricated exclusion, missed clear exclusion, wrong benefit-of-doubt call on a hard criterion).

If you land on NEUTRAL / DISAGREE / STRONGLY_DISAGREE, also pick a disagreement category:
- false_exclusion — MM-AI said Yes! (excluded) when the patient is NOT clearly excluded.
- missed_exclusion — MM-AI said No! (not excluded) when the patient IS clearly excluded.
- reasoning_flaw — the verdict may be defensible but the reasoning is substantively wrong (misread of patient or trial text).
- other — a problem that doesn't fit the above.
If Agreement is STRONGLY_AGREE or AGREE, category must be n/a.

Your response MUST end with exactly these two lines, in this order, and nothing after them:
Agreement: X
Disagreement category: Y
```

### 5. Finalize

```bash
python "${CLAUDE_SKILL_DIR}/scripts/finalize.py" \
  --work_dir <work_dir> \
  --output <output_path>
```

Appends `boilerplate_judge_response`, `boilerplate_judge_agreement`,
`boilerplate_judge_category`.

### 6. Report

Echo the script's summary lines plus the reasoning/verdict column names
used and the output path.

## Notes

- Do not write any file inside this skill directory at any step.
- Parser inspects the last ~600 chars of the response — the two verdict
  lines must be at the tail.
- The verdict column is rendered as `Yes!` / `No!` in the slim row file
  regardless of its original type (string or numeric).
