---
name: eval-trialcheck
description: Judge whether MM-AI agrees with itself on patient x trial-space reasoning. Reads MM-AI's reasoning and 0-5 score and produces a 5-point Likert agreement plus a disagreement category. Distinct from the rubric-replicating claude-trialcheck-oracle skill. Use when the user asks to evaluate MM-AI trialcheck reasoning with Claude as judge, or mentions vllm_parallel_trialcheck / eligibility_result eval.
disable-model-invocation: true
argument-hint: [trialcheck_output_parquet_or_csv]
allowed-tools: Read Write Bash(python *) Bash(python3 *) Bash(ls *) Bash(mkdir *) Bash(wc *) Bash(: *)
---

# eval-trialcheck

You are acting as an **Opus-4.7 judge** of MM-AI's trial-check reasoning.
Use **ultrathink** on every row — the task is not to re-score the case
from scratch (the `claude-trialcheck-oracle` skill does that) but to read
MM-AI's reasoning chain and judge whether it supports MM-AI's final score.

## Workflow

### 1. Resolve the input file

Input path comes from `$ARGUMENTS`. If empty or non-existent, ask the user.
Must end in `.parquet` or `.csv`.

Required columns:
- `patient_summary`
- `this_space`
- MM-AI reasoning column — auto-detected from `mm_ai_reasoning`,
  `trialcheck_llm_response`, `llama_response`, `llm_response`.
- MM-AI score column — auto-detected from `mm_ai_score`,
  `eligibility_result`, `score`.

### 2. Collect parameters from the user

Use AskUserQuestion for:

- **sample_size** — default 20.
- **random_seed** — default 42.
- **reasoning_col** — optional override; default "" (auto-detect).
- **score_col** — optional override; default "" (auto-detect).
- **work_dir** — default `<parent-of-input>/trialcheck_judge_work/`.
- **output_path** — default `<input_stem>_judged.<ext>`.

If auto-detection will fail (you don't see any of the alias names in the
file), ask the user which columns to use and pass them as
`--reasoning_col` / `--score_col`.

**Invariant — never violate.** Neither `work_dir` nor `output_path` may be
inside this skill directory.

### 3. Prepare the sample

```bash
python "${CLAUDE_SKILL_DIR}/scripts/prepare_sample.py" \
  --input <input_path> \
  --n <sample_size> \
  --seed <random_seed> \
  --work_dir <work_dir> \
  [--reasoning_col <name>] [--score_col <name>]
```

Strips leading "N." numbering from `this_space` (mirroring
`llm_check_trials.py`), writes `staging.parquet`, `cols.txt`, and
`<work_dir>/rows/row_NN.txt` — each containing four blocks: PATIENT SUMMARY,
TRIAL SPACE, MM-AI REASONING, MM-AI FINAL SCORE.

### 4. Judge each row

```bash
: > <work_dir>/responses.jsonl
```

For each row, Read `<work_dir>/rows/row_NN.txt`. Assemble the prompt below
verbatim. End your response with exactly two lines:

```
Agreement: <STRONGLY_AGREE|AGREE|NEUTRAL|DISAGREE|STRONGLY_DISAGREE>
Disagreement category: <reasoning_flaw|wrong_score_right_logic|missed_biomarker|missed_cancer_type|other|n/a>
```

Use `n/a` for the category when Agreement is STRONGLY_AGREE or AGREE.

Append with the stable helper:

```bash
python "${CLAUDE_SKILL_DIR}/scripts/append_response.py" \
  --work_dir <work_dir> --row_id <int> <<'TRIALCHECK_JUDGE_EOF'
<your full response, ending with the two verdict lines>
TRIALCHECK_JUDGE_EOF
```

#### System message (verbatim)

```
Reasoning: high
```

#### User message template (verbatim — fill in the four placeholders)

```
You are a brilliant oncologist serving as a judge in an AI evaluation. Another AI system (MM-AI) has been asked to decide whether a clinical trial space is a reasonable consideration for a specific patient, and to score how targeted the trial is for that patient on a 0–5 rubric. You will be shown the patient summary, the trial space, MM-AI's free-text reasoning, and MM-AI's final score.

Scoring rubric (for reference — this is what MM-AI was using):
Start with 0.
1) REASONABLENESS (0 or 1 point): +1 if the trial is a reasonable consideration (patient does not clearly meet an exclusion criterion such as wrong cancer type, wrong age group, wrong sex, excluded biomarker). If the trial is NOT reasonable, the final score is 0 — skip the remaining categories.
2) CANCER TYPE SPECIFICITY (+1): +1 if the trial specifies the patient's cancer type (e.g. "breast cancer") rather than "solid tumors"/"any cancer".
3) CANCER BURDEN/STAGE SPECIFICITY (+1): +1 if the trial specifies a particular stage/burden that matches the patient.
4) PRIOR TREATMENT SPECIFICITY (+1): +1 if the trial has specific prior-treatment requirements and the patient's history matches them.
5) BIOMARKER SPECIFICITY (+1): +1 if the trial requires a specific biomarker and the patient is known to have it.

MMAI's directive specifically told it to:
- IGNORE washout periods when judging reasonableness — assume the patient could wait. Do NOT penalize MMAI for not down-scoring a trial on washout grounds.
- IGNORE today's calendar date and reason from the most recent information in the patient summary as if that were "now". Do NOT penalize MMAI for not flagging that the most recent note is months or years old.
- NOT provide ethical judgments and NOT comment on resource constraints, feasibility, cost, or insurance. Do NOT penalize MMAI for omitting these considerations.
- Treat known absence of a required biomarker as making the trial NOT a reasonable consideration. MMAI was given the worked example that KRAS and EGFR driver mutations in lung cancer are mutually exclusive, so a documented KRAS mutation may be treated as evidence the patient does NOT have an EGFR mutation. Reasoning of that shape (using known biology to infer absence of a mutually exclusive biomarker) is acceptable.
- Skip the remaining specificity categories and finalize at score 0 whenever the trial fails the reasonableness gate.

Here is the patient summary:
{patient_summary}

Here is the trial space:
{trial_space}

Here is MM-AI's reasoning:
{mm_ai_reasoning}

Here is MM-AI's final score (0–5):
{mm_ai_score}

Your job: decide whether MM-AI's reasoning supports MM-AI's score, and whether both are correct given the patient/trial pair AND the directive above. Do NOT simply re-score from scratch — focus on MM-AI's chain of thought, catching hallucinations, criterion mis-reads, misattributed biomarkers, etc.

Reason step by step:
- Is MM-AI's reading of the patient accurate? (cancer type, burden, prior treatments, biomarkers)
- Is MM-AI's reading of the trial space accurate? (cancer type specified? biomarker required? stage specified?)
- Does MM-AI's rubric application land on the right score given its own reasoning AND the directive rules above (ignore washouts, ignore today's date, skip specificity points after a 0 on reasonableness)?
- If you spot a problem, is it (a) a flaw in the reasoning that happens to land on the right score, (b) reasoning that is basically correct but arrives at a wrong score, or (c) something else?

Resolve agreement on a 5-point Likert scale:
- STRONGLY_AGREE — reasoning is accurate, score is correct, no material problems.
- AGREE — reasoning and score are essentially right; at most minor nits.
- NEUTRAL — reasoning has notable problems but the final score is defensible, or vice versa in a way that roughly cancels out.
- DISAGREE — meaningful problem in reasoning AND/OR score; a different score would clearly be more defensible.
- STRONGLY_DISAGREE — the reasoning is wrong on a key point (wrong cancer type, wrong biomarker call, fabricated information) or the score is clearly off by 2+ points.

If you land on NEUTRAL / DISAGREE / STRONGLY_DISAGREE, also pick a disagreement category:
- reasoning_flaw — the reasoning is flawed (misread patient, misread trial, logic error).
- wrong_score_right_logic — reasoning is fine but the rubric was applied incorrectly to produce the score.
- missed_biomarker — MM-AI missed or misattributed a biomarker (required, present, absent).
- missed_cancer_type — MM-AI missed or misattributed the cancer type.
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

Appends `trialcheck_judge_response`, `trialcheck_judge_agreement`,
`trialcheck_judge_category`.

### 6. Report

Echo the script's summary lines plus the reasoning/score column names used
and the output path.

## Notes

- Do not write any file inside this skill directory at any step.
- Parser inspects the last ~600 chars of the response — the two verdict
  lines must be at the tail.
- This skill is NOT the same as `claude-trialcheck-oracle`. The oracle
  re-scores the case from scratch; this skill judges MM-AI's own reasoning.
