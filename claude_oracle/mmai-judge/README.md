# mmai-judge

Claude Code plugin providing Opus-4.7-as-judge evaluation skills for the
MatchMiner-AI training pipeline, plus the rubric-replicating trialcheck oracle.

## Skills

| Skill | Input | Verdict |
| --- | --- | --- |
| `eval-patient-summary` | Two parquet/CSV files — per-patient summaries + raw clinical notes — joined on `patient_id` and sorted by date inside the skill | One of `AGREE`, `OMISSION`, `INCORRECT`, `OMISSION_AND_INCORRECT` |
| `eval-trial-spaces` | One file with `nct_id`, `trial_text`, `extracted_spaces` | Two labels: `mutually_exclusive` and `exhaustive` — each `YES` / `PARTIAL` / `NO` |
| `eval-trialcheck` | One file with `patient_summary`, `this_space`, MM-AI reasoning + 0–5 score columns | 5-point Likert agreement + disagreement category |
| `eval-boilerplate` | One file with `patient_boilerplate_text`, `trial_boilerplate_text`, MM-AI reasoning + Yes!/No! verdict | 5-point Likert agreement + disagreement category |
| `claude-trialcheck-oracle` | One file with `patient_summary`, `this_space` | `Final score: X` (0–5), parsed into `eligibility_result` / `eligibility_verdict` matching `llm_check_trials.py` |

All skills share the same scaffold: `prepare_sample.py` slices a work
directory with slim per-row text files, the model reads each row and emits an
extended-thinking response ending in the verdict line(s), `append_response.py`
writes one JSONL record per row under a stable command prefix, and
`finalize.py` merges the responses back into the sampled dataframe and parses
the verdicts.

## Install

Local development:

```bash
claude --plugin-dir /path/to/mmai-judge
```

Skills are auto-namespaced as `/mmai-judge:eval-<name>`.

## Notes

- `claude-trialcheck-oracle` and `eval-trialcheck` are *not* duplicates.
  The oracle is a rubric re-scorer — it replicates `llm_check_trials.py` from
  scratch and emits a 0–5 integer. `eval-trialcheck` reads MM-AI's own
  reasoning and decides whether the judge agrees with MM-AI's score, emitting
  a Likert agreement + disagreement category.
- Each skill refuses to write `work_dir` or `output_path` inside its own
  skill directory.
- Each skill expects the caller to use extended thinking during scoring
  (the SKILL.md opens with an `ultrathink` directive).
