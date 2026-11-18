# Dual frontier LLM judges with manual adjudication

User, 2026-11-18: LLM-as-judge evaluation of three pipeline steps, with two frontier judges and the user
adjudicating wherever they disagree. The judges are Claude Opus 5.5 on Vertex AI (project profile-notes) and
GPT-6.1 Sol on Azure.

| Component | One item | The judges | Units compared |
|---|---|---|---|
| `extraction` | one trial's ctgov cohort extraction (non-PHI) | audit the screen, the cohort set (names, merges) and every criteria line (fidelity, key/boilerplate tag, inclusion/exclusion tag), plus missing criteria, against the listing and the extractor's own frozen prompts | screen; cohort set; per line: fidelity (ok/minor/major), boilerplate tag, type tag; per cohort: missing criteria (major or not); overall |
| `summary` | one index event's summary (PHI) | check every numbered statement of the final summary against the whole record (as the summarizer read it), then list omissions of what the summarizer was told to capture | per statement: ok/minor/major; per section: a major omission or not; overall |
| `decision` | one teacher request (PHI): a match pair (5 questions), a boilerplate pair, or one criteria line | answer it **blind**, exactly as the Nemotron teacher was asked, under the label contract the decision model was trained on | one yes/no per question, in the decision model's polarity |

The agreed or adjudicated value of each unit is the reference. For extraction and summaries it gives the
pipeline's error rates. For the decision questions it scores the teacher (accuracy) and the decision model
(AUROC, AUPRC, Brier).

## Files

| File | Runs on | Needs |
|---|---|---|
| `build_items.py` | camus (reads the fidelity run, notes, summaries and frozen prompts) | gptoss3 environment |
| `judge_worker.py` | any machine that can read and write the run directory | Python 3.10+; `anthropic[vertex]` and Google ADC for `vertex`; `az login` (or `AZURE_OPENAI_API_KEY`) for `azure` |
| `compare.py`, `adjudicate.py`, `metrics.py` | any machine that can read and write the run directory (kehl-lab) | Python 3.10+ standard library only |
| `judging.py` | shared helpers | |
| `rubric_{extraction,summary,decision}.py`, `prompts/judge.*.txt` | the three rubrics: sampling, rendering, schema, validation, units and the app's view | |
| `static/adjudicate.html` | the adjudication page | |

## A run directory

`v23/data/phi/dual_judge_<cohort>_<date>/` (PHI: directories 700, files 600). It contains:
- `build.json`;
- `items/` and `requests/` (one request per item; both judges get the same bytes);
- `responses/<judge>/` (append-only, one row per attempt);
- `compare/units.jsonl` and `compare/summary.json`;
- `adjudications.jsonl` (append-only; the latest decision per unit wins);
- `metrics/metrics.json` and `metrics/report.md`;
- `logs/`.

The smoke run is `v23/data/phi/dual_judge_soc_validation_20261118/`.

## Steps

```bash
TRAIN=/ksg/kehl_mm_data/mmai/v23/matchminer-ai-training-kehl/real_eval_code/dual_judge
RUN=/ksg/kehl_mm_data/mmai/v23/data/phi/dual_judge_soc_validation_20261118

# 1. build (camus): sample and render; ends with each judge's --dry-run price
/data1/ken/envs/gptoss3/bin/python $TRAIN/build_items.py --run-dir $RUN \
  --fidelity-dir /ksg/kehl_mm_data/mmai/v23/data/phi/real_eval_fidelity_soc_validation_20261117 \
  --denominators /ksg/kehl_mm_data/mmai/v23/data/phi/real_eval_denominators_20261114

# 2. judges (resumable; --limit 1 for a live test, --dry-run to price, --max-usd caps spend, default 75)
python3 $TRAIN/judge_worker.py --run-dir $RUN --judge vertex
python3 $TRAIN/judge_worker.py --run-dir $RUN --judge azure \
  --endpoint https://<resource>.openai.azure.com/openai/v1 --deployment <GPT-6.1 Sol deployment name> \
  --usd-in <price per M input> --usd-out <price per M output>

# 3. compare (rerun whenever answers arrive), 4. adjudicate, 5. metrics (any time, also mid-adjudication)
python3 $TRAIN/compare.py --run-dir $RUN
python3 $TRAIN/adjudicate.py --run-dir $RUN        # open http://127.0.0.1:8791 on the same machine
python3 $TRAIN/metrics.py --run-dir $RUN           # metrics/report.md
```

### On kehl-lab
kehl-lab sees camus's `/ksg` over sshfs as uid 1000, while the files belong to camus's uid 50777. The run
directory is 600/700 PHI, so kehl-lab can neither read nor write it until the mount maps the user. Add
`uid=1000,gid=1000` to kehl-lab's sshfs mount options and remount (needs sudo; it interrupts anything running from
`/ksg`). Do not loosen the run directory's permissions instead. After that, steps 2–5 run there unchanged with
kehl-lab's own `python3`.
- The Vertex judge needs `pip install 'anthropic[vertex]'` and ADC for profile-notes.
- The Azure judge needs `az login`, or `AZURE_OPENAI_API_KEY`.
- The app binds to 127.0.0.1 and refuses other Host headers. Open it in a browser on kehl-lab itself, or tunnel
  with `ssh -L 8791:127.0.0.1:8791`.

## How the judging works

**Both judges get the same request.** The system text is the rubric, plus the rules the judged step was given, plus
the answer's JSON schema.
- The profile-notes organization policy forbids structured outputs for partner models on Vertex
  (`constraints/vertexai.allowedPartnerModelFeatures`, 2026-11-18). So Opus answers to the schema written in the
  prompt.
- Azure additionally enforces the schema (strict `json_schema`).
- Neither judge gets sampling parameters. Effort is `high`.

**Every answer is validated by code:**
- the schema and enums;
- coverage: every cohort and line, or every statement, exactly once;
- the teacher contract's polarity rule for criteria lines.

An invalid answer is retried once with the errors appended. Refusals, truncation and twice-invalid answers are
final; that item's units go to the queue as `single`. A judge never falls back to another model.

**Quotes are checked.** Every evidence quote is looked up in the source and marked found or not found.

**Comparison** reduces each answer to units:
- `agreed`: becomes the reference.
- `disputed`: queued.
- `single`: the other judge failed; queued.
- `awaiting`: the other judge has not run yet; not queued.
- 10% of agreed units are queued as an `audit`, to estimate how often the judges agree and are both wrong.
  "Nothing missing" units are not audited.
- Severity kinds (omissions, missing criteria, overall) compare "major" against "not major".

**The app** shows the source on the left (notes with a date index and search, the listing, or the teacher task)
and the item's queued units on the right.
- The judges appear as Judge 1 and Judge 2 in an order fixed per item, revealed after you decide.
- For decision questions, the teacher's answer and the model's P(yes) are also revealed only after you decide.
- Keys: j/k move between units, 1/2 take a judge's value, a/b/c… pick a choice, n focuses the note, s skips,
  [ and ] change item.
- A decision made before a judge's answer changed is flagged as stale.

**Metrics** weight items by sampling weight. 95% intervals bootstrap over items.

## Smoke run, 2026-11-18 (validation SOC)

**What it samples from:**
- **Extraction:** 10 of the 500 trials in the ctgov trial sample.
- **Summaries:** 8 of the 500 query index events, in four contiguous note-volume strata (197 / 182 / 108 / 13
  events).
- **Decision questions:** 52 teacher requests:
  - 16 match pairs (80 question answers) of 34,595;
  - 12 boilerplate combinations of 33,135;
  - 24 criteria lines of 10,132.

  Each family is stratified by teacher answer × decision P. Criteria lines carry two-stage weights.

**Opus on Vertex, run from camus.** All answers were valid on the first attempt. The current items cost $8.01:
- extraction $1.40 ($0.14 per trial);
- summaries $4.94 ($0.62 per event; cost grows with note volume, about $0.07 plus $2.2 per million characters);
- decision questions $1.67 ($0.032 per request).

A further $5.41 went to summary items that the stratum fix replaced. The Azure judge has not run yet: it needs the
machine where `az` is logged in.
