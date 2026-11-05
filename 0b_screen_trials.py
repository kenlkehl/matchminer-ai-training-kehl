#!/usr/bin/env python3
"""Screen the registry trials training can sample: registry rule, DFCI holdout, cancer-treatment screen.

Production training draws its trial sample (0c_sample_trial_spaces.py --screened-trials) from the
trials this step accepts, in this order (user instruction, 2026-11-04):

1. Registry rule: 0b_extract_trial_cohorts.select_trials keeps interventional early phase I-IV
   trials first posted since --start-year from the parsed ClinicalTrials.gov export, and assigns
   each trial's split by NCT ID.
2. DFCI holdout: trials open at DFCI between --dfci-first-year and --dfci-last-year
   (0c_sample_trial_spaces.dfci_nct_ids_open_during) are excluded.
3. Cancer-treatment screen: every remaining listing (title, brief summary and eligibility criteria
   as one document) gets the inference package's screen_trials, the first step of the cohort
   workflow, with the same prompt and settings as cohort extraction. Every attempt's answer and
   reasoning is appended to <work-dir>/screen.jsonl as it returns and reused on rerun.

Writes --output, one row per trial the registry rule kept (nct_id, split, dates, dfci_holdout,
cancer_treatment_trial, screen_error), and <output>.screening.json with the count and IDs removed at
each step, the prompt hash, model, settings and commits.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from trial_cohort_format import file_stamp, read_clinical_csv  # noqa: E402


def load_stage(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem.replace('-', '_'), ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_commit(path):
    return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None


def main(args):
    extract, sample = load_stage('0b_extract_trial_cohorts.py'), load_stage('0c_sample_trial_spaces.py')
    sys.path.insert(0, str(args.inference_repo.resolve() / 'src'))
    if args.api_key_env != 'OPENAI_API_KEY':
        os.environ['OPENAI_API_KEY'] = os.environ.get(args.api_key_env, '')
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 4 * args.workers + 1024
    if soft != resource.RLIM_INFINITY and soft < needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed if hard == resource.RLIM_INFINITY else min(needed, hard), hard))
    import matchminer_ai
    from matchminer_ai.trials import screen_trials
    from matchminer_ai.trials.cohort_screen import SCREEN_FILE, SCREEN_SCHEMA_VERSION

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / 'screen.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        export = read_clinical_csv(args.input)
        selected = extract.select_trials(export, start_year=args.start_year, seed=args.seed)
        held_out = sample.dfci_nct_ids_open_during(pd.read_csv(args.dfci_trials_csv), args.dfci_first_year,
                                                   args.dfci_last_year)
        selected['dfci_holdout'] = selected.nct_id.isin(held_out)
        candidates = selected.loc[~selected.dfci_holdout]
        print(f'{len(export)} exported trials; {len(selected)} meet the registry rule; '
              f'{int(selected.dfci_holdout.sum())} open at DFCI {args.dfci_first_year}-{args.dfci_last_year} '
              f'are held out; screening {len(candidates)}', flush=True)
        config = extract.make_config(args)
        decisions, metadata = screen_trials(
            candidates[['nct_id', 'title', 'brief_summary', 'eligibility_criteria']].rename(columns={'nct_id': 'trial_id'}),
            config=config, output_dir=args.work_dir, return_metadata=True)
        decisions = decisions.rename(columns={'trial_id': 'nct_id'})
        if metadata['screen_errors'] and not args.allow_failures:
            raise SystemExit(f"{len(metadata['screen_errors'])} trials have no validated screen answer "
                             f"(see {args.work_dir / SCREEN_FILE}); rerun to retry them, or pass --allow-failures")
        columns = ['nct_id', 'split', 'study_first_post_date', 'study_start_date', 'overall_status', 'phases', 'dfci_holdout']
        output = selected[columns].merge(decisions, on='nct_id', how='left', validate='one_to_one')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f'.{args.output.name}.{os.getpid()}.tmp')
        output.to_csv(temporary, index=False)
        temporary.replace(args.output)
        screened = output.loc[~output.dfci_holdout]
        receipt = {
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'input': str(args.input.resolve()), 'input_stamp': file_stamp(args.input),
            'output': str(args.output.resolve()), 'output_stamp': file_stamp(args.output),
            'exported_trials': len(export),
            'registry_rule': dict(start_year=args.start_year, study_type='INTERVENTIONAL',
                                  phases=sorted(extract.PHASES), trials=len(selected), split_seed=args.seed),
            'dfci_holdout': dict(path=str(Path(args.dfci_trials_csv).resolve()), first_year=args.dfci_first_year,
                                 last_year=args.dfci_last_year, trials=int(selected.dfci_holdout.sum()),
                                 nct_ids=sorted(selected.nct_id[selected.dfci_holdout])),
            'screen': dict(schema_version=SCREEN_SCHEMA_VERSION, prompt_sha256=metadata['prompt_sha256'],
                           decisions=str((args.work_dir / SCREEN_FILE).resolve()), screened=len(screened),
                           accepted=int(screened.cancer_treatment_trial.eq('yes').sum()),
                           rejected=int(screened.cancer_treatment_trial.eq('no').sum()),
                           rejected_nct_ids=sorted(screened.nct_id[screened.cancer_treatment_trial.eq('no')]),
                           errors=metadata['screen_errors'], reused=metadata['reused'],
                           structured_outputs=metadata['structured_outputs']),
            'model': args.model, 'tokenizer_model': args.tokenizer_model, 'max_tokens': args.max_tokens,
            'max_attempts': args.max_attempts,
            'training_commit': git_commit(ROOT), 'inference_commit': git_commit(Path(matchminer_ai.__file__).parent),
        }
        Path(f'{args.output}.screening.json').write_text(json.dumps(receipt, indent=2) + '\n')
        screen = receipt['screen']
        print(f"screened {screen['screened']}: {screen['accepted']} cancer treatment trials, {screen['rejected']} not, "
              f"{len(screen['errors'])} without an answer; wrote {args.output}", flush=True)


def parser():
    extract = load_stage('0b_extract_trial_cohorts.py')
    p = extract.parser()
    p.description = __doc__.split('\n')[0]
    data = ROOT.parent / 'data/no_phi'
    p.set_defaults(work_dir=data / 'trial_screen')
    p.add_argument('--output', type=Path, default=data / 'screened_trials.csv')
    p.add_argument('--dfci-trials-csv', type=Path, default=data / 'dfci_trials_summary_10-18-26.csv')
    p.add_argument('--dfci-first-year', type=int, default=2025)
    p.add_argument('--dfci-last-year', type=int, default=2026)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
