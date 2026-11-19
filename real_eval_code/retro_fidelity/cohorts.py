#!/usr/bin/env python3
"""Stage extraction: the trial sample's ctgov cohorts, made exactly as training made its ctgov cohort tables.

Training's ctgov version (n=5000 run, RUN_REPORT_N5000_20261104.md section 4) was made in two steps, and this stage
runs the same two frozen programs with the same settings, not the current code: the cohort-name and criteria
prompts have been reworded since (inference a048f47), so the current 0d_extract_ctgov_cohorts.py would not
reproduce them.

1. Screen, cohort names, verbatim criteria and merge: ``0b_extract_trial_cohorts.py`` of the run's frozen source
   ``training_run_screened_sample5000_20261104/source_cohorts_fast`` (training 70918c1, inference 76e273c), as the
   run's ctgov-cohorts stage ran it (5 attempts, 50,000 output tokens, structured outputs, Nemotron's sampling
   profile), with ``--no-deboilerplate-cohorts``: its v3 boilerplate step was superseded by step 2.
2. The per-line boilerplate and inclusion/exclusion tags (``trial-cohorts-deboilerplate-v4``):
   ``scripts/rerun_deboilerplate.py --version ctgov`` of ``synthetic_dataset_n5000_screened_20261106/source``
   (training 40ce8c1, inference f792632), on step 1's merged cohorts, as the run's v4 rerun did.

With ``--source SNAPSHOT`` (since 2026-11-18, user instruction: regenerate the retrospective evaluations' cohorts with
the revised cohort prompts) the stage instead runs one program: the snapshot's own ``0d_extract_ctgov_cohorts.py``,
which screens, names, extracts, merges and tags every line (v4) in one pass with that snapshot's inference package.
The snapshot is a ``git archive`` of both repositories with a ``commits.json`` (for example
``data/no_phi/cohort_extraction_source_20261118``, inference ba77a65); its outputs are the same files without
``deboilerplate_v4.json``.

Input: the sampled listings (nct_id, title, brief_summary, eligibility_criteria; literal text kept), written to
``trials.jsonl``. Outputs: ``trial_cohort_lineitems.csv`` (merged), ``..._unmerged.csv``, ``trials_with_cohorts.csv``
(one audit row per trial), ``trial_cohort_lineitems_deboilerplated.csv`` (``cohort_key_criteria``,
``cohort_boilerplate_criteria``, ``cohort_criteria_items``) and ``deboilerplate_v4.json``. Both programs keep every
answer and its reasoning in their checkpoints (``extraction/checkpoints``, ``deboilerplate/``) and resume from them.
ctgov text is public, but this directory sits beside the PHI outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hold_gb300, log, private, private_dir, run_stage, stage, write_json  # noqa: E402

VERSION = 'retro-fidelity-extraction-v1'
SNAPSHOT_VERSION = 'retro-fidelity-extraction-snapshot-v1'
SNAPSHOT_PROMPTS = ['trial.cohorts.screen.user.txt', 'trial.cohorts.list.user.txt', 'trial.cohorts.criteria.user.txt',
                    'trial.cohorts.merge.user.txt', 'trial.cohorts.deboilerplate.user.txt']
DATA = Path('/ksg/kehl_mm_data/mmai/v23/data/no_phi')
STEP1 = DATA / 'training_run_screened_sample5000_20261104/source_cohorts_fast'
STEP2 = DATA / 'synthetic_dataset_n5000_screened_20261106/source'
TOKENIZER = DATA / ('synthetic_protocolcohorts_v11_cohortsv2_nemotron35_20261029/tokenizer/'
                    'NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4')
MODEL = 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4'
OUTPUTS = ['trials.jsonl', 'trials_with_cohorts.csv', 'trial_cohorts.json', 'trial_cohort_lineitems.csv',
           'trial_cohort_lineitems_unmerged.csv', 'trial_cohort_lineitems_deboilerplated.csv', 'deboilerplate_v4.json',
           'extraction_summary.json']


def source_digest(root, names):
    """SHA-256 of the small frozen program files (never data files)."""
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def commands(args, output):
    python = args.python
    step1 = [python, str(STEP1 / 'matchminer-ai-training-kehl/0b_extract_trial_cohorts.py'),
             '--trials-jsonl', str(output / 'trials.jsonl'),
             '--output-trials', str(output / 'trials_with_cohorts.csv'),
             '--output-cohorts', str(output / 'trial_cohort_lineitems.csv'),
             '--output-json', str(output / 'trial_cohorts.json'),
             '--output-unmerged-cohorts', str(output / 'trial_cohort_lineitems_unmerged.csv'),
             '--no-deboilerplate-cohorts', '--work-dir', str(output / 'extraction'),
             '--inference-repo', str(STEP1 / 'matchminer-ai-inference-kehl'),
             '--workers', '1536', '--max-attempts', '5', '--max-tokens', '50000',
             '--max_concurrent_per_server', '512', '--request_timeout', '7200', '--api-key-env', 'OPENAI_API_KEY',
             '--allow-failures', '--model', MODEL, '--tokenizer-model', str(TOKENIZER),
             '--server_urls_file', str(args.servers)]
    step2 = [python, '-I', str(STEP2 / 'scripts/rerun_deboilerplate.py'), '--version', 'ctgov',
             '--input', str(output / 'trial_cohort_lineitems.csv'),
             '--output', str(output / 'trial_cohort_lineitems_deboilerplated'),
             '--decisions-dir', str(output / 'deboilerplate'), '--metadata', str(output / 'deboilerplate_v4.json'),
             '--server-urls-file', str(args.servers), '--inference-repo', str(STEP2 / 'matchminer-ai-inference-kehl'),
             '--model', MODEL, '--tokenizer-model', str(TOKENIZER), '--chunks-in-flight', '4']
    return step1, step2


def snapshot_command(args, output):
    """One pass with the snapshot's current extractor: screen, names, criteria, merge and v4 line tags."""
    source = Path(args.source)
    return [args.python, str(source / 'matchminer-ai-training-kehl/0d_extract_ctgov_cohorts.py'),
            '--trials-jsonl', str(output / 'trials.jsonl'),
            '--output-trials', str(output / 'trials_with_cohorts.csv'),
            '--output-cohorts', str(output / 'trial_cohort_lineitems.csv'),
            '--output-json', str(output / 'trial_cohorts.json'),
            '--output-unmerged-cohorts', str(output / 'trial_cohort_lineitems_unmerged.csv'),
            '--output-deboilerplated-cohorts', str(output / 'trial_cohort_lineitems_deboilerplated.csv'),
            '--work-dir', str(output / 'extraction'),
            '--inference-repo', str(source / 'matchminer-ai-inference-kehl'),
            '--workers', str(args.workers), '--max-attempts', '5', '--max-tokens', '50000',
            '--max_concurrent_per_server', str(args.workers), '--request_timeout', '7200',
            '--api-key-env', 'OPENAI_API_KEY', '--allow-failures', '--model', MODEL, '--tokenizer-model', str(TOKENIZER),
            '--server_urls_file', str(args.servers)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trials', required=True, help='Sampled listings CSV (soc_trial_sample.csv)')
    p.add_argument('--servers', required=True, help='Endpoint registry (each row with its own model)')
    p.add_argument('--output', required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--holders', help='OA-pause holder registry (common.hold_gb300)')
    p.add_argument('--source', help='A code snapshot (git archive of both repositories with commits.json) whose current '
                                    'extractor makes the cohorts in one pass, instead of the frozen 2026-11-04 programs')
    p.add_argument('--workers', type=int, default=512, help='Requests in flight with --source (part of the identity)')
    stage.add_check_arg(p)
    args = p.parse_args(argv)
    private()
    from trial_cohort_format import file_stamp, read_clinical_csv
    output = Path(args.output)
    identity = dict(version=VERSION, trials=file_stamp(args.trials), model=MODEL,
                    step1=source_digest(STEP1, ['matchminer-ai-training-kehl/0b_extract_trial_cohorts.py',
                                                'matchminer-ai-training-kehl/trial_cohort_format.py']),
                    step2=source_digest(STEP2, ['scripts/rerun_deboilerplate.py',
                                                'matchminer-ai-training-kehl/ctgov_llm.py']),
                    step1_commits=json.loads((STEP1 / 'commits.json').read_text()),
                    step2_commits=json.loads((STEP2 / 'commits.json').read_text()))
    for side in ('step1_commits', 'step2_commits'):
        identity[side] = {repo: value['commit'] for repo, value in identity[side].items() if isinstance(value, dict)}
    outputs = OUTPUTS
    if args.source:
        source = Path(args.source)
        prompts = 'matchminer-ai-inference-kehl/src/matchminer_ai/prompts/'
        identity = dict(version=SNAPSHOT_VERSION, trials=file_stamp(args.trials), model=MODEL, workers=args.workers,
                        program=source_digest(source, ['matchminer-ai-training-kehl/0d_extract_ctgov_cohorts.py',
                                                       'matchminer-ai-training-kehl/ctgov_llm.py',
                                                       'matchminer-ai-training-kehl/trial_cohort_format.py']),
                        prompts=source_digest(source, [prompts + name for name in SNAPSHOT_PROMPTS]),
                        commits={repo: value['commit'] for repo, value in
                                 json.loads((source / 'commits.json').read_text()).items() if isinstance(value, dict)})
        outputs = [name for name in OUTPUTS if name != 'deboilerplate_v4.json']

    def work():
        hold_gb300(args.holders, 'cohort extraction (cohorts.py)')
        private_dir(output)
        listings = read_clinical_csv(args.trials, dtype=str)
        frame = listings[['nct_id', 'title', 'brief_summary', 'eligibility_criteria']].fillna('')
        if frame.nct_id.duplicated().any():
            raise ValueError('Duplicate NCT IDs in the trial sample')
        path = output / 'trials.jsonl'
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                         for row in frame.sort_values('nct_id').to_dict('records')))
            temporary.replace(path)
        environment = dict(os.environ, TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        environment.setdefault('OPENAI_API_KEY', 'not-needed')
        for number, command in enumerate([snapshot_command(args, output)] if args.source else commands(args, output), 1):
            log(f'extraction step {number} started')
            with (output / f'step{number}.log').open('a') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=environment,
                               cwd=Path(command[2] if command[1] == '-I' else command[1]).parent)
            log(f'extraction step {number} finished')
        audit = read_clinical_csv(output / 'trials_with_cohorts.csv')
        merged = read_clinical_csv(output / 'trial_cohort_lineitems.csv')
        clean = read_clinical_csv(output / 'trial_cohort_lineitems_deboilerplated.csv')
        unmerged = read_clinical_csv(output / 'trial_cohort_lineitems_unmerged.csv')
        if args.source:
            items = [json.loads(value) for value in clean.cohort_criteria_items.dropna() if str(value).strip()]
            manifest = json.loads((output / 'extraction' / 'manifest.json').read_text())['identity']
            metadata = dict(items=sum(len(i) for i in items), schema_version=(manifest.get('deboilerplate') or {}).get('schema'),
                            prompt_sha256=(manifest.get('deboilerplate') or {}).get('prompt_sha256'))
        else:
            metadata = json.loads((output / 'deboilerplate_v4.json').read_text())
        summary = dict(trials=len(audit), cancer_treatment=int(audit.cancer_treatment_trial.eq('yes').sum()),
                       not_cancer_treatment=int(audit.cancer_treatment_trial.eq('no').sum()),
                       failed=int(audit.extraction_error.notna().sum()), cohorts_before_merge=len(unmerged),
                       cohorts=len(merged), deboilerplated_cohorts=len(clean),
                       trials_with_cohorts=int(merged.trial_id.nunique()), lines=metadata.get('items'),
                       deboilerplate_schema=metadata.get('schema_version'),
                       deboilerplate_prompt_sha256=metadata.get('prompt_sha256'),
                       cohorts_kept_as_extracted=len(metadata.get('deboilerplate_errors') or {}),
                       cohorts_with_unclassified_lines=len(metadata.get('unclassified_items') or {}))
        write_json(output / 'extraction_summary.json', summary)
        log(f'extraction: {summary}')
        return summary

    run_stage(output, 'retro-extraction', identity, outputs, work, check=args.check)


if __name__ == '__main__':
    main()
