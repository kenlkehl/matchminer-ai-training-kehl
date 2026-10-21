#!/usr/bin/env python3
"""Resumable CT.gov → synthetic notes → summaries → model training orchestration.

The default starts at training for compatibility with existing train_all commands.
Select an earlier --start-at and optional --stop-after to run data production.
Serving/VM ownership stays with the external supervisor for upstream stages.
"""
import argparse
from dataclasses import dataclass, asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import train_from_summaries as training

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / 'data/no_phi'
STAGES = ('parse-trials', 'extract-spaces', 'sample-trials', 'generate-synthetic',
          'prepare-notes', 'summarize-patients', 'prepare-training', 'train')


@dataclass
class Stage:
    name: str
    command: list[str]
    inputs: list[Path]
    outputs: list[Path]
    sources: list[Path]


def parser():
    p = training.parser()
    p.description = __doc__
    p.add_argument('--start-at', choices=STAGES, default='train')
    p.add_argument('--stop-after', choices=STAGES, default='train')
    p.add_argument('--pipeline-dir', type=Path, help='Upstream state/log directory; default RUN_DIR/upstream')
    p.add_argument('--ctgov-json', type=Path)
    p.add_argument('--parsed-trials', type=Path)
    p.add_argument('--screening-input', type=Path)
    p.add_argument('--start-year', type=int, default=2020)
    p.add_argument('--sampled-spaces', type=Path)
    p.add_argument('--sample-trials', type=int, default=3000)
    p.add_argument('--data-seed', type=int, default=42)
    p.add_argument('--dfci-trials-csv', type=Path, default=DATA_ROOT / 'dfci_trials_summary_10-18-26.csv')
    p.add_argument('--generation-dir', type=Path)
    p.add_argument('--raw-synthetic-notes', type=Path)
    p.add_argument('--server-urls-file', type=Path, help='Externally managed live pool for upstream LLM stages')
    p.add_argument('--concurrency-control-file', type=Path)
    p.add_argument('--generation-model', help='API model ID/alias; defaults to the teacher model/alias')
    p.add_argument('--generation-tokenizer', help='HF tokenizer/model-family ID; defaults to --teacher-model')
    p.add_argument('--download-dir', type=Path)
    p.add_argument('--generation-context', type=int, default=250000)
    p.add_argument('--generation-concurrency-per-server', type=int, default=1024)
    p.add_argument('--generation-pending-requests', type=int, default=2560)
    p.add_argument('--generation-request-timeout', type=float, default=7200)
    p.add_argument('--generation-max-attempts', type=int, default=16)
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    p.add_argument('--wait-for-generation', action='store_true',
                   help='Observe the existing generation job until complete; never launch or interrupt it')
    p.add_argument('--generation-wait-timeout', type=float, default=0, help='Seconds; 0 waits indefinitely')
    p.add_argument('--synthetic-date-anchor', default='2025-01-01')
    p.add_argument('--summary-chunk-size', type=int, default=50000)
    p.add_argument('--summary-chunk-overlap', type=int, default=500)
    p.add_argument('--summary-max-tokens', type=int, default=20000)
    return p


def resolve_paths(args):
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    defaults = dict(pipeline_dir=args.run_dir / 'upstream', parsed_trials=args.data_dir / 'ctgov_trials.csv',
                    trial_spaces=args.data_dir / 'trial_space_lineitems.csv',
                    generation_dir=args.data_dir / 'synthetic_generation',
                    synthetic_notes=args.data_dir / 'all_synthetic_notes.parquet',
                    patient_summaries=args.data_dir / 'patient_summaries_with_spaces.parquet')
    for key, value in defaults.items():
        if args.start_at == 'train' and key in ('trial_spaces', 'synthetic_notes', 'patient_summaries'):
            continue
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.sampled_spaces = args.sampled_spaces or args.generation_dir / 'sample_trial_space_lineitems.csv'
    args.raw_synthetic_notes = args.raw_synthetic_notes or args.generation_dir / 'all_synthetic_notes.parquet'
    args.generation_model = args.generation_model or args.teacher_served_model or args.teacher_model
    args.generation_tokenizer = args.generation_tokenizer or args.teacher_model
    if args.start_at != 'train' and args.server_urls_file and not args.teacher_url and not args.teacher_servers_file:
        args.teacher_servers_file = args.server_urls_file
        args.teacher_served_model = args.teacher_served_model or args.generation_model
        args.teacher_concurrency_file = args.teacher_concurrency_file or args.concurrency_control_file
        if args.teacher_concurrency_file:
            args.label_concurrency_per_server = args.generation_concurrency_per_server
    return args


def training_command(args):
    command = [sys.executable, str(ROOT / 'train_from_summaries.py')]
    for action in training.parser()._actions:
        if action.dest in ('help', 'dry_run'):
            continue
        value = getattr(args, action.dest)
        if value is None:
            continue
        option = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                command.append(option)
        elif isinstance(action, argparse._AppendAction):
            for item in value:
                command += [option, str(item)]
        else:
            command += [option, str(value)]
    return command


def build_plan(args):
    if STAGES.index(args.start_at) > STAGES.index(args.stop_after):
        raise ValueError('--start-at must precede --stop-after')
    selected = STAGES[STAGES.index(args.start_at):STAGES.index(args.stop_after) + 1]
    if args.wait_for_generation and 'generate-synthetic' not in selected:
        raise ValueError('--wait-for-generation requires the generate-synthetic stage in the selected range')
    d, g, work = args.data_dir, args.generation_dir, args.pipeline_dir
    py = sys.executable
    registry = args.server_urls_file or args.teacher_servers_file
    remote = ['--server_urls_file', str(registry)] if registry else []
    remote += ['--max_concurrent_per_server', str(args.generation_concurrency_per_server),
               '--request_timeout', str(args.generation_request_timeout),
               '--max_attempts', str(args.generation_max_attempts), '--api-key-env', args.api_key_env]
    if args.concurrency_control_file:
        remote += ['--concurrency-control-file', str(args.concurrency_control_file)]
    model = ['--model', args.generation_model, '--tokenizer-model', args.generation_tokenizer]
    cache = ['--download-dir', str(args.download_dir)] if args.download_dir else []
    source = lambda *names: [ROOT / name for name in names]
    extraction = [py, str(ROOT / '0b_create_trial_spaces.py'), '--input', str(args.parsed_trials),
                  '--start-year', str(args.start_year), '--seed', str(args.data_seed),
                  '--output-trials', str(d / 'trials_with_spaces.csv'), '--output-spaces', str(args.trial_spaces),
                  '--output-screening', str(d / 'trial_treatment_screening.csv'),
                  '--work-dir', str(work / 'extraction'), '--max-model-len', str(args.generation_context),
                  *model, *cache, *remote]
    if args.screening_input:
        extraction += ['--screening-input', str(args.screening_input)]
    generated = [g / name for name in ('all_synthetic_notes.parquet', 'all_synthetic_events.parquet', 'manifest.json')]
    patient_map = d / 'synthetic_patient_metadata.parquet'
    raw_summaries = d / 'patient_summaries.parquet'
    candidates = [
        Stage('parse-trials', [py, str(ROOT / '0a_parse_ctgov_json.py'), '--input', str(args.ctgov_json),
                              '--output', str(args.parsed_trials)],
              [args.ctgov_json] if args.ctgov_json else [], [args.parsed_trials], source('0a_parse_ctgov_json.py', 'ctgov_metadata.py')),
        Stage('extract-spaces', extraction, [args.parsed_trials] + ([args.screening_input] if args.screening_input else []),
              [args.trial_spaces, d / 'trials_with_spaces.csv'],
              source('0b_create_trial_spaces.py', 'trial_hierarchy.py', 'trial_hierarchy_runner.py',
                     'trial_space_export.py', 'trial_treatment_screening.py') + sorted((ROOT / 'trial_prompts').glob('*.txt'))),
        Stage('sample-trials', [py, str(ROOT / '0c_sample_trial_spaces.py'), '--input-csv', str(args.trial_spaces),
              '--output-csv', str(args.sampled_spaces), '--dfci-trials-csv', str(args.dfci_trials_csv),
              '--n-trials', str(args.sample_trials), '--seed', str(args.data_seed)],
              [args.trial_spaces, args.dfci_trials_csv], [args.sampled_spaces, args.sampled_spaces.with_suffix('.sampling.json')],
              source('0c_sample_trial_spaces.py', 'trial_space_format.py')),
        Stage('generate-synthetic', [py, str(ROOT / 'generate_synthetic_data.py'), '--input-csv', str(args.sampled_spaces),
              '--output-dir', str(g), '--seed', str(args.data_seed), '--max-model-len', str(args.generation_context),
              '--pending-requests', str(args.generation_pending_requests), *model, *cache, *remote],
              [args.sampled_spaces], generated,
              source('generate_synthetic_data.py', '1a_make_synthetic_enrollee_prompts.py',
                     '1b_make_synthetic_negative_enrollee_prompts.py', '2_make_synthetic_notes_sharded.py',
                     'remote_vllm_pool.py', 'vllm_reasoning_utils.py')),
        Stage('prepare-notes', [py, str(ROOT / 'prepare_synthetic_training.py'), 'notes',
              '--input', str(args.raw_synthetic_notes), '--sampled-spaces', str(args.sampled_spaces),
              '--output', str(args.synthetic_notes), '--patients', str(patient_map),
              '--date-anchor', args.synthetic_date_anchor],
              [args.raw_synthetic_notes, args.sampled_spaces], [args.synthetic_notes, patient_map], source('prepare_synthetic_training.py')),
        Stage('summarize-patients', [py, str(ROOT / '6_summarize_patients.py'),
              '--input_parquet', str(args.synthetic_notes), '--output_parquet', str(d / 'patient_serial_summaries.parquet'),
              '--patient_summaries_parquet', str(raw_summaries), '--shard_dir', str(work / 'summaries'),
              '--max_model_len', str(args.generation_context), '--chunk_size', str(args.summary_chunk_size),
              '--chunk_overlap', str(args.summary_chunk_overlap), '--max_tokens', str(args.summary_max_tokens),
              *model, *(['--download_dir', str(args.download_dir)] if args.download_dir else []), *remote],
              [args.synthetic_notes], [raw_summaries, d / 'patient_serial_summaries.parquet'],
              source('6_summarize_patients.py', 'remote_vllm_pool.py', 'vllm_reasoning_utils.py')),
        Stage('prepare-training', [py, str(ROOT / 'prepare_synthetic_training.py'), 'summaries',
              '--input', str(raw_summaries), '--patients', str(patient_map), '--output', str(args.patient_summaries)],
              [raw_summaries, patient_map], [args.patient_summaries], source('prepare_synthetic_training.py')),
        Stage('train', training_command(args), list(training.input_paths(args).values()),
              training.stages(args)[-1].outputs, source('train_from_summaries.py')),
    ]
    return candidates[STAGES.index(args.start_at):STAGES.index(args.stop_after) + 1]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, default=str, indent=2) + '\n')
    temporary.replace(path)


def file_record(path, cache):
    path = Path(path).resolve()
    stat = path.stat()
    key = str(path)
    prior = cache.get(key, {})
    if prior.get('bytes') == stat.st_size and prior.get('mtime_ns') == stat.st_mtime_ns:
        return prior['identity']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    identity = dict(path=key, bytes=stat.st_size, sha256=digest.hexdigest())
    cache[key] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, identity=identity)
    return identity


def generation_active(directory):
    path = Path(directory) / 'generation.lock'
    if not path.exists():
        return False
    with path.open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def wait_generation(args):
    """Read only: the existing supervisor remains responsible for restart/recovery."""
    manifest = json.loads((args.generation_dir / 'manifest.json').read_text())
    if manifest['input_sha256'] != file_record(args.sampled_spaces, {})['sha256']:
        raise ValueError('Running generation uses different sampled trial spaces')
    deadline = time.monotonic() + args.generation_wait_timeout if args.generation_wait_timeout else None
    while True:
        status = json.loads((args.generation_dir / 'status.json').read_text())
        if status['stage'] == 'completed':
            return
        if status['stage'] == 'failed':
            raise RuntimeError('Existing synthetic generation failed; see its status/log before continuing')
        if deadline and time.monotonic() >= deadline:
            raise TimeoutError('Timed out waiting for existing synthetic generation; it was left running')
        print(f"[wait] synthetic generation: {status['stage']}, saved={status.get('stage_completed', 0)}", flush=True)
        time.sleep(30)


def execute(args, plan):
    # Upstream outputs must remain in the explicitly non-PHI workspace.
    for path in (args.data_dir, args.generation_dir, args.pipeline_dir):
        if not path.is_relative_to(DATA_ROOT.resolve()):
            raise ValueError('Data-production directories must be inside data/no_phi')
    if args.synthetic_notes == args.raw_synthetic_notes:
        raise ValueError('Prepared notes must use a different path from raw generated notes')
    if any(stage.name == 'generate-synthetic' for stage in plan) and args.raw_synthetic_notes != args.generation_dir / 'all_synthetic_notes.parquet':
        raise ValueError('Generation writes raw notes inside --generation-dir; use --raw-synthetic-notes only when starting later')
    if any(stage.name == 'parse-trials' for stage in plan) and args.ctgov_json is None:
        raise ValueError('--ctgov-json is required when starting at parse-trials')
    llm_stages = {stage.name for stage in plan} & {'extract-spaces', 'generate-synthetic', 'summarize-patients'}
    if llm_stages - ({'generate-synthetic'} if args.wait_for_generation else set()) and not (args.server_urls_file or args.teacher_servers_file):
        raise ValueError('Upstream LLM stages require --server-urls-file for an existing serving pool')
    args.pipeline_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.pipeline_dir / 'file_identities.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    with (args.pipeline_dir / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, PYTHONPATH=str(args.inference_repo / 'src') + os.pathsep + str(ROOT),
                   TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        for stage in plan:
            status_path = args.pipeline_dir / 'status.json'
            try:
                identity = dict(command=stage.command, inputs=[file_record(p, cache) for p in stage.inputs],
                                sources=[file_record(p, cache) for p in [Path(__file__), *stage.sources]])
                fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                request = args.pipeline_dir / 'requests' / f'{stage.name}.json'
                receipt = args.pipeline_dir / 'completed' / f'{stage.name}.json'
                if request.exists() and json.loads(request.read_text())['fingerprint'] != fingerprint:
                    raise ValueError(f'Inputs, code, or options changed for {stage.name}; use a fresh pipeline directory')
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if saved['fingerprint'] != fingerprint or saved['outputs'] != [file_record(p, cache) for p in stage.outputs]:
                        raise ValueError(f'Changed or missing completed output for {stage.name}')
                    print(f'[skip] {stage.name}', flush=True)
                    continue
                atomic_json(request, dict(fingerprint=fingerprint, **identity))
                atomic_json(cache_path, cache)
                atomic_json(status_path, dict(stage=stage.name, status='running'))
                if stage.name == 'sample-trials' and generation_active(args.generation_dir):
                    raise RuntimeError('Cannot resample an active generation run; start at generate-synthetic with --wait-for-generation')
                print(f'[start] {stage.name}', flush=True)
                if stage.name == 'generate-synthetic' and (args.wait_for_generation or generation_active(args.generation_dir)):
                    wait_generation(args)
                else:
                    log = args.pipeline_dir / 'logs' / f'{stage.name}.log'
                    log.parent.mkdir(exist_ok=True)
                    for output in stage.outputs:
                        output.parent.mkdir(parents=True, exist_ok=True)
                    with log.open('a') as stream:
                        subprocess.run(stage.command, cwd=ROOT, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
                outputs = [file_record(p, cache) for p in stage.outputs]
                atomic_json(receipt, dict(fingerprint=fingerprint, outputs=outputs))
                atomic_json(cache_path, cache)
                print(f'[done] {stage.name}', flush=True)
            except BaseException as exc:
                atomic_json(status_path, dict(stage=stage.name, status='failed', error=str(exc)))
                raise
        atomic_json(args.pipeline_dir / 'status.json', dict(status='completed', stop_after=args.stop_after))


def main(argv=None):
    raw = sys.argv[1:] if argv is None else argv
    args = resolve_paths(parser().parse_args(raw))
    plan = build_plan(args)
    if args.dry_run:
        display = []
        for stage in plan:
            display.extend([asdict(s) for s in training.stages(args)] if stage.name == 'train' else [asdict(stage)])
        print(json.dumps(display, default=str, indent=2))
    elif args.start_at == args.stop_after == 'train':
        # Preserve the established trainer's own locking, identities and resume.
        subprocess.run(training_command(args), cwd=ROOT, check=True)
    else:
        execute(args, plan)


if __name__ == '__main__':
    main()
