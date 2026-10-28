import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import prepare_synthetic_training as prepare
import run_pipeline as pipeline
import train_from_summaries as training
from tests.test_synthetic_generation import SPACE


def arguments(tmp_path, *extra):
    root = tmp_path / 'data/no_phi'
    return pipeline.resolve_paths(pipeline.parser().parse_args([
        '--data-dir', str(root / 'prepared'), '--run-dir', str(root / 'training'),
        '--generation-dir', str(root / 'generation'), *extra]))


def test_full_plan_connects_all_stages_and_live_pool(tmp_path):
    args = arguments(tmp_path, '--start-at', 'parse-trials', '--ctgov-json', str(tmp_path / 'trials.json'),
                     '--server-urls-file', str(tmp_path / 'servers.json'),
                     '--concurrency-control-file', str(tmp_path / 'limits.json'),
                     '--generation-model', 'gemma4-31b', '--generation-tokenizer', 'vendor/tokenizer')
    plan = pipeline.build_plan(args)
    assert [s.name for s in plan] == list(pipeline.STAGES)
    assert plan[1].command[plan[1].command.index('--start-year') + 1] == '2020'
    assert plan[2].command[plan[2].command.index('--n-trials') + 1] == '5000'
    assert plan[1].command[1].endswith('0b_extract_trial_cohorts.py')
    assert plan[1].outputs[0] == plan[2].inputs[0]
    assert plan[2].outputs[0] == plan[3].inputs[0]
    assert '--screening-input' not in plan[1].command
    from generate_synthetic_data import load_stage
    extraction = load_stage('0b_extract_trial_cohorts.py').parser().parse_args(plan[1].command[2:])
    assert extraction.max_attempts == 3 and extraction.max_tokens == 50000
    assert extraction.inference_repo == args.inference_repo
    assert plan[3].outputs[0] == plan[4].inputs[0]
    assert plan[4].outputs[0] == plan[5].inputs[0]
    assert plan[5].outputs[0] == plan[6].inputs[0]
    assert plan[6].outputs[0] == training.input_paths(args)['patient_summaries']
    assert args.sampled_spaces in plan[-1].inputs
    train_command = plan[-1].command
    assert train_command[train_command.index('--trial-spaces') + 1] == str(args.sampled_spaces)
    for stage in (plan[1], plan[3], plan[5]):
        assert stage.command[stage.command.index('--model') + 1] == 'gemma4-31b'
        assert stage.command[stage.command.index('--tokenizer-model') + 1] == 'vendor/tokenizer'
        assert stage.command[stage.command.index('--server_urls_file') + 1] == str(args.server_urls_file)
        if stage.name != 'extract-cohorts':
            assert '--concurrency-control-file' in stage.command
    # Extraction uses inference-owned adaptive limits and live registry caps.
    assert '--max_concurrent_per_server' in plan[1].command
    downstream = training.stages(args)
    for stage in downstream:
        if stage.teacher:
            assert str(args.server_urls_file) in stage.command
    summary_distillation = next(s for s in downstream if s.name == 'prepare-oncoreasoning')
    assert summary_distillation.command[summary_distillation.command.index('--notes') + 1] == str(args.synthetic_notes)


def test_pipeline_passes_note_preparation_workers(tmp_path):
    args = arguments(tmp_path, '--start-at', 'generate-synthetic', '--stop-after', 'generate-synthetic',
                     '--generation-note-preparation-workers', '3')
    command = pipeline.build_plan(args)[0].command
    assert command[command.index('--note-preparation-workers') + 1] == '3'


@pytest.mark.parametrize('wrapper', ['train_all.sh', 'train_all_gcp.sh'])
def test_shell_training_defaults_use_fast_shared_colbert(tmp_path, wrapper):
    result = subprocess.run(['bash', str(pipeline.ROOT / wrapper), '--dry-run', '--gpus', '0,1,2,3'],
        env={**os.environ, 'PYTHON': sys.executable}, text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    mines = [row['command'] for row in plan if row['name'].startswith('mine-')]
    assert len(mines) == 3
    for command in mines:
        assert '--nproc_per_node=4' in command
        assert command[command.index('--pair-batch-size') + 1] == '512'
        assert command[command.index('--score-query-batches') + 1] == '4'
        assert command[command.index('--device-cache-gb') + 1] == '8'
        assert command[command.index('--top-k-cohort-to-patient') + 1] == '40'
    assert mines[0][mines[0].index('--model') + 1] == 'lightonai/GTE-ModernColBERT-v1'
    fits = [row['command'] for row in plan if row['name'].startswith('train-trialspace-')]
    assert len(fits) == 2
    for command in fits:
        assert command[command.index('--loss') + 1] == 'legacy_ranking'
        assert command[command.index('--learning-rate') + 1] == '2e-05'
        assert command[command.index('--lr-scheduler') + 1] == 'linear'
        assert command[command.index('--warmup-ratio') + 1] == '0.01'
        assert command[command.index('--epochs') + 1] == '3'


def test_training_only_retains_original_default_identity(tmp_path):
    args = arguments(tmp_path)
    command = pipeline.training_command(args)
    parsed = pipeline.training.parser().parse_args(command[2:])
    assert parsed.synthetic_training_splits == ['train', 'val', 'test']
    assert '--trial-spaces' not in command
    assert '--patient-summaries' not in command
    assert '--synthetic-notes' not in command
    assert '--teacher-servers-file' not in command
    assert len(pipeline.build_plan(args)) == 1


def test_full_dry_run_shows_the_sampled_cohort_input(tmp_path, capsys):
    pipeline.main(['--start-at', 'sample-trials', '--data-dir', str(tmp_path), '--dry-run'])
    stages = {stage['name']: stage for stage in json.loads(capsys.readouterr().out)}
    sample = stages['sample-trials']['outputs'][0]
    prepare_command = stages['prepare-trialspace']['command']
    assert prepare_command[prepare_command.index('--cohorts') + 1] == sample


@pytest.mark.parametrize('wrapper', ['train_all.sh', 'train_all_gcp.sh', 'temp_train_gcp.sh'])
def test_shell_dry_run_includes_generation_and_stops_at_requested_stage(tmp_path, wrapper):
    output = subprocess.check_output([
        'bash', str(pipeline.ROOT / wrapper), '--start-at', 'sample-trials',
        '--stop-after', 'prepare-training', '--data-dir', str(tmp_path / 'not-created'), '--dry-run'],
        text=True, env={**os.environ, 'PYTHON': sys.executable})
    plan = json.loads(output)
    assert [s['name'] for s in plan] == list(pipeline.STAGES[2:7])
    assert not (tmp_path / 'not-created').exists()


def synthetic_files(tmp_path):
    spaces = tmp_path / 'sample.csv'
    pd.DataFrame([dict(space_index=i, nct_id=f'NCT{i}', this_space=SPACE,
                       trial_boilerplate_text='Infection', split=split)
                  for i, split in enumerate(['train', 'val'])]).to_csv(spaces, index=False)
    rows = []
    for patient in range(20):
        for event, age in [(2, 50), (8, 52)]:
            rows.append(dict(row_id=len(rows), pseudo_mrn=patient + 1, patient_id=f'patient-{patient}',
                space_index=patient // 10, nct_id=f'NCT{patient // 10}',
                patient_class='positive' if patient % 10 < 5 else 'negative', event_number=event,
                event_text=f'At age {age}, lesion measured < 1 mm.', event_type='clinical_note',
                synthetic_note='A fabricated clinical note.', reasoning='Saved reasoning'))
    source = tmp_path / 'raw.parquet'
    pd.DataFrame(rows[::-1]).to_parquet(source, index=False)
    return source, spaces


def test_preparation_preserves_ids_content_intervals_and_trial_cosplits(tmp_path):
    source, spaces = synthetic_files(tmp_path)
    original = source.read_bytes()
    notes, patients = tmp_path / 'prepared.parquet', tmp_path / 'patients.parquet'
    report = prepare.prepare_notes(source, spaces, notes, patients)
    assert source.read_bytes() == original
    frame, meta = pd.read_parquet(notes), pd.read_parquet(patients)
    assert report['notes'] == 40 and report['patients'] == 20
    assert frame.reasoning.eq('Saved reasoning').all()
    assert frame.event_text.str.contains('< 1 mm', regex=False).all()
    for _, group in frame.sort_values('event_number').groupby('patient_id'):
        assert (pd.to_datetime(group.date).iloc[1] - pd.to_datetime(group.date).iloc[0]).days == 730
        assert group.date.iloc[-1] == '2025-01-01'
        assert group.date_provenance.eq('synthetic_age_and_event_order').all()
    assert meta.groupby('nct_id').split.first().to_dict() == {'NCT0': 'train', 'NCT1': 'validation'}
    assert meta.groupby(['space_index', 'patient_class']).size().eq(5).all()
    summaries = tmp_path / 'summaries.parquet'
    pd.DataFrame([dict(pseudo_mrn=patient, patient_summary='Synthetic summary',
                       patient_boilerplate_text='No active infection') for patient in meta.pseudo_mrn]
                ).to_parquet(summaries, index=False)
    result = tmp_path / 'patient_summaries_with_spaces.parquet'
    prepare.prepare_summaries(summaries, patients, result)
    merged = pd.read_parquet(result)
    assert 'patient_boilerplate_text' not in merged
    assert merged.patient_summary.eq('Synthetic summary\n\nOther relevant history:\nNo active infection').all()
    assert merged.this_space.eq(SPACE).all()
    assert set(merged.pseudo_mrn) == set(frame.pseudo_mrn.astype(str))
    from make_top_matches import load_retrieval_tables
    mined_patients, mined_spaces = load_retrieval_tables(SimpleNamespace(
        patients_parquet=str(result), trials_file=str(spaces), splits=['train', 'val', 'test']))
    assert len(mined_patients) == 20 and len(mined_spaces) == 2
    assert set(mined_patients.split) == {'train'}
    assert set(mined_patients.source_split) == {'train', 'val'}


def test_dates_preserve_supplied_timestamps_and_label_ordinal_fallback():
    dates = prepare.ordered_dates([(1, 2, 50, '2021-01-02'), (2, 8, 52, '2023-03-04')], '2025-01-01')
    assert [d[1][:10] for d in dates] == ['2021-01-02', '2023-03-04']
    assert all(d[2] == 'provided' for d in dates)
    fallback = prepare.ordered_dates([(1, 2, None, None), (2, 8, None, None)], '2025-01-01')
    assert fallback == [(1, '2024-12-31', 'synthetic_event_order_only'), (2, '2025-01-01', 'synthetic_event_order_only')]
    with pytest.raises(ValueError, match='Partially dated'):
        prepare.ordered_dates([(1, 2, 50, None), (2, 8, 52, '2023-03-04')], '2025-01-01')


def test_preparation_preserves_week_intervals(tmp_path):
    source, spaces = synthetic_files(tmp_path)
    frame = pd.read_parquet(source)
    frame['event_text'] = frame.event_number.map({
        2: 'At age 50 years and 21 weeks, cancer was diagnosed.',
        8: 'At age 50 years and 24 weeks, treatment began.',
    })
    frame.to_parquet(source, index=False)
    notes, patients = tmp_path / 'prepared.parquet', tmp_path / 'patients.parquet'
    prepare.prepare_notes(source, spaces, notes, patients)
    for _, group in pd.read_parquet(notes).sort_values('event_number').groupby('patient_id'):
        assert (pd.to_datetime(group.date).iloc[1] - pd.to_datetime(group.date).iloc[0]).days == 21
        assert group.date_provenance.eq('synthetic_age_and_event_order').all()


@pytest.mark.parametrize('text,expected', [
    ('At age 50 years and 21 weeks, diagnosed.', 50 + 147 / 365.2425),
    ('At age 0 years and 26 weeks, diagnosed.', 182 / 365.2425),
    ('At age 61 years and 0 weeks, diagnosed.', 61),
    ('At age 18 months, diagnosed.', 1.5),
    ('At age 50.5 years, diagnosed.', 50.5),
    ('At age 50, diagnosed.', 50),
])
def test_age_parsing_supports_weeks_and_legacy_formats(text, expected):
    assert prepare.age_years(text) == pytest.approx(expected)


def test_summarization_loads_only_note_text_ids_and_dates(tmp_path, monkeypatch):
    from generate_synthetic_data import load_stage
    summary = load_stage('6_summarize_patients.py')
    source = tmp_path / 'notes.parquet'
    pd.DataFrame([dict(pseudo_mrn=1, date='2025-01-01', synthetic_note='A fabricated note.',
                       masked_text='Full event history', reasoning='Full reasoning trace')]).to_parquet(source)
    read = summary.pd.read_parquet
    calls = []
    def read_columns(path, **kwargs):
        calls.append(kwargs.get('columns'))
        return read(path, **kwargs)
    monkeypatch.setattr(summary.pd, 'read_parquet', read_columns)
    frame = summary.load_summary_notes(source, 'pseudo_mrn', 'date', 'synthetic_note')
    assert calls == [['pseudo_mrn', 'date', 'synthetic_note']]
    assert frame.columns.tolist() == calls[0]
    assert frame.synthetic_note.tolist() == ['A fabricated note.']


def test_preparation_rejects_split_conflicts_and_missing_summaries(tmp_path):
    source, spaces = synthetic_files(tmp_path)
    notes, patients = tmp_path / 'prepared.parquet', tmp_path / 'patients.parquet'
    prepare.prepare_notes(source, spaces, notes, patients)
    old = notes.read_bytes()
    frame = pd.read_parquet(source)
    frame['split'] = 'train'
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError, match='splits disagree'):
        prepare.prepare_notes(source, spaces, notes, patients)
    assert notes.read_bytes() == old
    summary = tmp_path / 'missing.parquet'
    pd.DataFrame([dict(pseudo_mrn='1', patient_summary='Synthetic summary')]).to_parquet(summary, index=False)
    with pytest.raises(ValueError, match='coverage'):
        prepare.prepare_summaries(summary, patients, tmp_path / 'out.parquet')


def test_stage_resume_checks_inputs_and_outputs_without_repeating_work(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'DATA_ROOT', tmp_path / 'data/no_phi')
    source = tmp_path / 'ctgov.json'
    source.write_text('[]')
    args = arguments(tmp_path, '--start-at', 'parse-trials', '--stop-after', 'parse-trials', '--ctgov-json', str(source))
    plan = pipeline.build_plan(args)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        args.parsed_trials.write_text('nct_id\nNCT1\n')
    monkeypatch.setattr(pipeline.subprocess, 'run', run)
    pipeline.execute(args, plan)
    pipeline.execute(args, plan)
    assert len(calls) == 1
    args.parsed_trials.write_text('changed output')
    with pytest.raises(ValueError, match='Changed or missing completed output'):
        pipeline.execute(args, plan)
    assert len(calls) == 1
    source.write_text('[{}]')
    with pytest.raises(ValueError, match='changed'):
        pipeline.execute(args, plan)
    assert len(calls) == 1


def test_attach_wait_is_read_only_and_never_launches_a_second_generator(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'DATA_ROOT', tmp_path / 'data/no_phi')
    args = arguments(tmp_path, '--start-at', 'generate-synthetic', '--stop-after', 'generate-synthetic', '--wait-for-generation')
    args.generation_dir.mkdir(parents=True)
    args.sampled_spaces.write_text('nct_id,this_space\nNCT1,space\n')
    manifest = {'input_stamp': pipeline.file_stamp(args.sampled_spaces)}
    (args.generation_dir / 'manifest.json').write_text(json.dumps(manifest))
    status = args.generation_dir / 'status.json'
    status.write_text(json.dumps({'stage': 'events', 'stage_completed': 10}))
    sleeps = []
    def advance(seconds):
        sleeps.append(seconds)
        status.write_text(json.dumps({'stage': 'completed'}))
        for output in pipeline.build_plan(args)[0].outputs:
            if not output.exists():
                output.write_text('completed output')
    monkeypatch.setattr(pipeline.time, 'sleep', advance)
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('Must not launch a duplicate generator'))
    with (args.generation_dir / 'generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert pipeline.generation_active(args.generation_dir)
        pipeline.execute(args, pipeline.build_plan(args))
        assert pipeline.generation_active(args.generation_dir)
    assert sleeps == [30]
    assert json.loads((args.generation_dir / 'manifest.json').read_text()) == manifest


def test_external_teacher_registry_is_not_rewritten(tmp_path, monkeypatch):
    registry = tmp_path / 'servers.json'
    entries = [dict(url=f'http://server-{i}/v1', model_root=root) for i, root in enumerate(['vendor/fp8', 'vendor/nvfp4'])]
    registry.write_text(json.dumps(dict(ready=True, servers=entries)))
    original = registry.read_bytes()
    args = training.parser().parse_args(['--teacher-servers-file', str(registry), '--teacher-served-model', 'gemma4-31b'])
    class Response:
        def __init__(self, url): self.url = url
        def __enter__(self): return self
        def __exit__(self, *exc): pass
        def read(self):
            root = next(e['model_root'] for e in entries if self.url.startswith(e['url']))
            return json.dumps({'data': [dict(id='gemma4-31b', root=root, max_model_len=262144)]}).encode()
    monkeypatch.setattr(training.urllib.request, 'urlopen', lambda url, **kw: Response(url))
    with training.managed_teacher_pool(args):
        assert registry.read_bytes() == original
    assert registry.read_bytes() == original
