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


def pipeline_stage(filename):
    from generate_synthetic_data import load_stage
    return load_stage(filename)


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
    stage = {s.name: s for s in plan}
    screen = stage['screen-trials'].command
    assert screen[1].endswith('0b_screen_trials.py') and screen[screen.index('--start-year') + 1] == '2020'
    assert [screen[screen.index(flag) + 1] for flag in ('--dfci-first-year', '--dfci-last-year')] == ['2025', '2026']
    assert screen[screen.index('--dfci-trials-csv') + 1] == str(args.dfci_trials_csv)
    assert stage['sample-trials'].command[stage['sample-trials'].command.index('--n-trials') + 1] == '5000'
    # Registry rule, DFCI holdout and screen -> seeded draw -> synthetic protocols -> both versions' cohorts.
    assert stage['screen-trials'].outputs[0] == args.screened_trials == stage['sample-trials'].inputs[0]
    sample = stage['sample-trials'].command
    assert sample[sample.index('--screened-trials') + 1] == str(args.screened_trials)
    assert [sample[sample.index(flag) + 1] for flag in ('--n-trials', '--seed')] == ['5000', '42']
    assert stage['sample-trials'].outputs[0] == args.registry_sample == stage['synthetic-protocols'].inputs[0]
    seeds = stage['synthetic-protocols'].outputs[0]
    assert seeds in stage['protocol-cohorts'].inputs and seeds == stage['registry-cohorts'].inputs[0]
    assert {stage['protocol-cohorts'].outputs[i] for i in (0, 1, 4)} <= set(stage['generation-cohorts'].inputs)
    assert args.registry_sample in stage['generation-cohorts'].inputs
    # Patients come from the protocol cohorts, boilerplate included; retrieval and training use both
    # versions with the boilerplate removed.
    assert stage['generation-cohorts'].outputs[0] == args.sampled_spaces == stage['generate-synthetic'].inputs[0]
    clean = stage['generation-cohorts'].outputs[2]
    assert clean == args.sampled_spaces.with_name('sample_trial_cohort_lineitems_deboilerplated.csv')
    generation = stage['generation-cohorts'].command
    assert generation[generation.index('--retrieval-output') + 1] == str(clean) and '--cohorts-file' not in generation
    assert {clean, *(stage['registry-cohorts'].outputs[i] for i in (1, 4))} <= set(stage['retrieval-cohorts'].inputs)
    assert args.sampled_spaces not in stage['retrieval-cohorts'].inputs
    assert stage['retrieval-cohorts'].outputs[0] == args.retrieval_cohorts
    from generate_synthetic_data import load_stage
    extract = load_stage('0b_extract_trial_cohorts.py')
    screening = pipeline_stage('0b_screen_trials.py').parser().parse_args(screen[2:])
    assert screening.max_attempts == 3 and screening.max_tokens == 50000
    assert screening.inference_repo == args.inference_repo and screening.output == args.screened_trials
    registry = extract.parser().parse_args(stage['registry-cohorts'].command[2:])
    assert registry.trials_jsonl == seeds and registry.allow_failures
    # Cohort rows are kept both before and after merging.
    assert extract.output_paths(registry)[3] == registry.output_unmerged_cohorts == stage['registry-cohorts'].outputs[3]
    assert stage['protocol-cohorts'].outputs[3] == args.protocol_dir / 'cohort_extraction/cohorts_unmerged.parquet'
    # Registry rows are de-boilerplated too, and retrieval reads them beside the de-boilerplated protocol cohorts.
    assert extract.output_paths(registry)[4] == registry.output_deboilerplated_cohorts == stage['registry-cohorts'].outputs[4]
    retrieval = stage['retrieval-cohorts'].command
    assert retrieval[retrieval.index('--registry-cohorts') + 1] == str(stage['registry-cohorts'].outputs[4])
    assert retrieval[retrieval.index('--protocol-cohorts') + 1] == str(clean)
    moved = arguments(tmp_path, '--start-at', 'protocol-cohorts', '--protocol-dir', str(tmp_path / 'data/no_phi/batch'),
                      '--protocol-cohort-dir', str(tmp_path / 'data/no_phi/rerun/protocol_cohorts'))
    rerun = {s.name: s for s in pipeline.build_plan(moved)}
    assert rerun['protocol-cohorts'].outputs[0] == tmp_path / 'data/no_phi/rerun/protocol_cohorts/cohorts.parquet'
    # Patients are seeded from the merged cohorts with all their criteria; retrieval reads them without boilerplate.
    assert rerun['protocol-cohorts'].outputs[4] == tmp_path / 'data/no_phi/rerun/protocol_cohorts/cohorts_deboilerplated.parquet'
    assert rerun['generation-cohorts'].inputs[:2] == [rerun['protocol-cohorts'].outputs[i] for i in (0, 4)]
    assert stage['generate-synthetic'].outputs[0] == stage['prepare-notes'].inputs[0]
    assert stage['prepare-notes'].outputs[0] == stage['summarize-patients'].inputs[0]
    assert stage['summarize-patients'].outputs[0] == stage['prepare-training'].inputs[0]
    assert stage['prepare-training'].outputs[0] == training.input_paths(args)['patient_summaries']
    assert args.retrieval_cohorts in plan[-1].inputs
    train_command = plan[-1].command
    assert train_command[train_command.index('--trial-spaces') + 1] == str(args.retrieval_cohorts)
    for name in ('screen-trials', 'registry-cohorts', 'generate-synthetic', 'summarize-patients'):
        command = stage[name].command
        assert command[command.index('--model') + 1] == 'gemma4-31b'
        assert command[command.index('--tokenizer-model') + 1] == 'vendor/tokenizer'
        assert command[command.index('--server_urls_file') + 1] == str(args.server_urls_file)
        if name not in ('screen-trials', 'registry-cohorts'):
            assert '--concurrency-control-file' in command
    for name in ('synthetic-protocols', 'protocol-cohorts'):
        command = stage[name].command
        assert command[command.index('--model-name') + 1] == 'vendor/tokenizer'
        assert command[command.index('--server-urls-file') + 1] == str(args.server_urls_file)
        assert '--allow-failures' in command
    # Extraction uses inference-owned adaptive limits and live registry caps.
    assert '--max_concurrent_per_server' in stage['screen-trials'].command
    downstream = training.stages(args)
    for item in downstream:
        if item.teacher:
            assert str(args.server_urls_file) in item.command
    # Labeling shares the pool but never the generation limits.
    labels = [s.command for s in downstream if s.name.startswith('label-trialspace-')]
    assert labels and all(str(args.concurrency_control_file) not in command for command in labels)
    assert all(command[command.index('--max_concurrent_per_server') + 1] == '768' for command in labels)
    own = arguments(tmp_path, '--start-at', 'parse-trials', '--server-urls-file', str(tmp_path / 'servers.json'),
                    '--concurrency-control-file', str(tmp_path / 'limits.json'),
                    '--teacher-concurrency-file', str(tmp_path / 'label_limits.json'))
    for item in training.stages(own):
        if item.name.startswith('label-trialspace-'):
            assert item.command[item.command.index('--concurrency-control-file') + 1] == str(tmp_path / 'label_limits.json')
    summary_distillation = next(s for s in downstream if s.name == 'prepare-oncoreasoning')
    assert summary_distillation.command[summary_distillation.command.index('--notes') + 1] == str(args.synthetic_notes)


def test_pipeline_passes_note_preparation_workers(tmp_path):
    args = arguments(tmp_path, '--start-at', 'generate-synthetic', '--stop-after', 'generate-synthetic',
                     '--generation-note-preparation-workers', '3')
    command = pipeline.build_plan(args)[0].command
    assert command[command.index('--note-preparation-workers') + 1] == '3'


def dry_run_plan(wrapper, *options):
    result = subprocess.run(['bash', str(pipeline.ROOT / wrapper), '--dry-run', '--gpus', '0,1,2,3', *options],
        env={**os.environ, 'PYTHON': sys.executable}, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def commands(plan, prefix):
    return [row['command'] for row in plan if row['name'].startswith(prefix)]


@pytest.mark.parametrize('wrapper', ['train_all.sh', 'train_all_gcp.sh'])
def test_shell_training_defaults_use_embeddinggemma_and_three_label_rounds(tmp_path, wrapper):
    plan = dry_run_plan(wrapper)
    names = [row['name'] for row in plan]
    assert names[:names.index('catalog')] == ['prepare-trialspace',
        'mine-1', 'label-trialspace-1', 'project-trialspace-1', 'train-trialspace-1',
        'mine-2', 'label-trialspace-2', 'project-trialspace-2', 'train-trialspace-2',
        'mine-3', 'label-trialspace-3', 'label-boilerplate', 'train-trial-checker', 'train-boilerplate-checker']
    mines = commands(plan, 'mine-')
    for command in mines:
        assert command[1:3] == ['trialspace_comparison.py', 'dense-mine']
        assert [command[command.index(flag) + 1] for flag in
                ('--max-tokens', '--patient-k', '--cohort-k', '--lexical-k', '--random-k', '--patient-trial-sample')] == \
            ['8192', '20', '40', '5', '5', '500']
    assert mines[0][mines[0].index('--model') + 1] == 'google/embeddinggemma-2'
    assert mines[0][mines[0].index('--revision') + 1] == '914f7f89142e33e77833254d9c9b90c3cef7303b'
    assert mines[1][mines[1].index('--model') + 1].endswith('trialspace_round1')
    # Each round draws a fresh trial sample per patient; round 1 keeps the original draws.
    assert '--trial-sample-round' not in mines[0]
    assert [command[command.index('--trial-sample-round') + 1] for command in mines[1:]] == ['2', '3']
    labels = commands(plan, 'label-trialspace-')
    assert '--reuse-labels' not in labels[0]
    for round_number, command in enumerate(labels[1:], 2):
        assert command[command.index('--reuse-labels') + 1].endswith(f'trialspace_round{round_number - 1}/teacher')
    for command in labels:
        assert command[command.index('--max_attempts') + 1] == '16'
        assert '--scheduler-adaptive-concurrency' in command
        assert command[command.index('--task') + 1] == 'match'
    # BoilerplateChecker labels: the separate prompt on every unique boilerplate combination of rounds 1-3.
    boilerplate = commands(plan, 'label-boilerplate')[0]
    assert boilerplate[boilerplate.index('--task') + 1] == 'boilerplate'
    assert boilerplate[boilerplate.index('--candidates') + 1].endswith('trialspace_round3/mining/candidates.jsonl')
    combined = boilerplate[boilerplate.index('--combine-candidates') + 1:][:2]
    assert [path.split('/')[-3] for path in combined] == ['trialspace_round1', 'trialspace_round2']
    # One multi-head TrialChecker on every round's match labels; a separate BoilerplateChecker.
    trial, separate = commands(plan, 'train-trial-checker')[0], commands(plan, 'train-boilerplate-checker')[0]
    assert trial[trial.index('train_checkers.py') + 1] == 'trial'
    assert [path.split('/')[-2] for path in trial[trial.index('--labels') + 1:][:3]] == [
        f'trialspace_round{n}' for n in (1, 2, 3)]
    assert separate[separate.index('train_checkers.py') + 1] == 'boilerplate'
    assert separate[separate.index('--labels') + 1].endswith('boilerplate_teacher')
    for command in (trial, separate):
        assert [command[command.index(flag) + 1] for flag in ('--base-model', '--max-length')] == [
            'answerdotai/ModernBERT-large', '8192']
    fits = commands(plan, 'train-trialspace-')
    assert len(fits) == 2
    for command in fits:
        assert command[1:6] == ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', 'finetune_embedder.py']
        assert '--mask-known-positives' in command
        assert float(command[command.index('--epochs') + 1]) == 1
        assert [command[command.index(flag) + 1] for flag in ('--batch-size', '--loss-mini-batch-size', '--prompt-profile')] == \
            ['96', '32', 'trialspace']
    assert fits[0][fits[0].index('--base-model') + 1] == 'google/embeddinggemma-2'
    assert fits[1][fits[1].index('--base-model') + 1].endswith('trialspace_round1')
    catalog = commands(plan, 'catalog')[0]
    assert catalog[catalog.index('--candidate-files') + 1].endswith('trialspace_round3/mining/patient_to_cohort.parquet')


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


def test_full_dry_run_trains_on_both_trial_versions(tmp_path, capsys):
    pipeline.main(['--start-at', 'sample-trials', '--data-dir', str(tmp_path), '--dry-run'])
    stages = {stage['name']: stage for stage in json.loads(capsys.readouterr().out)}
    combined = stages['retrieval-cohorts']['outputs'][0]
    prepare_command = stages['prepare-trialspace']['command']
    assert prepare_command[prepare_command.index('--cohorts') + 1] == combined


@pytest.mark.parametrize('wrapper', ['train_all.sh', 'train_all_gcp.sh', 'temp_train_gcp.sh'])
def test_shell_dry_run_includes_generation_and_stops_at_requested_stage(tmp_path, wrapper):
    output = subprocess.check_output([
        'bash', str(pipeline.ROOT / wrapper), '--start-at', 'sample-trials',
        '--stop-after', 'prepare-training', '--data-dir', str(tmp_path / 'not-created'), '--dry-run'],
        text=True, env={**os.environ, 'PYTHON': sys.executable})
    plan = json.loads(output)
    assert [s['name'] for s in plan] == list(pipeline.STAGES[2:pipeline.STAGES.index('prepare-training') + 1])
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
    # The key section and the boilerplate section stay apart.
    assert merged.patient_summary.eq('Synthetic summary').all()
    assert merged.patient_boilerplate_text.eq('No active infection').all()
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
