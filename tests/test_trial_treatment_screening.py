import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.test_trial_selection import REGISTRY_METADATA, load_script
from trial_treatment_screening import load_screening_results, parse_screening_output, screening_record

spaces = load_script('screened_trial_spaces', '0b_create_trial_spaces.py')


class FakeRequestSession:
    """Adapt these transport fixtures to the persistent per-request interface."""

    def __init__(self, registry, work_fn, **kwargs):
        self.registry, self.work_fn = registry, work_fn

    async def __aenter__(self):
        async def entries():
            return [None]
        self.registry.active_entries = entries
        return self

    async def submit(self, request):
        import remote_vllm_pool as pool
        results = []
        await pool.run_pool(
            work_items=[(0, request)], work_fn=self.work_fn,
            registry=self.registry, shard_writer=lambda rows, _: results.extend(rows),
            starting_shard_idx=0,
        )
        return results[0][1]

    async def __aexit__(self, *exc):
        await self.registry.stop()


@pytest.mark.parametrize('final,finish', [
    ('{}', 'stop'), ('[]', 'stop'), ('not JSON', 'stop'),
    ('{"is_cancer_treatment_trial": "false", "rationale": "Supportive"}', 'stop'),
    ('{"is_cancer_treatment_trial": 0, "rationale": "Supportive"}', 'stop'),
    ('{"is_cancer_treatment_trial": false, "rationale": " "}', 'stop'),
    ('{"is_cancer_treatment_trial": true, "rationale": "Treatment"}', 'length'),
])
def test_invalid_response_is_never_an_exclusion(final, finish):
    with pytest.raises(ValueError):
        parse_screening_output(final, finish)


@pytest.fixture
def endpoint_run(tmp_path, monkeypatch, request):
    """Exercise both real stage dispatchers with a fake completion transport."""
    import transformers
    import remote_vllm_pool as pool
    import vllm_reasoning_utils

    source = tmp_path / 'trials.csv'
    pd.DataFrame([
        dict(nct_id='NCT00000001', study_type='INTERVENTIONAL', phases='["EARLY_PHASE1"]', trial_text='ANTI_CANCER', **REGISTRY_METADATA),
        dict(nct_id='NCT00000002', study_type='INTERVENTIONAL', phases='["PHASE2"]', trial_text='SUPPORTIVE', study_first_post_date='2021-05-01'),
        dict(nct_id='NCT00000003', study_type='INTERVENTIONAL', phases='["PHASE1", "PHASE2"]', trial_text='DIAGNOSTIC', study_first_post_date='2022-05-01'),
        dict(nct_id='NCT00000004', study_type='OBSERVATIONAL', phases='["PHASE2"]', trial_text='NEVER_SEND', study_first_post_date='2023-05-01'),
    ]).to_csv(source, index=False)
    model, tokenizer_model = getattr(request, 'param', (
        'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4', None))
    state = SimpleNamespace(calls=[], reject_all=False, fail=False, registries=[], tokenizers=[])

    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

        def apply_chat_template(self, conversation, **kwargs):
            assert kwargs['enable_thinking'] is True
            return json.dumps(conversation)

    def load_tokenizer(name, **kwargs):
        state.tokenizers.append(name)
        return Tokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', load_tokenizer)
    monkeypatch.setattr(vllm_reasoning_utils, 'parse_reasoning_output', lambda raw, *a, **k: ('Reasoning', raw))

    async def create(**kwargs):
        messages = json.loads(kwargs['prompt'])
        screening = 'is_cancer_treatment_trial' in messages[0]['content']
        text = messages[-1]['content']
        assert 'NEVER_SEND' not in text
        state.calls.append(('screen' if screening else 'extract', text))
        assert kwargs['model'] == model
        assert kwargs['temperature'] == 1.0 and kwargs['top_p'] == .95
        assert kwargs['extra_body']['top_k'] == (64 if tokenizer_model else -1)
        assert kwargs['extra_body']['min_p'] == 0.0
        assert kwargs['presence_penalty'] == 0.0
        assert kwargs['extra_body']['repetition_penalty'] == 1.0
        if screening:
            assert kwargs['max_tokens'] == 8192
            final = json.dumps({'is_cancer_treatment_trial': 'ANTI_CANCER' in text and not state.reject_all,
                                'rationale': 'Test treatment-intent decision.'})
            if state.fail and 'DIAGNOSTIC' in text:
                final = 'invalid response'
        else:
            assert kwargs['max_tokens'] == 50000
            assert len([c for c in state.calls if c[0] == 'screen']) >= 3
            assert 'ANTI_CANCER' in text
            if 'CURRENT FIELD:' in text:
                field = text.split('CURRENT FIELD: ')[1].splitlines()[0]
                value = {'age': '18+', 'sex': 'Both', 'cancer_type': 'lung cancer',
                         'histology': 'adenocarcinoma', 'disease_state': 'metastatic'}.get(field, 'NA')
                final = json.dumps({'branches': [{'value': value, 'scope': 'all eligible patients'}], 'prune_reason': ''})
            elif 'Extract general exclusion criteria' in text:
                final = '{"exclusions": ["Active infection."]}'
            else:
                final = ('1. Age range allowed: 18+. Sex allowed: Any. Cancer type allowed: lung cancer. '
                     'Histology allowed: NA. Cancer burden allowed: metastatic. Prior treatment required: NA. '
                     'Prior treatment excluded: NA. Biomarkers required: NA. Biomarkers excluded: NA.\n'
                     'Boilerplate exclusions:\nActive infection.')
        return SimpleNamespace(choices=[SimpleNamespace(text=final, finish_reason='stop')])

    async def stop_registry():
        pass

    def registry(args):
        state.registries.append((args.server_urls, args.server_urls_file, args.max_concurrent_per_server))
        return SimpleNamespace(stop=stop_registry)

    async def run_pool(*, work_items, work_fn, registry, shard_writer, starting_shard_idx, **kwargs):
        assert registry.stop is stop_registry
        client = SimpleNamespace(completions=SimpleNamespace(create=create))
        payload = []
        for work_id, request in work_items:
            try:
                result = await work_fn(client, request)
            except ValueError as exc:
                result = f'ERROR: {exc}'
            payload.append((work_id, result))
        shard_writer(payload, starting_shard_idx)

    monkeypatch.setattr(pool, 'build_registry_from_args', registry)
    monkeypatch.setattr(pool, 'run_pool', run_pool)
    monkeypatch.setattr(pool, 'RequestSession', FakeRequestSession)
    argv = ['0b', '--input', str(source), '--work-dir', str(tmp_path/'work'),
            '--extraction-method', 'one-shot',
            '--output-trials', str(tmp_path/'out.csv'), '--output-spaces', str(tmp_path/'spaces.csv'),
            '--model', model,
            '--server_urls', 'http://requested-server:8000/v1', '--max_concurrent_per_server', '256']
    if tokenizer_model:
        argv.extend(['--tokenizer-model', tokenizer_model])
    monkeypatch.setattr(sys, 'argv', argv)
    return state, tmp_path, argv


@pytest.mark.parametrize('endpoint_run', [
    ('opaque-served-alias', 'RedHatAI/Gemma-4-31B-IT-FP8-Dynamic'),
], indirect=True)
def test_gemma_alias_uses_family_sampling_in_screening_and_extraction(endpoint_run):
    state, path, argv = endpoint_run
    spaces.main()
    assert [call[0] for call in state.calls] == ['screen'] * 3 + ['extract']
    assert state.tokenizers == ['RedHatAI/Gemma-4-31B-IT-FP8-Dynamic'] * 2
    # The tokenizer is part of checkpoint identity even when sampling matches.
    argv[argv.index('--tokenizer-model') + 1] = 'google/gemma-4-31B-it'
    with pytest.raises(ValueError, match='configuration changed'):
        spaces.main()
    assert len(state.calls) == 4


def test_complete_screen_precedes_extraction_and_resume_reuses_negatives(endpoint_run):
    state, path, _ = endpoint_run
    spaces.main()
    assert [c[0] for c in state.calls] == ['screen'] * 3 + ['extract']
    assert state.registries == [('http://requested-server:8000/v1', None, 256)] * 2
    assert pd.read_csv(path/'out.csv').nct_id.tolist() == ['NCT00000001']
    assert pd.read_csv(path/'spaces.csv').nct_id.tolist() == ['NCT00000001']
    for name in ['out.csv', 'spaces.csv']:
        assert pd.read_csv(path/name).iloc[0][list(REGISTRY_METADATA)].to_dict() == REGISTRY_METADATA
    audit = pd.read_csv(path/'trial_treatment_screening.csv')
    assert audit.treatment_screen_model.tolist() == [state.tokenizers[0]] * 3
    assert audit.is_cancer_treatment_trial.tolist() == [True, False, False]
    assert audit.treatment_screen_rationale.notna().all()
    assert audit.treatment_screen_reasoning.tolist() == ['Reasoning'] * 3
    spaces.main()
    assert len(state.calls) == 4


@pytest.mark.parametrize('reject_all', [False, True])
def test_screening_only_saves_model_without_extraction_and_can_resume(endpoint_run, reject_all):
    state, path, argv = endpoint_run
    state.reject_all = reject_all
    argv.append('--screening-only')
    spaces.main()
    assert [c[0] for c in state.calls] == ['screen'] * 3
    audit = pd.read_csv(path/'trial_treatment_screening.csv')
    model = argv[argv.index('--model') + 1]
    assert audit.treatment_screen_model.tolist() == [model] * 3
    assert audit.is_cancer_treatment_trial.tolist() == [not reject_all, False, False]
    for shard in (path/'work'/'treatment_screening').glob('remote_output_shard_*.parquet'):
        assert pd.read_parquet(shard).treatment_screen_model.eq(model).all()
    assert not (path/'work'/'space_extraction').exists()
    assert not (path/'out.csv').exists()
    assert not (path/'spaces.csv').exists()
    spaces.main()
    assert len(state.calls) == 3
    argv.remove('--screening-only')
    spaces.main()
    assert [c[0] for c in state.calls] == ['screen'] * 3 + ([] if reject_all else ['extract'])
    assert (path/'out.csv').exists() and (path/'spaces.csv').exists()


def test_saved_screening_allows_another_extraction_model_without_rescreening(endpoint_run, monkeypatch):
    state, path, argv = endpoint_run
    argv.append('--screening-only')
    spaces.main()
    audit = path/'trial_treatment_screening.csv'
    saved = audit.read_bytes()
    argv.remove('--screening-only')
    argv.extend(['--screening-input', str(audit)])
    argv[argv.index('--work-dir') + 1] = str(path/'extraction_work')
    argv[argv.index('--model') + 1] = 'google/gemma-4-26B-A4B-it'

    def extract(args, trials, base_cols, reasoning_parser, *, screening=False):
        assert not screening
        assert args.model == 'google/gemma-4-26B-A4B-it'
        assert trials.nct_id.tolist() == ['NCT00000001']
        result = trials.copy()
        final = ('1. Age range allowed: 18+. Sex allowed: Both. Cancer type allowed: lung cancer. '
                 'Histology allowed: adenocarcinoma. Disease state allowed: metastatic. '
                 'Prior treatment required: NA. Prior treatment excluded: NA. '
                 'Biomarkers required: NA. Biomarkers excluded: NA.')
        for name in ['space_text', 'space_output_no_reasoning', 'space_reasoning_and_output']:
            result[name] = final
        result['trial_boilerplate_text'] = 'Active infection.'
        output = Path(args.work_dir)/'remote_output_shard_000000.parquet'
        result.to_parquet(output, index=False)
        return [str(output)]

    monkeypatch.setattr(spaces, 'remote_main', extract)
    spaces.main()
    assert [call[0] for call in state.calls] == ['screen'] * 3
    assert audit.read_bytes() == saved
    assert not (path/'extraction_work/treatment_screening').exists()
    assert pd.read_csv(path/'spaces.csv').nct_id.tolist() == ['NCT00000001']
    manifest = json.loads((path/'extraction_work/screening_input_manifest.json').read_text())
    assert manifest['screening_models'] == [state.tokenizers[0]]


@pytest.fixture
def saved_screening(tmp_path):
    trials = pd.DataFrame([
        {'nct_id': 'NCT00000001', 'trial_text': 'Treatment trial', 'split': 'train'},
        {'nct_id': 'NCT00000002', 'trial_text': 'Supportive study', 'split': 'val'},
    ])
    rows = []
    for index, row in enumerate(trials.to_dict('records')):
        rows.append({**row, **screening_record('', json.dumps({
            'is_cancer_treatment_trial': index == 0, 'rationale': 'Test decision',
        }), 'stop'), 'treatment_screen_model': 'screening-model'})
    path = tmp_path/'screening.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return trials, path, tmp_path


def test_saved_screening_reorders_by_id_and_guards_resume_provenance(saved_screening):
    trials, path, work = saved_screening
    audit = pd.read_csv(path).iloc[::-1]
    audit.to_csv(path, index=False)
    for _ in range(2):
        assert load_screening_results(trials, path, work).nct_id.tolist() == ['NCT00000001']
    audit['treatment_screen_model'] = 'different-screening-model'
    audit.to_csv(path, index=False)
    with pytest.raises(ValueError, match='results changed'):
        load_screening_results(trials, path, work)


@pytest.mark.parametrize('change', ['missing_row', 'duplicate', 'text', 'split', 'status',
                                   'response', 'decision', 'numeric_decision', 'missing_model', 'column'])
def test_saved_screening_rejects_incomplete_or_inconsistent_audits(saved_screening, change):
    trials, path, work = saved_screening
    audit = pd.read_csv(path)
    if change == 'missing_row':
        audit = audit.iloc[:1]
    elif change == 'duplicate':
        audit = pd.concat([audit, audit.iloc[:1]])
    elif change in {'text', 'split', 'status', 'response', 'missing_model'}:
        column, value = {
            'text': ('trial_text', 'Changed trial'), 'split': ('split', 'test'),
            'status': ('treatment_screen_status', 'error'),
            'response': ('treatment_screen_output', 'invalid JSON'),
            'missing_model': ('treatment_screen_model', ''),
        }[change]
        audit.loc[0, column] = value
    elif change == 'decision':
        audit.loc[0, 'is_cancer_treatment_trial'] = False
    elif change == 'numeric_decision':
        audit['is_cancer_treatment_trial'] = [1, 0]
    else:
        audit = audit.drop(columns='treatment_screen_model')
    audit.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_screening_results(trials, path, work)
    assert not (work/'screening_input_manifest.json').exists()


def test_all_rejected_produces_empty_outputs_without_extraction(endpoint_run):
    state, path, _ = endpoint_run
    state.reject_all = True
    spaces.main()
    assert [c[0] for c in state.calls] == ['screen'] * 3
    assert pd.read_csv(path/'out.csv').empty
    assert pd.read_csv(path/'spaces.csv').empty
    for name in ['out.csv', 'spaces.csv']:
        assert set(REGISTRY_METADATA) <= set(pd.read_csv(path/name).columns)
    assert len(pd.read_csv(path/'trial_treatment_screening.csv')) == 3


def test_failed_screen_blocks_extraction_and_only_failed_trial_retries(endpoint_run):
    state, path, _ = endpoint_run
    state.fail = True
    with pytest.raises(RuntimeError, match='screening incomplete'):
        spaces.main()
    assert [c[0] for c in state.calls] == ['screen'] * 3
    assert not (path/'out.csv').exists()
    audit = pd.read_csv(path/'trial_treatment_screening.csv')
    assert audit.treatment_screen_status.tolist() == ['ok', 'ok', 'error']
    assert audit.treatment_screen_model.eq(state.tokenizers[0]).all()
    assert pd.isna(audit.iloc[2].is_cancer_treatment_trial)
    state.fail = False
    spaces.main()
    assert [c[0] for c in state.calls] == ['screen'] * 4 + ['extract']
    assert 'DIAGNOSTIC' in state.calls[3][1]
    assert len(pd.read_csv(path/'trial_treatment_screening.csv')) == 3


@pytest.mark.parametrize('change', ['prompt', 'model', 'sampling', 'input'])
def test_resume_rejects_changed_inputs_or_screening_policy(endpoint_run, monkeypatch, change):
    import trial_treatment_screening as screening
    state, path, argv = endpoint_run
    spaces.main()
    if change == 'prompt':
        monkeypatch.setattr(screening, 'SCREENING_PROMPT', screening.SCREENING_PROMPT + '\nNew policy')
    elif change == 'model':
        argv[argv.index('--model') + 1] = 'google/gemma-4-31b-it'
    elif change == 'sampling':
        argv.extend(['--temperature', '0.2'])
    else:
        rows = pd.read_csv(path/'trials.csv')
        rows.loc[0, 'trial_text'] += ' changed'
        rows.to_csv(path/'trials.csv', index=False)
    with pytest.raises(ValueError, match='new --work-dir'):
        spaces.main()
    assert len(state.calls) == 4


def test_server_file_and_custom_audit_location_are_honored(endpoint_run):
    state, path, argv = endpoint_run
    argv[argv.index('--server_urls')] = '--server_urls_file'
    argv[argv.index('--server_urls_file') + 1] = str(path/'servers.txt')
    argv.extend(['--output-screening', str(path/'custom_audit.csv')])
    spaces.main()
    assert state.registries == [(None, str(path/'servers.txt'), 256)] * 2
    assert len(pd.read_csv(path/'custom_audit.csv')) == 3
    assert not (path/'trial_treatment_screening.csv').exists()


@pytest.mark.parametrize('finish', ['stop', 'length'])
def test_local_screen_validates_before_persisting_and_reuses_completed_shards(tmp_path, monkeypatch, finish):
    import vllm_reasoning_utils
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    calls = []

    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def get_tokenizer(self):
            return SimpleNamespace(apply_chat_template=lambda conversation, **kwargs: json.dumps(conversation))
        def generate(self, prompts, sampling):
            assert all('is_cancer_treatment_trial' in p for p in prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(
                text='{"is_cancer_treatment_trial": false, "rationale": "Diagnostic-only study"}',
                finish_reason=finish)]) for p in prompts]

    class Process:
        def __init__(self, target, kwargs, **ignored):
            self.target, self.kwargs, self.exitcode = target, kwargs, None
        def start(self):
            try:
                self.target(**self.kwargs)
            except ValueError:
                self.exitcode = 1
            else:
                self.exitcode = 0
        def join(self):
            pass

    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: kwargs))
    monkeypatch.setattr(vllm_reasoning_utils, 'parse_reasoning_output', lambda raw, *a: ('reasoning', raw))
    monkeypatch.setattr(spaces.mp, 'get_context', lambda method: SimpleNamespace(Process=Process))
    args = SimpleNamespace(work_dir=str(tmp_path), gpus='0,1', gpus_per_instance=1,
        model='configured-model', download_dir=None, max_model_len=10000, max_num_seqs=8,
        gpu_mem_util=.9, temperature=1, top_p=.95, top_k=-1, min_p=0,
        presence_penalty=0, repetition_penalty=1, max_tokens=8192, batch_size=10)
    trials = pd.DataFrame([dict(nct_id='NCT00000001', trial_text='Diagnostic study')])
    if finish == 'length':
        with pytest.raises(RuntimeError, match='workers failed'):
            spaces.local_main(args, trials, trials.columns.tolist(), 'parser', screening=True)
        assert not list(tmp_path.glob('output_shard*'))
    else:
        for _ in range(2):
            paths = spaces.local_main(args, trials, trials.columns.tolist(), 'parser', screening=True)
            assert len(paths) == 1
            assert not bool(pd.read_parquet(paths[0]).iloc[0].is_cancer_treatment_trial)
            assert pd.read_parquet(paths[0]).treatment_screen_model.tolist() == ['configured-model']
        assert len(calls) == 1
        assert calls[0]['model'] == 'configured-model'
