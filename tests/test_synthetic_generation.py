import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import generate_synthetic_data as generation
from remote_vllm_pool import OutputValidationError


SPACE = ('Age range allowed: 18+. Sex allowed: Any. Cancer primary site allowed: Lung. '
         'Histology allowed: Adenocarcinoma. Disease state allowed: Metastatic. '
         'Prior treatment requirements: No prior systemic therapy. Biomarker requirements: EGFR mutation.')
HISTORIES = '\n<new_patient>\n'.join(
    f'<demographics> Patient {i}, aged 60.\n<diagnosis> Metastatic lung adenocarcinoma.\n'
    '<clinical_note> At age 60, cancer progressed.\n<ngs_report> EGFR mutation detected.'
    for i in range(5)
)


class Tokenizer:
    chat_template = 'fake-template'

    def apply_chat_template(self, conversation, **kwargs):
        assert kwargs['enable_thinking'] is True
        return '\n'.join(message['content'] for message in conversation)

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=self.encode(text))

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


def args_for(tmp_path):
    return argparse.Namespace(input_csv=str(tmp_path / 'sample.csv'), output_dir=str(tmp_path),
        model='gemma4-31b', tokenizer_model='RedHatAI/Gemma-4-31B-IT-FP8-Dynamic',
        seed=42, seed_max_tokens=16000, event_max_tokens=20000, note_max_tokens=15000, stop_after_events=False, max_model_len=250000,
        spaces_per_note_chunk=1, pending_requests=4, results_per_shard=3,
        max_attempts=2, request_timeout=7200, reasoning_parser='gemma4')


@pytest.mark.parametrize('filename', ['1a_make_synthetic_enrollee_prompts.py',
                                     '1b_make_synthetic_negative_enrollee_prompts.py'])
def test_prompt_modules_reject_standalone_execution_without_side_effects(tmp_path, filename):
    module = generation.load_stage(filename)
    assert callable(module.build_seed_messages) and callable(module.build_patient_messages)
    result = subprocess.run([sys.executable, '-S', module.__file__, '--help'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'import-only prompt module' in result.stderr
    assert 'generate_synthetic_data.py' in result.stderr
    assert not result.stdout
    assert not list(tmp_path.iterdir())


def test_patient_ids_do_not_collide_between_spaces_or_classes():
    records = [dict(space_index=space, patient_class=kind, nct_id='NCT1', answer=HISTORIES)
               for space in range(2) for kind in ('positive', 'negative')]
    frame = generation.event_rows(records, 'run')
    assert frame.pseudo_mrn.nunique() == frame.patient_id.nunique() == 20
    pd.testing.assert_frame_equal(frame, generation.event_rows(records, 'run'))
    assert set(frame.pseudo_mrn) == set(range(1, 21))


@pytest.mark.parametrize('answer,finish', [(HISTORIES, 'length'), ('', 'stop'),
                                          ('ERROR: failed', 'stop'), (HISTORIES.split('<new_patient>')[0], 'stop')])
def test_invalid_event_outputs_are_not_accepted(answer, finish):
    with pytest.raises(OutputValidationError):
        generation.validate_completion(answer, {'finish_reason': finish}, 'events')


@pytest.mark.parametrize('finding', [
    'Margins were close (< 1mm). No tumor at ink.',
    'PD-L1 TPS <1%; tumor purity >=20%; TMB <=5 mutations/Mb.',
    'Lesion size < 5 mm and > 2 mm, with no new metastases.',
    'Value <0.01; reference range 0 <= value < 1.',
])
@pytest.mark.parametrize('separator', ['\n', ''])
def test_event_inequalities_survive_both_parsers_and_note_inputs(finding, separator):
    histories = '\n<new_patient>\n'.join(
        separator.join([f'<demographics> Synthetic patient {i}, aged 60.',
                        '<diagnosis> Metastatic cancer.',
                        f'<pathology_report> {finding}',
                        '<clinical_note> Cancer is stable.'])
        for i in range(5)
    )
    record = dict(space_index=0, patient_class='positive', nct_id='NCT1', answer=histories)
    notes = generation.load_stage('2_make_synthetic_notes_sharded.py')
    current = generation.event_rows([record], 'run')
    legacy = notes.process_clinical_notes_from_df(pd.DataFrame([
        dict(space_index=0, synthetic_patient_prompts=histories)]))
    for events in (current, legacy):
        assert events.patient_id.nunique() == 5
        assert len(events) == 20
        assert events.loc[events.event_type.eq('pathology_report'), 'event_text'].tolist() == [finding] * 5
        prompts = notes.create_masked_dataset(events)
        assert len(prompts) == 10
        assert prompts.masked_text.map(lambda text: finding in text).all()
        for text in prompts.loc[prompts.event_type.eq('pathology_report'), 'masked_text']:
            assert f'<pathology_report>{finding} <END EVENT CORRESPONDING TO SYNTHETIC NOTE>' in text
    assert current.event_text.tolist() == legacy.event_text.tolist()


@pytest.mark.parametrize('bad_event', ['<unknown_tag> Invalid event.', '<pathology_report>'])
def test_tag_boundaries_still_reject_unknown_and_empty_events(bad_event):
    answer = HISTORIES.replace('<clinical_note>', bad_event + '\n<clinical_note>', 1)
    with pytest.raises(OutputValidationError, match='Unknown event tag or empty event'):
        generation.parse_histories(answer)


@pytest.mark.parametrize('kind', ['positive', 'negative'])
def test_user_instructions_and_single_seed_reach_the_new_stages(tmp_path, kind):
    instance = generation.Generator(args_for(tmp_path), Tokenizer())
    item = dict(key=f'{kind}:0', space_index=0, this_space=SPACE, trial_boilerplate_text='Active infection',
                patient_class=kind, seed_text='<demographics> Alice, female, age 60.\n<diagnosis> At age 60 years, lung cancer.')
    module = instance.positive if kind == 'positive' else instance.negative
    seed_prompt = instance.build_payload(item, 'seeds')['prompt']
    event_prompt = instance.build_payload(item, 'patient_events')['prompt']
    assert 'exactly five patients' in seed_prompt
    assert 'exactly one patient' in event_prompt
    assert item['seed_text'] in event_prompt
    assert 'Generate the complete event chronology' not in seed_prompt
    assert '(Beginning of example)' not in seed_prompt
    for prompt in (seed_prompt, event_prompt):
        assert module.CONSISTENCY_INSTRUCTIONS in prompt
        assert module.DIAGNOSIS_INSTRUCTIONS in prompt
        assert 'Active infection' in prompt
    assert module.EVENT_INSTRUCTIONS in event_prompt
    if kind == 'negative':
        assert 'narrowly misses at least one supplied criterion' in event_prompt
        assert 'who enrolled' not in event_prompt
    item['masked_text'] = 'A synthetic event <BEGIN EVENT CORRESPONDING TO SYNTHETIC NOTE> clinical note'
    assert instance.build_payload(item, 'notes')['prompt'] == instance.notes.build_prompts(
        [item['masked_text']], Tokenizer(), max_ctx_tokens=250000)[0]


SEEDS = '\n<new_patient>\n'.join(
    f'<demographics> Patient {i}, aged 60.\n<diagnosis> At age 60 years, metastatic lung adenocarcinoma.'
    for i in range(5)
)


def expand_seed(seed):
    return seed + '\n<clinical_note> At age 60 years, cancer progressed.\n<ngs_report> At age 60 years, EGFR mutation detected.'


@pytest.mark.parametrize('change,message', [
    (lambda s: s.replace('Patient 0', 'Patient 1'), 'distinct patient'),
    (lambda s: s.replace('<diagnosis>', '<clinical_note>', 1), 'diagnosis lines only'),
    (lambda s: s.split('<new_patient>')[0], 'five patient seeds'),
])
def test_seed_validation_rejects_wrong_count_identity_or_event_types(change, message):
    with pytest.raises(OutputValidationError, match=message):
        generation.parse_seeds(change(SEEDS))


@pytest.mark.parametrize('change,message', [
    (lambda s: s.replace('Patient 0', 'Somebody else'), 'seed demographics'),
    (lambda s: s.replace('At age 60 years, EGFR', 'At age 45 years, EGFR'), 'chronological order'),
    (lambda s: s + '\n<new_patient>\n' + s, 'exactly one patient'),
    (lambda s: s + '\n<clinical_note> At age 61 years, evaluated for study entry.', 'study entry'),
])
def test_individual_validation_preserves_seed_chronology_and_no_trial_leakage(change, message):
    seed = generation.parse_seeds(SEEDS)[0]
    with pytest.raises(OutputValidationError, match=message):
        generation.parse_patient_history(change(expand_seed(seed)), seed)


def test_seed_and_patient_markdown_fences_preserve_inequalities():
    seeds = generation.parse_seeds('```text\n' + SEEDS.replace('adenocarcinoma.', 'adenocarcinoma; PD-L1 <1%.') + '\n```')
    answer = '```text\n' + expand_seed(seeds[0]) + '\n```'
    events = generation.parse_patient_history(answer, seeds[0])
    assert 'PD-L1 <1%' in events[1][1]
    assert '```' not in events[-1][1]


def test_missing_terminal_period_is_restored_without_relaxing_seed_facts():
    seed = generation.parse_seeds(SEEDS)[0]
    answer = expand_seed(seed).replace('adenocarcinoma.\n', 'adenocarcinoma\n')
    events = generation.parse_patient_history(answer, seed)
    assert events[1] == generation.tagged_events(seed)[1]
    with pytest.raises(OutputValidationError, match='seed demographics'):
        generation.parse_patient_history(answer.replace('adenocarcinoma\n', 'squamous carcinoma\n'), seed)


def test_balanced_event_closers_are_formatting_but_malformed_markup_retries():
    closed = SEEDS.replace('aged 60.', 'aged 60.</demographics>').replace(
        'adenocarcinoma.', 'adenocarcinoma; PD-L1 <1%.</diagnosis>')
    seeds = generation.parse_seeds(closed)
    assert all('</' not in seed and 'PD-L1 <1%' in seed for seed in seeds)
    with pytest.raises(OutputValidationError, match='mismatched or nested'):
        generation.parse_seeds(closed.replace('</diagnosis>', '</systemic>', 1))


def fake_session(monkeypatch, complete):
    class Session:
        def __init__(self, registry, work, **kwargs): self.work = work
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def submit(self, payload):
            return await self.work(SimpleNamespace(base_url='http://test/v1/'), payload)
    monkeypatch.setattr(generation, 'make_completion_work_fn', lambda *args: complete)
    monkeypatch.setattr(generation, 'build_registry_from_args', lambda args: None)
    monkeypatch.setattr(generation, 'RequestSession', Session)


def test_partial_resume_keeps_seeds_and_completed_individual_patients(tmp_path, monkeypatch):
    spaces = pd.DataFrame([dict(nct_id='NCT1', this_space=SPACE, trial_boilerplate_text='', space_index=0)])
    calls, fail = [], [True]
    args = args_for(tmp_path)
    args.pending_requests = 1
    instance = generation.Generator(args, Tokenizer())

    async def complete(client, payload):
        calls.append((payload['stage'], payload['key']))
        if payload['key'] == 'positive:0:3' and fail[0]:
            raise RuntimeError('simulated interruption')
        if payload['stage'] == 'seeds':
            answer = SEEDS
        else:
            assert instance.seed_path(dict(space_index=0, patient_class=payload['key'].split(':')[0])).exists()
            answer = expand_seed(payload['seed_text'])
        return 'reasoning', answer, dict(finish_reason='stop', raw_text=answer)

    fake_session(monkeypatch, complete)
    with pytest.raises(ExceptionGroup, match='TaskGroup'):
        asyncio.run(instance.generate_patient_events(spaces))
    saved = pd.concat([pd.read_parquet(p) for p in generation.checkpoint_paths(tmp_path / 'patient_events')])
    assert set(saved.key) >= {'positive:0:0', 'positive:0:1', 'positive:0:2'}
    before = calls.copy()
    fail[0] = False
    asyncio.run(generation.Generator(args, Tokenizer()).generate_patient_events(spaces))
    resumed = calls[len(before):]
    assert ('seeds', 'positive:0') not in resumed
    assert not set(saved.key).intersection(key for stage, key in resumed)
    saved = pd.concat([pd.read_parquet(p) for p in generation.checkpoint_paths(tmp_path / 'patient_events')])
    assert len(saved) == saved.key.nunique() == 10


def test_patient_expansion_does_not_wait_for_all_seed_groups(tmp_path, monkeypatch):
    spaces = pd.DataFrame([dict(nct_id=f'NCT{i}', this_space=SPACE, trial_boilerplate_text='', space_index=i) for i in range(2)])

    async def scenario():
        expanded = asyncio.Event()
        async def complete(client, payload):
            if payload['stage'] == 'seeds':
                if payload['key'] == 'positive:0':
                    await expanded.wait()
                answer = SEEDS
            else:
                expanded.set()
                answer = expand_seed(payload['seed_text'])
            return 'reasoning', answer, dict(finish_reason='stop', raw_text=answer)
        fake_session(monkeypatch, complete)
        await asyncio.wait_for(generation.Generator(args_for(tmp_path), Tokenizer()).generate_patient_events(spaces), 10)
        assert expanded.is_set()
    asyncio.run(scenario())


def test_validation_retry_contains_exact_error_and_same_seed(tmp_path, monkeypatch):
    seed = generation.parse_seeds(SEEDS)[0]
    async def complete(client, payload):
        return '', expand_seed(seed).replace('At age 60 years, EGFR', 'At age 40 years, EGFR'), dict(finish_reason='stop', raw_text='invalid')
    fake_session(monkeypatch, complete)
    instance = generation.Generator(args_for(tmp_path), Tokenizer())
    payload = instance.build_payload(dict(key='positive:0:0', space_index=0, this_space=SPACE,
        trial_boilerplate_text='', patient_class='positive', seed_text=seed), 'patient_events')
    with pytest.raises(OutputValidationError, match='age 40 follows 60'):
        asyncio.run(instance.completion_work(tmp_path)(SimpleNamespace(base_url='http://test/v1/'), payload))
    assert 'age 40 follows 60' in payload['prompt']
    assert seed in payload['prompt']
    assert len(list(tmp_path.glob('invalid_*.json'))) == 1


def test_full_generation_and_resume_preserve_all_patients_and_reasoning(tmp_path, monkeypatch):
    pd.DataFrame([dict(nct_id='NCT1', this_space=SPACE, trial_boilerplate_text='', space_index=i)
                  for i in range(2)]).to_csv(tmp_path / 'sample.csv', index=False)
    calls = []

    def completion_factory(*args):
        async def complete(client, payload):
            calls.append(payload)
            answer = (SEEDS if payload['stage'] == 'seeds' else expand_seed(payload['seed_text'])
                      if payload['stage'] == 'patient_events' else 'A detailed synthetic clinical document.')
            return 'Thinking text', answer, {'finish_reason': 'stop', 'raw_text': 'RAW:' + answer}
        return complete

    class Session:
        def __init__(self, registry, work, **kwargs):
            self.work = work

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def submit(self, payload):
            return await self.work(SimpleNamespace(base_url='http://test/v1/'), payload)

    monkeypatch.setattr(generation, 'make_completion_work_fn', completion_factory)
    monkeypatch.setattr(generation, 'build_registry_from_args', lambda args: None)
    monkeypatch.setattr(generation, 'RequestSession', Session)
    instance = generation.Generator(args_for(tmp_path), Tokenizer())
    asyncio.run(instance.run())
    assert len(calls) == 4 + 20 + 40
    assert sum(p['stage'] == 'seeds' for p in calls) == 4
    assert sum(p['stage'] == 'patient_events' for p in calls) == 20
    for payload in calls:
        if payload['stage'] == 'patient_events':
            assert payload['seed_text'].count('<demographics>') == 1
            assert '<new_patient>' not in payload['seed_text']
    notes = pd.read_parquet(tmp_path / 'all_synthetic_notes.parquet')
    events = pd.read_parquet(tmp_path / 'all_synthetic_events.parquet')
    assert len(notes) == 40 and len(events) == 80
    assert notes.pseudo_mrn.nunique() == 20
    assert notes.row_id.nunique() == 40
    assert set(notes.patient_id) == set(events.patient_id)
    assert set(notes.patient_class) == {'positive', 'negative'}
    assert notes.reasoning.eq('Thinking text').all()
    assert notes.synth_note_reasoning_and_note.str.startswith('RAW:').all()
    for kind in ('positive', 'negative'):
        output = pd.read_csv(tmp_path / f'trial_spaces_with_{kind}_prompts.csv')
        assert len(output) == 2
        assert output.synthetic_patient_prompts.eq('\n<new_patient>\n'.join(expand_seed(s) for s in generation.parse_seeds(SEEDS))).all()
        assert output.patient_provenance_json.map(lambda text: len(json.loads(text))).eq(5).all()
    before = {p: p.read_bytes() for p in (tmp_path / 'note_shards').rglob('*.parquet')}
    asyncio.run(generation.Generator(args_for(tmp_path), Tokenizer()).run())
    assert len(calls) == 64
    assert before == {p: p.read_bytes() for p in before}
    assert json.loads((tmp_path / 'status.json').read_text())['stage'] == 'completed'


def test_sampling_retains_every_space_and_user_dfci_holdout(tmp_path):
    module = generation.load_stage('0c_sample_trial_spaces.py')
    input_csv, dfci_csv, output_csv = [tmp_path / name for name in ('input.csv', 'dfci.csv', 'sample.csv')]
    frame = pd.DataFrame([dict(nct_id=f'NCT{i}', this_space=SPACE, space_number=j)
                          for i in range(5) for j in range(i + 1)])
    frame.to_csv(input_csv, index=False)
    pd.DataFrame([dict(nct_id='NCT0', status='OPEN TO ACCRUAL', open_date='2020-01', close_date='2030-01')]
                 ).to_csv(dfci_csv, index=False)
    sample = module.sample_trial_spaces(input_csv, output_csv, dfci_csv, n_trials=3, chunksize=2)
    assert sample.nct_id.nunique() == 3 and 'NCT0' not in set(sample.nct_id)
    for trial, rows in sample.groupby('nct_id'):
        assert len(rows) == len(frame.loc[frame.nct_id.eq(trial)])
    assert sample.space_index.tolist() == list(range(len(sample)))
    same = module.sample_trial_spaces(input_csv, output_csv, dfci_csv, n_trials=3, chunksize=4)
    pd.testing.assert_frame_equal(sample, same)
