import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
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
        seed=42, event_max_tokens=40000, note_max_tokens=15000, max_model_len=250000,
        spaces_per_note_chunk=1, pending_requests=4, results_per_shard=3,
        max_attempts=2, request_timeout=7200, reasoning_parser='gemma4')


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


def test_existing_prompts_are_used_exactly(tmp_path):
    instance = generation.Generator(args_for(tmp_path), Tokenizer())
    item = dict(key='positive:0', space_index=0, this_space=SPACE, trial_boilerplate_text='Active infection',
                patient_class='positive')
    prompt = instance.build_payload(item, 'events')['prompt']
    import random
    messages = instance.positive.build_messages(SPACE + '\nNo history of:\nActive infection', random.Random(42))
    assert prompt == '\n'.join(message['content'] for message in messages)
    item['masked_text'] = 'A synthetic event <BEGIN EVENT CORRESPONDING TO SYNTHETIC NOTE> clinical note'
    assert instance.build_payload(item, 'notes')['prompt'] == instance.notes.build_prompts(
        [item['masked_text']], Tokenizer(), max_ctx_tokens=250000)[0]


def test_full_generation_and_resume_preserve_all_patients_and_reasoning(tmp_path, monkeypatch):
    pd.DataFrame([dict(nct_id='NCT1', this_space=SPACE, trial_boilerplate_text='', space_index=i)
                  for i in range(2)]).to_csv(tmp_path / 'sample.csv', index=False)
    calls = []

    def completion_factory(*args):
        async def complete(client, payload):
            calls.append(payload)
            answer = HISTORIES if payload['stage'] == 'events' else 'A detailed synthetic clinical document.'
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
    assert len(calls) == 4 + 40
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
        assert len(output) == 2 and output.synthetic_patient_prompts.eq(HISTORIES).all()
    before = {p: p.read_bytes() for p in (tmp_path / 'note_shards').rglob('*.parquet')}
    asyncio.run(generation.Generator(args_for(tmp_path), Tokenizer()).run())
    assert len(calls) == 44
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
