import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import re
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
    name_or_path = 'gemma4-31b'

    def apply_chat_template(self, conversation, **kwargs):
        if 'Beam' in self.name_or_path:
            assert kwargs == dict(add_generation_prompt=True, tokenize=False, reasoning_effort='medium')
        else:
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


@pytest.mark.parametrize('limits,expected_change', [
    ((4096, 1048576), (34816, 1048576)),
    ((131072, 1048576), None),
    ((generation.resource.RLIM_INFINITY, generation.resource.RLIM_INFINITY), None),
    ((4096, generation.resource.RLIM_INFINITY), (34816, generation.resource.RLIM_INFINITY)),
])
def test_fleet_generation_reserves_socket_and_checkpoint_capacity(monkeypatch, limits, expected_change):
    changes = []
    monkeypatch.setattr(generation.resource, 'getrlimit', lambda kind: limits)
    monkeypatch.setattr(generation.resource, 'setrlimit', lambda kind, value: changes.append(value))
    generation.ensure_file_descriptor_capacity(16384)
    assert changes == ([] if expected_change is None else [expected_change])


def test_insufficient_hard_file_limit_fails_before_loading_or_creating_output(tmp_path, monkeypatch):
    output = tmp_path / 'generation'
    monkeypatch.setattr(generation, 'parse_args', lambda: SimpleNamespace(
        pending_requests=16384, output_dir=str(output)))
    monkeypatch.setattr(generation.resource, 'getrlimit', lambda kind: (4096, 8192))
    with pytest.raises(RuntimeError, match='Raise the hard limit or reduce --pending-requests'):
        generation.main()
    assert not output.exists()


@pytest.mark.parametrize('model,backend', [
    ('reflection/Beam-1.0-501B-A23B-NVFP4', 'fast'),
    ('/tmp/tokenizers/Beam-1.0-501B-A23B-NVFP4', 'fast'),
    ('google/gemma-4-31B-it', 'auto'),
])
def test_generation_loads_beam_tokenizer_without_loading_model_architecture(monkeypatch, model, backend):
    import transformers
    calls = []
    for attribute, label in [('PreTrainedTokenizerFast', 'fast'), ('AutoTokenizer', 'auto')]:
        monkeypatch.setattr(transformers, attribute, SimpleNamespace(
            from_pretrained=lambda model, *, cache_dir, label=label: calls.append((label, model, cache_dir))))
    generation.load_generation_tokenizer(model, '/tmp/tokenizer-cache')
    assert calls == [(backend, model, '/tmp/tokenizer-cache')]


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
@pytest.mark.parametrize('model', ['gemma4-31b', 'reflection/Beam-1.0-501B-A23B-NVFP4'])
def test_user_instructions_and_single_seed_reach_the_new_stages(tmp_path, kind, model):
    tokenizer = Tokenizer()
    tokenizer.name_or_path = model
    args = args_for(tmp_path)
    args.tokenizer_model = model
    instance = generation.Generator(args, tokenizer)
    item = dict(key=f'{kind}:0', space_index=0, this_space=SPACE, trial_boilerplate_text='Active infection',
                patient_class=kind, seed_text='Alice is female and was diagnosed with lung cancer at 60 years and 0 weeks.')
    module = instance.positive if kind == 'positive' else instance.negative
    seed_prompt = instance.build_payload(item, 'seeds')['prompt']
    event_prompt = instance.build_payload(item, 'patient_events')['prompt']
    assert 'exactly five patients' in seed_prompt
    assert 'exactly one patient' in event_prompt
    assert item['seed_text'] in event_prompt
    assert 'Generate the complete event chronology' not in seed_prompt
    assert '(Beginning of example)' not in seed_prompt
    assert 'short prose paragraphs' in seed_prompt
    from synthetic_event_prompts import SEED_FORMAT_EXAMPLE
    assert SEED_FORMAT_EXAMPLE in seed_prompt
    assert 'Do not copy this example into your output' in seed_prompt
    assert 'reuse its patient identity' in seed_prompt
    assert SEED_FORMAT_EXAMPLE not in event_prompt
    assert '<demographics>' not in seed_prompt and '<diagnosis>' not in seed_prompt
    assert 'verbatim' not in event_prompt
    assert 'exactly one <demographics> line' in event_prompt
    assert 'age at the earliest event' in event_prompt
    assert 'The first clinical event does not have to be a diagnosis' in event_prompt
    assert 'including events at the same age in causal order' in event_prompt
    assert 'Immediately after the tag' in event_prompt
    assert SPACE in seed_prompt and 'Active infection' in seed_prompt
    assert module.ELIGIBILITY_INSTRUCTIONS in seed_prompt
    assert SPACE not in event_prompt and 'Active infection' not in event_prompt
    assert 'Eligibility criteria:' not in event_prompt
    assert module.ELIGIBILITY_INSTRUCTIONS not in event_prompt
    assert 'intended fit or mismatch' not in event_prompt
    assert 'supplied criterion' not in event_prompt and 'supplied criteria' not in event_prompt
    for prompt in (seed_prompt, event_prompt):
        assert module.CONSISTENCY_INSTRUCTIONS in prompt
        assert module.DIAGNOSIS_INSTRUCTIONS in prompt
        assert '50 years and 21 weeks' in prompt
        assert '0 through 51' in prompt
        assert 'organs that would not have been visualized' in prompt
        assert 'presence or absence of brain metastases' in prompt
    assert module.EVENT_INSTRUCTIONS in event_prompt
    assert 'At age N years and W weeks,' in event_prompt
    example = module.EVENT_INSTRUCTIONS.split('(Beginning of example)\n')[1].split('(End of example)')[0]
    example_events = generation.parse_patient_history(
        example, 'John Smith, male, was diagnosed with CNS lymphoma at 70 years and 2 weeks.')
    example_kinds = [event[0] for event in example_events]
    assert example_kinds[:6] == ['demographics', 'imaging_report', 'surgery',
                                'pathology_report', 'imaging_report', 'diagnosis']
    if kind == 'negative':
        assert 'would narrowly miss at least one' in seed_prompt
        assert 'who enrolled' not in event_prompt
    # No trial fields are needed or accepted by the patient prompt builder.
    minimal = dict(key=item['key'], patient_class=kind, seed_text=item['seed_text'])
    assert instance.build_payload(minimal, 'patient_events')['prompt'] == event_prompt
    minimal['patient_class'] = 'negative' if kind == 'positive' else 'positive'
    assert instance.build_payload(minimal, 'patient_events')['prompt'] == event_prompt
    item['masked_text'] = 'A synthetic event <BEGIN EVENT CORRESPONDING TO SYNTHETIC NOTE> clinical note'
    assert instance.build_payload(item, 'notes')['prompt'] == instance.notes.build_prompts(
        [item['masked_text']], tokenizer, max_ctx_tokens=250000)[0]


SEEDS = '\n<new_patient>\n'.join(
    f'Patient {i} is female and was diagnosed at 60 years and 0 weeks with metastatic lung adenocarcinoma.'
    for i in range(5)
)


def expand_seed(seed):
    patient, age, diagnosis = re.fullmatch(
        r'(.*?) is female and was diagnosed at (.*?) with (.*)', seed).groups()
    return (f'<demographics> {patient}, female, aged {age}.\n'
            f'<diagnosis> At age {age}, {diagnosis}\n'
            f'<clinical_note> At age {age}, cancer progressed.\n'
            f'<ngs_report> At age {age}, EGFR mutation detected.')


@pytest.mark.parametrize('change,message', [
    (lambda s: s.replace('Patient 0', 'Patient 1'), 'must be distinct'),
    (lambda s: '<demographics>' + s, 'short prose without event tags'),
    (lambda s: s.split('<new_patient>')[0], 'five patient seeds'),
])
def test_seed_validation_rejects_wrong_count_identity_or_event_types(change, message):
    with pytest.raises(OutputValidationError, match=message):
        generation.parse_seeds(change(SEEDS))


@pytest.mark.parametrize('replacement', ['60 years', '60 years and 52 weeks', '60 years and -1 weeks',
                                        '-1 years and 0 weeks', '60.5 years and 0 weeks'])
def test_seed_ages_require_years_and_valid_weeks(replacement):
    with pytest.raises(OutputValidationError, match='weeks'):
        generation.parse_seeds(SEEDS.replace('60 years and 0 weeks', replacement))


def test_prose_seed_can_mention_prior_diagnosis_after_current_diagnosis():
    answer = SEEDS.replace('adenocarcinoma.',
                          'adenocarcinoma. A prior breast cancer was diagnosed at 40 years and 21 weeks.')
    seeds = generation.parse_seeds(answer)
    assert len(seeds) == 5
    assert generation.seed_diagnosis_ages(seeds[0]) == {(60, 0), (40, 21)}


@pytest.mark.parametrize('duration', [
    'The disease-free interval was 1 year and 37 weeks.',
    'The prior cancer has been in remission for 4 years and 48 weeks.',
    'There has been no recurrence for 5 years.',
])
def test_seed_durations_do_not_require_extra_diagnosis_events(duration):
    seed = generation.parse_seeds(SEEDS)[0] + ' ' + duration
    assert generation.seed_diagnosis_ages(seed) == {(60, 0)}
    assert len(generation.parse_patient_history(expand_seed(generation.parse_seeds(SEEDS)[0]), seed)) == 4


@pytest.mark.parametrize('age_cue', ['at', 'at age', 'at age of', 'at the age of', 'aged', 'age'])
def test_pediatric_age_is_required_even_when_seed_also_contains_a_duration(age_cue):
    seed = (f'Patient 0 was diagnosed with leukemia {age_cue} 1 year and 37 weeks. '
            'Remission lasted 4 years and 49 weeks.')
    assert generation.seed_diagnosis_ages(seed) == {(1, 37)}
    with pytest.raises(OutputValidationError, match='assigned ages: 1 years and 37 weeks'):
        generation.parse_patient_history(expand_seed(generation.parse_seeds(SEEDS)[0]), seed)


def test_duration_cannot_substitute_for_a_missing_or_malformed_diagnosis_age():
    for diagnosis in ['Patient 0 has lung cancer.', 'Patient 0 was diagnosed at 60 years.']:
        with pytest.raises(OutputValidationError, match='Give each diagnosis age'):
            generation.seed_diagnosis_ages(diagnosis + ' Remission lasted 4 years and 49 weeks.')


def test_dated_treatment_completion_is_not_an_extra_diagnosis():
    seed = ('A synthetic patient was diagnosed with pancreatic cancer at 58 years and 4 weeks; '
            'received adjuvant chemotherapy completed at 58 years and 40 weeks, '
            'subsequently diagnosed with liver metastasis at 59 years and 12 weeks.')
    assert generation.seed_diagnosis_ages(seed) == {(58, 4), (59, 12)}
    answer = ('<demographics> Synthetic patient, male, aged 58 years and 4 weeks.\n'
              '<diagnosis> At age 58 years and 4 weeks, pancreatic cancer.\n'
              '<systemic> At age 58 years and 40 weeks, completed adjuvant chemotherapy.\n'
              '<diagnosis> At age 59 years and 12 weeks, metastatic recurrence.\n'
              '<clinical_note> At age 59 years and 13 weeks, treatment discussion.')
    assert len(generation.parse_patient_history(answer, seed)) == 5
    with pytest.raises(OutputValidationError, match='treatment/procedure events'):
        generation.parse_patient_history(answer.replace('58 years and 40 weeks', '58 years and 41 weeks'), seed)
    with pytest.raises(OutputValidationError, match='seed diagnoses'):
        generation.parse_patient_history(answer.replace('59 years and 12 weeks', '59 years and 11 weeks'), seed)


def test_procedure_age_does_not_invent_a_diagnosis_timestamp():
    seed = 'Synthetic patient underwent prostatectomy at 75 years and 4 weeks; pathology confirmed prostate cancer.'
    assert generation.seed_age_anchors(seed) == (set(), {(75, 4)})
    answer = ('<demographics> Synthetic patient, male, aged 75 years and 4 weeks.\n'
              '<surgery> At age 75 years and 4 weeks, prostatectomy.\n'
              '<pathology_report> At age 75 years and 5 weeks, prostate cancer.\n'
              '<diagnosis> At age 75 years and 5 weeks, prostate cancer.')
    assert len(generation.parse_patient_history(answer, seed)) == 4


@pytest.mark.parametrize('next_age,valid', [
    ('60 years and 10 weeks', False),
    ('60 years and 21 weeks', True),
    ('60 years and 51 weeks', True),
    ('61 years and 0 weeks', True),
    ('60 years', False),
    ('60 years and 52 weeks', False),
])
def test_patient_chronology_compares_years_and_weeks(next_age, valid):
    seed = generation.parse_seeds(SEEDS.replace('60 years and 0 weeks', '60 years and 21 weeks'))[0]
    answer = expand_seed(seed) + f'\n<clinical_note> At age {next_age}, cancer progressed.'
    if valid:
        assert generation.parse_patient_history(answer, seed)[-1][1].startswith(f'At age {next_age},')
    else:
        with pytest.raises(OutputValidationError, match='weeks'):
            generation.parse_patient_history(answer, seed)


@pytest.mark.parametrize('change,message', [
    (lambda s: s.replace('At age 60 years and 0 weeks, EGFR', 'At age 45 years and 0 weeks, EGFR'), 'chronological order'),
    (lambda s: s + '\n<new_patient>\n' + s, 'exactly one patient'),
    (lambda s: s + '\n<clinical_note> At age 61 years and 0 weeks, evaluated for study entry.', 'study entry'),
])
def test_individual_validation_preserves_chronology_and_no_trial_leakage(change, message):
    seed = generation.parse_seeds(SEEDS)[0]
    with pytest.raises(OutputValidationError, match=message):
        generation.parse_patient_history(change(expand_seed(seed)), seed)


def test_seed_and_patient_markdown_fences_preserve_inequalities():
    seeds = generation.parse_seeds('```text\n' + SEEDS.replace('adenocarcinoma.', 'adenocarcinoma; PD-L1 <1%.') + '\n```')
    answer = '```text\n' + expand_seed(seeds[0]) + '\n```'
    events = generation.parse_patient_history(answer, seeds[0])
    assert 'PD-L1 <1%' in events[1][1]
    assert '```' not in events[-1][1]


def test_patient_wording_is_generated_freely_from_prose_seed():
    seed = generation.parse_seeds(SEEDS)[0]
    answer = expand_seed(seed).replace('Patient 0, female, aged', 'Patient 0 is a woman aged').replace(
        'metastatic lung adenocarcinoma.', 'lung adenocarcinoma with distant metastases')
    events = generation.parse_patient_history(answer, seed)
    assert 'woman' in events[0][1] and 'distant metastases' in events[1][1]


def test_patient_preserves_diagnosis_age_anchors_and_demographics_start_age():
    seed = generation.parse_seeds(SEEDS)[0] + ' Prior breast cancer was diagnosed at 40 years and 21 weeks.'
    with pytest.raises(OutputValidationError, match='assigned ages: 40 years and 21 weeks'):
        generation.parse_patient_history(expand_seed(generation.parse_seeds(SEEDS)[0]), seed)
    answer = ('<demographics> Patient 0, female, aged 40 years and 21 weeks.\n'
              '<diagnosis> At age 40 years and 21 weeks, breast cancer.\n'
              + expand_seed(generation.parse_seeds(SEEDS)[0]).split('\n', 1)[1])
    assert len(generation.parse_patient_history(answer, seed)) == 5
    with pytest.raises(OutputValidationError, match='chronological order'):
        generation.parse_patient_history(answer.replace('aged 40 years and 21 weeks', 'aged 60 years and 0 weeks'), seed)


@pytest.mark.parametrize('change,message', [
    (lambda s: '\n'.join(s.splitlines()[1:]), 'initial demographics'),
    (lambda s: s + '\n' + s.splitlines()[0], 'initial demographics'),
    (lambda s: s.replace('<diagnosis>', '<clinical_note>'), 'diagnosis events'),
])
def test_patient_still_requires_one_initial_demographics_and_diagnoses(change, message):
    seed = generation.parse_seeds(SEEDS)[0]
    with pytest.raises(OutputValidationError, match=message):
        generation.parse_patient_history(change(expand_seed(seed)), seed)


def test_balanced_event_closers_are_formatting_but_malformed_markup_retries():
    seed = generation.parse_seeds(SEEDS)[0]
    closed = expand_seed(seed).replace('aged 60 years and 0 weeks.', 'aged 60 years and 0 weeks.</demographics>').replace(
        'adenocarcinoma.', 'adenocarcinoma; PD-L1 <1%.</diagnosis>')
    events = generation.parse_patient_history(closed, seed)
    assert 'PD-L1 <1%' in events[1][1]
    assert all('</' not in text for _, text in events)
    with pytest.raises(OutputValidationError, match='mismatched or nested'):
        generation.parse_patient_history(closed.replace('</diagnosis>', '</systemic>', 1), seed)


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
    with pytest.raises(RuntimeError, match='1 generation items remain unresolved'):
        asyncio.run(instance.generate_patient_events(spaces))
    saved = pd.concat([pd.read_parquet(p) for p in generation.checkpoint_paths(tmp_path / 'patient_events')])
    assert set(saved.key) >= {'positive:0:0', 'positive:0:1', 'positive:0:2'}
    assert len(saved) == 9
    assert (tmp_path / 'generation_failures/positive_0_3.json').exists()
    before = calls.copy()
    fail[0] = False
    asyncio.run(generation.Generator(args, Tokenizer()).generate_patient_events(spaces))
    resumed = calls[len(before):]
    assert ('seeds', 'positive:0') not in resumed
    assert not set(saved.key).intersection(key for stage, key in resumed)
    saved = pd.concat([pd.read_parquet(p) for p in generation.checkpoint_paths(tmp_path / 'patient_events')])
    assert len(saved) == saved.key.nunique() == 10
    assert not list((tmp_path / 'generation_failures').glob('*.json'))


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


@pytest.mark.parametrize('model', ['gemma4-31b', 'reflection/Beam-1.0-501B-A23B-NVFP4'])
def test_validation_retry_contains_exact_error_and_same_seed(tmp_path, monkeypatch, model):
    seed = generation.parse_seeds(SEEDS)[0]
    async def complete(client, payload):
        return '', expand_seed(seed).replace('At age 60 years and 0 weeks, EGFR', 'At age 40 years and 0 weeks, EGFR'), dict(finish_reason='stop', raw_text='invalid')
    fake_session(monkeypatch, complete)
    tokenizer = Tokenizer()
    tokenizer.name_or_path = model
    args = args_for(tmp_path)
    args.tokenizer_model = model
    instance = generation.Generator(args, tokenizer)
    payload = instance.build_payload(dict(key='positive:0:0', space_index=0, this_space=SPACE,
        trial_boilerplate_text='', patient_class='positive', seed_text=seed), 'patient_events')
    with pytest.raises(OutputValidationError, match='age 40 years and 0 weeks follows 60 years and 0 weeks'):
        asyncio.run(instance.completion_work(tmp_path)(SimpleNamespace(base_url='http://test/v1/'), payload))
    assert 'age 40 years and 0 weeks follows 60 years and 0 weeks' in payload['prompt']
    assert seed in payload['prompt']
    assert 'age 40 years and 0 weeks follows 60 years and 0 weeks' in payload['messages'][-1]['content']
    assert len(list(tmp_path.glob('invalid_*.json'))) == 1


@pytest.mark.parametrize('stage', ['seeds', 'patient_events', 'notes'])
def test_beam_generation_parses_native_raw_completions_and_saves_reasoning(tmp_path, stage):
    from openai.types import Completion
    model = 'reflection/Beam-1.0-501B-A23B-NVFP4'
    args = args_for(tmp_path)
    args.model = args.tokenizer_model = model
    args.reasoning_parser = 'auto'
    tokenizer = Tokenizer()
    tokenizer.name_or_path = model
    instance = generation.Generator(args, tokenizer)
    seed = generation.parse_seeds(SEEDS)[0]
    answer = SEEDS if stage == 'seeds' else expand_seed(seed) if stage == 'patient_events' else 'Synthetic note.'
    raw = ('<|channel|>analysis<|message|>Saved thinking.<|end|><|start|>assistant'
           '<|channel|>final<|message|>' + answer + '<|return|>')
    response = Completion.model_validate(dict(id='test', object='text_completion', created=0,
        model=model, choices=[dict(index=0, finish_reason='stop', text=raw, logprobs=None)]))
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return response

    payload = instance.build_payload(dict(key='positive:0:0', space_index=0, this_space=SPACE,
        trial_boilerplate_text='', patient_class='positive', seed_text=seed, masked_text='Synthetic event.'), stage)
    client = SimpleNamespace(base_url='http://test/v1/', completions=SimpleNamespace(create=create))
    result = asyncio.run(instance.completion_work(tmp_path)(client, payload))
    assert calls[0]['prompt'] == payload['prompt']
    assert 'messages' not in calls[0]
    assert calls[0]['extra_body']['skip_special_tokens'] is False
    assert payload['chat_template_kwargs'] == {'reasoning_effort': 'medium'}
    assert calls[0]['temperature'] == .7 and calls[0]['top_p'] == .99
    assert result['reasoning'] == 'Saved thinking.'
    assert result['transport'] == 'completions'
    assert result['raw_response'] == raw
    assert '<|channel|>' not in result['answer']


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
            assert '<demographics>' not in payload['seed_text']
            assert 'is female and was diagnosed' in payload['seed_text']
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
