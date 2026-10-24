import asyncio
import json
from types import SimpleNamespace

import pytest

from remote_vllm_pool import (
    AdaptiveSemaphore, CompletionSampling, ServerEntry, _dispatch_one,
    make_completion_work_fn,
)
from tests.test_trial_selection import load_script

spaces = load_script('retry_trial_spaces', 'legacy_trial_space_extraction.py')


@pytest.mark.parametrize('initial_thinking', [True, False])
@pytest.mark.parametrize('recover', [True, False])
def test_pool_feedback_switches_thinking_and_preserves_failures(tmp_path, monkeypatch, initial_thinking, recover):
    import vllm_reasoning_utils
    requests, parser_modes = [], []

    class Tokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            return json.dumps({'messages': conversation, 'thinking': kwargs['enable_thinking']})

    def parse(raw, *args, **kwargs):
        parser_modes.append(kwargs['enable_thinking'])
        return 'test reasoning', raw

    monkeypatch.setattr(vllm_reasoning_utils, 'parse_reasoning_output', parse)

    async def create(**kwargs):
        requests.append(json.loads(kwargs['prompt']))
        final = '{"cancer_treatment_trial": unquoted}'
        if recover and len(requests) == 3:
            final = '{"cancer_treatment_trial": "no"}'
        return SimpleNamespace(choices=[SimpleNamespace(text=final, finish_reason='stop')])

    async def scenario():
        tokenizer = Tokenizer()
        completion = make_completion_work_fn(CompletionSampling(model='test'), 'test', tokenizer)
        work = spaces.make_trial_completion_work_fn(completion, tokenizer, tmp_path, screening=True)
        payload = dict(nct_id='NCT00000001', trial_text='Synthetic supportive study', max_tokens=100,
                       enable_thinking=initial_thinking,
                       prompt=tokenizer.apply_chat_template(spaces.build_screening_messages('Synthetic supportive study'),
                                                            enable_thinking=initial_thinking))
        sem = AdaptiveSemaphore(16, success_threshold=10)
        sem.limit = 8
        entry = ServerEntry(url='http://unused', client=SimpleNamespace(completions=SimpleNamespace(create=create)), semaphore=sem)
        queue, saved = asyncio.Queue(), []

        async def add(work_id, result):
            saved.append((work_id, result))

        item_id, attempt = 1, 0
        while True:
            await sem.acquire()
            await _dispatch_one(entry, item_id, payload, attempt, work_fn=work,
                                work_queue=queue, results=SimpleNamespace(add=add), max_attempts=5,
                                error_placeholder=lambda payload, exc: f'ERROR: {exc}')
            assert sem.limit == 8  # Formatting failures remain neutral.
            if queue.empty():
                break
            item_id, payload, attempt = queue.get_nowait()
        assert len(saved) == 1
        if recover:
            assert saved[0][1]['is_cancer_treatment_trial'] is False
        else:
            assert saved[0][1].startswith('ERROR:')

    asyncio.run(scenario())
    expected = [initial_thinking, initial_thinking, not initial_thinking]
    if not recover:
        expected += [not initial_thinking, initial_thinking]
    assert parser_modes == expected == [r['thinking'] for r in requests]
    for r in requests[1:]:
        messages = r['messages']
        assert len(messages) == 3  # Combined prompt plus one failed answer/feedback pair.
        assert 'Synthetic supportive study' in messages[0]['content']
        assert 'unquoted' in messages[1]['content']
        assert 'Expecting value: line 1 column' in messages[2]['content']
    records = [json.loads(line) for line in (tmp_path/'validation_failures.jsonl').read_text().splitlines()]
    assert records[0]['raw_text'].endswith('unquoted}')
    assert records[0]['finish_reason'] == 'stop'
    assert records[1]['next_enable_thinking'] is not initial_thinking
    assert records[-1]['event'] == ('recovered' if recover else 'validation_failure')


def test_transport_errors_do_not_change_feedback_or_thinking(tmp_path):
    async def completion(client, payload):
        raise TimeoutError('connection timeout')

    async def scenario():
        payload = {'prompt': 'original', 'enable_thinking': False}
        original = dict(payload)
        work = spaces.make_trial_completion_work_fn(completion, None, tmp_path, screening=True)
        with pytest.raises(TimeoutError):
            await work(None, payload)
        assert payload == original
        assert not (tmp_path/'validation_failures.jsonl').exists()

    asyncio.run(scenario())


def test_extraction_feedback_is_bounded_and_does_not_change_other_items(tmp_path):
    calls = []
    class Tokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            calls.append((conversation, kwargs['enable_thinking']))
            return json.dumps(conversation)

    async def completion(client, payload):
        return 'reasoning', 'X' * 13000, {'finish_reason': 'stop', 'raw_text': 'X' * 13000}

    async def scenario():
        work = spaces.make_trial_completion_work_fn(completion, Tokenizer(), tmp_path)
        bad = dict(nct_id='NCT00000001', trial_text='Synthetic cancer trial', prompt='initial')
        other = dict(nct_id='NCT00000002', trial_text='Different trial', prompt='other')
        for _ in range(2):
            with pytest.raises(ValueError, match='Boilerplate exclusions'):
                await work(None, bad)
        assert other['prompt'] == 'other' and 'validation_failures' not in other
        messages, thinking = calls[-1]
        assert thinking is False
        assert messages[0]['content'] != 'Reasoning: high.'
        assert len(messages[-2]['content']) == 12000
        assert 'Boilerplate exclusions:' in messages[-1]['content']

    asyncio.run(scenario())


def test_reasoning_parser_receives_thinking_mode(monkeypatch):
    import sys
    from vllm_reasoning_utils import parse_reasoning_output
    seen = []

    class Parser:
        def __init__(self, tokenizer, **kwargs):
            self.options = kwargs
        def extract_reasoning(self, text, request):
            seen.append((self.options, request.chat_template_kwargs))
            return '', text

    monkeypatch.setitem(sys.modules, 'vllm.reasoning', SimpleNamespace(
        register_lazy_reasoning_parsers=lambda: None,
        ReasoningParserManager=SimpleNamespace(get_reasoning_parser=lambda name: Parser)))
    raw = '{"text":"literal <|im_end|> inside a string"}<|im_end|>\n'
    assert parse_reasoning_output(raw, 'nemotron_v3', None, enable_thinking=False) == (
        '', '{"text":"literal <|im_end|> inside a string"}')
    assert seen == [({'chat_template_kwargs': {'enable_thinking': False}}, {'enable_thinking': False})]
