import asyncio
import json
from types import SimpleNamespace

import pytest
from openai.types.completion import Completion

from remote_vllm_pool import CompletionSampling, make_completion_work_fn
from tests.test_trial_selection import load_script

spaces = load_script('beam_trial_spaces', '0b_create_trial_spaces.py')


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def apply_chat_template(self, conversation, **kwargs):
        return json.dumps(conversation) + json.dumps(kwargs)


def test_beam_native_parser_and_feedback_respect_context_and_effort(tmp_path):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        final = 'bad JSON' if len(calls) < 3 else json.dumps({
            'is_cancer_treatment_trial': True, 'rationale': 'Tests direct anticancer treatment.'})
        raw = ('<|channel|>analysis<|message|>Reasoning trace<|end|>'
               '<|start|>assistant<|channel|>final<|message|>' + final + '<|return|>')
        return Completion.model_validate(dict(id='test', object='text_completion', created=0,
            model='reflection/Beam-1.0-501B-A23B-NVFP4', choices=[dict(index=0, finish_reason='stop',
                text=raw, logprobs=None)]))

    async def scenario():
        tokenizer = Tokenizer()
        completion = make_completion_work_fn(CompletionSampling(
            model='reflection/Beam-1.0-501B-A23B-NVFP4', temperature=.7,
            top_p=.99, top_k=-1, repetition_penalty=1.0), 'beam', tokenizer)
        work = spaces.make_trial_completion_work_fn(completion, tokenizer, tmp_path,
            screening=True, max_model_len=6000, always_reasons=True)
        messages = spaces.build_screening_messages('Synthetic anticancer trial')
        payload = dict(nct_id='NCT00000001', trial_text='Synthetic anticancer trial',
            max_tokens=45000, messages=messages,
            chat_template_kwargs={'reasoning_effort': 'medium'},
            prompt=tokenizer.apply_chat_template(messages, reasoning_effort='medium',
                tokenize=False, add_generation_prompt=True))
        client = SimpleNamespace(completions=SimpleNamespace(create=create))
        budgets = []
        for _ in range(3):
            budgets.append(6000 - len(payload['prompt']) - 256)
            try:
                result = await work(client, payload)
            except ValueError:
                continue
        assert result['is_cancer_treatment_trial'] is True
        assert result['treatment_screen_reasoning'] == 'Reasoning trace'
        assert [c['max_tokens'] for c in calls] == budgets

    asyncio.run(scenario())
    for call, effort in zip(calls, ['medium', 'medium', 'low'], strict=True):
        assert f'"reasoning_effort": "{effort}"' in call['prompt']
        assert '"enable_thinking"' not in call['prompt']
        assert call['extra_body']['skip_special_tokens'] is False
        assert call['temperature'] == .7 and call['top_p'] == .99
        assert call['extra_body']['top_k'] == -1
    assert 'failed validation' in calls[1]['prompt']
    assert 'bad JSON' in calls[1]['prompt']
    audit = [json.loads(line) for line in (tmp_path/'validation_failures.jsonl').read_text().splitlines()]
    assert audit[1]['next_reasoning_effort'] == 'low'


def test_oversize_source_is_rejected_without_truncating_or_sending(tmp_path):
    async def unexpected(*args):
        pytest.fail('Oversize source must not reach the endpoint')

    async def scenario():
        work = spaces.make_trial_completion_work_fn(unexpected, Tokenizer(), tmp_path,
                                                    max_model_len=1000)
        payload = dict(nct_id='NCT00000001', prompt='x' * 1000, max_tokens=100)
        with pytest.raises(ValueError, match='leaves no output budget'):
            await work(None, payload)
        assert payload['prompt'] == 'x' * 1000

    asyncio.run(scenario())
