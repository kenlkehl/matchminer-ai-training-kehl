"""Model-family defaults must reach real request and local-generation paths."""
import argparse
import asyncio
from types import SimpleNamespace

import pytest

from vllm_reasoning_utils import add_sampling_cli_args, resolve_sampling_defaults, sampling_from_args


@pytest.mark.parametrize('model,top_k,presence', [
    ('google/gemma-4-31B-it', 64, 0.0),
    ('RedHatAI/Gemma-4-31B-IT-FP8-Dynamic', 64, 0.0),
    ('gemma4-31b', 64, 0.0),
    ('nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4', -1, 0.0),
    ('Qwen/Qwen3.5-35B-A3B', 20, 1.5),
    ('Inferact/Qwen3.8-Flash-Next-NVFP4', 20, 0.0),
])
def test_detected_family_replaces_legacy_greedy_and_penalty_defaults(model, top_k, presence):
    result = resolve_sampling_defaults(model, fallback=dict(
        temperature=0.0, top_k=1, top_p=0.2, repetition_penalty=1.1, presence_penalty=1.5))
    assert result == dict(temperature=1.0, top_p=0.95, top_k=top_k,
                         min_p=0.0, presence_penalty=presence, repetition_penalty=1.0)


def test_unknown_family_keeps_task_fallback_and_explicit_zero_wins():
    fallback = dict(temperature=0.2, top_k=1, repetition_penalty=1.1)
    assert resolve_sampling_defaults('unregistered', fallback=fallback) == fallback
    parser = argparse.ArgumentParser()
    add_sampling_cli_args(parser)
    args = parser.parse_args(['--temperature', '0', '--top_k', '7', '--presence-penalty', '0'])
    result = sampling_from_args(args, model='Qwen/Qwen3.5-4B', fallback=fallback)
    assert result['temperature'] == 0 and result['top_k'] == 7 and result['presence_penalty'] == 0
    assert result['top_p'] == .95


@pytest.mark.parametrize('thinking,temp,top_p,presence', [(True, 1.0, .95, 0), (False, .7, .8, 1.5)])
def test_teacher_request_sampling_tracks_thinking_mode(monkeypatch, thinking, temp, top_p, presence):
    from oncoreasoning_training import teacher
    captured = []
    monkeypatch.setattr(teacher, 'EndpointPool', lambda *a, **kw: SimpleNamespace(
        request=lambda **request: captured.append(request)))
    teacher.TeacherPool(model='Qwen/Qwen3.8-27B', thinking=thinking)([{'role': 'user', 'content': 'Fabricated'}])
    request = captured[0]
    assert request['temperature'] == temp and request['top_p'] == top_p
    assert request['presence_penalty'] == presence
    assert request['extra_body']['chat_template_kwargs']['enable_thinking'] is thinking


@pytest.mark.parametrize('model,top_k', [
    ('RedHatAI/Gemma-4-31B-IT-FP8-Dynamic', 64),
    ('nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4', -1),
])
def test_production_synthetic_generator_uses_tokenizer_family_sampling(tmp_path, model, top_k):
    from generate_synthetic_data import Generator
    from tests.test_synthetic_generation import Tokenizer, args_for

    args = args_for(tmp_path)
    args.model, args.tokenizer_model = 'served-alias', model
    sampling = Generator(args, Tokenizer()).sampling
    assert sampling.model == 'served-alias'
    assert sampling.temperature == 1.0 and sampling.top_p == .95
    assert sampling.top_k == top_k and sampling.repetition_penalty == 1.0


@pytest.mark.parametrize('temperature', [None, 0.0])
def test_synthetic_labeling_direct_and_pool_requests_agree(tmp_path, monkeypatch, temperature):
    from synthetic_extractors import vllm_labeling_common as common
    import remote_vllm_pool as pool

    direct, pooled = [], []
    def http(method, url, **kwargs):
        direct.append(kwargs['payload'])
        return {'choices': [{'message': {'content': '{}'}}]}
    monkeypatch.setattr(common, 'http_json', http)
    common.chat_completion(base_url='http://test/v1', model='served-alias',
        sampling_model='RedHatAI/Gemma-4-31B-IT-FP8-Dynamic', messages=[],
        temperature=temperature, max_tokens=100, timeout=1, api_key=None, json_mode=True)

    parser = argparse.ArgumentParser()
    common.add_common_args(parser, default_text_field='text', default_output=str(tmp_path/'out.jsonl'))
    args = parser.parse_args(['--input', str(tmp_path/'input.jsonl'), '--model',
        'RedHatAI/Gemma-4-31B-IT-FP8-Dynamic', '--served-model-name', 'served-alias'])
    args.temperature = temperature
    async def stop():
        pass
    async def create(**kwargs):
        pooled.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
    async def run_pool(**kwargs):
        _, record = kwargs['work_items'][0]
        await kwargs['work_fn'](SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create))), record)
    monkeypatch.setattr(pool, 'build_registry_from_args', lambda args: SimpleNamespace(stop=stop))
    monkeypatch.setattr(pool, 'run_pool', run_pool)
    asyncio.run(common.run_remote_labeling_pool(args=args,
        records=[common.Record(0, 'synthetic', 'Fabricated', {})],
        build_messages=lambda *a: [], shard_dir=tmp_path, starting_shard_idx=0))
    flattened = {**pooled[0], **pooled[0]['extra_body']}
    for key in ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty', 'chat_template_kwargs'):
        assert direct[0][key] == flattened[key]
    assert flattened['model'] == 'served-alias'
    assert flattened['temperature'] == (1.0 if temperature is None else temperature)
    assert flattened['top_k'] == 64
