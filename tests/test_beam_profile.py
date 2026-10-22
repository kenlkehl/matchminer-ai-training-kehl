"""Training consumers use the shared Beam profile without changing endpoints."""
from types import SimpleNamespace

import pytest

from vllm_reasoning_utils import resolve_parser_name, resolve_sampling_defaults, resolve_tool_call_parser, sampling_from_args

MODEL = "reflection/Beam-1.0-501B-A23B-NVFP4"


@pytest.mark.parametrize("model", [MODEL, MODEL.replace("NVFP4", "FP8"), "Beam-1.0-501B-A23B", "reflection-beam", "beam"])
def test_beam_defaults_replace_task_fallbacks_and_resolve_parsers(model):
    result = resolve_sampling_defaults(model, fallback=dict(temperature=1.0, top_p=.95,
        top_k=64, presence_penalty=1.5, repetition_penalty=1.1))
    assert result == dict(temperature=.7, top_p=.99, top_k=-1, min_p=0.0,
                          presence_penalty=0.0, repetition_penalty=1.0)
    assert resolve_parser_name(model) == "beam"
    assert resolve_tool_call_parser(model) == "beam"
    assert resolve_parser_name(model, "custom") == "custom"


def test_alias_tokenizer_model_and_explicit_zero_override():
    args = SimpleNamespace(model="opaque", tokenizer_model=MODEL, temperature=0.0)
    assert sampling_from_args(args)["temperature"] == 0.0
    assert sampling_from_args(args)["top_p"] == .99
    assert sampling_from_args(args)["top_k"] == -1


def test_beam_teacher_request_uses_shared_profile(monkeypatch):
    from oncoreasoning_training import teacher
    captured = []
    monkeypatch.setattr(teacher, 'EndpointPool', lambda *a, **kw: SimpleNamespace(
        request=lambda **request: captured.append(request)))
    teacher.TeacherPool(model=MODEL)([{"role": "user", "content": "Synthetic input"}])
    request = captured[0]
    assert request['model'] == MODEL
    assert request['temperature'] == .7 and request['top_p'] == .99
    assert request['extra_body']['top_k'] == -1
    assert 'reasoning_effort' not in request
    assert request['extra_body']['chat_template_kwargs'] == {'reasoning_effort': 'medium'}


def test_direct_extractor_request_uses_beam_profile_and_served_alias(monkeypatch):
    from synthetic_extractors import vllm_labeling_common as common
    captured = []

    def http(method, url, **kwargs):
        captured.append(kwargs['payload'])
        return {'choices': [{'message': {'content': '{}'}}]}

    monkeypatch.setattr(common, 'http_json', http)
    common.chat_completion(base_url='http://test/v1', model='opaque-alias',
        sampling_model=MODEL, messages=[], temperature=None, max_tokens=100,
        timeout=1, api_key=None, json_mode=True)
    request = captured[0]
    assert request['model'] == 'opaque-alias'
    assert request['temperature'] == .7 and request['top_p'] == .99
    assert request['top_k'] == -1 and request['repetition_penalty'] == 1.0
    assert request['chat_template_kwargs'] == {'reasoning_effort': 'medium'}


def test_primary_runner_selects_beam_reasoning_and_tool_parsers():
    import train_from_summaries as runner
    args = runner.parser().parse_args(['--teacher-model', MODEL])
    command = runner.teacher_command(args, 0)
    assert command[command.index('--reasoning-parser') + 1] == 'beam'
    assert command[command.index('--tool-call-parser') + 1] == 'beam'


def test_direct_extractor_server_selects_beam_parser_and_keeps_overrides():
    from synthetic_extractors import vllm_labeling_common as common

    args = SimpleNamespace(model=MODEL, reasoning_parser="auto")
    assert common.guess_reasoning_parser(MODEL) == "beam"
    assert common.inject_reasoning_args(args, []) == ["--reasoning-parser", "beam"]
    explicit = ["--reasoning-parser", "custom"]
    assert common.inject_reasoning_args(args, explicit) == explicit
    args.reasoning_parser = "none"
    assert common.inject_reasoning_args(args, []) == []


def test_beam_template_effort_defaults_and_overrides():
    from vllm_reasoning_utils import resolve_chat_template_kwargs
    assert resolve_chat_template_kwargs(MODEL) == {"reasoning_effort": "medium"}
    assert resolve_chat_template_kwargs(MODEL, enable_thinking=False) == {"reasoning_effort": "low"}
    assert resolve_chat_template_kwargs(MODEL, reasoning_effort="max") == {"reasoning_effort": "max"}


@pytest.mark.parametrize('stage', ['screening', 'extraction'])
def test_beam_checkpoint_manifest_rejects_old_effort(tmp_path, stage):
    import json
    from tests.test_trial_selection import load_script
    from trial_treatment_screening import check_screening_manifest

    args = SimpleNamespace(model=MODEL, work_dir=str(tmp_path), max_model_len=100000,
        temperature=.7, top_p=.99, top_k=-1, min_p=0., presence_penalty=0.,
        repetition_penalty=1., max_tokens=32000, screening_max_tokens=8000)
    if stage == 'screening':
        check = lambda: check_screening_manifest(args, 'beam', tmp_path)
    else:
        script = load_script('beam_manifest_spaces', '0b_create_trial_spaces.py')
        check = lambda: script.check_extraction_manifest(args, 'beam')
    check()
    path = tmp_path / f'{stage}_manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest['chat_template_kwargs'] == {'reasoning_effort': 'medium'}
    check()
    manifest['chat_template_kwargs']['reasoning_effort'] = 'max'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='changed'):
        check()
