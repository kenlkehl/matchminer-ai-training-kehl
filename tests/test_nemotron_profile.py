import importlib.util
from pathlib import Path

from vllm_reasoning_utils import resolve_parser_name, resolve_sampling_defaults, resolve_tool_call_parser

MODEL = 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4'

def test_training_uses_shared_model_profile():
    assert resolve_parser_name(MODEL) == 'nemotron_v3'
    assert resolve_tool_call_parser(MODEL) == 'qwen3_coder'
    assert resolve_parser_name(MODEL, 'custom') == 'custom'
    settings = resolve_sampling_defaults(MODEL, temperature=0.3, top_k=None)
    assert settings['temperature'] == 0.3
    assert settings['top_p'] == 0.95 and settings['top_k'] == -1
    assert settings['presence_penalty'] == 0.0


def test_remote_trial_spaces_do_not_reparse_reasoning(monkeypatch):
    spec = importlib.util.spec_from_file_location('trial_spaces', Path(__file__).parents[1] / 'legacy_trial_space_extraction.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def must_not_parse(*args):
        raise AssertionError('Final answer was parsed a second time')
    monkeypatch.setattr('vllm_reasoning_utils.parse_reasoning_output', must_not_parse)
    final = '1. Age range allowed: 18+.\nBoilerplate exclusions:\nActive infection.'
    answer, spaces, exclusions = module.postprocess_outputs([final], 'nemotron_v3', None, already_parsed=True)
    assert answer == [final]
    assert spaces == ['1. Age range allowed: 18+.\n']
    assert exclusions == ['Active infection.']


def test_primary_runner_uses_nemotron_parsers():
    import train_from_summaries as runner
    args = runner.parser().parse_args(['--teacher-model', MODEL])
    command = runner.teacher_command(args, 0)
    assert command[command.index('--reasoning-parser') + 1] == 'nemotron_v3'
    assert command[command.index('--tool-call-parser') + 1] == 'qwen3_coder'
    labels = [stage for stage in runner.stages(args) if stage.name.startswith('label-')]
    assert labels
    for stage in labels:
        # The ColBERT labeler resolves the parser from the underlying tokenizer,
        # independently of any endpoint's served-model alias.
        tokenizer = stage.command[stage.command.index('--teacher-tokenizer') + 1]
        assert tokenizer == MODEL
        assert resolve_parser_name(tokenizer) == 'nemotron_v3'


def test_teacher_sampling_preserves_explicit_temperature(monkeypatch):
    from oncoreasoning_training import teacher
    monkeypatch.setattr(teacher, 'EndpointPool', lambda *a, **k: None)
    pool = teacher.TeacherPool(model=MODEL)
    assert pool.parameters['temperature'] == 1.0
    assert pool.parameters['top_p'] == 0.95
    assert pool.parameters['extra_body']['top_k'] == -1
    assert teacher.TeacherPool(model=MODEL, temperature=0.2).parameters['temperature'] == 0.2
    assert teacher.TeacherPool().parameters['temperature'] == 1.0
    assert teacher.TeacherPool(model='unregistered-model').parameters['temperature'] == 0.2
