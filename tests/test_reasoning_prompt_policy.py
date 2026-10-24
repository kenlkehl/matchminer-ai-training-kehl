"""Exercise actual training prompt builders for GPT-OSS and other teachers."""
import json
from types import SimpleNamespace

import pytest

from tests.test_trial_selection import load_script
from tests.test_synthetic_generation import args_for
import generate_synthetic_data as generation
import llm_check_trials


class Tokenizer:
    chat_template = 'test-template'

    def __init__(self, model):
        self.name_or_path = model

    def apply_chat_template(self, conversation, **kwargs):
        if 'Beam' in self.name_or_path:
            assert kwargs['reasoning_effort'] == 'medium'
            assert 'enable_thinking' not in kwargs
        else:
            assert kwargs['enable_thinking'] is True
        return json.dumps(conversation)

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=self.encode(text))

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


@pytest.mark.parametrize('model,expected', [
    ('openai/gpt-oss-120b', True), ('openai/gpt-oss-20b', True),
    ('nvidia/Gemma-4-31B-IT-NVFP4', False),
    ('nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4', False),
    ('Qwen/Qwen3.6-27B', False), ('unknown/model', False),
    ('reflection/Beam-1.0-501B-A23B-NVFP4', False),
])
def test_generation_and_checking_headers_follow_teacher(model, expected, tmp_path):
    tokenizer = Tokenizer(model)
    notes = load_script('reasoning_policy_notes', '2_make_synthetic_notes_sharded.py')
    boilerplate = load_script('reasoning_policy_boilerplate', '14_check_boilerplate.py')
    spaces = load_script('reasoning_policy_spaces', 'legacy_trial_space_extraction.py')
    args = args_for(tmp_path)
    args.model = 'served-alias'
    args.tokenizer_model = model
    generator = generation.Generator(args, tokenizer)
    prompts = [
        notes.build_prompts(['Synthetic history.'], tokenizer)[0],
        boilerplate.build_prompts(['No infection.'], ['Active infection.'], tokenizer)[0],
        llm_check_trials.build_trial_check_prompt(tokenizer, 'Synthetic patient.', 'Synthetic trial.'),
        json.dumps(spaces.build_messages('Synthetic trial.', model)),
        generator.build_payload(dict(key='1', masked_text='Synthetic history.'), 'notes')['prompt'],
    ]
    for prompt in prompts:
        assert ('Reasoning: high' in json.loads(prompt)[0]['content']) is expected
