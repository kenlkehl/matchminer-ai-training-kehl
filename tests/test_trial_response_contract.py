"""Keep aggregate labeling and individual response questions on one rubric."""
from types import SimpleNamespace
import sys

import pytest

import llm_check_trials as labeling
from oncoreasoning_training import contracts as c


def test_local_and_remote_scoring_render_the_canonical_prompt(monkeypatch):
    calls = []

    def render(**kwargs):
        calls.append(kwargs)
        return kwargs["conversation"][-1]["content"]

    tokenizer = SimpleNamespace(apply_chat_template=render)
    patient = "Fabricated patient: stable in January, progression in March. Marker {unknown}."
    trial = "Requires currently non-progressive metastatic Ewing sarcoma."
    remote_prompt = labeling.build_trial_check_prompt(tokenizer, patient, trial)
    generated = []

    def generate(prompts, params):
        generated.extend(prompts)
        return [SimpleNamespace(outputs=[SimpleNamespace(text="Final score: 0")])]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "vllm_reasoning_utils", SimpleNamespace(
        parse_reasoning_output=lambda *args: ("", "Final score: 0")))
    model = SimpleNamespace(get_tokenizer=lambda: tokenizer, generate=generate)
    result = labeling.ask_about_trials_loosely([patient], [trial], model, "gemma4")
    source = c.load_prompt_sources()["llm_match_quality.user.txt"]
    assert generated == [remote_prompt] == [source.strip().format(
        patient_summary=patient, trial_summary=trial)]
    assert result[3:] == ([0], ["Score:0"])
    assert all(call["enable_thinking"] is True for call in calls)


@pytest.mark.parametrize("component", c.TRIAL_COMPONENTS)
def test_response_rules_are_shared_only_with_relevant_binary_components(component):
    sources = c.load_prompt_sources()
    messages = c.build_question_messages({
        "family": "trialchecking", "component": component,
        "patient_summary": "Fabricated patient: chemotherapy given; response undocumented.",
        "trial_summary": "Requires progression on chemotherapy or treatment intolerance.",
    }, sources)
    text = messages[-1]["content"]
    shared = c.trial_response_rules(sources)
    assert text.count(shared) == int(component in c.RESPONSE_COMPONENTS)
    specificity = c.trial_specificity_rules(sources)
    assert text.count(specificity) == int(component in c.SPECIFICITY_COMPONENTS)
    assert c.SPECIFICITY_RULES_HEADER not in shared
    assert text.count("\nQUESTION\n") == 1
    assert "A. Yes\nB. No" in text and "Final score:" not in text
    assert "very next token" in messages[0]["content"]
    if component in c.RESPONSE_COMPONENTS:
        assert "Missing response information is not proof of either response or progression" in text
        assert "Changing or stopping treatment does not, by itself" in text
        assert "Progression and intolerance are not interchangeable" in text


def test_aggregate_labeling_rejects_an_unreviewed_rubric(monkeypatch):
    from importlib import resources
    altered = SimpleNamespace(read_text=lambda **kwargs: "unreviewed rubric")
    monkeypatch.setattr(resources, "files", lambda *args: SimpleNamespace(joinpath=lambda *args: altered))
    labeling.trial_check_template.cache_clear()
    try:
        with pytest.raises(ValueError, match="TrialChecker rubric changed"):
            labeling.trial_check_template()
    finally:
        labeling.trial_check_template.cache_clear()
