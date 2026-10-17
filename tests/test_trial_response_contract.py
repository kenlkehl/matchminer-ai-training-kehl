"""Keep aggregate labeling and individual questions on the current cached rubric."""
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
    monkeypatch.setattr("vllm_reasoning_utils.parse_reasoning_output",
                        lambda *args: ("", "Final score: 0"))
    model = SimpleNamespace(get_tokenizer=lambda: tokenizer, generate=generate)
    result = labeling.ask_about_trials_loosely([patient], [trial], model, "gemma4")
    source = c.load_prompt_sources()["llm_match_quality.user.txt"]
    assert generated == [remote_prompt] == [source.strip().format(
        patient_summary=patient, trial_summary=trial)]
    assert result[3:] == ([0], ["Score:0"])
    assert all(call["enable_thinking"] is True for call in calls)


@pytest.mark.parametrize("component", c.TRIAL_COMPONENTS)
def test_current_rules_are_scoped_to_one_binary_component_before_patient_data(component):
    sources = c.load_prompt_sources()
    messages = c.build_question_messages({
        "family": "trialchecking", "component": component,
        "patient_summary": "Fabricated patient: chemotherapy given; response undocumented.",
        "trial_summary": "Requires progression on chemotherapy or treatment intolerance.",
    }, sources)
    text = messages[-1]["content"]
    for owner in c.MATCHING_RULE_MARKERS:
        shared = c.trial_matching_rules(sources, owner)
        assert text.count(shared) == int(component == owner)
    specificity = c.trial_specificity_rules(sources)
    assert text.count(specificity) == int(component in c.SPECIFICITY_COMPONENTS)
    assert text.count("QUESTION\n") == 1
    assert "A. Yes\nB. No" in text and "Final score:" not in text
    assert "very next token" in messages[0]["content"]
    changed = c.build_question_messages({
        "family": "trialchecking", "component": component,
        "patient_summary": "Different fabricated patient",
        "trial_summary": "Different fabricated trial",
    }, sources)[-1]["content"]
    prefix, _, context = text.partition("PATIENT SUMMARY\n")
    assert prefix == changed.partition("PATIENT SUMMARY\n")[0]
    assert "DECISION RULE" in prefix and "OPTIONS" in prefix
    assert "Fabricated patient" in context and "TRIAL SPACE" in context
    assert context.endswith("Answer with the single letter first, then an explanation.")
    if component == "biomarker_matching":
        assert "standard of care" in prefix and "novel biomarkers" in prefix
        assert "unknown required biomarker alone is not a clear mismatch" not in prefix
    if component == "treatment_history_matching":
        assert "required response or progression" in prefix and "washout" in prefix
        assert "but not during the specified line, the trial is not reasonable" in prefix
        assert "patient received the treatment in that line, trial is not reasonable" in prefix
    if component == "prior_treatment_specificity":
        assert "Exposure in a different line of therapy is insufficient" in prefix


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
