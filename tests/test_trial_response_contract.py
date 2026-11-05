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
    # The legacy aggregate labeler keeps the single-score rubric, read as the package ships it.
    source = (c.DEFAULT_INFERENCE_REPO / "src/matchminer_ai/prompts/llm_match_quality.user.txt").read_text(encoding="utf-8")
    assert generated == [remote_prompt] == [source.strip().format(
        patient_summary=patient, trial_summary=trial)]
    assert result[3:] == ([0], ["Score:0"])
    assert all(call["enable_thinking"] is True for call in calls)


@pytest.mark.parametrize("component", c.TRIAL_COMPONENTS)
def test_trialchecker_questions_are_the_five_match_labels_on_key_text_one_per_request(component):
    sources = c.load_prompt_sources()
    preamble, rules = c.match_label_rules(sources)
    messages = c.build_question_messages({
        "family": "trialchecking", "component": component,
        "patient_summary": "Fabricated patient: chemotherapy given; response undocumented.",
        "trial_summary": "Requires progression on chemotherapy or treatment intolerance.",
    }, sources)
    text = messages[-1]["content"]
    question, rule = rules[component]
    # Cut verbatim from the teacher's five-answer prompt: this label's question and rules only.
    assert f"QUESTION\n{question}\n" in text and rule in text and preamble in text
    for other, (_, other_rule) in rules.items():
        assert (other_rule in text) == (other == component)
    assert "Answer five questions" not in text and "OUTPUT FORMAT" not in text
    assert "A. Yes\nB. No" in text and "very next token" in messages[0]["content"]
    changed = c.build_question_messages({
        "family": "trialchecking", "component": component,
        "patient_summary": "Different fabricated patient", "trial_summary": "Different fabricated trial",
    }, sources)[-1]["content"]
    prefix, _, context = text.partition("PATIENT HISTORY\n")
    assert prefix == changed.partition("PATIENT HISTORY\n")[0]  # instructions first: a shared cacheable prefix
    assert "COHORT KEY ELIGIBILITY CRITERIA" in context
    assert context.endswith("Answer with the single letter first, then an explanation.")


def test_boilerplatechecker_questions_take_each_boilerplate_line_in_context():
    sources = c.load_prompt_sources()
    criteria = "Inclusion criteria:\nECOG 0-1.\nExclusion criteria:\nAny of the following:\n  - Active hepatitis B."
    text = c.build_question_messages(dict(family="boilerplatechecking", component="x", item_number=3,
        patient_boilerplate="ECOG 1. Hepatitis B core antibody positive, DNA undetectable.",
        boilerplate_criteria=criteria), sources)[-1]["content"]
    assert ("COHORT BOILERPLATE CRITERIA\nInclusion criteria:\n[1] ECOG 0-1.\nExclusion criteria:\n"
            "[2] Any of the following:\n[3]   - Active hepatitis B.") in text
    assert text.endswith("CRITERION TO ASSESS\n[3] - Active hepatitis B.\n\nAnswer with the single letter first, then an explanation.")
    assert "inclusion requirements as well as exclusions" in text and "Missing or undocumented information is not a failure" in text
    assert "PATIENT OTHER RELEVANT HISTORY\nECOG 1." in text


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


@pytest.mark.parametrize("root", ["nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
                                  "RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8"])
def test_gated_labels_accept_either_approved_nemotron_quantization(root):
    from trialspace.labeling import validate_served_teacher
    served = [dict(id="nemotron-3.5-super-vl-120b", root=root)]
    assert validate_served_teacher(served, requested="nemotron-3.5-super-vl-120b",
                                   tokenizer="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")["root"] == root


def test_gated_labels_reject_a_teacher_from_another_model_family():
    from trialspace.labeling import validate_served_teacher
    served = [dict(id="nemotron-3.5-super-vl-120b", root="nvidia/Gemma-4-31B-IT-NVFP4")]
    with pytest.raises(ValueError, match="differs from the configured model family"):
        validate_served_teacher(served, requested="nemotron-3.5-super-vl-120b",
                                tokenizer="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
