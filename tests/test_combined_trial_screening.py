"""Training shares the inference prompt while retaining old audit readers."""
import json

import pytest

from matchminer_ai.trials.cohort_schema import build_cohort_messages
from trial_treatment_screening import (
    build_screening_messages, parse_screening_output, screening_record,
)


def test_training_uses_the_same_combined_prompt():
    text = "Fabricated trial with literal {braces}."
    assert build_screening_messages(text) == build_cohort_messages(text)


@pytest.mark.parametrize("decision", [
    {"cancer_treatment_trial": "no"},
    {"cancer_treatment_trial": "yes", "multiple_cohorts": "no"},
    {"cancer_treatment_trial": "yes", "multiple_cohorts": "yes", "cohorts": ["A criteria", "B criteria"],
     "common_eligibility_criteria": "Shared inclusion and exclusion criteria."},
])
def test_live_adapter_retains_full_combined_result(decision):
    result = screening_record("", json.dumps(decision), "stop")
    assert result["is_cancer_treatment_trial"] == (decision["cancer_treatment_trial"] == "yes")
    assert json.loads(result["trial_cohort_output"]) == decision
    assert result["treatment_screen_rationale"].startswith("Combined prompt")


def test_historical_json_can_be_read_but_is_rejected_for_new_requests():
    old = '{"is_cancer_treatment_trial":false,"rationale":"Diagnostic only"}'
    assert parse_screening_output(old, "stop")["is_cancer_treatment_trial"] is False
    with pytest.raises(ValueError, match="Live requests require"):
        screening_record("", old, "stop")


def test_historical_v1_cohorts_are_readable_but_live_calls_require_shared_field():
    old = json.dumps({"cancer_treatment_trial": "yes", "multiple_cohorts": "yes",
                      "cohorts": ["Complete A criteria", "Complete B criteria"]})
    assert parse_screening_output(old, "stop")["is_cancer_treatment_trial"] is True
    with pytest.raises(ValueError, match="common_eligibility_criteria"):
        screening_record("", old, "stop")


@pytest.mark.parametrize("finish", ["length", "error", None])
def test_incomplete_new_response_is_not_an_exclusion(finish):
    with pytest.raises(ValueError):
        screening_record("", '{"cancer_treatment_trial":"no"}', finish)
