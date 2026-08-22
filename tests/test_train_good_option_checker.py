from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import train_good_option_checker as good_option
from remote_vllm_pool import (
    CompletionSampling,
    DynamicServerRegistry,
    make_completion_work_fn,
)


STUDY = {
    "protocolSection": {
        "identificationModule": {"briefTitle": "Drug A plus Drug B study"},
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {"phases": ["PHASE2"]},
        "descriptionModule": {"briefSummary": "A combination study."},
        "armsInterventionsModule": {
            "interventions": [
                {
                    "type": "DRUG",
                    "name": "Drug A",
                    "description": "A targeted agent.",
                    "otherNames": ["Agent A"],
                },
                {
                    "type": "BIOLOGICAL",
                    "name": "Drug B",
                    "description": "An antibody.",
                },
                {"type": "DRUG", "name": "Placebo"},
                {"type": "PROCEDURE", "name": "Tumor biopsy"},
            ]
        },
    }
}


RCT_STUDY = {
    "protocolSection": {
        "identificationModule": {"briefTitle": "Novel-X versus standard therapy"},
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {"phases": ["PHASE3"]},
        "descriptionModule": {"briefSummary": "A randomized treatment study."},
        "armsInterventionsModule": {
            "armGroups": [
                {
                    "label": "Experimental arm",
                    "type": "EXPERIMENTAL",
                    "description": "Participants receive Novel-X.",
                    "interventionNames": ["DRUG: Novel-X"],
                },
                {
                    "label": "Standard-of-care arm",
                    "type": "ACTIVE_COMPARATOR",
                    "description": "Participants receive Standard-Y.",
                    "interventionNames": ["DRUG: Standard-Y"],
                },
            ],
            "interventions": [
                {
                    "type": "DRUG",
                    "name": "Novel-X tablets",
                    "description": "Novel-X is administered orally.",
                    "armGroupLabels": ["Experimental arm"],
                },
                {
                    "type": "DRUG",
                    "name": "Standard-Y",
                    "description": "Standard systemic therapy.",
                    "armGroupLabels": ["Standard-of-care arm"],
                },
            ],
        },
    }
}


def _candidate_frame(*, duplicate: bool = False) -> pd.DataFrame:
    rows = [
        {
            "patient_summary": "Synthetic patient with cancer A.",
            "nct_id": "NCT12345678",
            "this_space": "1. Cancer A treated with Drug A.",
            "split": "train",
        },
        {
            "patient_summary": "Synthetic patient with cancer B.",
            "nct_id": "NCT87654321",
            "this_space": "Cancer B treated with Drug B.",
            "split": "train",
        },
    ]
    if duplicate:
        rows.append(dict(rows[0]))
    return pd.DataFrame(rows)


def _rubric_response() -> dict[str, object]:
    return {
        "patient_disease_type": "Cancer A",
        "drug_assessments": [
            {
                "drug_name": "Drug A",
                "targeted_biomarkers": ["Marker A"],
                "disease_type_benefit": {
                    "point": 1,
                    "rationale": "Human benefit was reported in Cancer A.",
                    "evidence_labels": ["S1"],
                },
                "common_biomarker_in_disease": {
                    "point": 1,
                    "rationale": "Marker A is common in Cancer A.",
                    "evidence_labels": ["S2", "INVENTED"],
                },
                "patient_biomarker_targeted": {
                    "point": 1,
                    "rationale": "The tumor documents Marker A and Drug A targets it.",
                    "evidence_labels": ["PATIENT", "CT"],
                },
                "biomarker_targeted_benefit": {
                    "point": 0,
                    "rationale": (
                        "No human biomarker-directed benefit evidence was supplied."
                    ),
                    "evidence_labels": [],
                },
            }
        ],
        "key_uncertainties": ["Small study"],
    }


def test_drug_query_api_structurally_excludes_patient_context() -> None:
    query_parameters = inspect.signature(
        good_option.build_drug_search_queries
    ).parameters
    experimental_query_parameters = inspect.signature(
        good_option.build_experimental_drug_search_queries
    ).parameters
    normalization_parameters = inspect.signature(
        good_option.build_drug_name_normalization_messages
    ).parameters
    expression_parameters = inspect.signature(
        good_option.build_biomarker_expression_search_queries
    ).parameters
    research_parameters = inspect.signature(good_option.research_trials).parameters
    enrichment_parameters = inspect.signature(
        good_option.enrich_trial_with_biomarker_expression_research
    ).parameters

    assert list(query_parameters) == ["interventions"]
    assert list(experimental_query_parameters) == ["interventions"]
    assert list(normalization_parameters) == ["interventions"]
    assert list(expression_parameters) == ["interventions"]
    assert "patient_summary" not in research_parameters
    assert "patient_history" not in research_parameters
    assert "patient_summary" not in enrichment_parameters
    assert "clinical_space_summary" not in enrichment_parameters
    assert good_option.research_trials.__module__ == "matchminer_ai.help_me_choose"


def test_teacher_canonicalizes_registry_qualifiers_without_inventing_names() -> None:
    interventions = (
        good_option.DrugIntervention(
            name="Dose-Escalation (Part One) Agent-X capsule",
            intervention_type="DRUG",
            description="Participants receive Agent-X orally.",
        ),
        good_option.DrugIntervention(
            name="Dose-Expansion (Part Two) Agent-X capsule",
            intervention_type="DRUG",
        ),
        good_option.DrugIntervention(
            name="Combination cohort: Agent-Y plus Agent-Z infusion",
            intervention_type="DRUG",
        ),
    )
    response = {
        "interventions": [
                {
                    "source_index": 0,
                    "experimental_role": "investigational",
                    "canonical_drug_names": ["Agent-X"],
                "rationale": "Agent-X appears in the registry name.",
            },
                {
                    "source_index": 1,
                    "experimental_role": "investigational",
                    "canonical_drug_names": ["Agent-X"],
                "rationale": "Agent-X appears in the registry name.",
            },
                {
                    "source_index": 2,
                    "experimental_role": "investigational",
                    "canonical_drug_names": ["Agent-Y", "Agent-Z"],
                "rationale": "Both names appear in the registry name.",
            },
        ]
    }

    parsed = good_option.parse_drug_name_normalization_response(
        json.dumps(response),
        nct_id="NCT12345678",
        interventions=interventions,
    )

    assert parsed.status == "ok"
    assert [item.name for item in parsed.canonical_interventions] == [
        "Agent-X",
        "Agent-Y",
        "Agent-Z",
    ]
    assert all("Dose-" not in item.name for item in parsed.canonical_interventions)
    assert json.loads(parsed.mappings_json)[1]["canonical_drug_names"] == ["Agent-X"]


def test_registry_arm_context_forces_comparator_only_drug_exclusion() -> None:
    interventions = good_option.extract_registry_drug_interventions(RCT_STUDY)

    assert good_option._registry_arm_types(interventions[0]) == {"EXPERIMENTAL"}
    assert good_option._registry_arm_types(interventions[1]) == {
        "ACTIVE_COMPARATOR"
    }
    prompt = good_option.build_drug_name_normalization_messages(interventions)
    assert "Standard-of-care arm" in prompt[1]["content"]

    response = {
        "interventions": [
            {
                "source_index": 0,
                "experimental_role": "investigational",
                "canonical_drug_names": ["Novel-X"],
                "rationale": "Novel-X is assigned to the experimental arm.",
            },
            {
                "source_index": 1,
                "experimental_role": "investigational",
                "canonical_drug_names": ["Standard-Y"],
                "rationale": "Incorrectly selected.",
            },
        ]
    }

    parsed = good_option.parse_drug_name_normalization_response(
        json.dumps(response),
        nct_id="NCT12345678",
        interventions=interventions,
    )

    assert [item.name for item in parsed.canonical_interventions] == ["Novel-X"]
    mappings = json.loads(parsed.mappings_json)
    assert mappings[1]["experimental_role"] == "not_investigational"
    assert mappings[1]["control_only_exclusion"] is True
    assert mappings[1]["canonical_drug_names"] == []


def test_registry_fetch_retains_public_arm_context_without_web_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_ids: list[str] = []

    async def fake_fetch(nct_id: str, *, client: object):
        del client
        requested_ids.append(nct_id)
        return RCT_STUDY

    monkeypatch.setattr(good_option.help_me_choose, "fetch_trial_study", fake_fetch)
    records = asyncio.run(
        good_option.fetch_trial_registry_research(
            ["NCT12345678"],
            max_concurrency=1,
            request_timeout=1,
        )
    )

    assert requested_ids == ["NCT12345678"]
    assert len(records) == 1
    assert good_option._registry_arm_types(records[0].interventions[0]) == {
        "EXPERIMENTAL"
    }
    assert good_option._registry_arm_types(records[0].interventions[1]) == {
        "ACTIVE_COMPARATOR"
    }
    assert records[0].search_results == ()


def test_registry_fetch_retries_http_429_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def fake_fetch(nct_id: str, *, client: object):
        nonlocal attempts
        del client
        attempts += 1
        if attempts == 1:
            request = good_option.httpx.Request(
                "GET",
                f"https://clinicaltrials.gov/api/v2/studies/{nct_id}",
            )
            response = good_option.httpx.Response(
                429,
                headers={"Retry-After": "0"},
                request=request,
            )
            raise good_option.httpx.HTTPStatusError(
                "Too Many Requests",
                request=request,
                response=response,
            )
        return RCT_STUDY

    monkeypatch.setattr(good_option.help_me_choose, "fetch_trial_study", fake_fetch)
    records = asyncio.run(
        good_option.fetch_trial_registry_research(
            ["NCT12345678"],
            max_concurrency=1,
            request_timeout=1,
            max_attempts=2,
            minimum_request_interval=0,
            initial_backoff=0,
            maximum_backoff=0,
        )
    )

    assert attempts == 2
    assert good_option.research_status(records[0]) != "registry_lookup_failed"
    assert records[0].title == "Novel-X versus standard therapy"


def test_registry_fetch_does_not_retry_nontransient_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def fake_fetch(nct_id: str, *, client: object):
        nonlocal attempts
        del client
        attempts += 1
        request = good_option.httpx.Request(
            "GET",
            f"https://clinicaltrials.gov/api/v2/studies/{nct_id}",
        )
        response = good_option.httpx.Response(404, request=request)
        raise good_option.httpx.HTTPStatusError(
            "Not Found",
            request=request,
            response=response,
        )

    monkeypatch.setattr(good_option.help_me_choose, "fetch_trial_study", fake_fetch)
    records = asyncio.run(
        good_option.fetch_trial_registry_research(
            ["NCT12345678"],
            max_concurrency=1,
            request_timeout=1,
            max_attempts=10,
            minimum_request_interval=0,
        )
    )

    assert attempts == 1
    assert good_option.research_status(records[0]) == "registry_lookup_failed"


@pytest.mark.parametrize(
    ("reasoning_parser", "expected_thinking"),
    [("qwen3", True), ("gemma4", False)],
)
def test_drug_normalization_enables_thinking_for_qwen_only(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_parser: str,
    expected_thinking: bool,
) -> None:
    thinking_values: list[bool] = []

    class FakeTokenizer:
        def apply_chat_template(
            self,
            conversation,
            *,
            add_generation_prompt,
            tokenize,
            enable_thinking,
        ):
            del conversation, add_generation_prompt, tokenize
            thinking_values.append(enable_thinking)
            return "rendered prompt"

    response = {
        "interventions": [
            {
                "source_index": 0,
                "experimental_role": "investigational",
                "canonical_drug_names": ["Novel-X"],
                "rationale": "The registry assigns Novel-X to the experimental arm.",
            },
            {
                "source_index": 1,
                "experimental_role": "not_investigational",
                "canonical_drug_names": [],
                "rationale": "Standard-Y is an active comparator.",
            },
        ]
    }

    async def fake_run_pool(*, work_items, shard_writer, **_kwargs):
        shard_writer(
            [(work_items[0][0], ("private reasoning", json.dumps(response)))],
            0,
        )
        return 1

    monkeypatch.setattr("remote_vllm_pool.run_pool", fake_run_pool)
    registry_item = good_option.trial_registry_research_from_study(
        "NCT12345678",
        RCT_STUDY,
    )
    runtime = good_option.TeacherRuntime(
        tokenizer=FakeTokenizer(),
        registry=object(),
        work_fn=object(),
        local_servers=[],
        reasoning_parser=reasoning_parser,
    )

    parsed = asyncio.run(
        good_option.canonicalize_trial_interventions_with_teacher(
            [registry_item],
            runtime=runtime,
            max_new_tokens=8000,
            max_attempts=2,
        )
    )

    assert thinking_values == [expected_thinking]
    assert parsed["NCT12345678"].status == "ok"
    assert [item.name for item in parsed["NCT12345678"].canonical_interventions] == [
        "Novel-X"
    ]


def test_qwen_normalization_repairs_empty_final_answer_with_larger_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thinking_values: list[bool] = []
    rendered_conversations: list[list[dict[str, str]]] = []
    requested_tokens: list[int] = []

    class FakeTokenizer:
        def apply_chat_template(
            self,
            conversation,
            *,
            add_generation_prompt,
            tokenize,
            enable_thinking,
        ):
            del add_generation_prompt, tokenize
            thinking_values.append(enable_thinking)
            rendered_conversations.append(conversation)
            return "rendered prompt"

    response = {
        "interventions": [
            {
                "source_index": 0,
                "experimental_role": "investigational",
                "canonical_drug_names": ["Novel-X"],
                "rationale": "Novel-X is assigned to the experimental arm.",
            },
            {
                "source_index": 1,
                "experimental_role": "not_investigational",
                "canonical_drug_names": [],
                "rationale": "Standard-Y is the active comparator.",
            },
        ]
    }
    pool_calls = 0

    async def fake_run_pool(*, work_items, shard_writer, **_kwargs):
        nonlocal pool_calls
        pool_calls += 1
        requested_tokens.append(work_items[0][1]["max_tokens"])
        if pool_calls == 1:
            result = (
                "r" * 8_000,
                "",
                {"finish_reason": "length", "raw_text_char_count": 8_000},
            )
        else:
            result = (
                "concise reasoning",
                json.dumps(response),
                {"finish_reason": "stop", "raw_text_char_count": 900},
            )
        shard_writer([(work_items[0][0], result)], 0)
        return 1

    monkeypatch.setattr("remote_vllm_pool.run_pool", fake_run_pool)
    registry_item = good_option.trial_registry_research_from_study(
        "NCT12345678",
        RCT_STUDY,
    )
    runtime = good_option.TeacherRuntime(
        tokenizer=FakeTokenizer(),
        registry=object(),
        work_fn=object(),
        local_servers=[],
        reasoning_parser="qwen3",
    )

    parsed = asyncio.run(
        good_option.canonicalize_trial_interventions_with_teacher(
            [registry_item],
            runtime=runtime,
            max_new_tokens=8_000,
            retry_max_new_tokens=24_000,
            parse_retries=1,
            max_attempts=2,
        )
    )["NCT12345678"]

    assert thinking_values == [True, True]
    assert requested_tokens == [8_000, 24_000]
    assert "previous attempt produced no final answer" in str(
        rendered_conversations[1]
    ).lower()
    assert parsed.status == "ok"
    assert parsed.attempt_count == 2
    assert parsed.finish_reason == "stop"
    attempts = json.loads(parsed.attempts_json)
    assert [item["status"] for item in attempts] == ["parse_failed", "ok"]
    assert attempts[0]["finish_reason"] == "length"
    assert attempts[0]["reasoning_char_count"] == 8_000


def test_experimental_arm_drug_is_preserved_after_many_comparators() -> None:
    study = json.loads(json.dumps(RCT_STUDY))
    module = study["protocolSection"]["armsInterventionsModule"]
    controls = [
        {
            "type": "DRUG",
            "name": f"Standard-{index}",
            "armGroupLabels": ["Standard-of-care arm"],
        }
        for index in range(8)
    ]
    novel = module["interventions"][0]
    module["interventions"] = [*controls, novel]

    interventions = good_option.extract_registry_drug_interventions(study)

    assert interventions[0].name == "Novel-X tablets"
    assert "Novel-X tablets" in {item.name for item in interventions}


def test_noninvestigational_standard_background_drug_is_not_searched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interventions = good_option.extract_registry_drug_interventions(RCT_STUDY)
    registry = good_option.trial_registry_research_from_study(
        "NCT12345678",
        RCT_STUDY,
    )
    normalization = good_option.parse_drug_name_normalization_response(
        json.dumps(
            {
                "interventions": [
                    {
                        "source_index": 0,
                        "experimental_role": "investigational",
                        "canonical_drug_names": ["Novel-X"],
                        "rationale": "Experimental arm agent.",
                    },
                    {
                        "source_index": 1,
                        "experimental_role": "not_investigational",
                        "canonical_drug_names": [],
                        "rationale": "Active comparator only.",
                    },
                ]
            }
        ),
        nct_id=registry.nct_id,
        interventions=interventions,
    )
    captured_queries: list[str] = []

    def fake_search(queries):
        captured_queries.extend(queries)
        return (), ()

    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(good_option.asyncio, "to_thread", run_inline)
    researched = asyncio.run(
        good_option.research_canonical_drug_names(
            registry,
            normalization,
            search_function=fake_search,
        )
    )

    assert [item.name for item in researched.interventions] == ["Novel-X"]
    assert captured_queries == [
        '"Novel-X" oncology mechanism efficacy safety clinical trial'
    ]
    assert all("Standard-Y" not in query for query in captured_queries)


def test_control_only_trial_is_terminal_without_web_search() -> None:
    comparator = good_option.extract_registry_drug_interventions(RCT_STUDY)[1]
    registry = good_option.TrialDrugResearch(
        nct_id="NCT12345678",
        interventions=(comparator,),
    )
    normalization = good_option.parse_drug_name_normalization_response(
        json.dumps(
            {
                "interventions": [
                    {
                        "source_index": 0,
                        "experimental_role": "investigational",
                        "canonical_drug_names": ["Standard-Y"],
                        "rationale": "Incorrect teacher selection.",
                    }
                ]
            }
        ),
        nct_id=registry.nct_id,
        interventions=registry.interventions,
    )
    captured_queries: list[str] = []

    def fake_search(queries):
        captured_queries.extend(queries)
        return (), ()

    researched = asyncio.run(
        good_option.research_canonical_drug_names(
            registry,
            normalization,
            search_function=fake_search,
        )
    )

    assert normalization.status == "no_experimental_interventions"
    assert researched.interventions == ()
    assert captured_queries == []
    assert good_option.research_status(researched) == (
        "no_experimental_drug_intervention"
    )


def test_valid_all_uncertain_answer_is_terminal_not_a_quality_failure() -> None:
    intervention = good_option.DrugIntervention(
        name="HSCT with conditioning regimen",
        intervention_type="BIOLOGICAL",
    )
    registry = good_option.TrialDrugResearch(
        nct_id="NCT12345678",
        interventions=(intervention,),
    )
    normalization = good_option.parse_drug_name_normalization_response(
        json.dumps(
            {
                "interventions": [
                    {
                        "source_index": 0,
                        "experimental_role": "uncertain",
                        "canonical_drug_names": [],
                        "rationale": (
                            "The registry does not identify a named experimental "
                            "drug in this regimen."
                        ),
                    }
                ]
            }
        ),
        nct_id=registry.nct_id,
        interventions=registry.interventions,
    )
    captured_queries: list[str] = []

    def fake_search(queries):
        captured_queries.extend(queries)
        return (), ()

    good_option.validate_normalization_batch(
        [registry],
        {registry.nct_id: normalization},
        maximum_failure_fraction=0.0,
        context="all-uncertain test",
    )
    researched = asyncio.run(
        good_option.research_canonical_drug_names(
            registry,
            normalization,
            search_function=fake_search,
        )
    )
    record = good_option.research_to_record(
        researched,
        normalization=normalization,
    )
    summary = good_option.validate_research_quality(
        [record],
        maximum_registry_failure_fraction=0.0,
        maximum_normalization_failure_fraction=0.0,
        context="all-uncertain test",
    )

    assert normalization.status == "no_identifiable_experimental_drug"
    assert captured_queries == []
    assert good_option.research_status(researched) == (
        "no_experimental_drug_intervention"
    )
    assert summary.normalization_failures == 0


def test_trial_without_structured_drug_is_terminal_without_web_search() -> None:
    registry = good_option.TrialDrugResearch(nct_id="NCT12345678")
    normalization = good_option.parse_drug_name_normalization_response(
        "",
        nct_id=registry.nct_id,
        interventions=(),
    )

    def unexpected_search(_queries):
        raise AssertionError("A no-drug trial must not issue a web search.")

    researched = asyncio.run(
        good_option.research_canonical_drug_names(
            registry,
            normalization,
            search_function=unexpected_search,
        )
    )

    assert normalization.status == "no_interventions"
    assert researched.interventions == ()
    assert good_option.research_status(researched) == (
        "no_experimental_drug_intervention"
    )


def test_teacher_name_not_supported_by_registry_text_falls_back() -> None:
    intervention = good_option.DrugIntervention(
        name="Dose cohort: Agent-X tablet",
        intervention_type="DRUG",
    )
    response = {
        "interventions": [
            {
                "source_index": 0,
                "experimental_role": "investigational",
                "canonical_drug_names": ["Invented-Y"],
                "rationale": "Unsupported guess.",
            }
        ]
    }

    parsed = good_option.parse_drug_name_normalization_response(
        json.dumps(response),
        nct_id="NCT12345678",
        interventions=(intervention,),
    )

    assert parsed.status == "partial_fallback"
    assert parsed.canonical_interventions == (intervention,)
    assert "unsupported name" in parsed.parse_error
    assert json.loads(parsed.mappings_json)[0]["used_fallback"] is True


def test_canonical_drug_names_become_literal_search_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_intervention = good_option.DrugIntervention(
        name="Dose-Escalation (Part One) Agent-X capsule",
        intervention_type="DRUG",
    )
    registry = good_option.TrialDrugResearch(
        nct_id="NCT12345678",
        interventions=(registry_intervention,),
    )
    normalization = good_option.parse_drug_name_normalization_response(
        json.dumps(
            {
                "interventions": [
                    {
                        "source_index": 0,
                        "experimental_role": "investigational",
                        "canonical_drug_names": ["Agent-X"],
                        "rationale": "Supported by the registry name.",
                    }
                ]
            }
        ),
        nct_id=registry.nct_id,
        interventions=registry.interventions,
    )
    captured_queries: list[str] = []

    def fake_search(queries):
        captured_queries.extend(queries)
        return (), ()

    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(good_option.asyncio, "to_thread", run_inline)

    researched = asyncio.run(
        good_option.research_canonical_drug_names(
            registry,
            normalization,
            search_function=fake_search,
        )
    )

    assert captured_queries == [
        '"Agent-X" oncology mechanism efficacy safety clinical trial'
    ]
    assert [item.name for item in researched.interventions] == ["Agent-X"]
    assert all("Dose-Escalation" not in query for query in captured_queries)


def test_biomarker_expression_queries_use_only_drug_names() -> None:
    queries = good_option.build_biomarker_expression_search_queries(
        good_option.extract_drug_interventions(STUDY)
    )

    assert queries == (
        '"Drug A" oncology molecular target biomarker expression prevalence '
        "across cancer types",
        '"Drug B" oncology molecular target biomarker expression prevalence '
        "across cancer types",
    )


def test_web_queries_are_chunked_so_later_drugs_are_not_starved() -> None:
    calls: list[tuple[str, ...]] = []

    def fake_search(queries):
        calls.append(tuple(queries))
        return (
            tuple(
                good_option.DrugSearchResult(
                    query=query,
                    title=query,
                    snippet="Evidence",
                    url=f"https://example.org/{index}-{len(calls)}",
                )
                for index, query in enumerate(queries)
            ),
            (),
        )

    queries = tuple(f"drug-{index}" for index in range(7))
    results, notices = good_option._search_query_chunks(fake_search, queries)

    assert [len(call) for call in calls] == [3, 3, 1]
    assert [item.query for item in results] == list(queries)
    assert notices == ()


def test_training_queries_preserve_every_selected_canonical_drug() -> None:
    interventions = tuple(
        good_option.DrugIntervention(f"Experimental-{index}", "DRUG")
        for index in range(10)
    )

    baseline = good_option.build_experimental_drug_search_queries(interventions)
    expression = good_option.build_biomarker_expression_search_queries(interventions)

    assert len(baseline) == 10
    assert len(expression) == 10
    assert "Experimental-9" in baseline[-1]
    assert "Experimental-9" in expression[-1]


def test_extracts_only_non_placebo_drug_interventions() -> None:
    interventions = good_option.extract_drug_interventions(STUDY)

    assert [item.name for item in interventions] == ["Drug A", "Drug B"]
    assert [item.intervention_type for item in interventions] == [
        "DRUG",
        "BIOLOGICAL",
    ]


def test_patient_text_enters_after_drug_only_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_queries: list[str] = []

    async def fake_fetch(_nct_id: str, *, client: object):
        del client
        return STUDY

    def fake_search(queries):
        captured_queries.extend(queries)
        return (
            (
                good_option.DrugSearchResult(
                    query=queries[0],
                    title="Drug A results",
                    snippet="Reported findings.",
                    url="https://example.org/drug-a",
                ),
            ),
            (),
        )

    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(good_option.help_me_choose, "fetch_trial_study", fake_fetch)
    monkeypatch.setattr(good_option.asyncio, "to_thread", run_inline)
    research = asyncio.run(
        good_option.research_trial_drugs(
            "NCT12345678",
            client=object(),
            search_function=fake_search,
        )
    )
    research = asyncio.run(
        good_option.enrich_trial_with_biomarker_expression_research(
            research,
            search_function=fake_search,
        )
    )
    marker = "PRIVATE_PATIENT_MARKER"
    messages = good_option.build_good_option_messages(
        patient_summary=f"Synthetic patient {marker}",
        research=research,
    )
    prompt_text = messages[1]["content"].lower()

    assert captured_queries
    assert all(marker not in query for query in captured_queries)
    assert any("biomarker expression prevalence" in query for query in captured_queries)
    assert marker in messages[1]["content"]
    assert "four-criterion evidence rubric" in messages[0]["content"].lower()
    assert "do not score eligibility" in messages[0]["content"].lower()
    assert "untrusted" in messages[0]["content"].lower()
    assert "patient_disease_type" in messages[1]["content"]
    assert "full relevant disease and histology population" in prompt_text
    assert "defined independently of the biomarker being scored" in prompt_text
    assert (
        "common within a biomarker-positive subgroup does not establish" in prompt_text
    )
    assert "state the population and denominator in the rationale" in prompt_text


def test_good_option_response_parser_derives_four_point_score() -> None:
    parsed = good_option.parse_good_option_response(
        json.dumps(_rubric_response()),
        allowed_evidence_labels={"PATIENT", "CT", "S1", "S2"},
        biomarker_expression_evidence_labels={"S2"},
    )

    assert parsed.status == "ok"
    assert parsed.drug_count == 1
    assert parsed.total_points == 3
    assert parsed.max_points == 4
    assert parsed.score_0_1 == pytest.approx(0.75)
    assert parsed.point_disease_type_benefit == 1
    assert parsed.point_common_biomarker_in_disease == 1
    assert parsed.point_patient_biomarker_targeted == 1
    assert parsed.point_biomarker_targeted_benefit == 0
    assert json.loads(parsed.uncertainties_json) == ["Small study"]
    assert json.loads(parsed.evidence_common_biomarker_in_disease_json) == [
        {"drug_name": "Drug A", "evidence_labels": ["S2"]}
    ]


def test_multidrug_response_is_normalized_by_four_times_drug_count() -> None:
    response = _rubric_response()
    second = json.loads(json.dumps(response["drug_assessments"][0]))  # type: ignore[index]
    second["drug_name"] = "Drug B"
    second["targeted_biomarkers"] = ["Marker B"]
    second["common_biomarker_in_disease"]["point"] = 0
    second["common_biomarker_in_disease"]["evidence_labels"] = []
    second["patient_biomarker_targeted"]["point"] = 0
    second["patient_biomarker_targeted"]["evidence_labels"] = []
    response["drug_assessments"].append(second)  # type: ignore[union-attr]

    parsed = good_option.parse_good_option_response(
        json.dumps(response),
        expected_drug_names=["Drug A", "Drug B"],
        allowed_evidence_labels={"PATIENT", "CT", "S1", "S2"},
        biomarker_expression_evidence_labels={"S2"},
    )

    assert parsed.status == "ok"
    assert parsed.drug_count == 2
    assert parsed.total_points == 4
    assert parsed.max_points == 8
    assert parsed.score_0_1 == pytest.approx(0.5)
    assert len(json.loads(parsed.drug_assessments_json)) == 2


def test_multidrug_response_requires_exactly_one_assessment_per_drug() -> None:
    parsed = good_option.parse_good_option_response(
        json.dumps(_rubric_response()),
        expected_drug_names=["Drug A", "Drug B"],
    )

    assert parsed.status == "parse_failed"
    assert "missing=['Drug B']" in parsed.parse_error


def test_multi_patient_request_shares_trial_context_and_parses_each_case() -> None:
    research = good_option.TrialDrugResearch(
        nct_id="NCT12345678",
        title="Synthetic Drug A trial",
        interventions=(good_option.DrugIntervention("Drug A", "DRUG"),),
    )
    messages = good_option.build_good_option_batch_messages(
        patient_cases=[
            ("case-a", "Synthetic patient A with Cancer A."),
            ("case-b", "Synthetic patient B with Cancer B."),
        ],
        research=research,
    )
    prompt = messages[1]["content"]

    assert prompt.count('"candidate_trial"') == 1
    assert prompt.count('"case-a"') == 1
    assert prompt.count('"case-b"') == 1
    assert "Synthetic patient A" in prompt
    assert "Synthetic patient B" in prompt

    first = _rubric_response()
    second = json.loads(json.dumps(first))
    second["patient_disease_type"] = "Cancer B"
    response = {
        "patient_trials": [
            {"candidate_id": "case-a", **first},
            {"candidate_id": "case-b", **second},
        ]
    }
    parsed = good_option.parse_good_option_batch_response(
        json.dumps(response),
        expected_candidate_ids=["case-a", "case-b"],
        expected_drug_names=["Drug A"],
        allowed_evidence_labels={"PATIENT", "CT", "S1", "S2"},
        biomarker_expression_evidence_labels={"S2"},
    )

    assert parsed["case-a"].status == "ok"
    assert parsed["case-b"].status == "ok"
    assert parsed["case-a"].patient_disease_type == "Cancer A"
    assert parsed["case-b"].patient_disease_type == "Cancer B"


def test_multi_patient_response_marks_only_missing_case_as_failed() -> None:
    response = {
        "patient_trials": [
            {"candidate_id": "case-a", **_rubric_response()},
        ]
    }

    parsed = good_option.parse_good_option_batch_response(
        json.dumps(response),
        expected_candidate_ids=["case-a", "case-b"],
        expected_drug_names=["Drug A"],
        allowed_evidence_labels={"PATIENT", "CT", "S1", "S2"},
        biomarker_expression_evidence_labels={"S2"},
    )

    assert parsed["case-a"].status == "ok"
    assert parsed["case-b"].status == "parse_failed"
    assert "Missing patient_trials item" in parsed["case-b"].parse_error


def test_label_stage_batches_patient_trials_sharing_one_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = pd.DataFrame(
        [
            {
                "patient_summary": "Synthetic patient A with Cancer A.",
                "nct_id": "NCT12345678",
                "this_space": "1. First matching space.",
                "split": "train",
            },
            {
                "patient_summary": "Synthetic patient A with Cancer A.",
                "nct_id": "NCT12345678",
                "this_space": "2. Duplicate trial through another space.",
                "split": "train",
            },
            {
                "patient_summary": "Synthetic patient B with Cancer B.",
                "nct_id": "NCT12345678",
                "this_space": "3. Third matching space.",
                "split": "train",
            },
        ]
    )
    candidate_path = tmp_path / "top_patients_tocheck_round1.parquet"
    candidates.to_parquet(candidate_path, index=False)
    research = good_option.TrialDrugResearch(
        nct_id="NCT12345678",
        title="Drug A trial",
        interventions=(good_option.DrugIntervention("Drug A", "DRUG"),),
        search_results=(
            good_option.DrugSearchResult(
                query='"Drug A" oncology mechanism efficacy safety clinical trial',
                title="Benefit",
                snippet="Human benefit in Cancer A.",
                url="https://example.org/benefit",
            ),
            good_option.DrugSearchResult(
                query=(
                    '"Drug A" oncology molecular target biomarker expression '
                    "prevalence across cancer types"
                ),
                title="Expression",
                snippet="Marker A prevalence.",
                url="https://example.org/expression",
            ),
        ),
    )
    research_output = tmp_path / "research.parquet"
    pd.DataFrame([good_option.research_to_record(research)]).to_parquet(
        research_output,
        index=False,
    )

    class FakeTokenizer:
        def apply_chat_template(self, conversation, **_kwargs):
            return conversation[-1]["content"]

    submitted_prompt_count = 0

    async def fake_run_pool(*, work_items, shard_writer, starting_shard_idx, **_kwargs):
        nonlocal submitted_prompt_count
        payload = []
        for prompt_id, request in work_items:
            submitted_prompt_count += 1
            _instructions, prompt_json = request["prompt"].split("\n\n", 1)
            prompt_payload = json.loads(prompt_json)
            response_rows = []
            for patient in prompt_payload[
                "patient_trials_private_to_configured_llm"
            ]:
                response_rows.append(
                    {
                        "candidate_id": patient["candidate_id"],
                        **_rubric_response(),
                    }
                )
            payload.append(
                (
                    prompt_id,
                    ("", json.dumps({"patient_trials": response_rows})),
                )
            )
        shard_writer(payload, starting_shard_idx)
        return len(payload)

    monkeypatch.setattr("remote_vllm_pool.run_pool", fake_run_pool)
    args = SimpleNamespace(
        research_output=str(research_output),
        research_shards_dir=str(tmp_path / "research_shards"),
        label_output=str(tmp_path / "labels.parquet"),
        label_shards_dir=str(tmp_path / "label_shards"),
        scan_batch_size=10,
        submission_batch_size=10,
        max_candidates=None,
        patients_per_request=4,
        max_batch_new_tokens=8000,
        max_new_tokens=2000,
        results_per_shard=200,
        max_attempts=2,
        store_reasoning=False,
        model="teacher-model",
    )
    runtime = good_option.TeacherRuntime(
        tokenizer=FakeTokenizer(),
        registry=object(),
        work_fn=object(),
        local_servers=[],
    )

    output = asyncio.run(
        good_option.run_label_stage(args, [candidate_path], runtime=runtime)
    )
    labels = pd.read_parquet(output)

    assert submitted_prompt_count == 1
    assert len(labels) == 2
    assert labels["candidate_id"].is_unique
    assert "this_space" not in labels.columns
    assert labels["good_option_label_status"].eq("ok").all()


def test_label_stage_refuses_research_failure_as_patient_label(
    tmp_path: Path,
) -> None:
    candidates = pd.DataFrame(
        [
            {
                "patient_summary": "Synthetic patient A.",
                "nct_id": "NCT12345678",
                "this_space": "Synthetic space A.",
                "split": "train",
            },
            {
                "patient_summary": "Synthetic patient B.",
                "nct_id": "NCT87654321",
                "this_space": "Synthetic space B.",
                "split": "train",
            },
        ]
    )
    candidate_path = tmp_path / "top_patients_tocheck_round1.parquet"
    candidates.to_parquet(candidate_path, index=False)
    intervention = good_option.DrugIntervention("Experimental-X", "DRUG")
    good_normalization = good_option.DrugNameNormalization(
        nct_id="NCT12345678",
        registry_interventions=(intervention,),
        canonical_interventions=(intervention,),
        status="ok",
    )
    failed_normalization = good_option.DrugNameNormalization(
        nct_id="NCT87654321",
        registry_interventions=(intervention,),
        canonical_interventions=(),
        status="parse_failed",
        parse_error="Empty teacher answer.",
    )
    research_output = tmp_path / "research.parquet"
    pd.DataFrame(
        [
            good_option.research_to_record(
                good_option.TrialDrugResearch(
                    nct_id="NCT12345678",
                    interventions=(intervention,),
                ),
                normalization=good_normalization,
            ),
            good_option.research_to_record(
                good_option.TrialDrugResearch(nct_id="NCT87654321"),
                normalization=failed_normalization,
            ),
        ]
    ).to_parquet(research_output, index=False)
    args = SimpleNamespace(
        patients_per_request=1,
        max_batch_new_tokens=100,
        max_drug_assessments_per_request=1,
        research_output=str(research_output),
        research_shards_dir=str(tmp_path / "research_shards"),
        scan_batch_size=10,
        max_registry_failure_fraction=1.0,
        max_drug_normalization_failure_fraction=1.0,
    )

    with pytest.raises(
        RuntimeError,
        match="Infrastructure/teacher failures are not patient labels",
    ):
        asyncio.run(good_option.run_label_stage(args, [candidate_path]))

    assert not (tmp_path / "label_shards").exists()


def test_old_holistic_score_response_is_rejected() -> None:
    parsed = good_option.parse_good_option_response('{"score": 72}')

    assert parsed.status == "parse_failed"
    assert pd.isna(parsed.score_0_1)


@pytest.mark.parametrize("point", [-1, 2, "unknown", True])
def test_good_option_response_parser_rejects_invalid_points(point: object) -> None:
    response = _rubric_response()
    response["drug_assessments"][0]["disease_type_benefit"]["point"] = point  # type: ignore[index]
    parsed = good_option.parse_good_option_response(json.dumps(response))

    assert parsed.status == "parse_failed"
    assert pd.isna(parsed.score_0_1)


def test_awarded_point_requires_criterion_specific_evidence() -> None:
    response = _rubric_response()
    response["drug_assessments"][0]["common_biomarker_in_disease"][  # type: ignore[index]
        "evidence_labels"
    ] = ["PATIENT"]

    parsed = good_option.parse_good_option_response(json.dumps(response))

    assert parsed.status == "ok"
    assert parsed.point_common_biomarker_in_disease == 0
    assessments = json.loads(parsed.drug_assessments_json)
    assert (
        "Validator reset this point to 0"
        in assessments[0]["common_biomarker_in_disease"]["rationale"]
    )


def test_common_biomarker_point_requires_expression_research_source() -> None:
    parsed = good_option.parse_good_option_response(
        json.dumps(_rubric_response()),
        allowed_evidence_labels={"PATIENT", "CT", "S1", "S2"},
        biomarker_expression_evidence_labels={"S3"},
    )

    assert parsed.status == "ok"
    assert parsed.point_common_biomarker_in_disease == 0
    assessments = json.loads(parsed.drug_assessments_json)
    assert (
        "target-expression research source"
        in assessments[0]["common_biomarker_in_disease"]["rationale"]
    )


def test_candidate_stream_deduplicates_across_top_files(tmp_path: Path) -> None:
    first = tmp_path / "top_cohorts_tocheck_round1.parquet"
    second = tmp_path / "top_patients_tocheck_round1.parquet"
    _candidate_frame(duplicate=True).to_parquet(first, index=False)
    second_frame = _candidate_frame().iloc[[0]].copy()
    second_frame["this_space"] = "2. A second space for the same patient and trial."
    second_frame.to_parquet(second, index=False)

    batches = list(
        good_option.iter_unique_candidate_batches(
            [first, second],
            scan_batch_size=2,
            submission_batch_size=2,
        )
    )
    combined = pd.concat(batches, ignore_index=True)

    assert len(combined) == 2
    assert combined["candidate_id"].is_unique
    assert combined.loc[0, "this_space"] == "Cancer A treated with Drug A."


def test_custom_candidate_path_requires_non_phi_confirmation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "no_phi"
    data_dir.mkdir(parents=True)
    outside = tmp_path / "possibly_sensitive.parquet"
    _candidate_frame().to_parquet(outside, index=False)

    with pytest.raises(ValueError, match="confirm-inputs-are-non-phi"):
        good_option.resolve_candidate_paths(
            data_dir,
            [str(outside)],
            confirm_inputs_are_non_phi=False,
        )


def test_research_record_round_trip() -> None:
    research = good_option.TrialDrugResearch(
        nct_id="NCT12345678",
        title="Synthetic trial",
        phases=("PHASE2",),
        interventions=(good_option.DrugIntervention("Drug A", "DRUG", "Description"),),
        search_results=(
            good_option.DrugSearchResult(
                query='"Drug A" oncology mechanism efficacy safety clinical trial',
                title="Result",
                snippet="Snippet",
                url="https://example.org/result",
            ),
        ),
    )

    record = good_option.research_to_record(
        research,
        fetched_at_utc="2026-08-19T00:00:00+00:00",
    )
    restored = good_option.research_from_record(record)

    assert restored == research
    assert record["fetched_at_utc"] == "2026-08-19T00:00:00+00:00"
    assert len(record["research_implementation_sha256"]) == 64
    assert (
        record["biomarker_expression_query_version"]
        == good_option.BIOMARKER_EXPRESSION_QUERY_VERSION
    )
    assert (
        record["drug_name_normalization_prompt_version"]
        == good_option.DRUG_NAME_NORMALIZATION_PROMPT_VERSION
    )


def test_research_record_preserves_registry_and_canonical_names() -> None:
    registry_intervention = good_option.DrugIntervention(
        name="Dose-Escalation Agent-X capsule",
        intervention_type="DRUG",
    )
    canonical_intervention = good_option.DrugIntervention(
        name="Agent-X",
        intervention_type="DRUG",
    )
    normalization = good_option.DrugNameNormalization(
        nct_id="NCT12345678",
        registry_interventions=(registry_intervention,),
        canonical_interventions=(canonical_intervention,),
        mappings_json=json.dumps(
            [
                {
                    "source_index": 0,
                    "registry_name": registry_intervention.name,
                    "canonical_drug_names": [canonical_intervention.name],
                    "used_fallback": False,
                }
            ]
        ),
        status="ok",
        raw_response='{"interventions": []}',
    )
    research = good_option.TrialDrugResearch(
        nct_id="NCT12345678",
        interventions=(canonical_intervention,),
    )

    record = good_option.research_to_record(
        research,
        normalization=normalization,
        teacher_model="teacher-model",
    )

    assert json.loads(record["registry_interventions_json"])[0]["name"] == (
        registry_intervention.name
    )
    assert json.loads(record["interventions_json"])[0]["name"] == "Agent-X"
    assert record["drug_name_normalization_status"] == "ok"
    assert record["drug_name_normalization_teacher_model"] == "teacher-model"


def test_research_quality_gate_rejects_registry_outage() -> None:
    records = []
    for index in range(10):
        research = good_option.TrialDrugResearch(
            nct_id=f"NCT{index:08d}",
            notices=("ClinicalTrials.gov lookup failed: HTTP 429",),
        )
        records.append(good_option.research_to_record(research))

    with pytest.raises(RuntimeError, match="registry failure fraction 100.0%"):
        good_option.validate_research_quality(
            records,
            maximum_registry_failure_fraction=0.05,
            maximum_normalization_failure_fraction=0.10,
            context="test outage",
        )


def test_research_quality_gate_rejects_empty_teacher_answers() -> None:
    registry_intervention = good_option.DrugIntervention(
        "Experimental-X",
        "DRUG",
    )
    records = []
    for index in range(10):
        nct_id = f"NCT{index:08d}"
        normalization = good_option.DrugNameNormalization(
            nct_id=nct_id,
            registry_interventions=(registry_intervention,),
            canonical_interventions=(),
            status="parse_failed",
            parse_error="No valid interventions JSON object was found.",
        )
        records.append(
            good_option.research_to_record(
                good_option.TrialDrugResearch(nct_id=nct_id),
                normalization=normalization,
            )
        )

    with pytest.raises(
        RuntimeError,
        match="technical drug-normalization failure fraction 100.0%",
    ):
        good_option.validate_research_quality(
            records,
            maximum_registry_failure_fraction=0.05,
            maximum_normalization_failure_fraction=0.10,
            context="test empty answers",
        )


def test_technical_normalization_diagnostics_are_persisted(
    tmp_path: Path,
) -> None:
    intervention = good_option.DrugIntervention("Experimental-X", "DRUG")
    registry = good_option.TrialDrugResearch(
        nct_id="NCT12345678",
        title="Experimental-X trial",
        interventions=(intervention,),
    )
    attempts = [
        {
            "attempt": 1,
            "max_tokens": 8_000,
            "status": "parse_failed",
            "finish_reason": "length",
        }
    ]
    normalization = good_option.DrugNameNormalization(
        nct_id=registry.nct_id,
        registry_interventions=(intervention,),
        canonical_interventions=(),
        status="parse_failed",
        raw_response="",
        parse_error="No valid interventions JSON object was found.",
        attempt_count=1,
        attempts_json=json.dumps(attempts),
        finish_reason="length",
        reasoning_char_count=8_000,
    )

    output = good_option.write_normalization_failure_diagnostics(
        registry_items=[registry],
        normalizations={registry.nct_id: normalization},
        research_shards_dir=tmp_path / "research_shards",
        teacher_model="teacher-model",
        batch_number=1,
    )

    assert output is not None
    frame = pd.read_parquet(output)
    assert frame.loc[0, "nct_id"] == "NCT12345678"
    assert frame.loc[0, "normalization_finish_reason"] == "length"
    assert frame.loc[0, "normalization_attempt_count"] == 1
    assert json.loads(frame.loc[0, "normalization_attempts_json"]) == attempts
    assert "patient" not in " ".join(frame.columns).lower()


def test_research_quality_gate_allows_confirmed_control_only_trial() -> None:
    registry_intervention = good_option.DrugIntervention("Standard-Y", "DRUG")
    normalization = good_option.DrugNameNormalization(
        nct_id="NCT12345678",
        registry_interventions=(registry_intervention,),
        canonical_interventions=(),
        status="no_experimental_interventions",
    )
    research = good_option.TrialDrugResearch(
        nct_id="NCT12345678",
        notices=(
            "No investigational drug or biological agent was identified; "
            "comparator/background interventions were not searched.",
        ),
    )
    record = good_option.research_to_record(
        research,
        normalization=normalization,
    )

    summary = good_option.validate_research_quality(
        [record],
        maximum_registry_failure_fraction=0.05,
        maximum_normalization_failure_fraction=0.10,
        context="test control-only trial",
    )

    assert summary.confirmed_no_experimental_drug_trials == 1
    assert summary.normalization_failures == 0


def test_research_map_rejects_stale_expression_query_cache(tmp_path: Path) -> None:
    research = good_option.TrialDrugResearch(nct_id="NCT12345678")
    record = good_option.research_to_record(research)
    record["biomarker_expression_query_version"] = "stale-version"
    output = tmp_path / "research.parquet"
    pd.DataFrame([record]).to_parquet(output, index=False)

    loaded = good_option._research_map(output, tmp_path / "missing-shards")

    assert loaded == {}


def test_research_map_rejects_stale_name_normalization_cache(tmp_path: Path) -> None:
    research = good_option.TrialDrugResearch(nct_id="NCT12345678")
    record = good_option.research_to_record(research)
    record["drug_name_normalization_prompt_version"] = "stale-version"
    output = tmp_path / "research.parquet"
    pd.DataFrame([record]).to_parquet(output, index=False)

    loaded = good_option._research_map(output, tmp_path / "missing-shards")

    assert loaded == {}


def test_label_resume_uses_only_current_successful_schema(tmp_path: Path) -> None:
    output = tmp_path / "labels.parquet"
    pd.DataFrame(
        [
            {
                "candidate_id": "current",
                "good_option_label_status": "ok",
                "prompt_version": good_option.GOOD_OPTION_PROMPT_VERSION,
                "label_schema_version": good_option.GOOD_OPTION_LABEL_SCHEMA_VERSION,
            },
            {
                "candidate_id": "failed",
                "good_option_label_status": "parse_failed",
                "prompt_version": good_option.GOOD_OPTION_PROMPT_VERSION,
                "label_schema_version": good_option.GOOD_OPTION_LABEL_SCHEMA_VERSION,
            },
            {
                "candidate_id": "old",
                "good_option_label_status": "ok",
                "prompt_version": "old-prompt",
                "label_schema_version": "1",
            },
            {
                "candidate_id": "no-experimental-drug",
                "good_option_label_status": "no_experimental_drug_intervention",
                "prompt_version": good_option.GOOD_OPTION_PROMPT_VERSION,
                "label_schema_version": good_option.GOOD_OPTION_LABEL_SCHEMA_VERSION,
            },
        ]
    ).to_parquet(output, index=False)

    done = good_option.load_done_candidate_ids(output, tmp_path / "missing-shards")

    assert done == {"current", "no-experimental-drug"}


def test_checker_drug_context_excludes_teacher_web_snippets() -> None:
    research = good_option.TrialDrugResearch(
        nct_id="NCT12345678",
        title="Drug A study",
        interventions=(good_option.DrugIntervention("Drug A", "DRUG"),),
        search_results=(
            good_option.DrugSearchResult(
                query="drug-only query",
                title="Result",
                snippet="TEACHER_ONLY_SNIPPET",
                url="https://example.org/result",
            ),
        ),
    )

    context = good_option.build_trial_drug_context(research)

    assert "Drug A" in context
    assert "TEACHER_ONLY_SNIPPET" not in context


def test_training_frame_uses_patient_level_validation_split() -> None:
    labels = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "patient_summary": "Same synthetic patient",
                "this_space": "Space A",
                "trial_drug_context": "DRUG: Drug A",
                "split": "train",
                "drug_count": 1,
                "good_option_points": 1,
                "good_option_max_points": 4,
                "good_option_score": 0.25,
                "good_option_label_status": "ok",
                "point_disease_type_benefit": 1,
                "point_common_biomarker_in_disease": 0,
                "point_patient_biomarker_targeted": 0,
                "point_biomarker_targeted_benefit": 0,
            },
            {
                "candidate_id": "b",
                "patient_summary": "Same synthetic patient",
                "this_space": "Space B",
                "trial_drug_context": "DRUG: Drug B",
                "split": "train",
                "drug_count": 2,
                "good_option_points": 6,
                "good_option_max_points": 8,
                "good_option_score": 0.75,
                "good_option_label_status": "ok",
                "point_disease_type_benefit": 2,
                "point_common_biomarker_in_disease": 1,
                "point_patient_biomarker_targeted": 2,
                "point_biomarker_targeted_benefit": 1,
            },
            {
                "candidate_id": "bad",
                "patient_summary": "Another synthetic patient",
                "this_space": "Space C",
                "trial_drug_context": "DRUG: Drug C",
                "split": "train",
                "drug_count": 0,
                "good_option_points": -1,
                "good_option_max_points": -1,
                "good_option_score": float("nan"),
                "good_option_label_status": "parse_failed",
                "point_disease_type_benefit": -1,
                "point_common_biomarker_in_disease": -1,
                "point_patient_biomarker_targeted": -1,
                "point_biomarker_targeted_benefit": -1,
            },
            {
                "candidate_id": "inconsistent",
                "patient_summary": "Third synthetic patient",
                "this_space": "Space D",
                "trial_drug_context": "DRUG: Drug D",
                "split": "train",
                "drug_count": 1,
                "good_option_points": 2,
                "good_option_max_points": 4,
                "good_option_score": 0.75,
                "good_option_label_status": "ok",
                "point_disease_type_benefit": 1,
                "point_common_biomarker_in_disease": 1,
                "point_patient_biomarker_targeted": 0,
                "point_biomarker_targeted_benefit": 0,
            },
        ]
    )

    prepared = good_option.prepare_training_frame(
        labels,
        validation_fraction=0.5,
        seed=42,
    )

    assert len(prepared) == 2
    assert prepared["partition"].nunique() == 1
    assert prepared["label"].tolist() == pytest.approx([0.25, 0.75])
    assert not prepared["text"].str.contains("Clinical trial space:").any()
    assert prepared["text"].str.contains(
        "Registry investigational-drug context:"
    ).all()


def test_local_vllm_command_uses_openai_server_and_requested_parser() -> None:
    command = good_option.build_vllm_server_command(
        model="example/model",
        download_dir="/models",
        tensor_parallel_size=2,
        max_model_len=50_000,
        max_num_seqs=128,
        gpu_memory_utilization=0.9,
        port=8100,
        reasoning_parser="gemma4",
    )

    assert command[:3] == [
        good_option.sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    assert command[command.index("--reasoning-parser") + 1] == "gemma4"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"


def test_remote_registry_accepts_api_key_without_exposing_it() -> None:
    registry = DynamicServerRegistry(
        max_concurrent_per_server=1,
        request_timeout=10,
        static_urls=["http://localhost:8000/v1"],
        api_key="secret-test-value",
    )

    assert registry._api_key == "secret-test-value"
    assert "secret-test-value" not in repr(registry)


def test_completion_work_fn_optionally_returns_finish_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm_reasoning_utils.parse_reasoning_output",
        lambda text, parser_name, tokenizer: (
            f"reasoning:{parser_name}",
            f"answer:{text}:{tokenizer}",
        ),
    )

    requests: list[dict[str, object]] = []

    class FakeCompletions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(text="raw", finish_reason="length")]
            )

    client = SimpleNamespace(completions=FakeCompletions())
    work_fn = make_completion_work_fn(
        CompletionSampling(model="teacher", request_timeout=1),
        "qwen3",
        "tokenizer",
    )

    result = asyncio.run(
        work_fn(
            client,
            {
                "prompt": "prompt",
                "max_tokens": 100,
                "include_completion_metadata": True,
            },
        )
    )

    assert result == (
        "reasoning:qwen3",
        "answer:raw:tokenizer",
        {"finish_reason": "length", "raw_text_char_count": 3},
    )
    assert requests[0]["extra_body"]["repetition_penalty"] == 1.1


def test_good_option_cli_defaults_repetition_penalty_to_1_1() -> None:
    args = good_option.build_parser().parse_args(["generate"])

    assert args.repetition_penalty == 1.1
