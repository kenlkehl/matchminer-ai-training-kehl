from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

import train_good_option_checker as good_option
from remote_vllm_pool import DynamicServerRegistry


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
            "rationale": "No human biomarker-directed benefit evidence was supplied.",
            "evidence_labels": [],
        },
        "key_uncertainties": ["Small study"],
    }


def test_drug_query_api_structurally_excludes_patient_context() -> None:
    query_parameters = inspect.signature(
        good_option.build_drug_search_queries
    ).parameters
    expression_parameters = inspect.signature(
        good_option.build_biomarker_expression_search_queries
    ).parameters
    research_parameters = inspect.signature(good_option.research_trials).parameters
    enrichment_parameters = inspect.signature(
        good_option.enrich_trial_with_biomarker_expression_research
    ).parameters

    assert list(query_parameters) == ["interventions"]
    assert list(expression_parameters) == ["interventions"]
    assert "patient_summary" not in research_parameters
    assert "patient_history" not in research_parameters
    assert "patient_summary" not in enrichment_parameters
    assert "clinical_space_summary" not in enrichment_parameters
    assert good_option.research_trials.__module__ == "matchminer_ai.help_me_choose"


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
        clinical_space_summary="Cancer A treated with Drug A.",
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
    assert parsed.total_points == 3
    assert parsed.score_0_1 == pytest.approx(0.75)
    assert parsed.point_disease_type_benefit == 1
    assert parsed.point_common_biomarker_in_disease == 1
    assert parsed.point_patient_biomarker_targeted == 1
    assert parsed.point_biomarker_targeted_benefit == 0
    assert json.loads(parsed.uncertainties_json) == ["Small study"]
    assert json.loads(parsed.evidence_common_biomarker_in_disease_json) == ["S2"]


def test_old_holistic_score_response_is_rejected() -> None:
    parsed = good_option.parse_good_option_response('{"score": 72}')

    assert parsed.status == "parse_failed"
    assert pd.isna(parsed.score_0_1)


@pytest.mark.parametrize("point", [-1, 2, "unknown", True])
def test_good_option_response_parser_rejects_invalid_points(point: object) -> None:
    response = _rubric_response()
    response["disease_type_benefit"]["point"] = point  # type: ignore[index]
    parsed = good_option.parse_good_option_response(json.dumps(response))

    assert parsed.status == "parse_failed"
    assert pd.isna(parsed.score_0_1)


def test_awarded_point_requires_criterion_specific_evidence() -> None:
    response = _rubric_response()
    response["common_biomarker_in_disease"]["evidence_labels"] = [  # type: ignore[index]
        "PATIENT"
    ]

    parsed = good_option.parse_good_option_response(json.dumps(response))

    assert parsed.status == "parse_failed"
    assert "web source" in parsed.parse_error


def test_common_biomarker_point_requires_expression_research_source() -> None:
    parsed = good_option.parse_good_option_response(
        json.dumps(_rubric_response()),
        allowed_evidence_labels={"PATIENT", "CT", "S1", "S2"},
        biomarker_expression_evidence_labels={"S3"},
    )

    assert parsed.status == "parse_failed"
    assert "target-expression research source" in parsed.parse_error


def test_candidate_stream_deduplicates_across_top_files(tmp_path: Path) -> None:
    first = tmp_path / "top_cohorts_tocheck_round1.parquet"
    second = tmp_path / "top_patients_tocheck_round1.parquet"
    _candidate_frame(duplicate=True).to_parquet(first, index=False)
    _candidate_frame().iloc[[0]].to_parquet(second, index=False)

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


def test_research_map_rejects_stale_expression_query_cache(tmp_path: Path) -> None:
    research = good_option.TrialDrugResearch(nct_id="NCT12345678")
    record = good_option.research_to_record(research)
    record["biomarker_expression_query_version"] = "stale-version"
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
        ]
    ).to_parquet(output, index=False)

    done = good_option.load_done_candidate_ids(output, tmp_path / "missing-shards")

    assert done == {"current"}


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
                "good_option_points": 1,
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
                "good_option_points": 3,
                "good_option_score": 0.75,
                "good_option_label_status": "ok",
                "point_disease_type_benefit": 1,
                "point_common_biomarker_in_disease": 1,
                "point_patient_biomarker_targeted": 1,
                "point_biomarker_targeted_benefit": 0,
            },
            {
                "candidate_id": "bad",
                "patient_summary": "Another synthetic patient",
                "this_space": "Space C",
                "trial_drug_context": "DRUG: Drug C",
                "split": "train",
                "good_option_points": -1,
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
                "good_option_points": 2,
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
    assert prepared["text"].str.contains("Clinical trial space:").all()
    assert prepared["text"].str.contains("Registry drug context:").all()


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
