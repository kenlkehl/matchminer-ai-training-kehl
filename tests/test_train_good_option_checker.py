from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import train_good_option_checker as good_option
from matchminer_ai.matching import (
    RUBRIC_CRITERIA,
    build_good_option_messages,
)
from matchminer_ai.trials import (
    DrugSummary,
    GoodOptionCatalog,
    TrialDrugAssignment,
)


def _catalog() -> GoodOptionCatalog:
    summary = DrugSummary(
        drug_id="D1",
        preferred_name="Novel Agent",
        ncit_code="C1",
        research_status="complete",
        synthesis_status="ok",
        structured_facts={},
        good_option_summary="Novel Agent targets Marker A and has human evidence.",
        help_me_choose_summary="Novel Agent mechanism, efficacy, and safety.",
        evidence_count=3,
    )
    assignments = [
        TrialDrugAssignment(
            trial_id=trial_id,
            drug_id="D1",
            preferred_name="Novel Agent",
            registry_name="Novel Agent",
            intervention_type="DRUG",
            role="investigational",
            role_confidence="high",
            scoreable=True,
        )
        for trial_id in ("NCT12345678", "NCT87654321")
    ]
    return GoodOptionCatalog(
        path=Path("/tmp/catalog"),
        manifest={"compatibility_id": "compat-v2"},
        trial_registry=pd.DataFrame(
            [
                {"trial_id": trial_id, "registry_status": "ok"}
                for trial_id in ("NCT12345678", "NCT87654321")
            ]
        ),
        trial_drug_index=pd.DataFrame([item.to_record() for item in assignments]),
        drug_summaries=pd.DataFrame([summary.to_record()]),
        drug_evidence=pd.DataFrame(),
        drug_research_attempts=pd.DataFrame(),
    )


def _assessment(points: tuple[int, int, int, int]) -> dict[str, object]:
    value: dict[str, object] = {
        "drug_name": "Novel Agent",
        "targeted_biomarkers": ["Marker A"],
    }
    for criterion, point in zip(RUBRIC_CRITERIA, points, strict=True):
        value[criterion] = {
            "point": point,
            "rationale": f"Rationale for {criterion}.",
        }
    return value


def _labels(*, conflicting: bool = False) -> pd.DataFrame:
    rows = [
        {
            "candidate_id": "C1",
            "patient_id": "P1",
            "patient_group_id": "PG1",
            "patient_summary": "Synthetic patient with Marker A.",
            "nct_id": "NCT12345678",
            "good_option_status": "ok",
            "drug_assessments_json": json.dumps([_assessment((1, 1, 1, 0))]),
            "catalog_compatibility_id": "compat-v2",
            "prompt_version": good_option.GOOD_OPTION_PROMPT_VERSION,
            "label_schema_version": good_option.GOOD_OPTION_LABEL_SCHEMA_VERSION,
        },
        {
            "candidate_id": "C2",
            "patient_id": "P2",
            "patient_group_id": "PG2",
            "patient_summary": "Second synthetic patient.",
            "nct_id": "NCT87654321",
            "good_option_status": "ok",
            "drug_assessments_json": json.dumps([_assessment((0, 0, 0, 0))]),
            "catalog_compatibility_id": "compat-v2",
            "prompt_version": good_option.GOOD_OPTION_PROMPT_VERSION,
            "label_schema_version": good_option.GOOD_OPTION_LABEL_SCHEMA_VERSION,
        },
    ]
    if conflicting:
        rows.append(
            {
                **rows[0],
                "candidate_id": "C3",
                "nct_id": "NCT87654321",
                "drug_assessments_json": json.dumps([_assessment((0, 0, 0, 0))]),
            }
        )
    return pd.DataFrame(rows)


def test_candidate_loading_ignores_space_text_and_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "candidates.parquet"
    pd.DataFrame(
        [
            {
                "pseudo_mrn": "P1",
                "patient_summary": "Synthetic patient",
                "nct_id": "nct12345678",
                "this_space": "MUST_NOT_ENTER_GOOD_OPTION",
                "split": "train",
            },
            {
                "pseudo_mrn": "P1",
                "patient_summary": "Synthetic patient",
                "nct_id": "NCT12345678",
                "this_space": "Different space",
                "split": "train",
            },
        ]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="confirm-inputs-are-non-phi"):
        good_option.load_candidates([path])
    loaded = good_option.load_candidates(
        [path], confirm_inputs_are_non_phi=True
    )

    assert len(loaded) == 1
    assert loaded.loc[0, "nct_id"] == "NCT12345678"
    assert "this_space" not in loaded.columns


def test_nct_id_file_accepts_text_csv_and_parquet(tmp_path: Path) -> None:
    text = tmp_path / "ids.txt"
    text.write_text("NCT12345678\n# comment\nNCT87654321\n", encoding="utf-8")
    csv = tmp_path / "ids.csv"
    pd.DataFrame({"trial_id": ["NCT12345678"]}).to_csv(csv, index=False)
    parquet = tmp_path / "ids.parquet"
    pd.DataFrame({"nct_id": ["NCT87654321"]}).to_parquet(parquet, index=False)

    assert good_option._read_nct_ids_file(text) == [
        "NCT12345678",
        "NCT87654321",
    ]
    assert good_option._read_nct_ids_file(csv) == ["NCT12345678"]
    assert good_option._read_nct_ids_file(parquet) == ["NCT87654321"]


def test_catalog_id_derivation_reads_candidate_nct_column_only(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate_ids.parquet"
    pd.DataFrame({"nct_id": ["nct12345678", "NCT87654321"]}).to_parquet(
        candidate, index=False
    )
    args = good_option.build_parser().parse_args(
        ["catalog", "--candidate-files", str(candidate)]
    )

    assert good_option.resolve_nct_ids(args) == (
        "NCT12345678",
        "NCT87654321",
    )


def test_catalog_cli_passes_checkpoint_resume_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_build(nct_ids, output_path, **kwargs):
        captured.update(
            {
                "nct_ids": nct_ids,
                "output_path": output_path,
                **kwargs,
            }
        )
        return SimpleNamespace(
            path=Path(output_path),
            trial_registry=pd.DataFrame({"trial_id": list(nct_ids)}),
            drug_summaries=pd.DataFrame(),
        )

    monkeypatch.setattr(good_option, "build_good_option_catalog", fake_build)
    checkpoints = tmp_path / "checkpoints"
    args = good_option.build_parser().parse_args(
        [
            "catalog",
            "--nct-id",
            "NCT12345678",
            "--catalog",
            str(tmp_path / "catalog"),
            "--catalog-checkpoint-dir",
            str(checkpoints),
            "--reset-catalog-checkpoints",
        ]
    )

    good_option.run_catalog(args)

    assert captured["nct_ids"] == ("NCT12345678",)
    assert captured["checkpoint_path"] == str(checkpoints)
    assert captured["reset_checkpoint"] is True


def test_label_serialization_preserves_llm_inputs_and_outputs() -> None:
    source = pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "patient_id": "P1",
                "patient_group_id": "PG1",
                "patient_summary": "Synthetic patient with Marker A.",
                "nct_id": "NCT12345678",
                "split": "train",
            }
        ]
    )
    scored = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "NCT12345678",
                "good_option_status": "ok",
                "good_option_answer_text": '{"drug_assessments": []}',
                "good_option_reasoning_text": "Reasoning trace from the teacher.",
                "good_option_finish_reason": "stop",
            }
        ]
    )

    serialized = good_option._serialize_label_frame(
        source, scored, catalog=_catalog()
    )

    assert serialized.loc[0, "llm_response"] == '{"drug_assessments": []}'
    assert serialized.loc[0, "llm_reasoning"] == "Reasoning trace from the teacher."
    assert serialized.loc[0, "llm_finish_reason"] == "stop"
    assert serialized.loc[0, "llm_invocation_status"] == "called"
    assert serialized.loc[0, "llm_drug_information"] == (
        "Novel Agent targets Marker A and has human evidence."
    )
    rendered_user_prompt = build_good_option_messages(
        patient_summary=source.loc[0, "patient_summary"],
        drug_summaries=_catalog().scoreable_summaries_for_trial("NCT12345678"),
    )[1]["content"]
    assert (
        "SCOREABLE DRUG SUMMARIES\n"
        f"{serialized.loc[0, 'llm_drug_information']}\n\n"
        "RUBRIC\n"
    ) in rendered_user_prompt


def test_llm_invocation_status_distinguishes_intentional_skips() -> None:
    assert good_option._llm_invocation_status("ok") == "called"
    assert good_option._llm_invocation_status("parse_failed") == "called"
    assert (
        good_option._llm_invocation_status("no_scoreable_drug")
        == "not_called:no_scoreable_drug"
    )


def test_label_batch_delegates_feedback_retries_and_reasoning_off_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_score(pairs, **kwargs):
        captured["pairs"] = pairs
        captured.update(kwargs)
        return pd.DataFrame(
            [{"patient_id": "P1", "trial_id": "NCT12345678"}]
        )

    monkeypatch.setattr(good_option, "score_good_options_with_llm", fake_score)
    config = good_option.configure_teacher(
        good_option.build_parser().parse_args(["label"])
    )
    result = good_option._label_batch_with_retries(
        pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "nct_id": "NCT12345678",
                    "patient_summary": "Synthetic patient.",
                }
            ]
        ),
        catalog=_catalog(),
        config=config,
        max_attempts=3,
    )

    assert result.loc[0, "trial_id"] == "NCT12345678"
    assert captured["max_parse_attempts"] == 3
    assert captured["reasoning_off_fallback"] is True
    assert list(captured["pairs"].columns) == [
        "patient_id",
        "trial_id",
        "cancer_history_summary",
    ]


def test_existing_label_shards_backfill_llm_drug_information(tmp_path: Path) -> None:
    shard = tmp_path / "labels_000000.parquet"
    _labels().to_parquet(shard, index=False)

    assert (
        good_option._backfill_label_shard_drug_information(tmp_path, catalog=_catalog())
        == 1
    )
    backfilled = pd.read_parquet(shard)
    assert backfilled["llm_drug_information"].tolist() == [
        "Novel Agent targets Marker A and has human evidence.",
        "Novel Agent targets Marker A and has human evidence.",
    ]
    assert backfilled["llm_invocation_status"].tolist() == ["called", "called"]
    assert (
        good_option._backfill_label_shard_drug_information(tmp_path, catalog=_catalog())
        == 0
    )


def test_flattening_creates_one_four_target_row_per_patient_drug() -> None:
    flattened = good_option.flatten_patient_drug_labels(_labels(), _catalog())

    assert len(flattened) == 2
    assert flattened.loc[0, "drug_id"] == "D1"
    assert all(len(value) == 4 for value in flattened["labels"])
    assert all(
        isinstance(point, float)
        for values in flattened["labels"]
        for point in values
    )
    assert "NCT12345678" not in flattened.loc[0, "checker_text"]
    assert "http" not in flattened.loc[0, "checker_text"]
    assert flattened.loc[0, "checker_text"].index(
        "Patient cancer history"
    ) < flattened.loc[0, "checker_text"].index(
        "Investigational drug evidence summary"
    )


def test_conflicting_patient_drug_duplicates_fail() -> None:
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        good_option.flatten_patient_drug_labels(
            _labels(conflicting=True), _catalog()
        )


def test_catalog_compatibility_is_required_for_training_labels() -> None:
    labels = _labels()
    labels["catalog_compatibility_id"] = "old"
    with pytest.raises(ValueError, match="different compatibility"):
        good_option.flatten_patient_drug_labels(labels, _catalog())


def test_existing_label_shards_must_match_catalog_and_schema(tmp_path: Path) -> None:
    shard = tmp_path / "labels_000000.parquet"
    _labels().to_parquet(shard, index=False)

    assert good_option._existing_label_ids(
        tmp_path, catalog=_catalog()
    ) == {"C1", "C2"}

    incompatible = _labels()
    incompatible["prompt_version"] = "old"
    incompatible.to_parquet(shard, index=False)
    with pytest.raises(ValueError, match="incompatible prompt_version"):
        good_option._existing_label_ids(tmp_path, catalog=_catalog())


def test_existing_parse_failures_are_requeued_on_resume(tmp_path: Path) -> None:
    labels = _labels()
    labels.loc[labels["candidate_id"].eq("C2"), "good_option_status"] = "parse_failed"
    labels.to_parquet(tmp_path / "labels_000000.parquet", index=False)

    assert good_option._existing_label_ids(
        tmp_path, catalog=_catalog()
    ) == {"C1"}


def test_split_strategy_none_uses_every_row_for_training() -> None:
    frame = pd.DataFrame(
        {
            "patient_group_id": ["P1", "P2"],
            "drug_id": ["D1", "D2"],
        }
    )
    partitions = good_option.assign_partitions(
        frame,
        strategy="none",
        patient_validation_fraction=0.2,
        drug_validation_fraction=0.2,
        seed=42,
    )
    assert partitions.tolist() == ["train", "train"]


def test_patient_and_drug_holdouts_do_not_leak_into_training() -> None:
    frame = pd.DataFrame(
        [
            {
                "patient_group_id": f"P{patient}",
                "drug_id": f"D{drug}",
            }
            for patient in range(20)
            for drug in range(10)
        ]
    )
    frame["partition"] = good_option.assign_partitions(
        frame,
        strategy="patient_drug",
        patient_validation_fraction=0.35,
        drug_validation_fraction=0.35,
        seed=7,
    )
    assert set(frame["partition"]).issubset(
        {"train", "unseen_patient", "unseen_drug", "strict_validation"}
    )
    train_patients = set(
        frame.loc[frame["partition"].eq("train"), "patient_group_id"]
    )
    held_patients = set(
        frame.loc[
            frame["partition"].isin({"unseen_patient", "strict_validation"}),
            "patient_group_id",
        ]
    )
    train_drugs = set(frame.loc[frame["partition"].eq("train"), "drug_id"])
    held_drugs = set(
        frame.loc[
            frame["partition"].isin({"unseen_drug", "strict_validation"}),
            "drug_id",
        ]
    )
    assert train_patients.isdisjoint(held_patients)
    assert train_drugs.isdisjoint(held_drugs)


def test_four_logit_metrics_report_per_criterion_and_macro_auc() -> None:
    evaluation = SimpleNamespace(
        predictions=np.asarray(
            [[5.0, 5.0, -5.0, -5.0], [-5.0, -5.0, 5.0, 5.0]]
        ),
        label_ids=np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]]),
    )
    metrics = good_option.compute_metrics(evaluation)

    assert metrics["binary_accuracy"] == 1.0
    assert metrics["auroc_macro"] == 1.0
    assert all(f"auroc_{criterion}" in metrics for criterion in RUBRIC_CRITERIA)


def test_resume_checkpoint_must_match_catalog_and_split(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    split_manifest = {"split_fingerprint_sha256": "split-v2"}
    compatible = {
        "matchminer_catalog_compatibility_id": "compat-v2",
        "matchminer_checker_input_version": good_option.GOOD_OPTION_INPUT_VERSION,
        "matchminer_label_schema_version": good_option.GOOD_OPTION_LABEL_SCHEMA_VERSION,
        "matchminer_split_fingerprint_sha256": "split-v2",
    }
    (checkpoint / "config.json").write_text(
        json.dumps(compatible), encoding="utf-8"
    )

    good_option.validate_resume_checkpoint(
        checkpoint, catalog=_catalog(), split_manifest=split_manifest
    )

    incompatible = {**compatible, "matchminer_split_fingerprint_sha256": "old"}
    (checkpoint / "config.json").write_text(
        json.dumps(incompatible), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incompatible"):
        good_option.validate_resume_checkpoint(
            checkpoint, catalog=_catalog(), split_manifest=split_manifest
        )


def test_cli_exposes_hard_boundary_v2_stages_and_all_orchestration() -> None:
    parser = good_option.build_parser()

    catalog = parser.parse_args(["catalog", "--nct-id", "NCT12345678"])
    validate = parser.parse_args(["validate-catalog"])
    label = parser.parse_args(["label", "--max-candidates", "5"])
    train = parser.parse_args(["train", "--split-strategy", "none"])
    all_steps = parser.parse_args(["all", "--num-processes", "4"])

    assert catalog.catalog.endswith("good_option_catalog")
    assert validate.command == "validate-catalog"
    assert label.label_output.endswith("good_option_four_point_labels_v2.parquet")
    assert label.label_parse_attempts == 3
    assert train.output_dir.endswith("goodoptionchecker_four_point_v2")
    assert all_steps.num_processes == 4
    assert all_steps.handler is good_option.run_all
    with pytest.raises(SystemExit):
        parser.parse_args(["generate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["all", "--num-processes", "0"])


def test_all_runs_isolated_stages_in_order_and_forwards_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog"
    catalog_checkpoints = tmp_path / "catalog-checkpoints"
    args = good_option.build_parser().parse_args(
        [
            "all",
            "--catalog",
            str(catalog_path),
            "--catalog-checkpoint-dir",
            str(catalog_checkpoints),
            "--reset-catalog-checkpoints",
            "--nct-id",
            "NCT12345678",
            "--server-urls-file",
            "servers.json",
            "--max-candidates",
            "5",
            "--num-processes",
            "4",
            "--resume-from-checkpoint",
            "checkpoint-10",
            "--no-bf16",
        ]
    )
    expected = good_option._build_all_stage_commands(args)
    calls: list[tuple[list[str], bool]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        calls.append((command, check))

    monkeypatch.setattr(good_option.subprocess, "run", fake_run)
    good_option.run_all(args)

    assert calls == [(command, True) for _stage, command in expected]
    assert [stage for stage, _command in expected] == [
        "catalog",
        "validate-catalog",
        "label",
        "train",
    ]
    catalog, _validate, label, train = [command for _stage, command in expected]
    script = str(Path(good_option.__file__).resolve())
    for stage, command in expected:
        script_index = command.index(script)
        assert (
            good_option.build_parser().parse_args(command[script_index + 1 :]).command
            == stage
        )
    assert ["--nct-id", "NCT12345678"] == catalog[
        catalog.index("--nct-id") : catalog.index("--nct-id") + 2
    ]
    assert ["--catalog-checkpoint-dir", str(catalog_checkpoints)] == catalog[
        catalog.index("--catalog-checkpoint-dir") :
        catalog.index("--catalog-checkpoint-dir") + 2
    ]
    assert "--reset-catalog-checkpoints" in catalog
    assert ["--server-urls-file", "servers.json"] == label[
        label.index("--server-urls-file") : label.index("--server-urls-file") + 2
    ]
    assert ["--max-candidates", "5"] == label[
        label.index("--max-candidates") : label.index("--max-candidates") + 2
    ]
    assert train[1:3] == ["-m", "accelerate.commands.launch"]
    assert ["--num_processes", "4"] == train[
        train.index("--num_processes") : train.index("--num_processes") + 2
    ]
    assert "--no-bf16" in train
    assert ["--resume-from-checkpoint", "checkpoint-10"] == train[
        train.index("--resume-from-checkpoint") : train.index(
            "--resume-from-checkpoint"
        )
        + 2
    ]


def test_all_reuses_matching_completed_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog"
    catalog_path.mkdir()
    args = good_option.build_parser().parse_args(
        [
            "all",
            "--catalog",
            str(catalog_path),
            "--nct-id",
            "NCT12345678",
        ]
    )
    monkeypatch.setattr(
        good_option,
        "load_good_option_catalog",
        lambda _path: SimpleNamespace(
            trial_registry=pd.DataFrame({"trial_id": ["NCT12345678"]})
        ),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        good_option.subprocess,
        "run",
        lambda command, *, check: calls.append(command) if check else None,
    )

    good_option.run_all(args)

    script = str(Path(good_option.__file__).resolve())
    assert [command[command.index(script) + 1] for command in calls] == [
        "validate-catalog",
        "label",
        "train",
    ]


def test_all_reuses_catalog_that_covers_more_trials_than_this_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A catalog shared across corpora may hold trials this run does not need."""
    catalog_path = tmp_path / "catalog"
    catalog_path.mkdir()
    args = good_option.build_parser().parse_args(
        [
            "all",
            "--catalog",
            str(catalog_path),
            "--nct-id",
            "NCT12345678",
        ]
    )
    monkeypatch.setattr(
        good_option,
        "load_good_option_catalog",
        lambda _path: SimpleNamespace(
            trial_registry=pd.DataFrame(
                {"trial_id": ["NCT12345678", "NCT87654321"]}
            )
        ),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        good_option.subprocess,
        "run",
        lambda command, *, check: calls.append(command) if check else None,
    )

    good_option.run_all(args)

    script = str(Path(good_option.__file__).resolve())
    assert [command[command.index(script) + 1] for command in calls] == [
        "validate-catalog",
        "label",
        "train",
    ]


def test_all_rejects_completed_catalog_for_different_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog"
    catalog_path.mkdir()
    args = good_option.build_parser().parse_args(
        [
            "all",
            "--catalog",
            str(catalog_path),
            "--nct-id",
            "NCT12345678",
        ]
    )
    monkeypatch.setattr(
        good_option,
        "load_good_option_catalog",
        lambda _path: SimpleNamespace(
            trial_registry=pd.DataFrame({"trial_id": ["NCT87654321"]})
        ),
    )

    with pytest.raises(ValueError, match="missing 1 trial ID"):
        good_option.run_all(args)


def test_server_file_configures_shared_remote_teacher(tmp_path: Path) -> None:
    servers = tmp_path / "servers.json"
    servers.write_text(
        json.dumps(
            {
                "servers": [
                    {"url": "http://host-a:8000/v1"},
                    {"url": "http://host-b:8000/v1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    args = good_option.build_parser().parse_args(
        [
            "label",
            "--server-urls-file",
            str(servers),
            "--model",
            "teacher/model",
        ]
    )
    config = good_option.configure_teacher(args)

    assert config.remote["enabled"] is True
    assert config.remote["server_urls"] == [
        "http://host-a:8000/v1",
        "http://host-b:8000/v1",
    ]
    assert config.llm_good_option["remote"]["model_name"] == "teacher/model"
    assert (
        good_option._active_good_option_output_tokens(config)
        == good_option.MIN_GOOD_OPTION_OUTPUT_TOKENS
        == 100_000
    )
