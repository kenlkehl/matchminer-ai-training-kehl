from types import SimpleNamespace
import json
from unittest.mock import patch

import pandas as pd
import pytest

from make_top_matches import load_retrieval_tables, sample_candidates
from oncoreasoning_training import contracts as c
from oncoreasoning_training.create_all_training_data import component_tasks
from oncoreasoning_training.prepare_boilerplate import extract, validate_criteria
from oncoreasoning_training.teacher import DEFAULT_TEACHER, TeacherPool, server_urls
import train_from_summaries as runner


def test_mining_preserves_identity_and_split_and_samples_distinct_trials(tmp_path):
    patients = tmp_path / "patients.parquet"
    trials = tmp_path / "trials.csv"
    pd.DataFrame({"pseudo_mrn": ["p1", "p2", "p3"], "patient_summary": ["x", "y", "z"],
                  "split": ["train", "val", "test"]}).to_parquet(patients)
    pd.DataFrame({"nct_id": ["NCT1", "NCT1", "NCT2", "NCT3", "NCT4"],
                  "this_space": ["1. shared", "2. other", "shared", "val space", "test space"],
                  "split": ["train", "train", "train", "val", "test"]}).to_csv(trials, index=False)
    args = SimpleNamespace(patients_parquet=str(patients), trials_file=str(trials), splits=["train", "val"])
    pts, spaces = load_retrieval_tables(args)
    assert pts.pseudo_mrn.tolist() == ["p1", "p2"]
    assert spaces.nct_id.tolist() == ["NCT1", "NCT1", "NCT2", "NCT3"]
    pt_to_sp, sp_to_pt = sample_candidates(pts, spaces, 2, 500, 42)
    assert set(pt_to_sp[0]) == {0, 1, 2}  # Two NCT IDs expand to three spaces.
    assert set(pt_to_sp[1]) == {3}
    assert [list(indices) for indices in sp_to_pt] == [[0], [0], [0], [1]]


def test_boilerplate_extraction_requires_complete_verbatim_rules():
    source = "Boilerplate exclusions:\n1. Active pneumonitis, except resolved grade 1 disease.\n2. Creatinine clearance <30."
    criteria = ["Active pneumonitis, except resolved grade 1 disease.", "Creatinine clearance <30."]
    assert validate_criteria(source, criteria) == criteria
    for invalid in (criteria[:1], ["Active pneumonitis.", criteria[1]],
                    [criteria[0], "Creatinine clearance"]):
        with pytest.raises(ValueError):
            validate_criteria(source, invalid)
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
        reasoning="private reasoning", content="<think>private</think>" + json.dumps({"criteria": criteria})))])
    assert extract(source, lambda _: response) == criteria


def test_boilerplate_questions_are_individual_and_preserve_unknown_rule():
    pair = {"patient_id": "p", "patient_summary": "Synthetic summary", "patient_boilerplate": "Pneumonitis not documented",
            "trial_id": "NCT1", "trial_summary": "space", "split": "train",
            "exclusion_criteria": ["Active pneumonitis", "Severe heart failure"]}
    tasks = list(component_tasks(pair, c.load_prompt_sources()))
    questions = [task for task in tasks if task["family"] == "boilerplatechecking"]
    assert len(questions) == 2
    for task, excluded in zip(questions, reversed(pair["exclusion_criteria"])):
        prompt = str(task["messages"])
        assert excluded not in prompt
        assert "benefit of the doubt and answer No" in prompt
        assert "A. Yes" in prompt and "B. No" in prompt
        assert "very next token" in prompt


def test_pool_round_robins_the_same_teacher_with_thinking(tmp_path):
    path = tmp_path / "servers.json"
    path.write_text(json.dumps({"servers": [{"url": "http://127.0.0.1:8100/v1"}, {"url": "http://127.0.0.1:8101/v1"}]}))
    calls = []
    def client(**kwargs):
        def create(**request):
            calls.append((kwargs["base_url"], request))
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch("openai.OpenAI", side_effect=client):
        pool = TeacherPool(server_urls(file=path))
        for _ in range(3):
            pool([{"role": "user", "content": "fabricated"}])
    assert calls[0][0] == calls[2][0] != calls[1][0]
    assert all(request["model"] == DEFAULT_TEACHER and request["extra_body"]["chat_template_kwargs"]["enable_thinking"]
               for _, request in calls)


def test_integrated_plan_starts_with_base_mining_and_trains_only_two_models():
    args = runner.parser().parse_args([])
    plan = runner.stages(args)
    assert plan[0].name == "mine-1" and args.embedding_model in plan[0].command
    assert [stage.name for stage in plan if stage.name.startswith("train-")] == [
        "train-trialspace-1", "train-trialspace-2", "train-oncoreasoning"]
    assert plan[-1].command[-3:] == ["--fsdp", "--resume-from-checkpoint", "auto"]
    for stage in plan:
        if stage.teacher:
            assert DEFAULT_TEACHER in stage.command
    prepare = next(stage for stage in plan if stage.name == "prepare-oncoreasoning")
    assert "--boilerplate-components" in prepare.command and "--catalog" in prepare.command
    command = runner.teacher_command(args, 0)
    assert "127.0.0.1" in command and "modelopt" in command and "--language-model-only" in command
