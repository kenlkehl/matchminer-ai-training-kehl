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


def test_resume_ignores_checkpoint_interrupted_during_save(tmp_path):
    from training_checkpoints import latest_complete_checkpoint
    complete, partial = tmp_path / "checkpoint-100", tmp_path / "checkpoint-200"
    complete.mkdir(); partial.mkdir()
    (complete / "trainer_state.json").write_text('{"global_step": 100}')
    (partial / "trainer_state.json").write_text('{"global_step":')
    assert latest_complete_checkpoint(tmp_path) == str(complete)
    (partial / "trainer_state.json").write_text('{"global_step": 200}')
    assert latest_complete_checkpoint(tmp_path) == str(partial)


def test_teacher_discovers_its_own_pip_cuda_toolkit(monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda command, **_: (
        "/teacher/lib/python3.13/site-packages/nvidia/cu13\n"
        if command[0] == "/teacher/bin/python" else pytest.fail("Wrong teacher environment")))
    env = runner.teacher_environment(SimpleNamespace(vllm="/teacher/bin/vllm"))
    assert env["CUDA_HOME"] == "/teacher/lib/python3.13/site-packages/nvidia/cu13"
    assert env["PATH"].startswith(env["CUDA_HOME"] + "/bin:")


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
    assert command[command.index("--max-num-seqs") + 1] == "96"
    for stage in plan:
        if stage.name.startswith("label-"):
            assert stage.command[stage.command.index("--max_concurrent_per_server") + 1] == "96"


def test_adapter_upgrade_preserves_original_fingerprint_and_rejects_changed_inputs(tmp_path, monkeypatch):
    from oncoreasoning_training import pipeline_upgrade as upgrade
    previous = {"args": {"student_model": "gemma"}, "training_commit": "old", "inputs": ["source-hash"]}
    current = {**previous, "training_commit": "new"}
    with pytest.raises(ValueError, match="changed"):
        upgrade.compatible_identity(previous, current, tmp_path, tmp_path)
    receipt = {"kind": "separate-oncoreasoning-lora-v1", "original_identity_sha256": c.digest(previous),
               "approved_identity_sha256": c.digest(current)}
    (tmp_path / "oncoreasoning_code_upgrade.json").write_text(json.dumps(receipt))
    calls = []
    monkeypatch.setattr(upgrade, "validate_code_scope", lambda *args: calls.append(args))
    assert upgrade.compatible_identity(previous, current, tmp_path, tmp_path) == previous
    assert calls == [(tmp_path, "old", "new")]
    for invalid in ({**current, "inputs": ["changed"]}, {**current, "training_commit": "other"}):
        with pytest.raises(ValueError, match="changed"):
            upgrade.compatible_identity(previous, invalid, tmp_path, tmp_path)


def test_adapter_upgrade_rejects_upstream_code_or_plan_changes(monkeypatch):
    from oncoreasoning_training import pipeline_upgrade as upgrade
    old = "def stages(args):\n    return ['mine', 'train']\n\ndef main():\n    pass\n"
    def git(repo, *args):
        if args[:2] == ("diff", "--name-only"):
            return "train_from_summaries.py"
        if args[0] == "show":
            return old if args[1].startswith("old:") else old.replace("'mine'", "'changed'")
        return ""
    monkeypatch.setattr(upgrade, "git", git)
    with pytest.raises(ValueError, match="stage plan"):
        upgrade.validate_code_scope("repo", "old", "new")
    monkeypatch.setattr(upgrade, "git", lambda *_: "make_top_matches.py")
    with pytest.raises(ValueError, match="outside"):
        upgrade.validate_code_scope("repo", "old", "new")


def test_adapter_upgrade_refuses_started_training(tmp_path):
    from oncoreasoning_training.pipeline_upgrade import register_upgrade
    (tmp_path / "pipeline_manifest.json").write_text(json.dumps({"args": {"models_dir": str(tmp_path / "models")}}))
    (tmp_path / "status.json").write_text('{"stage":"train-oncoreasoning","status":"running"}')
    with pytest.raises(ValueError, match="before"):
        register_upgrade(tmp_path, tmp_path)


def test_execution_upgrade_only_normalizes_positive_scheduling_limits():
    import ast
    from oncoreasoning_training.pipeline_upgrade import SchedulingNormalizer
    def normalize(source):
        return ast.dump(SchedulingNormalizer().visit(ast.parse(source)))
    before = "['--max_concurrent_per_server', '4', '--max-num-seqs', '32', '--model', 'teacher']"
    after = before.replace("'4'", "'64'").replace("'32'", "'64'")
    assert normalize(before) == normalize(after)
    assert normalize(before) != normalize(after.replace("'teacher'", "'different'"))
    assert normalize(before) != normalize(after.replace("'64'", "'0'"))
