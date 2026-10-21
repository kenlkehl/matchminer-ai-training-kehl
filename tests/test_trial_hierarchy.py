import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.test_trial_treatment_screening import FakeRequestSession, spaces
from tests.test_trial_treatment_screening import endpoint_run as screening_endpoint_run
from trial_hierarchy import Completion, Hierarchy, load_prompts
from trial_hierarchy_runner import _prepare_payload, local_extract, prompt_bundle

endpoint_run = screening_endpoint_run


def enable_hierarchy(argv):
    position = argv.index("--extraction-method")
    del argv[position : position + 2]


def test_default_cli_screens_then_runs_seven_fields_and_separate_boilerplate(
    endpoint_run,
):
    state, path, argv = endpoint_run
    enable_hierarchy(argv)
    spaces.main()
    assert [c[0] for c in state.calls] == ["screen"] * 3 + ["extract"] * 8
    # Boilerplate is independent; concurrent cache lookups may dispatch it first.
    field_calls = [text for _, text in state.calls[3:] if "CURRENT FIELD:" in text]
    assert "CURRENT FIELD: age" in field_calls[0]
    assert (
        sum("Extract general exclusion criteria" in call[1] for call in state.calls)
        == 1
    )
    result = pd.read_csv(path / "spaces.csv")
    assert len(result) == 1
    assert result.space_number.tolist() == [1]
    assert result.trial_boilerplate_text.tolist() == ["Active infection."]
    assert "Histology allowed: adenocarcinoma." in result.this_space.iloc[0]
    nodes = path / "work/space_extraction/hierarchy_nodes"
    assert len(list(nodes.rglob("*.json"))) == 8
    spaces.main()
    assert len(state.calls) == 11


def test_remote_resume_reuses_flat_node_cache_after_directory_sharding(endpoint_run):
    state, path, argv = endpoint_run
    enable_hierarchy(argv)
    spaces.main()
    work = path / "work/space_extraction"
    nodes = work / "hierarchy_nodes"
    for checkpoint in list(nodes.rglob("*.json")):
        checkpoint.replace(nodes / checkpoint.name)
    for shard in work.glob("remote_output_shard_*.parquet"):
        shard.unlink()
    previous = pd.read_csv(path / "spaces.csv")
    previous_trials = pd.read_csv(path / "out.csv")
    spaces.main()
    assert len(state.calls) == 11
    result = pd.read_csv(path / "spaces.csv")
    pd.testing.assert_frame_equal(result, previous)
    assert result.reasoning_trials_path.tolist() == [str((path / "out.csv").resolve())]
    trials = pd.read_csv(path / "out.csv")
    assert trials.nct_id.tolist() == result.nct_id.tolist()
    # Independent boilerplate/field completions can appear in a different order.
    # Full reasoning is stored once per trial, outside the compact space rows.
    column = "space_reasoning_and_output"
    assert column not in result.columns
    pd.testing.assert_frame_equal(trials.drop(columns=column), previous_trials.drop(columns=column))
    before, after = (json.loads(frame[column].iloc[0]) for frame in (previous_trials, trials))
    assert {node['request_key']: node for node in before.pop('nodes')} == {
        node['request_key']: node for node in after.pop('nodes')
    }
    assert before == after


def test_sharded_nodes_still_require_a_prompt_manifest(endpoint_run):
    state, path, argv = endpoint_run
    enable_hierarchy(argv)
    spaces.main()
    work = path / "work/space_extraction"
    (work / "extraction_manifest.json").unlink()
    for shard in work.glob("remote_output_shard_*.parquet"):
        shard.unlink()
    with pytest.raises(ValueError, match="lack a prompt manifest"):
        spaces.main()
    assert len(state.calls) == 11


@pytest.mark.parametrize(
    "prompt_key", ["age", "anti_boilerplate", "omissions", "response", "boilerplate_response"]
)
def test_hierarchical_prompt_change_rejects_checkpoint_reuse_without_rescreening(
    endpoint_run, monkeypatch, prompt_key
):
    import trial_hierarchy_runner

    state, path, argv = endpoint_run
    enable_hierarchy(argv)
    spaces.main()
    old = prompt_bundle()
    monkeypatch.setattr(
        trial_hierarchy_runner,
        "prompt_bundle",
        lambda: {**old, prompt_key: old[prompt_key] + "\nChange"},
    )
    with pytest.raises(ValueError, match="fresh extraction checkpoints"):
        spaces.main()
    assert len(state.calls) == 11
    assert len(list((path / "work/treatment_screening").glob("*.parquet"))) == 1


def test_one_shot_checkpoints_cannot_silently_mix_with_hierarchy(endpoint_run):
    state, _, argv = endpoint_run
    spaces.main()
    enable_hierarchy(argv)
    with pytest.raises(ValueError, match="fresh extraction checkpoints"):
        spaces.main()
    assert len(state.calls) == 4


def test_remote_trial_batches_preserve_ids_and_one_registry(endpoint_run):
    state, path, argv = endpoint_run
    enable_hierarchy(argv)
    argv.extend(["--hierarchy-trial-batch-size", "1"])
    source = pd.read_csv(path / "trials.csv")
    for index in range(3):
        source.loc[index, "trial_text"] = f"ANTI_CANCER {index}"
    source.to_csv(path / "trials.csv", index=False)
    spaces.main()
    assert len(state.calls) == 27  # 3 screens, then 8 calls per trial.
    assert len(state.registries) == 2  # Screening and one shared extraction pool.
    assert pd.read_csv(path / "spaces.csv").nct_id.tolist() == [
        "NCT00000001",
        "NCT00000002",
        "NCT00000003",
    ]
    shards = path / "work/space_extraction"
    assert len(list(shards.glob("remote_output_shard_*.parquet"))) == 3


def test_failed_trial_does_not_cancel_other_trials_or_export_partial_csvs(endpoint_run, monkeypatch):
    import trial_hierarchy_runner
    state, path, argv = endpoint_run
    enable_hierarchy(argv)
    source = pd.read_csv(path/'trials.csv')
    for index in range(3):
        source.loc[index, 'trial_text'] = f'ANTI_CANCER {index}'
    source.to_csv(path/'trials.csv', index=False)
    original_get = trial_hierarchy_runner.ResponseCache.get

    def cached_prune(cache, request):
        if request.trial_index == 0 and request.field_name == 'cancer_primary_site':
            return Completion('{"branches": [], "prune_reason": "No cancer population in this synthetic case"}',
                              finish_reason='stop')
        return original_get(cache, request)

    monkeypatch.setattr(trial_hierarchy_runner.ResponseCache, 'get', cached_prune)
    with pytest.raises(RuntimeError, match='1 trials need review; 2 complete trials remain checkpointed'):
        spaces.main()
    directory = path/'work/space_extraction'
    assert set(json.loads((directory/'extraction_failures.json').read_text())) == {'NCT00000001'}
    assert len(list(directory.glob('remote_output_shard_*.parquet'))) == 2
    assert len(list((directory/'hierarchy_traces').glob('*.json'))) == 3
    assert not (path/'out.csv').exists() and not (path/'spaces.csv').exists()
    audit = pd.read_csv(path/'trial_treatment_screening.csv')
    assert audit.is_cancer_treatment_trial.tolist() == [True, True, True]


def test_shared_core_and_prompts_are_identical_to_inference():
    root = Path(__file__).resolve().parents[2]
    inference = root / "matchminer-ai-inference-kehl/src/matchminer_ai"
    if not inference.is_dir():
        pytest.skip("Sibling inference checkout required for cross-repository contract")
    assert (inference / "trials/hierarchy.py").read_bytes() == (
        root / "matchminer-ai-training-kehl/trial_hierarchy.py"
    ).read_bytes()
    assert (inference / "_trial_space_format.py").read_bytes() == (
        root / "matchminer-ai-training-kehl/trial_space_format.py"
    ).read_bytes()
    assert load_prompts(inference / "prompts") == prompt_bundle()
    assert (inference / "prompts/trial.user.primer.txt").read_bytes() == (
        root / "matchminer-ai-training-kehl/trial_prompts/trial.user.primer.txt"
    ).read_bytes()
    assert (
        spaces.PROMPT_HEADER
        == (
            root / "matchminer-ai-training-kehl/trial_prompts/trial.user.primer.txt"
        ).read_text()
    )


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def apply_chat_template(self, conversation, **kwargs):
        return json.dumps(conversation)


def test_per_node_context_budget_and_beam_effort():
    request = Hierarchy(0, "source", prompt_bundle()).requests()[0]
    first = _prepare_payload(request, Tokenizer(), 45000, 100000, beam=True)
    bounded_context = len(first["prompt"].split()) + 512
    fallback = _prepare_payload(
        request, Tokenizer(), 45000, bounded_context, beam=True, failures=2
    )
    assert first["chat_template_kwargs"] == {"reasoning_effort": "max"}
    assert fallback["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert (
        fallback["max_tokens"]
        == bounded_context - len(fallback["prompt"].split()) - 256
    )
    with pytest.raises(ValueError, match="no output budget"):
        _prepare_payload(request, Tokenizer(), 100, 1)


def test_local_hierarchy_keeps_model_resident_retries_and_resumes(
    tmp_path, monkeypatch
):
    import vllm_reasoning_utils

    monkeypatch.setattr(
        vllm_reasoning_utils,
        "parse_reasoning_output",
        lambda text, *a, **kw: ("trace", text),
    )
    (tmp_path / "extraction_manifest.json").write_text('{"test": "local"}')
    calls = []

    class LLM:
        def generate(self, prompts, sampling):
            results = []
            for prompt in prompts:
                calls.append(prompt)
                text = json.loads(prompt)[1]["content"]
                if len(calls) == 1:
                    final = "{bad json"
                elif "CURRENT FIELD:" not in text:
                    final = '{"exclusions": []}'
                else:
                    field = text.split("CURRENT FIELD: ")[1].splitlines()[0]
                    value = {
                        "age": "18+",
                        "sex": "Both",
                        "cancer_primary_site": "lung cancer",
                        "histology": "adenocarcinoma",
                        "disease_state": "metastatic",
                    }.get(field, "NA")
                    final = json.dumps(
                        {
                            "branches": [{"value": value, "scope": "all"}],
                            "prune_reason": "",
                        }
                    )
                results.append(
                    SimpleNamespace(
                        outputs=[SimpleNamespace(text=final, finish_reason="stop")]
                    )
                )
            return results

    args = SimpleNamespace(
        batch_size=10, max_attempts=3, max_tokens=1000, max_model_len=10000
    )
    trials = pd.DataFrame([{"nct_id": "T1", "trial_text": "synthetic trial"}])
    output = tmp_path / "output_shard_000.parquet"
    for _ in range(2):
        local_extract(
            LLM(),
            Tokenizer(),
            SimpleNamespace(max_tokens=1000),
            trials,
            args,
            "none",
            output,
        )
    assert len(calls) == 9
    assert "failed validation" in calls[1]
    assert (
        pd.read_parquet(output)
        .space_text.iloc[0]
        .startswith("1. Age range allowed: 18+.")
    )
    assert "validation_failures" in " ".join(p.name for p in tmp_path.iterdir())


def test_invalid_child_never_exports_successful_siblings():
    state = Hierarchy(0, "source", prompt_bundle())
    state.accept(
        state.requests(),
        [
            Completion(
                json.dumps(
                    {
                        "branches": [
                            {"value": "18+", "scope": "A"},
                            {"value": "40+", "scope": "B"},
                        ],
                        "prune_reason": "",
                    }
                )
            )
        ],
    )
    state.accept(
        state.requests(),
        [
            Completion(
                '{"branches": [{"value": "Both", "scope": "A"}], "prune_reason": ""}'
            ),
            Completion("bad json"),
        ],
    )
    with pytest.raises(ValueError):
        state.output()


def test_remote_validation_feedback_preserves_concurrency_classification_and_resumes(
    tmp_path, monkeypatch
):
    import transformers

    import remote_vllm_pool as pool
    import trial_hierarchy_runner as runner
    import vllm_reasoning_utils
    from remote_vllm_pool import OutputValidationError

    (tmp_path / "extraction_manifest.json").write_text('{"test": "remote retry"}')
    calls, templates, validation_errors = [], [], []

    class TrackingTokenizer(Tokenizer):
        def apply_chat_template(self, conversation, **kwargs):
            templates.append((conversation[1]["content"], kwargs))
            return super().apply_chat_template(conversation, **kwargs)

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: TrackingTokenizer(),
    )
    monkeypatch.setattr(
        vllm_reasoning_utils,
        "parse_reasoning_output",
        lambda text, *a, **kw: ("trace", text),
    )

    async def create(**kwargs):
        messages = json.loads(kwargs["prompt"])
        calls.append(messages)
        text = messages[1]["content"]
        age_calls = [c for c in calls if "CURRENT FIELD: age\n" in c[1]["content"]]
        if "CURRENT FIELD: age\n" in text and len(age_calls) <= 2:
            final = "bad JSON"
        elif "CURRENT FIELD:" not in text:
            final = '{"exclusions": []}'
        else:
            field = text.split("CURRENT FIELD: ")[1].splitlines()[0]
            value = {
                "age": "18+",
                "sex": "Both",
                "cancer_primary_site": "lung cancer",
                "histology": "adenocarcinoma",
                "disease_state": "metastatic",
            }.get(field, "NA")
            final = json.dumps(
                {"branches": [{"value": value, "scope": "all"}], "prune_reason": ""}
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(text=final, finish_reason="stop")]
        )

    async def stop():
        pass

    async def run_pool(*, work_items, work_fn, shard_writer, **kwargs):
        completed = []
        client = SimpleNamespace(completions=SimpleNamespace(create=create))
        for work_id, payload in work_items:
            for _ in range(3):
                try:
                    result = await work_fn(client, payload)
                except OutputValidationError as exc:
                    validation_errors.append(str(exc))
                    continue
                completed.append((work_id, result))
                break
            else:
                pytest.fail("Valid correction was not accepted")
        shard_writer(completed, 0)

    monkeypatch.setattr(
        pool, "build_registry_from_args", lambda args: SimpleNamespace(stop=stop)
    )
    monkeypatch.setattr(pool, "run_pool", run_pool)
    monkeypatch.setattr(pool, "RequestSession", FakeRequestSession)
    args = SimpleNamespace(
        model="nemotron",
        tokenizer_model="cached",
        download_dir=None,
        temperature=1.0,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=1000,
        max_model_len=10000,
        max_attempts=3,
        work_dir=str(tmp_path),
    )
    trials = pd.DataFrame([{"nct_id": "T1", "trial_text": "synthetic"}])
    runner.remote_main(args, trials, "nemotron_v3")
    assert len(calls) == 10
    assert len(validation_errors) == 2
    age_calls = [c for c in calls if "CURRENT FIELD: age\n" in c[1]["content"]]
    assert "failed validation" in age_calls[1][-1]["content"]
    age_templates = [p for text, p in templates if "CURRENT FIELD: age\n" in text]
    assert [p["enable_thinking"] for p in age_templates] == [True, True, False]
    runner.remote_main(args, trials, "nemotron_v3")
    assert len(calls) == 10


def test_fenced_remote_field_and_boilerplate_answers_need_no_retry(endpoint_run):
    state, path, argv = endpoint_run
    enable_hierarchy(argv)
    state.fenced = True
    spaces.main()
    assert len(state.calls) == 11  # Three screens plus seven fields and boilerplate.
    work = path / "work/space_extraction"
    assert not (work / "validation_failures.jsonl").exists()
    assert len(pd.read_csv(path / "spaces.csv")) == 1
    nodes = [json.loads(p.read_text()) for p in (work / "hierarchy_nodes").rglob("*.json")]
    assert len(nodes) == 8
    assert all(node["completion"]["final"].startswith("```json\n") for node in nodes)
    spaces.main()
    assert len(state.calls) == 11  # Fenced checkpoint responses also resume without calls.


def test_resume_reuses_completed_shard_without_node_replay_or_rewriting(endpoint_run, monkeypatch):
    import trial_hierarchy_runner as runner

    state, path, argv = endpoint_run
    enable_hierarchy(argv)
    spaces.main()
    shard = next((path / "work/space_extraction").glob("remote_output_shard_*.parquet"))
    before = (shard.read_bytes(), shard.stat().st_mtime_ns)

    def unexpected_read(*args):
        pytest.fail("Completed trial should not replay its node cache")

    monkeypatch.setattr(runner.ResponseCache, "get", unexpected_read)
    spaces.main()
    assert (shard.read_bytes(), shard.stat().st_mtime_ns) == before
    assert len(state.calls) == 11
    with pytest.raises(ValueError, match="different trial"):
        runner._completed_trial_path(shard.parent, 0, "wrong-trial")


def test_remote_cpu_work_and_checkpoint_io_keep_event_loop_responsive(endpoint_run, monkeypatch):
    import asyncio
    import threading
    import trial_hierarchy_runner as runner
    import vllm_reasoning_utils

    state, path, argv = endpoint_run
    enable_hierarchy(argv)
    loop = None
    original_schedule = runner.run_ready_hierarchies

    async def schedule(*args, **kwargs):
        nonlocal loop
        loop = asyncio.get_running_loop()
        return await original_schedule(*args, **kwargs)

    def checked_io(original):
        def wrapped(*args, **kwargs):
            heartbeat = threading.Event()
            loop.call_soon_threadsafe(heartbeat.set)
            assert heartbeat.wait(2), "Checkpoint I/O blocked the request loop"
            return original(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(runner, "run_ready_hierarchies", schedule)
    monkeypatch.setattr(runner, "_prepare_payload", checked_io(runner._prepare_payload))
    parse = vllm_reasoning_utils.parse_reasoning_output

    def checked_parse(*args, **kwargs):
        if loop is not None:
            return checked_io(parse)(*args, **kwargs)
        return parse(*args, **kwargs)

    monkeypatch.setattr(vllm_reasoning_utils, "parse_reasoning_output", checked_parse)
    monkeypatch.setattr(runner.ResponseCache, "get", checked_io(runner.ResponseCache.get))
    monkeypatch.setattr(runner.ResponseCache, "put", checked_io(runner.ResponseCache.put))
    monkeypatch.setattr(runner, "_save_results", checked_io(runner._save_results))
    spaces.main()
    assert len(state.calls) == 11
    assert len(pd.read_csv(path / "spaces.csv")) == 1
