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


def test_default_cli_screens_then_runs_nine_fields_and_separate_boilerplate(
    endpoint_run,
):
    state, path, argv = endpoint_run
    enable_hierarchy(argv)
    spaces.main()
    assert [c[0] for c in state.calls] == ["screen"] * 3 + ["extract"] * 10
    assert "CURRENT FIELD: age" in state.calls[3][1]
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
    assert len(list(nodes.glob("*.json"))) == 10
    spaces.main()
    assert len(state.calls) == 13


@pytest.mark.parametrize(
    "prompt_key", ["age", "definitions", "response", "boilerplate_response"]
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
    assert len(state.calls) == 13
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
    assert len(state.calls) == 33  # 3 screens, then 10 calls per trial.
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
        if request.trial_index == 0 and request.field_name == 'cancer_type':
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
                        "cancer_type": "lung cancer",
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
    assert len(calls) == 11
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
            templates.append(kwargs)
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
        if len(calls) <= 2:
            final = "bad JSON"
        elif "CURRENT FIELD:" not in text:
            final = '{"exclusions": []}'
        else:
            field = text.split("CURRENT FIELD: ")[1].splitlines()[0]
            value = {
                "age": "18+",
                "sex": "Both",
                "cancer_type": "lung cancer",
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
    assert len(calls) == 12
    assert len(validation_errors) == 2
    assert "failed validation" in calls[1][-1]["content"]
    assert [p["enable_thinking"] for p in templates[:3]] == [True, True, False]
    runner.remote_main(args, trials, "nemotron_v3")
    assert len(calls) == 12
