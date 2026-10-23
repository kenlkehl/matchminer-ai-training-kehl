import asyncio
import json
from types import SimpleNamespace

import pytest

import review_trial_spaces
from trial_space_refinement import format_space


@pytest.mark.parametrize("task", ["refinement", "residual-exclusions", "residual-criteria"])
def test_review_transport_retries_with_feedback_and_resumes_saved_response(tmp_path, monkeypatch, task):
    import openai
    import remote_vllm_pool
    import vllm_reasoning_utils as reasoning

    space = dict(age="18 years or older", sex="no restriction", cancer_primary_site="lung",
                 histology="adenocarcinoma", disease_state="advanced/metastatic",
                 prior_treatment_requirements="NA", biomarker_requirements="NA")
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps([dict(case_id="synthetic-1", nct_id="SYNTHETIC",
                                         trial_text="Synthetic adult lung study.", space_text=format_space(space))]))
    args = SimpleNamespace(task=task, input=input_file, output_dir=tmp_path / "results", endpoint="http://synthetic/v1",
                           model="synthetic", tokenizer="synthetic", concurrency=1, max_attempts=4,
                           max_tokens=1000, max_model_len=10000, timeout=10, api_key_env="SYNTHETIC_KEY", max_refinements=3)
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is True
            return json.dumps(messages)

        def encode(self, prompt, **kwargs):
            return list(range(100))

    class Client:
        def __init__(self, **kwargs):
            self.models = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="synthetic")], model_dump=lambda: {"data": [{"id": "synthetic"}]})

    def make_work(*args):
        async def work(client, payload):
            calls.append(payload)
            assert payload["enable_thinking"] is True
            if task == "refinement":
                spaces = [] if len(calls) == 1 else [space]
                final = json.dumps(dict(status="unchanged", spaces=spaces, explanation="Faithful to source."))
            elif task == "residual-exclusions":
                final = json.dumps({"exclusions": [42] if len(calls) == 1 else ["ECOG performance status greater than 1."]})
                assert "boilerplate" not in payload["prompt"].lower()
                assert "FULL AUTHORED PRIMER" not in payload["prompt"]
            else:
                final = json.dumps({"criteria": [42] if len(calls) == 1 else []})
                assert "AUTHORED CORE SPACE DEFINITIONS" in payload["prompt"]
            return "Saved reasoning", final, {"finish_reason": "stop", "raw_text": final}
        return work

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    monkeypatch.setattr(reasoning, "load_tokenizer", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(reasoning, "resolve_parser_name", lambda *a: "synthetic")
    monkeypatch.setattr(reasoning, "resolve_sampling_defaults", lambda *a, **k: {})
    monkeypatch.setattr(remote_vllm_pool, "make_completion_work_fn", make_work)
    asyncio.run(review_trial_spaces.run(args))
    assert len(calls) == 2
    expected_error = {"refinement": "cannot silently remove a space",
                      "residual-exclusions": "nonempty single-line string",
                      "residual-criteria": "Each criterion must contain"}[task]
    assert expected_error in calls[1]["prompt"]
    case_dir = args.output_dir / "case_0000"
    call_dir = case_dir / "check_00" if task == "residual-criteria" else case_dir
    assert json.loads((call_dir / "attempt_02.json").read_text())["reasoning"] == "Saved reasoning"
    # Restore from a validated raw response if interrupted before final result save.
    (case_dir / "result.json").unlink()
    if task == "residual-criteria":
        (call_dir / "result.json").unlink()
    asyncio.run(review_trial_spaces.run(args))
    assert len(calls) == 2
    assert json.loads((call_dir / "result.json").read_text())["attempts"] == 2
    asyncio.run(review_trial_spaces.run(args))
    assert len(calls) == 2
    args.max_tokens = 999
    with pytest.raises(ValueError, match="fresh output"):
        asyncio.run(review_trial_spaces.run(args))
