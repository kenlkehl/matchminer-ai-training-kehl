"""Fabricated-data regressions for final-only, answer-first distillation."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from oncoreasoning_training import contracts as c
from oncoreasoning_training import create_all_training_data as pipeline


class CharTokenizer:
    """Transparent token boundaries for chunk coverage and masking assertions."""
    chat_template = "test template with an explicit assistant boundary"
    model_max_length = 1000000
    name_or_path = "fabricated-character-tokenizer"

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def get_vocab(self):
        return {chr(i): i for i in range(256)}

    def __len__(self):
        return 256

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        assert tokenize is False
        rendered = "".join(f"<{row['role']}>\n{row['content']}<end>\n" for row in messages)
        return rendered + ("<assistant>\n" if add_generation_prompt else "")

    def save_pretrained(self, directory):
        Path(directory).mkdir(exist_ok=True)
        (Path(directory) / "fake_tokenizer.json").write_text("{}")


@pytest.fixture
def tokenizer():
    return CharTokenizer()


@pytest.fixture
def sources():
    return c.load_prompt_sources()


def response(answer, *, finish="stop", reasoning="TEACHER SECRET REASONING"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
        message=SimpleNamespace(content=answer, reasoning_content=reasoning))])


@dataclass(frozen=True)
class ClassEvidence:
    class_id: str
    drug_names: tuple


class Catalog:
    compatibility_id = "fabricated-catalog-v3"

    def trial_status(self, trial):
        return "ok"

    def scoreable_summaries_for_trial(self, trial):
        return (SimpleNamespace(drug_id="drug-1", preferred_name="Fabricated drug"),)

    def class_evidence_for_trial(self, trial):
        return (ClassEvidence("covering-class", ("Fabricated drug", "Other drug")),
                ClassEvidence("unrelated-class", ("Unrelated drug",)))


def evidence_packer(drugs, *, class_evidence):
    assert len(drugs) == 1
    assert len(class_evidence) == 1
    assert class_evidence[0].drug_names == ("Fabricated drug",)
    return "scope: agent: fabricated evidence. scope: class: class covers Fabricated drug.", []


def test_default_model_and_chunk_contract():
    args = pipeline.parser().parse_args(["prepare"])
    assert args.model_name == "google/gemma-4-E4B-it"
    assert args.tasks == ["summarization", "clinical_qa"]
    assert (args.chunk_min_tokens, args.chunk_max_tokens) == (10000, 50000)


def test_chunks_are_variable_reproducible_complete_and_date_aware(tokenizer):
    notes = [("2020-01-01", "a" * 45000), ("2020-02-01", "b" * 150000)]
    chunks = list(pipeline.variable_chunks(notes, tokenizer))
    assert chunks == list(pipeline.variable_chunks(notes, tokenizer))
    assert len({x["target_chunk_tokens"] for x in chunks}) > 1
    assert all(10000 <= x["target_chunk_tokens"] <= 50000 for x in chunks)
    assert all(x["chunk_tokens"] == len(x["chunk_text"]) <= 50000 for x in chunks)
    assert all(x["chunk_tokens"] >= 10000 for x in chunks[:-1])
    assert chunks[0]["token_start"] == 0
    for previous, current in zip(chunks, chunks[1:]):
        assert current["token_start"] == previous["token_end"] - 500
    text = chunks[0]["chunk_text"] + "".join(x["chunk_text"][500:] for x in chunks[1:])
    expected = "".join(f"=== Clinical Note dated {date} ===\n{note}\n\n" for date, note in notes)
    assert text == expected
    assert chunks[-1]["first_date"] == chunks[-1]["last_date"] == "2020-02-01"
    short = list(pipeline.variable_chunks([("2020-01-01", "short note")], tokenizer))
    assert len(short) == 1 and short[0]["chunk_tokens"] < 10000
    with pytest.raises(ValueError):
        list(pipeline.variable_chunks(notes, tokenizer, minimum=500, overlap=500))


def test_one_component_and_one_drug_per_question(sources):
    pair = {"patient_id": "synthetic-p1", "patient_summary": "Fabricated patient summary",
            "trial_id": "NCT00000000", "trial_summary": "Fabricated trial space", "split": "train"}
    tasks = list(pipeline.component_tasks(pair, sources, Catalog(), evidence_packer))
    assert len(tasks) == 14
    assert {x["component"] for x in tasks} == set(c.TRIAL_COMPONENTS) | set(c.GOOD_OPTION_COMPONENTS)
    for task in tasks:
        prompt = task["messages"][-1]["content"]
        assert prompt.count("\nQUESTION\n") == 1
        assert "A. Yes\nB. No" in prompt
        assert "C. " not in prompt and "Final score:" not in prompt
        assert "criterion 3" not in prompt  # criterion 4 is self-contained
        assert "very next token" in task["messages"][0]["content"]
    rules = c.good_option_rules(sources)
    assert "already received this drug" in rules["disease_type_benefit"][1]
    assert "20%" in rules["common_biomarker_in_disease"][1]
    assert "Absent or untested scores 0" in rules["patient_biomarker_targeted"][1]
    assert "Preclinical and mechanistic support fall short" in rules["biomarker_targeted_benefit"][1]
    assert "unknown required biomarker alone is not a clear mismatch" in c.TRIAL_COMPONENTS["biomarker_matching"][1]
    assert "unknown/untested status yields No" in c.TRIAL_COMPONENTS["biomarker_specificity"][1]
    assert "washout" in c.TRIAL_COMPONENTS["treatment_history_matching"][1]


def test_changed_rubrics_fail_instead_of_silently_changing_semantics(sources, tmp_path):
    root = tmp_path / "src/matchminer_ai/prompts"
    root.mkdir(parents=True)
    for name, text in sources.items():
        (root / name).write_text(text + (" new rule" if name == "llm_match_quality.user.txt" else ""))
    with pytest.raises(ValueError, match="TrialChecker rubric changed"):
        c.load_prompt_sources(tmp_path)
    sources["llm_good_option.rubric.txt"] = "new rubric"
    with pytest.raises(ValueError, match="rubric structure changed"):
        c.good_option_rules(sources)


@pytest.mark.parametrize("wrapped", [
    "<think>private analysis</think>A\nSupported final explanation.",
    "<|channel>thought\nprivate analysis<channel|>A\nSupported final explanation.",
    "assistantanalysis private analysis assistantfinal A\nSupported final explanation.",
    "A\nSupported final explanation.",
])
def test_only_final_answer_becomes_supervised_tokens(wrapped, tokenizer):
    messages = [{"role": "user", "content": "Fabricated question"}]
    row = c.tokenize_example(tokenizer, messages, wrapped, "clinical_qa", 1024)
    first = next(i for i, value in enumerate(row["labels"]) if value != -100)
    assert row["input_ids"][first] == ord("A")
    assert all(i == -100 for i in row["labels"][:first])
    assert tokenizer.decode(row["labels"][first:]) == "A\nSupported final explanation.<end>\n"
    assert "private analysis" not in tokenizer.decode(row["input_ids"])
    with pytest.raises(ValueError, match="truncation is prohibited"):
        c.tokenize_example(tokenizer, messages, wrapped, "clinical_qa", 10)


@pytest.mark.parametrize("bad", ["Yes, because evidence", "C\nUnknown", "A", "A. Yes\nExplanation", "<think>unfinished", "A\n<think>trace</think>explanation"])
def test_malformed_or_unfinished_targets_rejected(bad):
    with pytest.raises(ValueError):
        c.validate_answer(bad, "clinical_qa")


def test_teacher_ignores_separate_reasoning_and_retries_truncation():
    task = {"id": "synthetic-id", "category": "clinical_qa", "split": "train"}
    messages = [{"role": "user", "content": "Fabricated question"}]
    replies = iter([response("A\nPartial answer", finish="length"), response("B\nComplete answer")])
    result = pipeline.invoke_teacher(task, messages, lambda _: next(replies))
    assert result["answer"] == "B\nComplete answer"
    assert "TEACHER SECRET" not in json.dumps(result)
    with pytest.raises(RuntimeError, match="no target saved"):
        pipeline.invoke_teacher(task, messages, lambda _: response(None), attempts=1)


def test_summary_target_and_serial_input_never_contain_teacher_reasoning(tokenizer, sources, tmp_path):
    parent = {"answer": "<think>private planning</think>Completed running summary"}
    pipeline.atomic_json(pipeline.response_path(tmp_path, "parent"), parent)
    task = {"category": "summarization", "parent_id": "parent", "first_date": "2020-01-01",
            "last_date": "2020-01-02", "chunk_text": "New fabricated note"}
    messages = pipeline.task_messages(task, sources, tmp_path)
    assert "Completed running summary" in messages[-1]["content"]
    assert "private planning" not in str(messages)
    tokens = c.tokenize_example(tokenizer, messages, "<think>new private planning</think>Updated summary", "summarization", 20000)
    assert "planning" not in tokenizer.decode(tokens["input_ids"])
    assert tokenizer.decode([x for x in tokens["labels"] if x != -100]) == "Updated summary<end>\n"


def test_prepare_generate_build_and_safe_resume(tmp_path, tokenizer, monkeypatch):
    from datasets import load_from_disk
    from matchminer_ai import trials
    from matchminer_ai.matching import good_options
    monkeypatch.setattr(trials, "load_good_option_catalog", lambda *a, **kw: Catalog())
    monkeypatch.setattr(good_options, "pack_good_option_evidence", evidence_packer)
    notes, pairs = tmp_path / "notes.parquet", tmp_path / "pairs.parquet"
    pd.DataFrame([{"pseudo_mrn": "p1", "date": "2020-01-01", "synthetic_note": " ".join(f"Fabricated clinical event number {i}." for i in range(10))},
                  {"pseudo_mrn": "p2", "date": "2020-01-01", "synthetic_note": "Another fabricated clinical note."}]).to_parquet(notes)
    pd.DataFrame([{"pseudo_mrn": "p1", "nct_id": "NCT00000000", "this_space": "Trial space", "patient_summary": "Patient one", "split": "train"},
                  {"pseudo_mrn": "p2", "nct_id": "NCT00000000", "this_space": "Trial space", "patient_summary": "Patient two", "split": "validation"}]).to_parquet(pairs)
    directory = tmp_path / "run"
    prep = pipeline.parser().parse_args(["prepare", "--notes", str(notes), "--candidates", str(pairs),
        "--output-dir", str(directory), "--confirm-inputs-are-non-phi", "--chunk-min-tokens", "100",
        "--chunk-max-tokens", "150", "--chunk-overlap", "10"])
    pipeline.prepare(prep, tokenizer=tokenizer)
    tasks = list(pipeline.jsonl(directory / "requests.jsonl"))
    assert {t["split"] for t in tasks if t["patient_id"] == "p2"} == {"validation"}
    gen = pipeline.parser().parse_args(["generate", "--output-dir", str(directory), "--server-url", "http://teacher.invalid/v1", "--teacher-model", "fabricated-teacher"])
    calls = []
    def teacher(messages):
        calls.append(messages)
        if "NEXT CLINICAL RECORD SEGMENT" in messages[-1]["content"]:
            return response("<think>private summary planning</think>Updated fabricated summary")
        return response("<think>private analysis</think>A\nSupported fabricated explanation.")
    pipeline.generate(gen, tokenizer=tokenizer, call=teacher)
    assert len(calls) == len(tasks)
    assert any("PRIOR SUMMARY:\nUpdated fabricated summary" in m[-1]["content"] for m in calls)
    pipeline.generate(gen, tokenizer=tokenizer, call=lambda _: pytest.fail("Resume unexpectedly called teacher"))
    changed = deepcopy(gen)
    changed.teacher_model = "different-teacher"
    with pytest.raises(ValueError, match="Incompatible"):
        pipeline.generate(changed, tokenizer=tokenizer, call=teacher)
    build = pipeline.parser().parse_args(["build", "--output-dir", str(directory), "--max-seq-length", "20000"])
    pipeline.build(build, tokenizer=tokenizer)
    ds = load_from_disk(str(directory / "tokenized_dataset"))
    assert len(ds["train"]) + len(ds["validation"]) == len(tasks)
    for split in ds:
        for row in ds[split]:
            assert "private" not in tokenizer.decode(row["input_ids"])
            target = tokenizer.decode([x for x in row["labels"] if x != -100])
            assert target.startswith("A\n" if row["category"] == "clinical_qa" else "Updated fabricated summary")
    # A missing response must not silently remove a task from the student set.
    pipeline.response_path(directory / "responses", tasks[-1]['id']).unlink()
    with pytest.raises(ValueError, match="Missing teacher response"):
        list(pipeline.training_rows(directory, tokenizer, 20000))
    (directory / "requests.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="requests changed"):
        pipeline.load_prepared(directory)


def test_quick_mode_has_one_token_and_same_initial_choices(tokenizer):
    from oncoreasoning_training.preview_model import generation_options
    quick = generation_options(tokenizer, 10, quick=True)
    verbose = generation_options(tokenizer, 10, max_new_tokens=200)
    assert quick["max_new_tokens"] == 1 and verbose["max_new_tokens"] == 200
    assert quick["do_sample"] is verbose["do_sample"] is False
    assert quick["prefix_allowed_tokens_fn"](0, [1] * 10) == (ord("A"), ord("B"))
    assert verbose["prefix_allowed_tokens_fn"](0, [1] * 11) == list(range(256))


def test_tiny_gemma_text_loading_loss_gradient_and_save(tmp_path):
    import torch
    from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration, TrainingArguments
    from oncoreasoning_training.fine_tune_llm import load_student, AnswerOnlyTrainer
    cfg = Gemma4TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        hidden_size_per_layer_input=8, vocab_size_per_layer_input=128,
        layer_types=["sliding_attention", "full_attention"], max_position_embeddings=256,
        sliding_window=16, final_logit_softcapping=30.)
    original = Gemma4ForConditionalGeneration(Gemma4Config(text_config=cfg, vision_config=None, audio_config=None)).eval()
    base = tmp_path / "base"
    original.save_pretrained(base)
    student, _ = load_student(base, dtype=torch.float32, attention="eager")
    assert student.config.model_type == "gemma4_text"
    assert not hasattr(student.model, "vision_tower")
    assert not hasattr(student.model, "audio_tower")
    student.eval()
    ids = torch.tensor([[10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 0]])
    labels = torch.tensor([[-100, -100, -100, 13, 14, 15], [-100, -100, 12, 13, 14, -100]])
    torch.testing.assert_close(student(ids).logits, original(ids).logits)
    trainer = AnswerOnlyTrainer(model=student, args=TrainingArguments(output_dir=str(tmp_path / "train"), use_cpu=True, report_to="none"))
    batch = {"input_ids": ids, "labels": labels}
    reduced = trainer.compute_loss(student, batch)
    full = student(**batch).loss
    torch.testing.assert_close(reduced, full)
    reduced.backward()
    gradient = student.lm_head.weight.grad.clone()
    student.zero_grad()
    full.backward()
    torch.testing.assert_close(student.lm_head.weight.grad, gradient)
    # Token normalization remains correct when accumulating differently sized targets.
    student.zero_grad()
    accumulated = sum(trainer.compute_loss(student, {key: value[i:i+1] for key, value in batch.items()}, num_items_in_batch=6) for i in range(2))
    torch.testing.assert_close(accumulated, reduced)
    saved = tmp_path / "student"
    student.save_pretrained(saved)
    reloaded, _ = load_student(saved, dtype=torch.float32, attention="eager")
    torch.testing.assert_close(student(ids).logits, reloaded(ids).logits)


@pytest.mark.parametrize("task", ["clinical_qa", "both"])
def test_training_entrypoint_saves_and_resumes_text_artifact(tmp_path, monkeypatch, task):
    import torch
    from datasets import Dataset, DatasetDict
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration, PreTrainedTokenizerFast, TrainingArguments
    from oncoreasoning_training import fine_tune_llm as training
    from oncoreasoning_training.preview_model import generation_options
    base, data, output = (tmp_path / name for name in ("base", "data", "output"))
    vocab = {word: i for i, word in enumerate(["[PAD]", "[EOS]", "[UNK]", "system", "user", "assistant", "A", "B", "Question", "Answer", "Explanation", "Yes", "No"])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]", model_max_length=256,
        model_input_names=["input_ids", "attention_mask"],
        chat_template="{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' [EOS] ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}")
    cfg = Gemma4TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_kv_shared_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        hidden_size_per_layer_input=8, vocab_size_per_layer_input=128,
        layer_types=["sliding_attention", "full_attention"] * 2, max_position_embeddings=256,
        sliding_window=16, final_logit_softcapping=30.)
    Gemma4ForConditionalGeneration(Gemma4Config(text_config=cfg, vision_config=None, audio_config=None)).save_pretrained(base)
    tokenizer.save_pretrained(base)
    data.mkdir()
    tokenizer.save_pretrained(data / "tokenizer")
    messages = [{"role": "user", "content": "Question A Yes B No"}]
    row = c.tokenize_example(tokenizer, messages, "A\nExplanation", "clinical_qa", 128)
    rows = [{**row, "category": category} for category in c.TASKS for _ in range(2)]
    # Different summary targets let us verify that the adapters learn different weights.
    for item in rows[:2]:
        item["input_ids"] = item["input_ids"][:-2] + [12, 1]
        item["labels"] = item["labels"][:-2] + [12, 1]
    DatasetDict({split: Dataset.from_list(rows) for split in ("train", "validation")}).save_to_disk(str(data / "tokenized_dataset"))
    pipeline.atomic_json(data / "training_manifest.json", {"student_model": str(base), "format_version": c.FORMAT_VERSION,
        "tokenizer_fingerprint": c.tokenizer_fingerprint(tokenizer), "max_seq_length": 128})
    def cpu_args(**kwargs):
        kwargs.update(use_cpu=True, disable_tqdm=True)
        return TrainingArguments(**kwargs)
    monkeypatch.setattr("transformers.TrainingArguments", cpu_args)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    argv = ["fine_tune_llm.py", "--data-dir", str(data), "--output-dir", str(output),
            "--task", task, "--save-steps", "1", "--gradient-accumulation-steps", "1",
            "--warmup-ratio", "0", "--lora-rank", "2", "--max-eval-examples", "1"]
    monkeypatch.setattr("sys.argv", argv)
    training.main()
    assert (output / "oncoreasoning_contract.json").is_file()
    adapter = output / "clinical_qa" if task == "both" else output
    before = (adapter / "adapter_model.safetensors").read_bytes()
    if task == "both":
        collection = pipeline.read_json(output / "oncoreasoning_contract.json")
        assert collection["artifact_type"] == "adapter_collection"
        summary = output / "summarization"
        assert (summary / "adapter_model.safetensors").read_bytes() != before
        assert pipeline.read_json(summary / "oncoreasoning_contract.json")["tasks"] == ["summarization"]
    contract = pipeline.read_json(adapter / "oncoreasoning_contract.json")
    assert contract["tasks"] == ["clinical_qa"]
    assert contract["task_counts"] == {"train": 2, "validation": 2}
    assert contract["metrics"]["eval_loss"] > 0
    # Both skips its finished adapters; a single task can restore a real checkpoint.
    monkeypatch.setattr("sys.argv", argv + ["--resume-from-checkpoint", "auto" if task == "both" else str(adapter / "checkpoint-2")])
    training.main()
    assert (adapter / "adapter_model.safetensors").read_bytes() == before
    model, _ = training.load_student(base, dtype=torch.float32)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
    for name, parameter in model.get_base_model().named_parameters():
        if "lora_" not in name:
            torch.testing.assert_close(parameter, frozen[name.replace(".base_layer", "")], rtol=0, atol=0)
    assert any(torch.count_nonzero(p).item() for name, p in model.named_parameters() if "lora_B" in name)
    model.eval()
    prompt = tokenizer(c.render_prompt(tokenizer, messages), return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        quick = model.generate(**prompt, **generation_options(tokenizer, prompt.input_ids.shape[1], quick=True))
        verbose = model.generate(**prompt, **generation_options(tokenizer, prompt.input_ids.shape[1], max_new_tokens=3))
    assert quick.shape[1] == prompt.input_ids.shape[1] + 1
    assert quick[0, -1].item() in c.letter_token_ids(tokenizer).values()
    assert quick[0, -1].item() == verbose[0, prompt.input_ids.shape[1]].item()


def test_adapter_dataset_filter_keeps_tasks_and_splits_separate():
    from datasets import Dataset, DatasetDict
    from oncoreasoning_training.fine_tune_llm import select_task_dataset
    rows = [{"input_ids": [i], "attention_mask": [1], "labels": [i], "category": category}
            for i, category in enumerate(["summarization", "clinical_qa", "clinical_qa"])]
    data = DatasetDict({split: Dataset.from_list(rows) for split in ("train", "validation", "test")})
    summary, counts = select_task_dataset(data, "summarization", max_eval_examples=1)
    qa, qa_counts = select_task_dataset(data, "clinical_qa", max_eval_examples=1)
    assert summary["train"]["input_ids"] == [[0]]
    assert qa["train"]["input_ids"] == [[1], [2]]
    assert counts == {"train": 1, "validation": 1}
    assert qa_counts == {"train": 2, "validation": 2}
    assert len(qa["validation"]) == 1 and "test" not in qa
    assert qa["validation"]["input_ids"] == select_task_dataset(data, "clinical_qa", max_eval_examples=1)[0]["validation"]["input_ids"]
    with pytest.raises(ValueError, match="category"):
        select_task_dataset(data.remove_columns("category"), "clinical_qa")
    with pytest.raises(ValueError, match="No training"):
        select_task_dataset(DatasetDict(train=Dataset.from_list(rows[:1])), "clinical_qa")
