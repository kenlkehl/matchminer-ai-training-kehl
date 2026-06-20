import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / relative_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training = load_module(
    "oncoreasoning_training/create_all_training_data.py",
    "oncoreasoning_training_data",
)


class FakeTokenResult:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    def __init__(self):
        self.last_conversation = None

    def __call__(self, text, add_special_tokens=False):
        return FakeTokenResult(list(range(len(str(text).split()))))

    def decode(self, toks):
        return " ".join(f"tok{i}" for i in toks)

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
    ):
        self.last_conversation = conversation
        return "rendered"


class CharTokenizer:
    name_or_path = "char-tokenizer"

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def __call__(self, texts, max_length=None, truncation=False, add_special_tokens=False):
        if isinstance(texts, str):
            return FakeTokenResult(self.encode(texts, add_special_tokens=add_special_tokens))

        input_ids = []
        for text in texts:
            ids = self.encode(text)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            input_ids.append(ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
    ):
        rendered = []
        for message in conversation:
            if message["role"] == "assistant":
                rendered.append(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    + message["content"]
                )
            else:
                rendered.append(f"{message['role']}\n{message['content']}\n")
        if add_generation_prompt:
            rendered.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(rendered)


class OncoReasoningTrainingDataTests(unittest.TestCase):
    def test_summarization_prompt_matches_live_builder(self):
        live_summary = load_module("6_summarize_patients.py", "live_summary")
        row = pd.Series({
            "prior_summary": None,
            "first_date": "2020-01-01",
            "last_date": "2020-01-31",
            "chunk_text": "clinical note text",
            "new_summary_reasoning": "reasoning",
            "new_summary": "summary",
        })

        training_messages = training.build_summarization_messages(row)

        tokenizer = FakeTokenizer()
        live_summary.build_prompt_text(
            tokenizer,
            prior_summary=None,
            first_date="2020-01-01",
            last_date="2020-01-31",
            chunk_text="clinical note text",
            max_model_len=20000,
            model_name="openai/gpt-oss-120b",
        )

        self.assertEqual(
            training_messages[0]["content"],
            tokenizer.last_conversation[0]["content"],
        )
        self.assertEqual(
            training_messages[1]["content"],
            tokenizer.last_conversation[1]["content"],
        )
        self.assertEqual(
            training_messages[2]["content"],
            "<think>\nreasoning\n</think>\nsummary",
        )

    def test_trialspace_prompt_matches_live_builder(self):
        live_trialspaces = load_module("0b_create_trial_spaces.py", "live_trialspaces")
        row = pd.Series({
            "trial_text": "trial eligibility text",
            "space_reasoning_and_output": "raw response",
        })

        training_messages = training.build_trialspace_messages(row)
        live_messages = live_trialspaces.build_messages("trial eligibility text")

        self.assertEqual(
            training.TRIALSPACE_PROMPT_HEADER,
            live_trialspaces.PROMPT_HEADER,
        )
        self.assertEqual(
            training.TRIALSPACE_PROMPT_SUFFIX,
            live_trialspaces.PROMPT_SUFFIX,
        )
        self.assertEqual(training_messages[:2], live_messages)
        self.assertEqual(training_messages[2]["content"], "<think>\nraw response\n</think>")

    def test_reasoning_columns_are_included_when_present(self):
        boilerplate = training.build_boilerplate_messages(pd.Series({
            "patient_boilerplate_text": "patient",
            "trial_boilerplate_text": "trial",
            "boilerplate_check_llm_reasoning": "bp reasoning",
            "boilerplate_check_llm_response": "No!",
        }))
        trialcheck = training.build_trialcheck_messages(pd.Series({
            "patient_summary": "patient",
            "this_space": "space",
            "trialcheck_llm_reasoning": "tc reasoning",
            "trialcheck_llm_response": "Final score: 3",
        }))

        self.assertEqual(
            boilerplate[2]["content"],
            "<think>\nbp reasoning\n</think>\nNo!",
        )
        self.assertEqual(
            trialcheck[2]["content"],
            "<think>\ntc reasoning\n</think>\nFinal score: 3",
        )

    def test_final_response_fallback_when_reasoning_is_absent(self):
        boilerplate = training.build_boilerplate_messages({
            "patient_boilerplate_text": "patient",
            "trial_boilerplate_text": "trial",
            "boilerplate_check_llm_response": "No!",
        })
        trialcheck = training.build_trialcheck_messages({
            "patient_summary": "patient",
            "this_space": "space",
            "trialcheck_llm_response": "Final score: 3",
        })

        self.assertEqual(boilerplate[2]["content"], "No!")
        self.assertEqual(trialcheck[2]["content"], "Final score: 3")

    def test_summarization_loader_uses_current_pipeline_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "summary_shards").mkdir()

            pd.DataFrame([{
                "pseudo_mrn": "p1",
                "chunk_index": 0,
                "first_date": "2020-01-01",
                "last_date": "2020-01-31",
                "prior_summary": None,
                "new_summary_reasoning": "reasoning",
                "new_summary": "summary",
            }]).to_parquet(data_dir / "patient_serial_summaries.parquet")

            pd.DataFrame([{
                "patient_id": "p1",
                "local_idx": 0,
                "chunk_text": "note text",
            }]).to_parquet(data_dir / "summary_shards" / "prepared_chunks.parquet")

            loaded = training.load_summarization(str(data_dir))

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.loc[0, "chunk_text"], "note text")

    def test_default_paths_are_script_relative(self):
        self.assertTrue(training.DEFAULT_DATA_DIR.is_absolute())
        self.assertEqual(
            training.DEFAULT_OUTPUT_DIR,
            training.DEFAULT_DATA_DIR / "oncoreasoning_training_data",
        )

    def test_default_model_is_qwen35_4b(self):
        self.assertEqual(training.DEFAULT_MODEL_NAME, "Qwen/Qwen3.5-4B")

    def test_load_chat_tokenizer_preserves_native_pad_token(self):
        class TokenizerWithPad:
            pad_token = "<|endoftext|>"
            eos_token = "<|im_end|>"

        tokenizer = TokenizerWithPad()

        with patch.object(training.AutoTokenizer, "from_pretrained", return_value=tokenizer):
            loaded = training.load_chat_tokenizer("Qwen/Qwen3.5-4B")

        self.assertIs(loaded, tokenizer)
        self.assertEqual(loaded.pad_token, "<|endoftext|>")

    def test_streaming_tokenize_masks_prompt_before_assistant_header(self):
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        text = "prompt text " + assistant_header + "assistant answer"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.parquet"
            output = tmp_path / "tokenized.dataset"
            pd.DataFrame({"text": [text]}).to_parquet(source)

            count = training.streaming_tokenize(
                source_parquet=str(source),
                tokenizer=CharTokenizer(),
                max_seq_length=1000,
                output_path=str(output),
                batch_size=1,
                num_workers=1,
            )

            from datasets import Dataset

            dataset = Dataset.load_from_disk(str(output))
            row = dataset[0]

        self.assertEqual(count, 1)
        header_ids = CharTokenizer().encode(assistant_header)
        idx = training._find_last_subsequence(row["input_ids"], header_ids)
        mask_end = idx + len(header_ids)

        self.assertGreaterEqual(idx, 0)
        self.assertTrue(all(label == -100 for label in row["labels"][:mask_end]))
        self.assertEqual(row["labels"][mask_end:], row["input_ids"][mask_end:])


if __name__ == "__main__":
    unittest.main()
