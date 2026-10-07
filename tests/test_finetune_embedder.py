"""Offline checks with a tiny random EmbeddingGemma 2 and fabricated text."""

import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from packaging.version import Version


class TrialSpaceEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for package, minimum in (("transformers", "5.19"), ("sentence-transformers", "6.1")):
            if Version(importlib.metadata.version(package)) < Version(minimum):
                raise unittest.SkipTest(f"Requires {package}>={minimum}")

        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import (
            EmbeddingGemma2Config, EmbeddingGemma2Model, EmbeddingGemma2TextConfig,
            PreTrainedTokenizerFast,
        )
        import finetune_embedder

        cls.training = finetune_embedder
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.base = Path(cls.tmp.name) / "base"
        cls.base.mkdir()
        torch.manual_seed(42)
        vocab = {word: i for i, word in enumerate([
            "[PAD]", "[EOS]", "[BOS]", "[UNK]", "task", ":", "sentence", "similarity",
            "|", "query", "patient", "trial", "lung", "breast", "cancer", "text",
        ])}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 2), ("[EOS]", 1)],
        )
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]",
            bos_token="[BOS]", unk_token="[UNK]", model_max_length=8192,
            model_input_names=["input_ids", "attention_mask"],
            # The tiny offline fixture uses a plain tokenizer, without media processors.
            processor_class="PreTrainedTokenizerFast",
        ).save_pretrained(cls.base)
        text_config = EmbeddingGemma2TextConfig(
            vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, global_head_dim=8, hidden_size_per_layer_input=8,
            embedding_dim=16, max_position_embeddings=8192, sliding_window=16,
            layer_types=["sliding_attention", "full_attention"],
        )
        EmbeddingGemma2Model(EmbeddingGemma2Config(
            text_config=text_config, vision_config=None, audio_config=None,
        )).save_pretrained(cls.base)
        # Mirror Google's ST stack, including projected embedding dimension.
        modules = [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.base.modules.transformer.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.base.modules.normalize.Normalize"},
        ]
        (cls.base / "modules.json").write_text(json.dumps(modules))
        (cls.base / "sentence_bert_config.json").write_text(json.dumps({
            "transformer_task": "feature-extraction",
            "modality_config": {"text": {"method": "forward", "method_output_name": "last_hidden_state"}},
            "module_output_name": "token_embeddings",
        }))
        (cls.base / "1_Pooling").mkdir()
        (cls.base / "1_Pooling/config.json").write_text(json.dumps({
            "embedding_dimension": 16, "pooling_mode": "mean", "include_prompt": True,
        }))
        (cls.base / "2_Normalize").mkdir()

    def load_model(self):
        return self.training.load_text_model(str(self.base), "cpu")

    def test_text_only_loading_and_saved_inference_contract(self):
        import torch
        from sentence_transformers import SentenceTransformer

        model = self.load_model()
        self.assertIsNone(model[0].auto_model.vision_tower)
        self.assertIsNone(model[0].auto_model.audio_tower)
        self.assertEqual(next(model.parameters()).dtype, torch.float32)
        self.assertTrue(model[1].include_prompt)
        with tempfile.TemporaryDirectory() as saved:
            model.save(saved)
            self.training.validate_resume_checkpoint(saved)
            reloaded = SentenceTransformer(saved, device="cpu")
            texts = ["lung cancer patient", "lung cancer trial"]
            expected = model.encode(texts, convert_to_tensor=True)
            for method in (reloaded.encode, reloaded.encode_query, reloaded.encode_document):
                actual = method(texts, convert_to_tensor=True)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual.norm(dim=1), torch.ones(2))
            self.assertEqual(reloaded.max_seq_length, self.training.MAX_TOKENS)

    def test_training_prefix_matches_inference_and_truncates_once(self):
        import torch
        from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
        from datasets import Dataset

        model = self.load_model()
        model.max_seq_length = 24
        row = {"patient_summary": "patient " * 50, "this_space": "trial " * 50, "label": 1.0}
        with tempfile.TemporaryDirectory() as output:
            trainer = SentenceTransformerTrainer(
                model=model,
                args=SentenceTransformerTrainingArguments(
                    output_dir=output, prompts=self.training.PROMPT_PREFIX,
                    use_cpu=True, report_to=[],
                ),
                train_dataset=Dataset.from_list([row]),
            )
            batch = trainer.data_collator([row])
            for column in ("patient_summary", "this_space"):
                expected = model.preprocess([row[column]], prompt=self.training.PROMPT_PREFIX)
                ids = batch[f"{column}_input_ids"]
                torch.testing.assert_close(ids, expected["input_ids"])
                self.assertEqual(ids.shape[1], 24)
                # Both the prefix and special tokens must be present exactly once.
                self.assertEqual(ids[0].tolist().count(4), 1)  # "task"
                self.assertEqual(ids[0, 0].item(), 2)  # BOS
                self.assertEqual(ids[0, -1].item(), 1)  # EOS

    def test_mining_uses_saved_query_prompt_and_allows_explicit_override(self):
        import torch
        import make_top_matches

        model = self.load_model()
        texts = ["lung cancer patient", "lung cancer trial"]
        with patch.object(make_top_matches, "load_text_model", return_value=model), \
             patch.object(torch.cuda, "set_device"):
            for prompt in (None, "task: sentence similarity | query: ", ""):
                actual = make_top_matches._encode_worker(
                    texts, "cuda:0", str(self.base), 2, 32, prompt,
                )
                expected = model.encode(texts, prompt_name="query", convert_to_tensor=True)
                torch.testing.assert_close(torch.from_numpy(actual), expected)

    def test_rejects_old_model_family_and_incompatible_prompts(self):
        with tempfile.TemporaryDirectory() as old:
            (Path(old) / "config.json").write_text('{"model_type": "qwen3"}')
            with self.assertRaisesRegex(ValueError, "requires google/embeddinggemma-2"):
                self.training.load_text_model(old, "cpu")
        with tempfile.TemporaryDirectory() as saved:
            self.load_model().save(saved)
            path = Path(saved) / "config_sentence_transformers.json"
            config = json.loads(path.read_text())
            config["prompts"]["query"] = "query"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Incompatible TrialSpace checkpoint"):
                self.training.validate_resume_checkpoint(saved)

    def test_tiny_training_saves_and_resumes_both_losses(self):
        import pandas as pd
        from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformer
        import torch

        def tiny_args(**kwargs):
            kwargs.update(max_steps=2, per_device_train_batch_size=2, save_steps=1,
                          use_cpu=True, bf16=False, report_to=[], disable_tqdm=True)
            return SentenceTransformerTrainingArguments(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            parquet = path / "fabricated.parquet"
            pd.DataFrame({
                "patient_summary": ["lung cancer patient", "breast cancer patient"],
                "this_space": ["lung cancer trial", "lung cancer trial"],
                "eligibility_result": [5, 1],
            }).to_parquet(parquet)
            argv = ["finetune_embedder.py", "-m", str(self.base), "-i", str(parquet),
                    "-c", str(path / "checkpoints"), "-o", str(path / "model")]
            with patch("sys.argv", argv), \
                 patch("sentence_transformers.SentenceTransformerTrainingArguments", side_effect=tiny_args), \
                 patch.object(torch.cuda, "is_available", return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.training.main()
                self.training.main()  # Resume the saved checkpoint without a family/prompt mismatch.
            model = SentenceTransformer(str(path / "model"), device="cpu")
            self.training.validate_resume_checkpoint(path / "checkpoints/checkpoint-2")
            embeddings = model.encode(["lung cancer patient"], convert_to_tensor=True)
            self.assertTrue(torch.isfinite(embeddings).all())


if __name__ == "__main__":
    unittest.main()
