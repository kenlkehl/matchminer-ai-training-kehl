#!/usr/bin/env python3
"""Fine-tune the configurable text-only student on final-answer targets."""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from transformers import Trainer

from oncoreasoning_training import contracts as c
from oncoreasoning_training.create_all_training_data import atomic_json, read_json


class AnswerOnlyTrainer(Trainer):
    """Project only supervised positions to the large vocabulary.

    This is exactly the ordinary masked causal-LM loss, including the first
    letter and the entire explanation/summary. It avoids allocating vocabulary
    logits for up to 50K ignored input tokens. Transformers still handles token
    normalization across accumulation steps and distributed workers.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        shifted = F.pad(inputs["labels"], (0, 1), value=-100)[..., 1:]
        positions = shifted.ne(-100).any(dim=0).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            raise ValueError("Batch has no supervised answer tokens")
        return super().compute_loss(model, {
            **inputs, "logits_to_keep": positions,
            "shift_labels": shifted.index_select(-1, positions).contiguous(),
        }, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)


def load_student(model_name, *, dtype, attention="sdpa"):
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_name)
    text_config = config.get_text_config()
    # Gemma4Config also maps to a multimodal class in AutoModelForCausalLM.
    # Select Gemma4TextConfig and translate the published text subtree explicitly.
    mapping = {r"^model\.language_model\.": "model."} if config.model_type == "gemma4" else None
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_name, config=text_config, dtype=dtype, attn_implementation=attention,
        key_mapping=mapping, output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError("Student checkpoint did not load all text weights; refusing partially initialized training")
    if config.model_type == "gemma4":
        # Transformers 5 reverses user key mappings when saving by default.
        # We have permanently extracted the text model: retain its native names
        # in both Trainer checkpoints and the final artifact so ordinary
        # AutoModelForCausalLM reloads do not need the original multimodal map.
        model._weight_conversions = []
    if any(getattr(model.config, name, None) is not None for name in ("vision_config", "audio_config")):
        raise ValueError("Expected a text-only causal language model")
    if "logits_to_keep" not in inspect.signature(model.forward).parameters:
        raise ValueError("Student must support logits_to_keep for long-context answer-only loss")
    return model, text_config


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=c.DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-name", help="Defaults to the prepared manifest's model; changing it requires re-preparation")
    p.add_argument("--output-dir", type=Path, default=c.REPO_ROOT.parent / "models" / "oncoreasoning_gemma4_e4b_answer_first_v1")
    p.add_argument("--resume-from-checkpoint", help="Explicit checkpoint path; never resume a different run automatically")
    p.add_argument("--num-train-epochs", type=float, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attention", default="sdpa", choices=("sdpa", "flash_attention_2", "eager"))
    p.add_argument("--lora-rank", type=int, default=0, help="0 for full fine-tuning; e.g. 64 for LoRA")
    return p


def main():
    from datasets import load_from_disk
    from transformers import DataCollatorForSeq2Seq, TrainingArguments
    args = parser().parse_args()
    manifest = read_json(args.data_dir / "training_manifest.json")
    model_name = args.model_name or manifest["student_model"]
    if manifest["format_version"] != c.FORMAT_VERSION or model_name != manifest["student_model"]:
        raise ValueError("Model/format mismatch: prepare and build new data for the chosen base model")
    tokenizer = c.load_chat_tokenizer(str(args.data_dir / "tokenizer"))
    if c.tokenizer_fingerprint(tokenizer) != manifest["tokenizer_fingerprint"]:
        raise ValueError("Saved tokenizer does not match the training manifest")
    base_tokenizer = c.load_chat_tokenizer(model_name)
    if c.tokenizer_fingerprint(base_tokenizer) != manifest["tokenizer_fingerprint"]:
        raise ValueError("Base model tokenizer changed since data preparation")
    dataset = load_from_disk(str(args.data_dir / "tokenized_dataset"))
    columns = ("input_ids", "attention_mask", "labels")
    dataset = dataset.select_columns(list(columns))
    settings = {key: getattr(args, key) for key in (
        "num_train_epochs", "learning_rate", "batch_size", "gradient_accumulation_steps",
        "warmup_steps", "save_steps", "seed", "attention", "lora_rank")}
    run = {"training_manifest": manifest, "model_name": model_name, "settings": settings}
    run_path = args.output_dir / "oncoreasoning_training_run.json"
    if args.resume_from_checkpoint:
        if not run_path.exists() or read_json(run_path) != run:
            raise ValueError("Checkpoint resume requires the identical training manifest and settings")
        checkpoint = Path(args.resume_from_checkpoint).resolve()
        if not checkpoint.is_dir() or checkpoint.parent != args.output_dir.resolve():
            raise ValueError("Resume checkpoint must belong to this output directory")
    elif int(os.environ.get("RANK", "0")) == 0 and args.output_dir.exists() and any(args.output_dir.iterdir()):
        # Only rank zero checks a fresh destination. Another rank may arrive
        # here after rank zero has created the run manifest or Trainer folder.
        raise ValueError("Output directory is not empty; resume explicitly or select a fresh directory")
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # Full fine-tuning keeps float32 master weights. LoRA may freeze a bf16 base.
    dtype = torch.bfloat16 if bf16 and args.lora_rank else torch.float32
    model, text_config = load_student(model_name, dtype=dtype, attention=args.attention)
    if manifest["max_seq_length"] > text_config.max_position_embeddings:
        raise ValueError("Prepared sequence length exceeds the model context")
    if args.lora_rank:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
            lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
        model.enable_input_require_grads()
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, optim="adamw_torch", lr_scheduler_type="cosine", warmup_steps=args.warmup_steps,
        save_steps=args.save_steps, save_total_limit=2, logging_steps=10, report_to="none",
        eval_strategy="epoch" if "validation" in dataset else "no", prediction_loss_only=True,
        seed=args.seed, data_seed=args.seed, ddp_find_unused_parameters=False,
    )
    trainer = AnswerOnlyTrainer(model=model, args=training_args, processing_class=tokenizer,
        train_dataset=dataset["train"], eval_dataset=dataset.get("validation"),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8))
    if trainer.is_world_process_zero():
        atomic_json(run_path, run)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.config.use_cache = True
    model.generation_config.do_sample = False
    trainer.save_model(str(args.output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        atomic_json(args.output_dir / "oncoreasoning_contract.json", {
            "format_version": c.FORMAT_VERSION, "base_model": model_name,
            "enable_thinking": False, "answer_token_ids": c.letter_token_ids(tokenizer),
            "max_seq_length": manifest["max_seq_length"], "tasks": list(c.TASKS),
            "lora_adapter": bool(args.lora_rank),
        })


if __name__ == "__main__":
    main()
