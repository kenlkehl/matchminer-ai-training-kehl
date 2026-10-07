#!/usr/bin/env python3
"""Train independent summarization and clinical-QA LoRA adapters."""
from __future__ import annotations

import argparse
import gc
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
        result = super().compute_loss(model, {
            **inputs, "logits_to_keep": positions,
            "shift_labels": shifted.index_select(-1, positions).contiguous(),
        }, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        loss = result[0] if return_outputs else result
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError("Non-finite adapter loss; stopping before saving invalid weights")
        return result


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
    p.add_argument("--task", choices=("both", *c.TASKS), default="both",
                   help="Both trains two independent adapters, each from the same frozen base")
    p.add_argument("--output-dir", type=Path, default=c.REPO_ROOT.parent / "models" / "oncoreasoning_gemma4_e4b_answer_first_v2")
    p.add_argument("--resume-from-checkpoint", help="Checkpoint path, or auto to restart this identical run after interruption")
    p.add_argument("--num-train-epochs", type=float, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=0, help="Overrides the warmup ratio when positive")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--max-eval-examples", type=int, default=256,
                   help="Fixed seeded subset per task for checkpoint selection; full validation data are retained")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attention", default="sdpa", choices=("sdpa", "flash_attention_2", "eager"))
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--fsdp", action="store_true", help="Shard the frozen base and adapters with FSDP2")
    return p


def select_task_dataset(dataset, task, *, max_eval_examples=256, seed=42):
    """Filter by category before removing metadata, without decoding token columns."""
    from datasets import DatasetDict
    if task not in c.TASKS:
        raise ValueError("Select exactly one adapter task")
    selected, counts = {}, {}
    for split, rows in dataset.items():
        if split not in ("train", "validation"):
            continue
        if "category" not in rows.column_names:
            raise ValueError("Prepared dataset needs category labels to separate adapters")
        rows = rows.filter(lambda categories: [value == task for value in categories],
                           input_columns=["category"], batched=True)
        counts[split] = len(rows)
        if not len(rows):
            continue
        if split == "validation" and len(rows) > max_eval_examples:
            rows = rows.shuffle(seed=seed).select(range(max_eval_examples))
        selected[split] = rows.select_columns(["input_ids", "attention_mask", "labels"])
    if "train" not in selected:
        raise ValueError(f"No training examples for {task}")
    return DatasetDict(selected), counts


def add_adapter(model, args):
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules="all-linear", bias="none",
        init_lora_weights=True, task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    if not trainable or any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
        raise ValueError("Expected trainable LoRA weights and a completely frozen base")
    return model, {"trainable": trainable, "total": total}


def train_adapter(args):
    from datasets import load_from_disk
    from transformers import DataCollatorForSeq2Seq, TrainingArguments, set_seed
    set_seed(args.seed)
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
    dataset, counts = select_task_dataset(load_from_disk(str(args.data_dir / "tokenized_dataset")),
        args.task, max_eval_examples=args.max_eval_examples, seed=args.seed)
    settings = {key: getattr(args, key) for key in (
        "task", "num_train_epochs", "learning_rate", "batch_size", "gradient_accumulation_steps",
        "warmup_steps", "warmup_ratio", "weight_decay", "max_grad_norm", "save_steps",
        "max_eval_examples", "seed", "attention", "lora_rank", "lora_alpha", "lora_dropout", "fsdp")}
    run = {"training_manifest": manifest, "model_name": model_name, "settings": settings,
           "task_counts": counts, "eval_examples": len(dataset.get("validation", [])),
           "world_size": int(os.environ.get("WORLD_SIZE", "1"))}
    run_path = args.output_dir / "oncoreasoning_training_run.json"
    restarting = args.resume_from_checkpoint == "auto"
    if restarting:
        from training_checkpoints import latest_complete_checkpoint
        if run_path.exists() and read_json(run_path) != run:
            raise ValueError("Automatic resume requires the identical training manifest and settings")
        contract_path = args.output_dir / "oncoreasoning_contract.json"
        if contract_path.exists():
            contract = read_json(contract_path)
            if (not run_path.exists() or contract.get("training_run_sha256") != c.digest(run)
                    or not (args.output_dir / "adapter_model.safetensors").is_file()
                    or not (args.output_dir / "adapter_config.json").is_file()):
                raise ValueError("Completed adapter does not match this run")
            print(f"[skip adapter] {args.task}", flush=True)
            return contract
        args.resume_from_checkpoint = latest_complete_checkpoint(args.output_dir)
    if args.resume_from_checkpoint:
        if not run_path.exists() or read_json(run_path) != run:
            raise ValueError("Checkpoint resume requires the identical training manifest and settings")
        checkpoint = Path(args.resume_from_checkpoint).resolve()
        if not checkpoint.is_dir() or checkpoint.parent != args.output_dir.resolve():
            raise ValueError("Resume checkpoint must belong to this output directory")
    elif not (restarting and run_path.exists()) and int(os.environ.get("RANK", "0")) == 0 and args.output_dir.exists() and any(args.output_dir.iterdir()):
        # Only rank zero checks a fresh destination. Another rank may arrive
        # here after rank zero has created the run manifest or Trainer folder.
        raise ValueError("Output directory is not empty; resume explicitly or select a fresh directory")
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # FSDP2 groups must have a uniform storage dtype, including the FP32 LoRA
    # weights. Its BF16 policy casts forward computation without training the base.
    dtype = torch.bfloat16 if bf16 and not args.fsdp else torch.float32
    model, text_config = load_student(model_name, dtype=dtype, attention=args.attention)
    if manifest["max_seq_length"] > text_config.max_position_embeddings:
        raise ValueError("Prepared sequence length exceeds the model context")
    model, parameter_counts = add_adapter(model, args)
    print(f"[{args.task}] examples={counts}, parameters={parameter_counts}", flush=True)
    model.config.use_cache = False
    distributed = {}
    if args.fsdp:
        if text_config.model_type != "gemma4_text":
            raise ValueError("Review the FSDP wrapping policy when changing the student architecture")
        distributed = dict(fsdp="full_shard auto_wrap", fsdp_config={
            "version": 2, "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer"],
            "activation_checkpointing": True,
        })
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.fsdp, gradient_checkpointing_kwargs={"use_reentrant": False},
        # Transformers 5 accepts a fractional warmup_steps instead of warmup_ratio.
        bf16=bf16, optim="adamw_torch", lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps or args.warmup_ratio,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps, save_total_limit=2, logging_steps=10, report_to="none",
        eval_strategy="steps" if "validation" in dataset else "no", eval_steps=args.save_steps,
        load_best_model_at_end="validation" in dataset, metric_for_best_model="eval_loss",
        greater_is_better=False, prediction_loss_only=True, logging_nan_inf_filter=False,
        seed=args.seed, data_seed=args.seed, ddp_find_unused_parameters=False,
        **distributed,
    )
    trainer = AnswerOnlyTrainer(model=model, args=training_args, processing_class=tokenizer,
        train_dataset=dataset["train"], eval_dataset=dataset.get("validation"),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8))
    if trainer.is_world_process_zero():
        atomic_json(run_path, run)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = dict(result.metrics)
    if "validation" in dataset:
        metrics.update(trainer.evaluate())
    if any(not torch.isfinite(torch.tensor(value)) for key, value in metrics.items() if key.endswith("loss")):
        raise FloatingPointError("Non-finite final adapter metrics")
    model.config.use_cache = True
    model.generation_config.do_sample = False
    trainer.save_model(str(args.output_dir))
    contract = {
        "format_version": c.FORMAT_VERSION, "artifact_type": "lora_adapter", "base_model": model_name,
        "enable_thinking": False, "answer_token_ids": c.letter_token_ids(tokenizer),
        "max_seq_length": manifest["max_seq_length"], "tasks": [args.task], "lora_adapter": True,
        "training_run_sha256": c.digest(run), "parameter_counts": parameter_counts,
        "settings": settings, "task_counts": counts, "metrics": metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        atomic_json(args.output_dir / "oncoreasoning_contract.json", contract)
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.free_memory()
    return contract


def main():
    args = parser().parse_args()
    if (args.lora_rank < 1 or args.lora_alpha < 1 or not 0 <= args.lora_dropout < 1
            or args.learning_rate <= 0 or not 0 <= args.warmup_ratio < 1
            or args.max_grad_norm <= 0 or args.max_eval_examples < 1 or args.save_steps < 1
            or args.warmup_steps < 0 or args.weight_decay < 0 or args.num_train_epochs <= 0
            or args.batch_size < 1 or args.gradient_accumulation_steps < 1):
        raise ValueError("Invalid LoRA training hyperparameters; full-parameter tuning is no longer supported")
    if args.task != "both":
        train_adapter(args)
        return
    if args.resume_from_checkpoint not in (None, "auto"):
        raise ValueError("Use auto for two adapters, or select --task for a specific checkpoint")
    if args.output_dir.exists() and any(path.name not in (*c.TASKS, "oncoreasoning_contract.json")
                                      for path in args.output_dir.iterdir()):
        raise ValueError("Adapter collection output contains unrelated or old full-model files")
    # Keep the existing entrypoint and completion marker: an already-running
    # pipeline will execute this updated file only when it reaches training.
    contracts = {}
    for task in c.TASKS:
        single = argparse.Namespace(**{**vars(args), "task": task, "output_dir": args.output_dir / task})
        contracts[task] = train_adapter(single)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if int(os.environ.get("RANK", "0")) == 0:
        atomic_json(args.output_dir / "oncoreasoning_contract.json", {
            "format_version": c.FORMAT_VERSION, "artifact_type": "adapter_collection",
            "base_model": contracts[c.TASKS[0]]["base_model"], "tasks": list(c.TASKS),
            "adapters": {task: task for task in c.TASKS},
            "adapter_contract_sha256": {task: c.digest(contract) for task, contract in contracts.items()},
        })


if __name__ == "__main__":
    main()
