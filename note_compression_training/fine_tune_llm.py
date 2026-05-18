#!/usr/bin/env python3
"""Fine-tune LiquidAI/LFM2.5-1.2B-Thinking on note compression SFT data."""

from __future__ import annotations

import argparse
import inspect
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Thinking"
DEFAULT_DATASET = "../data/no_phi/note_compression_training_data/tokenized_training_data.dataset"
DEFAULT_OUTPUT = "../models/note_compression_lfm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SFT for the note compression Liquid thinking model."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--logging-dir", default="./logs/note_compression_lfm")
    parser.add_argument("--max-length", type=int, default=50000)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--lr-scheduler-type", default="cosine_with_restarts")
    parser.add_argument("--num-cycles", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-liger-kernel", action="store_true")
    parser.add_argument("--no-activation-offloading", action="store_true")

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def checkpoint_arg(output_dir: str, resume_from_checkpoint: str):
    if resume_from_checkpoint == "false":
        return False
    if resume_from_checkpoint != "auto":
        return resume_from_checkpoint
    if os.path.isdir(output_dir) and any(
        name.startswith("checkpoint-") for name in os.listdir(output_dir)
    ):
        return True
    return False


def build_lora_config(args: argparse.Namespace):
    if not args.use_lora:
        return None
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
        init_lora_weights="gaussian",
        modules_to_save=["lm_head"],
    )


def filter_kwargs(callable_obj, kwargs: dict) -> dict:
    params = set(inspect.signature(callable_obj).parameters)
    return {key: value for key, value in kwargs.items() if key in params}


def build_sft_config(args: argparse.Namespace, dtype) -> SFTConfig:
    config_kwargs = {
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "bf16": not args.no_bf16,
        "auto_find_batch_size": False,
        "save_total_limit": args.save_total_limit,
        "packing": False,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "save_steps": args.save_steps,
        "optim": args.optim,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "model_init_kwargs": {
            "torch_dtype": dtype,
            "attn_implementation": args.attn_implementation,
        },
        "lr_scheduler_kwargs": {"num_cycles": args.num_cycles},
        "logging_steps": args.logging_steps,
        "activation_offloading": not args.no_activation_offloading,
        "use_liger_kernel": not args.no_liger_kernel,
        "use_liger": not args.no_liger_kernel,
        "logging_dir": args.logging_dir,
        "output_dir": args.output_dir,
        "report_to": args.report_to,
        "max_length": args.max_length,
        "max_seq_length": args.max_length,
    }
    return SFTConfig(**filter_kwargs(SFTConfig.__init__, config_kwargs))


def main() -> None:
    args = parse_args()
    bf16 = not args.no_bf16

    print(f"Loading dataset: {args.dataset_dir}")
    dataset = Dataset.load_from_disk(args.dataset_dir)
    print(f"Dataset rows: {len(dataset)}")

    torch.backends.cuda.enable_flash_sdp(True)
    print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")

    dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = build_sft_config(args, dtype)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "args": sft_config,
        "peft_config": build_lora_config(args),
        "train_dataset": dataset,
        "data_collator": data_collator,
    }
    trainer = SFTTrainer(**filter_kwargs(SFTTrainer.__init__, trainer_kwargs))

    trainer.train(
        resume_from_checkpoint=checkpoint_arg(
            args.output_dir,
            args.resume_from_checkpoint,
        )
    )


if __name__ == "__main__":
    main()
