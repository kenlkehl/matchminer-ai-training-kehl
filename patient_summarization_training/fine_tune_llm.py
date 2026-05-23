#!/usr/bin/env python3
"""Fine-tune google/gemma-4-E2B-it on patient summarization SFT data."""

from __future__ import annotations

import argparse
import inspect
import os

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_TOKENIZER = "google/gemma-4-E2B-it"
DEFAULT_DATASET = "../data/no_phi/patient_summarization_training_data/tokenized_training_data.dataset"
DEFAULT_OUTPUT = "../models/patient_summarization_gemma4_e2b_it"
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
GEMMA4_LORA_TARGET_MODULES = (
    r".*language_model\.layers\.\d+\.(?:self_attn|mlp)\."
    r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SFT for the patient summarization Gemma 4 E2B-IT model."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--tokenizer-name",
        default=DEFAULT_TOKENIZER,
        help="Tokenizer to save with the fine-tuned model and use for padding.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--logging-dir",
        default="./logs/patient_summarization_gemma4_e2b_it",
    )
    parser.add_argument("--max-length", type=int, default=50000)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--lr-scheduler-type", default="cosine_with_restarts")
    parser.add_argument("--num-cycles", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
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
    parser.add_argument(
        "--activation-offloading",
        action="store_true",
        help="Enable TRL activation offloading. It is disabled by default for Gemma 4 DDP stability.",
    )
    parser.add_argument(
        "--no-activation-offloading",
        dest="activation_offloading",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--train-multimodal-towers",
        action="store_true",
        help="Keep Gemma 4 vision/audio towers trainable. Leave off for text-only SFT.",
    )
    parser.add_argument(
        "--no-tail-logits-only",
        action="store_true",
        help="Disable Gemma 4 loss-only logits slicing for pre-tokenized SFT batches.",
    )
    parser.add_argument(
        "--train-per-layer-embeddings",
        action="store_true",
        help="Keep Gemma 4 per-layer embedding parameters trainable.",
    )
    parser.add_argument(
        "--invalid-label-action",
        choices=("mask", "error"),
        default="mask",
        help="How to handle labels outside the model output vocabulary.",
    )

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


def load_model(args: argparse.Namespace, dtype):
    loader = (
        AutoModelForImageTextToText
        if "gemma-4" in args.model_name.lower()
        else AutoModelForCausalLM
    )
    return loader.from_pretrained(
        args.model_name,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
        trust_remote_code=True,
    )


def freeze_gemma4_text_only_modules(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if "gemma-4" not in args.model_name.lower():
        return

    frozen_params = 0
    module_names = []
    if not args.train_multimodal_towers:
        module_names.extend(
            [
                "model.vision_tower",
                "model.audio_tower",
                "model.embed_vision",
                "model.embed_audio",
            ]
        )
    if not args.train_per_layer_embeddings:
        module_names.extend(
            [
                "model.language_model.embed_tokens_per_layer",
                "model.language_model.per_layer_model_projection",
                "model.language_model.per_layer_projection_norm",
            ]
        )

    for module_name in module_names:
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            continue
        for param in module.parameters():
            if param.requires_grad:
                frozen_params += param.numel()
                param.requires_grad_(False)

    if frozen_params:
        print(f"Froze {frozen_params:,} Gemma 4 parameters for text-only SFT")


def build_lora_config(args: argparse.Namespace):
    if not args.use_lora:
        return None
    from peft import LoraConfig, TaskType

    target_modules = (
        GEMMA4_LORA_TARGET_MODULES
        if "gemma-4" in args.model_name.lower()
        else DEFAULT_LORA_TARGET_MODULES
    )
    config_kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
        "use_rslora": True,
        "init_lora_weights": "gaussian",
        "modules_to_save": ["lm_head"],
        "ensure_weight_tying": True,
    }
    return LoraConfig(**filter_kwargs(LoraConfig, config_kwargs))


def filter_kwargs(callable_obj, kwargs: dict) -> dict:
    params = set(inspect.signature(callable_obj).parameters)
    return {key: value for key, value in kwargs.items() if key in params}


class DataCollatorForCausalLMWithTailLogits:
    def __init__(
        self,
        tokenizer,
        max_length: int | None,
        pad_to_multiple_of: int | None = None,
        tail_logits_only: bool = False,
        input_vocab_size: int | None = None,
        label_vocab_size: int | None = None,
        invalid_label_action: str = "mask",
    ) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError(
                "Tokenizer must define a pad_token_id before building batches."
            )
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tail_logits_only = tail_logits_only
        self.input_vocab_size = input_vocab_size
        self.label_vocab_size = label_vocab_size
        self.invalid_label_action = invalid_label_action

    def __call__(self, features: list[dict]) -> dict:
        features = [self.truncate_feature(feature) for feature in features]
        batch = self.pad_features(features)
        self.validate_input_ids(batch)
        self.validate_labels(batch)
        if self.tail_logits_only:
            self.keep_only_supervised_tail_logits(batch)
        return batch

    def truncate_feature(self, feature: dict) -> dict:
        if not self.max_length:
            return feature

        truncated = dict(feature)
        seq_len = len(truncated["input_ids"])
        if seq_len <= self.max_length:
            return truncated

        for key in ("input_ids", "attention_mask", "labels"):
            if key in truncated:
                truncated[key] = truncated[key][-self.max_length :]
        return truncated

    def pad_features(self, features: list[dict]) -> dict:
        max_len = 0
        normalized = []
        for feature in features:
            input_ids = list(feature["input_ids"])
            attention_mask = list(feature.get("attention_mask", [1] * len(input_ids)))
            labels = list(feature["labels"])
            lengths = {
                "input_ids": len(input_ids),
                "attention_mask": len(attention_mask),
                "labels": len(labels),
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Feature has misaligned sequence lengths: {lengths}")
            max_len = max(max_len, len(input_ids))
            normalized.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in normalized:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * pad_len
            )
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)

        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }

    def validate_input_ids(self, batch: dict) -> None:
        if self.input_vocab_size is None or "input_ids" not in batch:
            return

        input_ids = batch["input_ids"]
        invalid = input_ids.ge(self.input_vocab_size) | input_ids.lt(0)
        if not invalid.any():
            return

        bad_ids = input_ids[invalid]
        raise ValueError(
            "Batch contains input_ids outside the model input vocabulary: "
            f"min={bad_ids.min().item()}, max={bad_ids.max().item()}, "
            f"input_vocab_size={self.input_vocab_size}"
        )

    def validate_labels(self, batch: dict) -> None:
        if self.label_vocab_size is None or "labels" not in batch:
            return

        labels = batch["labels"]
        valid_label_positions = labels.ne(-100)
        invalid = valid_label_positions & (
            labels.ge(self.label_vocab_size) | labels.lt(0)
        )
        if not invalid.any():
            return

        bad_ids = labels[invalid]
        message = (
            "Batch contains labels outside the model output vocabulary: "
            f"min={bad_ids.min().item()}, max={bad_ids.max().item()}, "
            f"label_vocab_size={self.label_vocab_size}, count={bad_ids.numel()}"
        )
        if self.invalid_label_action == "error":
            raise ValueError(message)

        print(f"{message}. Masking them to -100.")
        batch["labels"] = labels.masked_fill(invalid, -100)

    @staticmethod
    def keep_only_supervised_tail_logits(batch: dict) -> None:
        labels = batch.get("labels")
        if labels is None:
            return

        supervised = labels.ne(-100)
        if not supervised.any():
            batch["logits_to_keep"] = 1
            batch["labels"] = labels[:, -1:].contiguous()
            return

        seq_len = labels.shape[1]
        positions = (
            torch.arange(seq_len, device=labels.device)
            .unsqueeze(0)
            .expand_as(labels)
        )
        first_supervised = (
            torch.where(supervised, positions, seq_len)
            .min(dim=1)
            .values.min()
            .item()
        )
        first_logit = max(first_supervised - 1, 0)
        logits_to_keep = seq_len - first_logit
        if logits_to_keep < seq_len:
            batch["labels"] = labels[:, -logits_to_keep:].contiguous()
        batch["logits_to_keep"] = logits_to_keep


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    logits = logits.float()
    labels = F.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    if logits.shape[:-1] != shift_labels.shape:
        raise ValueError(
            "Logits and shifted labels are misaligned: "
            f"logits={tuple(logits.shape)}, shifted_labels={tuple(shift_labels.shape)}"
        )

    supervised = shift_labels.ne(ignore_index)
    if not supervised.any():
        return logits.sum() * 0.0

    vocab_size = logits.shape[-1]
    invalid = supervised & (shift_labels.lt(0) | shift_labels.ge(vocab_size))
    if invalid.any():
        bad_ids = shift_labels[invalid]
        raise ValueError(
            "Shifted labels contain ids outside the logits vocabulary: "
            f"min={bad_ids.min().item()}, max={bad_ids.max().item()}, "
            f"logits_vocab_size={vocab_size}, count={bad_ids.numel()}"
        )

    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1).to(logits.device),
        ignore_index=ignore_index,
        reduction=reduction,
    )
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


class CausalLMSFTTrainer(SFTTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        inputs["use_cache"] = False
        outputs = model(**inputs)
        loss = causal_lm_loss(
            outputs.logits,
            labels,
            num_items_in_batch=num_items_in_batch,
        )
        return (loss, outputs) if return_outputs else loss


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
            "trust_remote_code": True,
        },
        "lr_scheduler_kwargs": {"num_cycles": args.num_cycles},
        "logging_steps": args.logging_steps,
        "activation_offloading": args.activation_offloading,
        "use_liger_kernel": not args.no_liger_kernel,
        "use_liger": not args.no_liger_kernel,
        "logging_dir": args.logging_dir,
        "output_dir": args.output_dir,
        "report_to": args.report_to,
        "max_length": args.max_length,
        "max_seq_length": args.max_length,
    }
    return SFTConfig(**filter_kwargs(SFTConfig.__init__, config_kwargs))


def embedding_vocab_size(model: torch.nn.Module, output: bool = False) -> int | None:
    if output:
        embedding = model.get_output_embeddings()
    else:
        embedding = model.get_input_embeddings()

    if embedding is None:
        return None
    if hasattr(embedding, "weight"):
        return embedding.weight.shape[0]
    if hasattr(embedding, "out_features"):
        return embedding.out_features
    return None


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
    model = load_model(args, dtype)
    freeze_gemma4_text_only_modules(model, args)
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = build_sft_config(args, dtype)

    data_collator = DataCollatorForCausalLMWithTailLogits(
        tokenizer,
        max_length=args.max_length,
        pad_to_multiple_of=8,
        tail_logits_only=(
            "gemma-4" in args.model_name.lower()
            and not args.no_tail_logits_only
        ),
        input_vocab_size=embedding_vocab_size(model),
        label_vocab_size=embedding_vocab_size(model, output=True),
        invalid_label_action=args.invalid_label_action,
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
    trainer = CausalLMSFTTrainer(
        **filter_kwargs(CausalLMSFTTrainer.__init__, trainer_kwargs)
    )

    trainer.train(
        resume_from_checkpoint=checkpoint_arg(
            args.output_dir,
            args.resume_from_checkpoint,
        )
    )


if __name__ == "__main__":
    main()
