#!/usr/bin/env python3
"""Smoke-test a trained artifact's answer-first contract on a prepared QA prompt."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oncoreasoning_training import contracts as c


def generation_options(tokenizer, prompt_length, *, quick=False, max_new_tokens=1024):
    letters = tuple(c.letter_token_ids(tokenizer).values())
    vocabulary = list(range(len(tokenizer)))
    def allowed(batch_id, tokens):
        return letters if len(tokens) == prompt_length else vocabulary
    return {"do_sample": False, "max_new_tokens": 1 if quick else max_new_tokens,
            "prefix_allowed_tokens_fn": allowed}


def main():
    import torch
    from oncoreasoning_training.create_all_training_data import jsonl, read_json
    from oncoreasoning_training.fine_tune_llm import load_student
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True, help="Prepared requests.jsonl")
    p.add_argument("--request-id", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    metadata = read_json(args.model / "oncoreasoning_contract.json")
    if metadata.get("artifact_type") == "adapter_collection":
        args.model = args.model / metadata["adapters"]["clinical_qa"]
        metadata = read_json(args.model / "oncoreasoning_contract.json")
    if metadata["format_version"] != c.FORMAT_VERSION:
        raise ValueError("Unsupported artifact format")
    if "clinical_qa" not in metadata["tasks"]:
        raise ValueError("Select the clinical_qa adapter for answer-first previews")
    task = next((row for row in jsonl(args.requests) if row["id"] == args.request_id), None)
    if task is None or task["category"] != "clinical_qa":
        raise ValueError("Select a prepared clinical QA request")
    tokenizer = c.load_chat_tokenizer(str(args.model))
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    source = metadata["base_model"] if metadata["lora_adapter"] else str(args.model)
    model, _ = load_student(source, dtype=dtype)
    if metadata["lora_adapter"]:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model)
    model.to(args.device).eval()
    inputs = tokenizer(c.render_prompt(tokenizer, task["messages"]), add_special_tokens=False, return_tensors="pt").to(args.device)
    length = inputs.input_ids.shape[1]
    options = generation_options(tokenizer, length, quick=args.quick, max_new_tokens=args.max_new_tokens)
    if length + options["max_new_tokens"] > metadata["max_seq_length"]:
        raise ValueError("Request exceeds the trained context budget")
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    print(tokenizer.decode(output[0, length:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
