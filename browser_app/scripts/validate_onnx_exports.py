#!/usr/bin/env python3
"""Smoke-validate exported ONNX artifacts against their source model shapes.

This validation intentionally uses small fixed strings. It catches missing
tokenizer sidecars, incompatible ONNX graphs, and obvious output-shape errors.
It is not a numeric equivalence certification for quantized artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EXAMPLES = {
    "trial": (
        "Age range allowed: 18 years and older. Sex allowed: Both. Cancer type allowed: non-small cell lung cancer.",
        "Age: 68\nSex: Male\nCancer type: Non-small cell lung cancer\nBiomarkers: KRAS G12C mutation",
    ),
    "boilerplate": (
        "Patient history: ECOG 1. No active pneumonitis.\nTrial exclusions: Active pneumonitis; poor performance status.",
    ),
}


def validate_classifier(model_dir: Path, text: str) -> None:
    from transformers import AutoTokenizer
    import onnxruntime as ort

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True, max_length=512)
    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files in {model_dir}")
    for onnx_file in onnx_files:
        session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        feed = {name: value for name, value in inputs.items() if name in {i.name for i in session.get_inputs()}}
        outputs = session.run(None, feed)
        logits = np.asarray(outputs[0])
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise AssertionError(f"Unexpected classifier logits shape for {onnx_file.name}: {logits.shape}")
        print(f"[ok] {model_dir.name}/{onnx_file.name}: logits shape {logits.shape}")


def validate_feature_extractor(model_dir: Path, text: str) -> None:
    from transformers import AutoTokenizer
    import onnxruntime as ort

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True, max_length=512)
    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files in {model_dir}")
    for onnx_file in onnx_files:
        session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        feed = {name: value for name, value in inputs.items() if name in {i.name for i in session.get_inputs()}}
        outputs = session.run(None, feed)
        hidden = np.asarray(outputs[0])
        if hidden.ndim not in (2, 3):
            raise AssertionError(f"Unexpected feature output shape for {onnx_file.name}: {hidden.shape}")
        print(f"[ok] {model_dir.name}/{onnx_file.name}: feature shape {hidden.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="artifacts/models")
    args = parser.parse_args()

    root = Path(args.models_dir)
    validate_feature_extractor(root / "TrialSpace-0526-ONNX", EXAMPLES["trial"][1])
    validate_classifier(root / "TrialChecker-0526-ONNX", EXAMPLES["trial"][0] + "\nNow here is the patient summary:" + EXAMPLES["trial"][1])
    validate_classifier(root / "BoilerplateChecker-0526-ONNX", EXAMPLES["boilerplate"][0])


if __name__ == "__main__":
    main()
