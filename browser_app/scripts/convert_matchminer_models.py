#!/usr/bin/env python3
"""Export MatchMiner-AI Hugging Face models to browser-oriented ONNX folders.

The script writes artifacts under browser_app/artifacts/models by default. It
does not commit weights. After export, publish or serve the folders and update
public/models.manifest.json or the in-app model IDs to point at those ONNX
repositories/paths.

Expected source models:
  - ksg-dfci/TrialSpace-0526
  - ksg-dfci/TrialChecker-0526
  - ksg-dfci/BoilerplateChecker-0526
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = {
    "TrialSpace": {
        "source": "ksg-dfci/TrialSpace-0526",
        "task": "feature-extraction",
        "output_transform": "mean_pool_normalize",
    },
    "TrialChecker": {
        "source": "ksg-dfci/TrialChecker-0526",
        "task": "text-classification",
        "output_transform": "sigmoid",
    },
    "BoilerplateChecker": {
        "source": "ksg-dfci/BoilerplateChecker-0526",
        "task": "text-classification",
        "output_transform": "softmax_positive",
    },
}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def export_with_optimum(source: str, task: str, out_dir: Path, opset: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        "--model",
        source,
        "--task",
        task,
        "--opset",
        str(opset),
        str(out_dir),
    ]
    run(cmd)


def export_with_torch(source_or_snapshot: str | Path, task: str, out_dir: Path, opset: int) -> None:
    """Manual ONNX export.

    Optimum currently trips on SentenceTransformer wrappers for TrialSpace in
    this environment. A direct Transformers export is sufficient for the
    browser app: feature extraction consumes last_hidden_state and applies mean
    pooling in JS; classifiers consume logits and apply sigmoid/softmax in JS.
    """
    import torch
    from torch import nn
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    source = str(source_or_snapshot)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if task == "feature-extraction":
        model = AutoModel.from_pretrained(source, trust_remote_code=True, dtype=dtype)

        class Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_ids, attention_mask):
                output = self.inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=False,
                )
                return output[0]

    elif task == "text-classification":
        model = AutoModelForSequenceClassification.from_pretrained(source, trust_remote_code=True, dtype=dtype)

        class Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_ids, attention_mask):
                output = self.inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
                return output[0]

    else:
        raise ValueError(f"Unsupported manual export task: {task}")

    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    model.to(device)
    model.eval()
    wrapper = Wrapper(model).eval()
    dummy = tokenizer(
        "Age: 70\nSex: Male\nCancer type: lung cancer\nBiomarkers: KRAS G12C.",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    dummy = {key: value.to(device) for key, value in dummy.items()}
    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state" if task == "feature-extraction" else "logits"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        output_names[0]: {0: "batch"},
    }
    if task == "feature-extraction":
        dynamic_axes[output_names[0]][1] = "sequence"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(out_dir / "model.onnx"),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
            external_data=True,
        )


def quantize_dynamic(model_dir: Path, suffix: str = "_quantized") -> list[str]:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:  # pragma: no cover - depends on optional local env
        print(f"[warn] onnxruntime quantization unavailable: {exc}")
        return []

    created: list[str] = []
    for onnx_path in sorted(model_dir.glob("*.onnx")):
        if onnx_path.name.endswith(f"{suffix}.onnx"):
            continue
        if has_fp16_initializers(onnx_path):
            print(f"[quantize] skipping {onnx_path.name}: dynamic quantization from fp16 produces invalid ORT graphs")
            continue
        out_path = onnx_path.with_name(f"{onnx_path.stem}{suffix}.onnx")
        print(f"[quantize] {onnx_path.name} -> {out_path.name}")
        quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QInt8)
        fix_quantized_scale_initializers(out_path)
        created.append(out_path.name)
    return created


def has_fp16_initializers(onnx_path: Path) -> bool:
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(onnx_path), load_external_data=False)
    return any(initializer.data_type == TensorProto.FLOAT16 for initializer in model.graph.initializer)


def fix_quantized_scale_initializers(onnx_path: Path) -> None:
    """ORT dynamic quantization of fp16 exports can leave DQ scale tensors in fp16.

    ONNX Runtime rejects those graphs. DequantizeLinear scales should be float32
    for broad runtime compatibility, including ORT Web.
    """
    import onnx
    from onnx import TensorProto, numpy_helper

    model = onnx.load(str(onnx_path))
    changed = False
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.data_type == TensorProto.FLOAT16 and initializer.name.endswith("_scale"):
            array = numpy_helper.to_array(initializer).astype("float32")
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array, initializer.name))
            changed = True
    if changed:
        onnx.save(model, str(onnx_path))


def copy_tokenizer_sidecars(source_snapshot: Path, out_dir: Path) -> None:
    names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "modules.json",
        "sentence_bert_config.json",
        "config_sentence_transformers.json",
    ]
    for name in names:
        src = source_snapshot / name
        if src.exists() and not (out_dir / name).exists():
            shutil.copy2(src, out_dir / name)
    pooling = source_snapshot / "1_Pooling"
    if pooling.exists() and not (out_dir / "1_Pooling").exists():
        shutil.copytree(pooling, out_dir / "1_Pooling")


def snapshot_download(source: str, cache_dir: Path | None) -> Path:
    from huggingface_hub import snapshot_download as hf_snapshot_download

    path = hf_snapshot_download(repo_id=source, cache_dir=str(cache_dir) if cache_dir else None)
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/models", help="Directory for ONNX model folders.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--only", choices=sorted(DEFAULT_MODELS), nargs="*")
    args = parser.parse_args()

    root = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    root.mkdir(parents=True, exist_ok=True)

    selected = args.only or list(DEFAULT_MODELS)
    manifest = {
        "created_by": "scripts/convert_matchminer_models.py",
        "models": [],
    }

    for name in selected:
        spec = DEFAULT_MODELS[name]
        source = spec["source"]
        out_dir = root / f"{name}-0526-ONNX"
        print(f"\n=== {name}: {source} ===")
        snapshot = snapshot_download(source, cache_dir)
        export_with_torch(snapshot, spec["task"], out_dir, args.opset)
        copy_tokenizer_sidecars(snapshot, out_dir)
        quantized = [] if args.skip_quantize else quantize_dynamic(out_dir)
        (out_dir / "browser_model.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "source_model": source,
                    "task": spec["task"],
                    "output_transform": spec["output_transform"],
                    "opset": args.opset,
                    "quantized_files": quantized,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest["models"].append(
            {
                "name": name,
                "source": source,
                "path": str(out_dir),
                "task": spec["task"],
                "output_transform": spec["output_transform"],
            }
        )

    (root / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {root / 'conversion_manifest.json'}")


if __name__ == "__main__":
    main()
