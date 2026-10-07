"""Explicit approval records for changing only the pending adapter trainer.

The live parent keeps its original plan and manifest fingerprint. Preserve that
fingerprint on restart so its completion markers remain valid; require a receipt
for the exact new commit and prove that earlier stage commands are unchanged.
"""
from __future__ import annotations

import ast
import datetime as dt
import json
from pathlib import Path
import subprocess

from oncoreasoning_training.contracts import digest
from oncoreasoning_training.create_all_training_data import atomic_json

ALLOWED_PATHS = {
    "oncoreasoning_training/fine_tune_llm.py",
    "oncoreasoning_training/preview_model.py",
    "oncoreasoning_training/pipeline_upgrade.py",
    "oncoreasoning_training/README.md", "README.md", "train_from_summaries.py",
    "scripts/register_oncoreasoning_upgrade.py",
    "tests/test_oncoreasoning_training_data.py", "tests/test_oncoreasoning_adapters.py",
    "tests/test_gemma_pipeline.py",
}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def validate_code_scope(repo, old_commit, new_commit):
    changed = git(repo, "diff", "--name-only", old_commit, new_commit).splitlines()
    if not changed or set(changed) - ALLOWED_PATHS:
        raise ValueError("Upgrade changes files outside the pending adapter training scope")
    # The parser, stage plan and teacher lifecycle must be literally unchanged
    # as Python syntax. Only the runner's manifest/resume integration can change.
    def protected_functions(commit):
        tree = ast.parse(git(repo, "show", f"{commit}:train_from_summaries.py"))
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "main"}
    if protected_functions(old_commit) != protected_functions(new_commit):
        raise ValueError("Upgrade changes the running pipeline's stage plan or teacher lifecycle")
    git(repo, "diff", "--exit-code", "HEAD", "--")
    return changed


def register_upgrade(repo, run_dir):
    repo, run_dir = Path(repo), Path(run_dir)
    manifest = json.loads((run_dir / "pipeline_manifest.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())
    model_dir = Path(manifest["args"]["models_dir"]) / "oncoreasoning"
    if (status.get("status") == "complete" or status.get("stage") == "train-oncoreasoning"
            or (run_dir / "completed/train-oncoreasoning.json").exists()
            or (model_dir.exists() and any(model_dir.iterdir()))):
        raise ValueError("Register the adapter upgrade before OncoReasoning training begins")
    current = {**manifest, "training_commit": git(repo, "rev-parse", "HEAD")}
    changed = validate_code_scope(repo, manifest["training_commit"], current["training_commit"])
    receipt = {"kind": "separate-oncoreasoning-lora-v1", "original_identity_sha256": digest(manifest),
        "approved_identity_sha256": digest(current), "from_commit": manifest["training_commit"],
        "to_commit": current["training_commit"], "changed_paths": changed,
        "registered_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic_json(run_dir / "oncoreasoning_code_upgrade.json", receipt)
    return receipt


def compatible_identity(previous, current, run_dir, repo):
    if previous == current:
        return previous
    receipt_path = Path(run_dir) / "oncoreasoning_code_upgrade.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("kind") == "separate-oncoreasoning-lora-v1"
                and receipt.get("original_identity_sha256") == digest(previous)
                and receipt.get("approved_identity_sha256") == digest(current)
                and {**previous, "training_commit": current["training_commit"]} == current):
            validate_code_scope(repo, previous["training_commit"], current["training_commit"])
            print("Using registered OncoReasoning adapter upgrade; preserving completed stages", flush=True)
            return previous
    raise ValueError("Pipeline inputs/code/settings changed; use a fresh run directory or register the scoped adapter upgrade")
