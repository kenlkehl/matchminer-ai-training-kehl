#!/usr/bin/env python3
"""Register the pulled adapter-only upgrade without stopping the active pipeline."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oncoreasoning_training.pipeline_upgrade import register_upgrade

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = register_upgrade(ROOT, args.run_dir)
    print(f"Registered {receipt['from_commit']} -> {receipt['to_commit']}; active process left running")
