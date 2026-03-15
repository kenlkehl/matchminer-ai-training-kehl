#!/usr/bin/env python3
"""Convert existing all_training_data.parquet reasoning markers to <think>/</think> tags,
then re-tokenize the data.

Usage:
  python convert_to_think_tags.py path/to/all_training_data.parquet --max-seq-length 32000
"""
import argparse
import os
import re

import pandas as pd
from transformers import AutoTokenizer

from create_all_training_data import apply_think_tags, streaming_tokenize


def main():
    parser = argparse.ArgumentParser(
        description="Convert reasoning markers and re-tokenize training data"
    )
    parser.add_argument("input", help="Path to all_training_data.parquet")
    parser.add_argument("--output", help="Output parquet path (default: overwrite input)")
    parser.add_argument("--max-seq-length", type=int, required=True,
                        help="Max sequence length for tokenization")
    parser.add_argument("--model-name", type=str,
                        default="LiquidAI/LFM2.5-1.2B-Thinking",
                        help="Tokenizer model (default: LiquidAI/LFM2.5-1.2B-Thinking)")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1,
                        help="Number of parallel workers for tokenization (default: all CPUs)")
    parser.add_argument("--writer-batch-size", type=int, default=1000,
                        help="Writer batch size for tokenization (default: 1000)")
    args = parser.parse_args()

    output_parquet = args.output or args.input

    # Step 1: Replace reasoning markers
    print(f"Reading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df)} rows")

    print("Applying think tag replacements...")
    df['text'] = df['text'].apply(apply_think_tags)

    print(f"Writing to {output_parquet}...")
    df.to_parquet(output_parquet)
    print(f"  Wrote {len(df)} rows")
    del df

    # Step 2: Re-tokenize
    tokenized_path = os.path.join(os.path.dirname(output_parquet),
                                  'tokenized_training_data.dataset')

    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizing with max_length={args.max_seq_length} (streaming to disk)...")
    num_examples = streaming_tokenize(
        source_parquet=output_parquet,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        output_path=tokenized_path,
        batch_size=args.writer_batch_size,
        num_workers=args.num_workers,
    )
    print(f"Total examples: {num_examples}")
    print(f"Tokenized dataset: {tokenized_path}")
    print("Done!")


if __name__ == "__main__":
    main()
