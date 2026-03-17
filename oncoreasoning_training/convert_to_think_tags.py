#!/usr/bin/env python3
"""Convert existing all_training_data.parquet reasoning markers to <think>/</think> tags,
then re-tokenize the data.

Usage:
  python convert_to_think_tags.py path/to/all_training_data.parquet --max-seq-length 32000
"""
import argparse
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from create_all_training_data import apply_think_tags, streaming_tokenize

CONVERT_BATCH_SIZE = 10_000


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

    # Step 1: Replace reasoning markers (streaming — constant memory)
    pf = pq.ParquetFile(args.input)
    total_rows = pf.metadata.num_rows
    print(f"Reading {args.input} ({total_rows} rows)")

    # Write to a temp file when overwriting the input, then atomically swap
    overwriting = os.path.abspath(output_parquet) == os.path.abspath(args.input)
    if overwriting:
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix='.parquet', dir=os.path.dirname(os.path.abspath(output_parquet)))
        os.close(tmp_fd)
        write_path = tmp_path
    else:
        tmp_path = None
        write_path = output_parquet

    print("Applying think tag replacements (streaming)...")
    schema = pf.schema_arrow
    writer = pq.ParquetWriter(write_path, schema)
    rows_done = 0
    try:
        for batch in pf.iter_batches(batch_size=CONVERT_BATCH_SIZE):
            texts = batch.column("text").to_pylist()
            converted = [apply_think_tags(t) for t in texts]
            out_batch = pa.RecordBatch.from_pydict(
                {"text": converted}, schema=schema)
            writer.write_batch(out_batch)
            rows_done += len(texts)
            if rows_done % (CONVERT_BATCH_SIZE * 10) < CONVERT_BATCH_SIZE:
                print(f"  Processed {rows_done}/{total_rows} rows...")
        writer.close()
        if overwriting:
            os.replace(tmp_path, output_parquet)
            tmp_path = None  # successfully moved, don't clean up
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    print(f"  Wrote {rows_done} rows to {output_parquet}")

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
