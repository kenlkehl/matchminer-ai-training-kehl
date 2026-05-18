#!/usr/bin/env python3
"""Build single-task SFT data for note compression.

The source data is the final parquet produced by ``3_compress_notes.py``. Each
training row is rendered with the Liquid chat template and contains:

  system compression prompt
  user metadata + original synthetic note
  assistant <think>Gemma reasoning</think> final compressed text

The output parquet has one column, ``text``. The script can also tokenize that
parquet into a Hugging Face dataset with prompt-masked labels for SFT.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets.arrow_writer import ArrowWriter
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Thinking"
DEFAULT_INPUT = "../data/no_phi/compressed_synthetic_notes.parquet"
DEFAULT_OUTPUT_DIR = "../data/no_phi/note_compression_training_data"
DEFAULT_NUM_WORKERS = min(os.cpu_count() or 1, 32)


def load_compression_module():
    """Load ``3_compress_notes.py`` despite its digit-prefixed file name."""
    module_path = REPO_ROOT / "3_compress_notes.py"
    spec = importlib.util.spec_from_file_location("compress_notes", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_scalar(value: Any) -> Any | None:
    """Return None for pandas/Arrow missing values or empty strings."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if str(value).strip() == "":
        return None
    return value


def token_len(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def truncate_field(text: str, max_tokens: int, tokenizer) -> str:
    """Token-truncate text with a head+tail strategy."""
    toks = tokenizer(text, add_special_tokens=False).input_ids
    if len(toks) <= max_tokens:
        return text
    half = max(1, max_tokens // 2)
    return tokenizer.decode(toks[:half]) + " ... " + tokenizer.decode(toks[-half:])


def find_column(
    names: set[str],
    preferred: str,
    aliases: tuple[str, ...],
    label: str,
) -> str:
    if preferred in names:
        return preferred
    for alias in aliases:
        if alias in names:
            print(f"Column '{preferred}' not found; using '{alias}' for {label}.")
            return alias
    candidates = ", ".join([preferred, *aliases])
    raise ValueError(f"Could not find {label} column. Tried: {candidates}")


def build_metadata(
    row: dict[str, Any],
    row_number: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row_id = clean_scalar(row.get("row_idx"))
    fallback_doc_id = f"row{row_id if row_id is not None else row_number}"

    metadata: dict[str, Any] = {
        "document_id": clean_scalar(row.get(args.document_id_col)) or fallback_doc_id,
    }

    if args.patient_id_col in row:
        patient_id = clean_scalar(row.get(args.patient_id_col))
        if patient_id is not None:
            metadata["patient_id"] = patient_id
    if args.date_col in row:
        date_value = clean_scalar(row.get(args.date_col))
        if date_value is not None:
            metadata["date"] = date_value
    if args.note_type_col in row:
        note_type = clean_scalar(row.get(args.note_type_col))
        if note_type is not None:
            metadata["note_type"] = note_type

    return metadata


def build_messages(
    compression_module,
    document_text: str,
    reasoning: str,
    summary: str,
    metadata: dict[str, Any],
) -> list[dict[str, str]]:
    user_content = compression_module.build_document_user_prompt(
        document_text=document_text,
        metadata=metadata,
    )
    assistant_content = f"<think>\n{reasoning.strip()}\n</think>\n{summary.strip()}"
    return [
        {"role": "system", "content": compression_module.COMPRESSION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def render_training_text(
    compression_module,
    tokenizer,
    document_text: str,
    reasoning: str,
    summary: str,
    metadata: dict[str, Any],
    max_seq_length: int,
) -> tuple[str | None, bool]:
    """Render one example, truncating only the source note if needed."""
    original_document = document_text
    working_document = document_text
    was_truncated = False

    for _ in range(4):
        messages = build_messages(
            compression_module=compression_module,
            document_text=working_document,
            reasoning=reasoning,
            summary=summary,
            metadata=metadata,
        )
        text = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            enable_thinking=True,
        )
        total = token_len(text, tokenizer)
        if total <= max_seq_length:
            return text, was_truncated

        document_tokens = token_len(working_document, tokenizer)
        overhead = total - document_tokens
        budget = max_seq_length - overhead - 16
        if budget <= 0:
            return None, was_truncated

        working_document = truncate_field(original_document, budget, tokenizer)
        was_truncated = True

    return None, was_truncated


def source_columns(args: argparse.Namespace, schema_names: set[str]) -> list[str]:
    columns = [
        args.text_col,
        args.reasoning_col,
        args.summary_col,
        args.document_id_col,
        args.patient_id_col,
        args.date_col,
        args.note_type_col,
    ]
    if "row_idx" in schema_names:
        columns.append("row_idx")
    return list(dict.fromkeys(c for c in columns if c and c in schema_names))


def iter_source_rows(
    input_parquet: str,
    columns: list[str],
    batch_size: int,
):
    pf = pq.ParquetFile(input_parquet)
    row_number = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        values = batch.to_pydict()
        for i in range(batch.num_rows):
            yield row_number, {col: values[col][i] for col in columns}
            row_number += 1


def flush_texts(writer: pq.ParquetWriter, texts: list[str]) -> None:
    table = pa.Table.from_pydict(
        {"text": texts},
        schema=pa.schema([pa.field("text", pa.large_string())]),
    )
    writer.write_table(table)


def build_text_parquet(args: argparse.Namespace, tokenizer) -> tuple[int, int, int]:
    compression_module = load_compression_module()
    pf = pq.ParquetFile(args.input_parquet)
    schema_names = set(pf.schema_arrow.names)

    args.text_col = find_column(schema_names, args.text_col, ("synthetic_note",), "note text")
    args.reasoning_col = find_column(
        schema_names,
        args.reasoning_col,
        ("reasoning", "summary_reasoning"),
        "reasoning",
    )
    args.summary_col = find_column(schema_names, args.summary_col, ("summary",), "summary")

    out_path = Path(args.output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_path} exists. Pass --overwrite to replace it.")
        out_path.unlink()

    columns = source_columns(args, schema_names)
    print(f"Reading {args.input_parquet}")
    print(f"Rows: {pf.metadata.num_rows}")
    print(f"Using columns: {columns}")

    writer = pq.ParquetWriter(
        out_path,
        pa.schema([pa.field("text", pa.large_string())]),
    )

    written = 0
    dropped = 0
    truncated = 0
    pending: list[str] = []

    try:
        for row_number, row in iter_source_rows(
            args.input_parquet,
            columns=columns,
            batch_size=args.source_batch_size,
        ):
            if args.max_examples is not None and written >= args.max_examples:
                break

            document = clean_scalar(row.get(args.text_col))
            reasoning = clean_scalar(row.get(args.reasoning_col))
            summary = clean_scalar(row.get(args.summary_col))
            if document is None or summary is None:
                dropped += 1
                continue
            if not args.keep_error_summaries and str(summary).strip().startswith("ERROR:"):
                dropped += 1
                continue
            if reasoning is None and not args.keep_empty_reasoning:
                dropped += 1
                continue

            text, was_truncated = render_training_text(
                compression_module=compression_module,
                tokenizer=tokenizer,
                document_text=str(document),
                reasoning="" if reasoning is None else str(reasoning),
                summary=str(summary),
                metadata=build_metadata(row, row_number, args),
                max_seq_length=args.max_seq_length,
            )
            if text is None:
                dropped += 1
                continue
            pending.append(text)
            written += 1
            truncated += int(was_truncated)

            if len(pending) >= args.writer_batch_size:
                flush_texts(writer, pending)
                pending.clear()
                print(f"  Wrote {written} examples...")

        if pending:
            flush_texts(writer, pending)
    finally:
        writer.close()

    print(f"Text parquet: {out_path}")
    print(f"Built {written} examples; truncated {truncated}; dropped {dropped}.")
    return written, truncated, dropped


def find_last_subsequence(seq: list[int], subseq: list[int]) -> int:
    if not subseq:
        return len(seq)
    seq_bytes = struct.pack(f"{len(seq)}I", *seq)
    sub_bytes = struct.pack(f"{len(subseq)}I", *subseq)
    pos = seq_bytes.rfind(sub_bytes)
    if pos < 0:
        return -1
    return pos // 4


def assistant_header_ids(tokenizer) -> list[list[int]]:
    headers = [
        "<|im_start|>assistant\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ]
    encoded = []
    for header in headers:
        ids = tokenizer.encode(header, add_special_tokens=False)
        if ids:
            encoded.append(ids)
    return encoded


def mask_prompt(input_ids: list[int], header_options: list[list[int]]) -> tuple[list[int], bool]:
    labels = list(input_ids)
    best_idx = -1
    best_len = 0
    for header_ids in header_options:
        idx = find_last_subsequence(input_ids, header_ids)
        if idx > best_idx:
            best_idx = idx
            best_len = len(header_ids)
    if best_idx < 0:
        return labels, False
    mask_end = best_idx + best_len
    labels[:mask_end] = [-100] * mask_end
    return labels, True


def tokenize_worker(args_tuple):
    (
        source_parquet,
        row_group_indices,
        model_name,
        max_seq_length,
        arrow_path,
        header_options,
        batch_size,
        worker_idx,
        num_workers,
    ) = args_tuple
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pf = pq.ParquetFile(source_parquet)
    writer = ArrowWriter(path=arrow_path)
    total = sum(pf.metadata.row_group(i).num_rows for i in row_group_indices)
    rows_done = 0
    masked_count = 0
    unmasked_count = 0
    prefix = f"[Worker {worker_idx + 1}/{num_workers}] "

    for rg_idx in row_group_indices:
        texts = pf.read_row_group(rg_idx, columns=["text"]).column("text").to_pylist()
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]
            tokenized = tokenizer(
                batch_texts,
                max_length=max_seq_length,
                truncation=True,
            )
            labels = []
            for input_ids in tokenized["input_ids"]:
                row_labels, masked = mask_prompt(input_ids, header_options)
                labels.append(row_labels)
                masked_count += int(masked)
                unmasked_count += int(not masked)

            writer.write_batch(
                {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": labels,
                }
            )

            rows_done += len(batch_texts)
            if rows_done % (batch_size * 10) < batch_size:
                print(f"  {prefix}Tokenized {rows_done}/{total} examples...")

    num_examples, num_bytes = writer.finalize()
    print(f"  {prefix}Wrote {num_examples} examples ({num_bytes / 1e6:.1f} MB)")
    return num_examples, num_bytes, masked_count, unmasked_count


def streaming_tokenize(
    source_parquet: str,
    tokenizer,
    max_seq_length: int,
    output_path: str,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> int:
    output = Path(output_path)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(source_parquet)
    total_rows = pf.metadata.num_rows
    num_row_groups = pf.metadata.num_row_groups
    num_workers = max(1, min(num_workers, total_rows, num_row_groups))
    header_options = assistant_header_ids(tokenizer)

    if num_workers <= 1:
        worker_args = [
            (
                source_parquet,
                list(range(num_row_groups)),
                tokenizer.name_or_path,
                max_seq_length,
                str(output / "data-00000-of-00001.arrow"),
                header_options,
                batch_size,
                0,
                1,
            )
        ]
        results = [tokenize_worker(worker_args[0])]
        data_files = [{"filename": "data-00000-of-00001.arrow"}]
    else:
        assignments = [[] for _ in range(num_workers)]
        for rg_idx in range(num_row_groups):
            assignments[rg_idx % num_workers].append(rg_idx)

        worker_args = []
        for i, group in enumerate(assignments):
            if not group:
                continue
            worker_args.append(
                (
                    source_parquet,
                    group,
                    tokenizer.name_or_path,
                    max_seq_length,
                    str(output / f"data-{i:05d}-of-{num_workers:05d}.arrow"),
                    header_options,
                    batch_size,
                    i,
                    num_workers,
                )
            )
        print(f"Launching {len(worker_args)} tokenization workers...")
        with mp.Pool(len(worker_args)) as pool:
            results = pool.map(tokenize_worker, worker_args)
        data_files = [
            {"filename": f"data-{i:05d}-of-{num_workers:05d}.arrow"}
            for i in range(len(worker_args))
        ]

    num_examples = sum(r[0] for r in results)
    num_bytes = sum(r[1] for r in results)
    masked_count = sum(r[2] for r in results)
    unmasked_count = sum(r[3] for r in results)

    with open(output / "state.json", "w") as f:
        json.dump(
            {
                "_data_files": data_files,
                "_fingerprint": "note_compression_tokenized",
                "_format_columns": None,
                "_format_kwargs": {},
                "_format_type": None,
                "_output_all_columns": False,
                "_split": None,
            },
            f,
            indent=2,
        )
    with open(output / "dataset_info.json", "w") as f:
        json.dump({}, f)

    print(f"Tokenized dataset: {output}")
    print(f"Rows: {num_examples}; bytes: {num_bytes / 1e6:.1f} MB")
    print(f"Prompt-masked: {masked_count}; unmasked: {unmasked_count}")
    if unmasked_count:
        print("WARNING: some examples did not match a known assistant header.")
    return num_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Liquid SFT data for the note compression task."
    )
    parser.add_argument("--input-parquet", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-parquet", default=None)
    parser.add_argument("--tokenized-dir", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=50000)
    parser.add_argument("--text-col", default="synthetic_note")
    parser.add_argument("--reasoning-col", default="summary_reasoning")
    parser.add_argument("--summary-col", default="summary")
    parser.add_argument("--document-id-col", default="document_id")
    parser.add_argument("--patient-id-col", default="pseudo_mrn")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--note-type-col", default="note_type")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--source-batch-size", type=int, default=1000)
    parser.add_argument("--writer-batch-size", type=int, default=1000)
    parser.add_argument("--tokenize-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--keep-empty-reasoning", action="store_true")
    parser.add_argument("--keep-error-summaries", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    args.output_parquet = args.output_parquet or str(output_dir / "all_training_data.parquet")
    args.tokenized_dir = args.tokenized_dir or str(output_dir / "tokenized_training_data.dataset")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    build_text_parquet(args, tokenizer)

    if not args.skip_tokenize:
        streaming_tokenize(
            source_parquet=args.output_parquet,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            output_path=args.tokenized_dir,
            batch_size=args.tokenize_batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
