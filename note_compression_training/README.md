# Note Compression Training

Single-task SFT setup for training `LiquidAI/LFM2.5-1.2B-Thinking` to perform
the note compression task from `3_compress_notes.py`.

## Create Data

```bash
python note_compression_training/create_training_data.py \
  --input-parquet ../data/no_phi/compressed_synthetic_notes.parquet \
  --output-dir ../data/no_phi/note_compression_training_data \
  --model-name LiquidAI/LFM2.5-1.2B-Thinking \
  --max-seq-length 50000
```

This writes:

- `../data/no_phi/note_compression_training_data/all_training_data.parquet`
- `../data/no_phi/note_compression_training_data/tokenized_training_data.dataset`

Each `text` row is a Liquid chat-template conversation with the original
synthetic note in the user message and the Gemma compression reasoning plus
final compressed note in the assistant message.

## Train

```bash
python note_compression_training/fine_tune_llm.py \
  --dataset-dir ../data/no_phi/note_compression_training_data/tokenized_training_data.dataset \
  --output-dir ../models/note_compression_lfm
```

For a quick wiring test after data creation:

```bash
python note_compression_training/fine_tune_llm.py \
  --dataset-dir ../data/no_phi/note_compression_training_data/tokenized_training_data.dataset \
  --output-dir /tmp/note_compression_lfm_smoke \
  --max-steps 1
```
