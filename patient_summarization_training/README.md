# Patient Summarization Training

Single-task SFT setup for training `LiquidAI/LFM2.5-1.2B-Thinking` to perform
the iterative patient summarization task from `6_summarize_patients.py`.

## Create Data

```bash
python patient_summarization_training/create_training_data.py \
  --input-parquet ../data/no_phi/patient_serial_summaries.parquet \
  --chunks-parquet ../data/no_phi/summary_shards_compressed/prepared_chunks.parquet \
  --output-dir ../data/no_phi/patient_summarization_training_data \
  --model-name LiquidAI/LFM2.5-1.2B-Thinking \
  --max-seq-length 50000
```

This writes:

- `../data/no_phi/patient_summarization_training_data/all_training_data.parquet`
- `../data/no_phi/patient_summarization_training_data/tokenized_training_data.dataset`

Each `text` row is a Liquid chat-template conversation with the prior summary
and next clinical record chunk in the user message, followed by the reasoning
trace in `<think>...</think>` tags and the final updated patient summary in the
assistant message.

## Train

```bash
python patient_summarization_training/fine_tune_llm.py \
  --dataset-dir ../data/no_phi/patient_summarization_training_data/tokenized_training_data.dataset \
  --output-dir ../models/patient_summarization_lfm
```

For a quick wiring test after data creation:

```bash
python patient_summarization_training/fine_tune_llm.py \
  --dataset-dir ../data/no_phi/patient_summarization_training_data/tokenized_training_data.dataset \
  --output-dir /tmp/patient_summarization_lfm_smoke \
  --max-steps 1
```
