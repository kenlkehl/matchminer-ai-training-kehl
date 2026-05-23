# Patient Summarization Training

Single-task SFT setup for training `google/gemma-4-E2B-it` to perform
the iterative patient summarization task from `6_summarize_patients.py`.

## Create Data

```bash
python patient_summarization_training/create_training_data.py \
  --input-parquet ../data/no_phi/patient_serial_summaries.parquet \
  --chunks-parquet ../data/no_phi/summary_shards/prepared_chunks.parquet \
  --output-dir ../data/no_phi/patient_summarization_training_data \
  --model-name google/gemma-4-E2B-it \
  --tokenizer-name google/gemma-4-E2B-it \
  --reasoning-parser gemma4 \
  --max-seq-length 50000
```

This writes:

- `../data/no_phi/patient_summarization_training_data/all_training_data.parquet`
- `../data/no_phi/patient_summarization_training_data/tokenized_training_data.dataset`

Each `text` row is a Gemma 4 chat-format conversation with the prior summary
and next clinical record chunk in the user message, followed by the reasoning
trace in Gemma's `<|channel>thought ... <channel|>` format and the final
updated patient summary in the model turn.

## Train

```bash
python patient_summarization_training/fine_tune_llm.py \
  --dataset-dir ../data/no_phi/patient_summarization_training_data/tokenized_training_data.dataset \
  --model-name google/gemma-4-E2B-it \
  --tokenizer-name google/gemma-4-E2B-it \
  --output-dir ../models/patient_summarization_gemma4_e2b_it
```

For a quick wiring test after data creation:

```bash
python patient_summarization_training/fine_tune_llm.py \
  --dataset-dir ../data/no_phi/patient_summarization_training_data/tokenized_training_data.dataset \
  --output-dir /tmp/patient_summarization_gemma4_e2b_it_smoke \
  --max-steps 1
```
