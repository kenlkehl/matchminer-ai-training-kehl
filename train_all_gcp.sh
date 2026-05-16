#!/usr/bin/env bash
# GCP-orchestrated version of train_all.sh.
#
# Distributes vLLM inference across a configurable list of GCP worker
# instances (each 1–8 GPUs). The instance from which this script runs is
# the orchestrator; it never appears in the worker list. Workers are
# brought up before every vLLM step and torn down again (full instance
# stop) before each non-vLLM step listed by the user (make_top_matches,
# finetune_embedder, 15_train_modernbert_trial_checker,
# 16_train_modernbert_boilerplate_checker). When the script exits
# (success, failure, or Ctrl-C), all worker instances are stopped.
#
# Required env:
#   GCP_INSTANCES_FILE  Path to a file with one `name:zone` line per worker.
#
# Optional env:
#   PYTHON_ENV          Same path on every worker; default /home/kenneth_kehl/thisenv.
#   SERVERS_FILE        Path to the JSON file the orchestrator writes; default
#                       /tmp/mmai_gcp_servers.json.
#   INCLUDE_SELF        Comma-separated GPU IDs on the orchestrator to use as
#                       extra local vLLM servers (default "0,1,2,3,4,5,6,7" =
#                       use all 8 local GPUs). Set INCLUDE_SELF= (empty) to
#                       disable and dispatch only to remote workers.
#   MODEL               Override the HF model id (default same as train_all.sh).
#   REASONING_PARSER    Override vLLM reasoning parser (default 'auto').
#
# Single-machine mode is untouched: use train_all.sh as before.

set -uo pipefail

MODEL="${MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}"
REASONING_PARSER="${REASONING_PARSER:-auto}"

: "${GCP_INSTANCES_FILE:?GCP_INSTANCES_FILE must be set (file with name:zone per worker)}"
PYTHON_ENV="${PYTHON_ENV:-/home/kenneth_kehl/thisenv}"
SERVERS_FILE="${SERVERS_FILE:-/tmp/mmai_gcp_servers.json}"
INCLUDE_SELF="${INCLUDE_SELF:-0,1,2,3,4,5,6,7}"

ORCH_PID=""

cleanup() {
    if [[ -n "$ORCH_PID" ]]; then
        kill -TERM "$ORCH_PID" 2>/dev/null || true
        wait "$ORCH_PID" 2>/dev/null || true
        ORCH_PID=""
    fi
    echo "[train_all_gcp] final cleanup: stopping all worker instances"
    python gcp_vllm_orchestrator.py stop-instances \
        --instances-file "$GCP_INSTANCES_FILE" || true
}
trap cleanup EXIT INT TERM

# Bring up the vLLM cluster with the given config.
# Args: 1=max_model_len 2=max_num_seqs 3=gpu_memory_utilization 4=gpus_per_server (default 1)
start_vllm_cluster() {
    local mml="$1"
    local mns="$2"
    local gmu="$3"
    local gps="${4:-1}"

    local extra=()
    if [[ -n "$INCLUDE_SELF" ]]; then
        extra+=(--include-self "$INCLUDE_SELF")
    fi

    python gcp_vllm_orchestrator.py serve \
        --instances-file "$GCP_INSTANCES_FILE" \
        --python-env "$PYTHON_ENV" \
        --servers-file "$SERVERS_FILE" \
        --model "$MODEL" \
        --reasoning-parser "$REASONING_PARSER" \
        --gpus-per-server "$gps" \
        --max-model-len "$mml" \
        --max-num-seqs "$mns" \
        --gpu-memory-utilization "$gmu" \
        --download-dir "~/models" \
        "${extra[@]}" &
    ORCH_PID=$!

    python gcp_vllm_orchestrator.py wait-for-ready \
        --servers-file "$SERVERS_FILE" --timeout 1800
}

# Tear down vLLM processes on workers (instances remain RUNNING for cheap
# resume on the next vLLM step). Use stop_workers_fully to also stop the
# instances themselves.
stop_vllm_cluster() {
    if [[ -n "$ORCH_PID" ]]; then
        kill -TERM "$ORCH_PID" 2>/dev/null || true
        wait "$ORCH_PID" 2>/dev/null || true
        ORCH_PID=""
    fi
}

# Stop everything (vLLM processes + the worker VMs). Used before the
# non-vLLM steps the user listed: make_top_matches, finetune_embedder,
# 15_train_modernbert_trial_checker, 16_train_modernbert_boilerplate_checker.
stop_workers_fully() {
    stop_vllm_cluster
    python gcp_vllm_orchestrator.py stop-instances \
        --instances-file "$GCP_INSTANCES_FILE" || true
}

set -e

# ---------------------------------------------------------------------------
# Step 0a — local-only (no vLLM)
# ---------------------------------------------------------------------------

# pull the ctgov JSON the same way train_all.sh does
wget -P ../data/no_phi https://huggingface.co/datasets/ksg-dfci/mmai-synthetic/resolve/main/ctgov_interventional_phased_cancer_trials_11-3-25.json

python 0a_parse_ctgov_json.py

# ---------------------------------------------------------------------------
# Step 0b — vLLM (max_model_len=30000)
# ---------------------------------------------------------------------------
start_vllm_cluster 30000 900 0.90 1
python 0b_create_trial_spaces.py \
  --input ../data/no_phi/ctgov_trials.csv \
  --server_urls_file "$SERVERS_FILE" \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download-dir ~/models --max-model-len 30000 --max-tokens 20000 --gpu-mem-util 0.90
echo 0 done
stop_vllm_cluster

python 0c_sample_trial_spaces.py

# ---------------------------------------------------------------------------
# Step 1a / 1b — vLLM
# ---------------------------------------------------------------------------
start_vllm_cluster 50000 900 0.90 1
python 1a_make_synthetic_enrollee_prompts.py \
  --server_urls_file "$SERVERS_FILE" \
  --model-name "$MODEL" --reasoning-parser "$REASONING_PARSER"
echo 1a done

python 1b_make_synthetic_negative_enrollee_prompts.py \
  --server_urls_file "$SERVERS_FILE" \
  --model-name "$MODEL" --reasoning-parser "$REASONING_PARSER"
echo 1b done
stop_vllm_cluster

# ---------------------------------------------------------------------------
# Step 2a / 2b — vLLM (max_model_len=25000)
# ---------------------------------------------------------------------------
start_vllm_cluster 25000 900 0.90 1
python 2_make_synthetic_notes_sharded.py \
  --input_csv ../data/no_phi/trial_spaces_with_positive_prompts.csv \
  --out_dir ../data/no_phi/synthetic_notes \
  --server_urls_file "$SERVERS_FILE" \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 25000 --max_new_tokens 20000 \
  --batch_size 1000 --tp 1 --temperature 0.75 --top_p 0.5 \
  --perturb_prob 0.3
echo 2a done

python 2_make_synthetic_notes_sharded.py \
  --input_csv ../data/no_phi/trial_spaces_with_negative_prompts.csv \
  --out_dir ../data/no_phi/synthetic_negative_notes \
  --server_urls_file "$SERVERS_FILE" \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 25000 --max_new_tokens 20000 \
  --batch_size 1000 --tp 1 --temperature 0.75 --top_p 0.5 \
  --perturb_prob 0.3
echo 2b done
stop_vllm_cluster

# Step 2c — local merge (no vLLM)
aggregator=$(cat << 'EOF'
import pandas as pd

positive_notes = pd.read_parquet("../data/no_phi/synthetic_notes/synthetic_notes.parquet")
print(positive_notes.info())

negative_notes = pd.read_parquet("../data/no_phi/synthetic_negative_notes/synthetic_notes.parquet")
print(negative_notes.info())

spaces = pd.read_csv('../data/no_phi/trial_spaces_with_positive_prompts.csv')

negative_notes['pseudo_mrn'] = negative_notes.pseudo_mrn * 1000000

output = pd.concat([positive_notes, negative_notes], ignore_index=True)
output = pd.merge(output, spaces, on='space_index')
output.to_parquet("../data/no_phi/all_synthetic_notes.parquet")
EOF
)
python -c "$aggregator"

# ---------------------------------------------------------------------------
# Step 3 — vLLM (max_model_len=50000)
# ---------------------------------------------------------------------------
start_vllm_cluster 50000 900 0.90 1
python 3_compress_notes.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --output_parquet ../data/no_phi/compressed_synthetic_notes.parquet \
  --shard_dir ../data/no_phi/compressed_note_shards \
  --server_urls_file "$SERVERS_FILE" \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 --max_tokens 10000 \
  --max_concurrent_requests 50 \
  --patient_id_col pseudo_mrn --text_col synthetic_note
echo 3 done
stop_vllm_cluster

# Local rename / dedup (no vLLM)
aggregator=$(cat << 'EOF'
import pandas as pd
compressed = pd.read_parquet('../data/no_phi/compressed_synthetic_notes.parquet')
compressed = compressed.rename(columns={'patient_id': 'pseudo_mrn'})
summary_str = compressed['summary'].astype(str)
mask = (summary_str.str.strip() != '') & (~summary_str.str.startswith('ERROR:'))
compressed = compressed[mask].reset_index(drop=True)
compressed.to_parquet('../data/no_phi/compressed_synthetic_notes_for_patient_summary.parquet')
print(f"Kept {len(compressed)} compressed notes for patient summarization")
EOF
)
python -c "$aggregator"

# ---------------------------------------------------------------------------
# Step 6 — vLLM (max_model_len=100000)
# ---------------------------------------------------------------------------
start_vllm_cluster 100000 900 0.90 1
python 6_summarize_patients.py \
  --input_parquet ../data/no_phi/compressed_synthetic_notes_for_patient_summary.parquet \
  --patient_id_col pseudo_mrn --text_col summary \
  --output_parquet ../data/no_phi/patient_serial_summaries.parquet \
  --shard_dir ../data/no_phi/summary_shards_compressed \
  --server_urls_file "$SERVERS_FILE" \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 100000 --max_tokens 20000 \
  --chunk_size 50000 --chunk_overlap 500 \
  --max_concurrent_requests 25 \
  --generate_dates \
  --synthetic_start_date 2017-01-01 \
  --synthetic_min_days 0 --synthetic_max_days 180
echo 6 done
stop_vllm_cluster

aggregator=$(cat << 'EOF'
import pandas as pd
spaces = pd.read_csv('../data/no_phi/sample_trial_space_lineitems.csv')
summaries = pd.read_parquet('../data/no_phi/patient_summaries.parquet')
notes = pd.read_parquet('../data/no_phi/all_synthetic_notes.parquet')[['pseudo_mrn','space_index']].groupby('pseudo_mrn').first().reset_index()
summaries['pseudo_mrn'] = pd.to_numeric(summaries.pseudo_mrn)
summaries = pd.merge(summaries, notes, on='pseudo_mrn')
summaries = pd.merge(summaries, spaces, on='space_index')
summaries.to_parquet('../data/no_phi/patient_summaries_with_spaces.parquet')
EOF
)
python -c "$aggregator"

# ---------------------------------------------------------------------------
# Step 7 — vLLM
# ---------------------------------------------------------------------------
start_vllm_cluster 50000 900 0.95 1
python llm_check_trials.py \
 --input_parquet ../data/no_phi/patient_summaries_with_spaces.parquet \
 --out_dir ../data/no_phi/initial_trialcheck_outputs \
 --final_output space_specific_eligibility_checks.parquet \
 --server_urls_file "$SERVERS_FILE" \
 --prompt_batch_size 2000 \
 --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
 --download_dir ~/models \
 --max_model_len 50000 --gpu_memory_utilization 0.95
echo 7 done
stop_vllm_cluster

mv ../data/no_phi/initial_trialcheck_outputs/space_specific_eligibility_checks.parquet ../data/no_phi/space_specific_eligibility_checks.parquet

# ---------------------------------------------------------------------------
# Step 8 — finetune_embedder (no vLLM) → stop workers
# ---------------------------------------------------------------------------
stop_workers_fully
accelerate launch finetune_embedder.py \
  -i ../data/no_phi/space_specific_eligibility_checks.parquet \
  -c ~/models/initial_embedder_training \
  -m Qwen/Qwen3-Embedding-0.6B \
  -o ../models/pt_trial_summary_perspace_finetuned.model
echo 8 done

# ---------------------------------------------------------------------------
# Step 9a — make_top_matches (no vLLM) → workers stay stopped
# ---------------------------------------------------------------------------
python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model ../models/pt_trial_summary_perspace_finetuned.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 --top_k_patients 40 \
  --encode_batch_size 128 --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet ../data/no_phi/top_cohorts_tocheck_round1.parquet \
  --out_patients_parquet ../data/no_phi/top_patients_tocheck_round1.parquet
echo 9a done

# ---------------------------------------------------------------------------
# Steps 9b / 9c — vLLM
# ---------------------------------------------------------------------------
start_vllm_cluster 50000 900 0.95 1
python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_cohorts_tocheck_round1.parquet \
  --out_dir ../data/no_phi/round1_patientcentric_checks \
  --final_output top_cohorts_checked_round1.parquet \
  --server_urls_file "$SERVERS_FILE" \
  --prompt_batch_size 2000 \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 --gpu_memory_utilization 0.95
echo 9b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round1.parquet \
  --out_dir ../data/no_phi/round1_trialcentric_checks \
  --final_output top_patients_checked_round1.parquet \
  --server_urls_file "$SERVERS_FILE" \
  --prompt_batch_size 2000 \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 --gpu_memory_utilization 0.95
echo 9c done
stop_vllm_cluster

# ---------------------------------------------------------------------------
# Step 10 — finetune_embedder (no vLLM) → stop workers
# ---------------------------------------------------------------------------
stop_workers_fully
accelerate launch finetune_embedder.py \
   -i ../data/no_phi/round1_trialcentric_checks/top_patients_checked_round1.parquet \
   -i ../data/no_phi/round1_patientcentric_checks/top_cohorts_checked_round1.parquet \
   -c ../models/reranker1_training \
   -m ../models/pt_trial_summary_perspace_finetuned.model \
   -o ../models/reranker_round1.model
echo 10 done

# ---------------------------------------------------------------------------
# Step 11a — make_top_matches (no vLLM)
# ---------------------------------------------------------------------------
python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model ../models/reranker_round1.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 --top_k_patients 40 \
  --encode_batch_size 128 --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet ../data/no_phi/top_cohorts_tocheck_round2.parquet \
  --out_patients_parquet ../data/no_phi/top_patients_tocheck_round2.parquet
echo 11a done

# ---------------------------------------------------------------------------
# Steps 11b / 11c — vLLM
# ---------------------------------------------------------------------------
start_vllm_cluster 50000 900 0.95 1
python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_cohorts_tocheck_round2.parquet \
  --out_dir ../data/no_phi/round2_patientcentric_checks \
  --final_output top_cohorts_checked_round2.parquet \
  --server_urls_file "$SERVERS_FILE" \
  --prompt_batch_size 2000 \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 --gpu_memory_utilization 0.95
echo 11b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round2.parquet \
  --out_dir ../data/no_phi/round2_trialcentric_checks \
  --final_output top_patients_checked_round2.parquet \
  --server_urls_file "$SERVERS_FILE" \
  --prompt_batch_size 2000 \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 --gpu_memory_utilization 0.95
echo 11c done
stop_vllm_cluster

# ---------------------------------------------------------------------------
# Step 12 — finetune_embedder (no vLLM) → stop workers
# ---------------------------------------------------------------------------
stop_workers_fully
accelerate launch finetune_embedder.py \
   -i ../data/no_phi/round2_trialcentric_checks/top_patients_checked_round2.parquet \
   -i ../data/no_phi/round2_patientcentric_checks/top_cohorts_checked_round2.parquet \
   -c ../models/reranker2_training \
   -m ../models/reranker_round1.model \
   -o ../models/reranker_round2.model
echo 12 done

# ---------------------------------------------------------------------------
# Step 13a — make_top_matches (no vLLM)
# ---------------------------------------------------------------------------
python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model ../models/reranker_round2.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 --top_k_patients 40 \
  --encode_batch_size 128 --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet ../data/no_phi/top_cohorts_tocheck_round3.parquet \
  --out_patients_parquet ../data/no_phi/top_patients_tocheck_round3.parquet
echo 13a done

# ---------------------------------------------------------------------------
# Steps 13b / 13c — vLLM
# ---------------------------------------------------------------------------
start_vllm_cluster 50000 900 0.95 1
python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_cohorts_tocheck_round3.parquet \
  --out_dir ../data/no_phi/round3_patientcentric_checks \
  --final_output top_cohorts_checked_round3.parquet \
  --server_urls_file "$SERVERS_FILE" \
  --prompt_batch_size 2000 \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 --gpu_memory_utilization 0.95
echo 13b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round3.parquet \
  --out_dir ../data/no_phi/round3_trialcentric_checks \
  --final_output top_patients_checked_round3.parquet \
  --server_urls_file "$SERVERS_FILE" \
  --prompt_batch_size 2000 \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 --gpu_memory_utilization 0.95
echo 13c done

# ---------------------------------------------------------------------------
# Step 14 — vLLM (max_model_len=50000)
# ---------------------------------------------------------------------------
# (Same vLLM config as 13b/c — keep the cluster up.)
python 14_check_boilerplate.py \
  --server_urls_file "$SERVERS_FILE" \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --prompt_batch_size 1000 \
  --max_model_len 50000 --max_new_tokens 20000 \
  --gpu_memory_utilization 0.95 \
  --out_dir ../data/no_phi/boilerplate_checks
echo 14 done
stop_vllm_cluster

# ---------------------------------------------------------------------------
# Steps 15 / 16 — modernbert training (no vLLM) → stop workers
# ---------------------------------------------------------------------------
stop_workers_fully
accelerate launch --num_processes 8 15_train_modernbert_trial_checker.py
echo 15 done

accelerate launch --num_processes 8 16_train_modernbert_boilerplate_checker.py
echo 16 done

echo "[train_all_gcp] all 16 steps complete."
