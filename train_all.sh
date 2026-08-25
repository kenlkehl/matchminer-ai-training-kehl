# note: vllm steps can fail for parallelized scripts if there is no pre-compiled cache yet for that vllm config. if this happens, restarting script at that stage should allow training to proceed.

# Override MODEL / REASONING_PARSER via env to swap LLMs. REASONING_PARSER=auto
# infers from MODEL (see vllm_reasoning_utils.MODEL_TO_PARSER).
# Example: MODEL=Qwen/Qwen3.6-27B-FP8 REASONING_PARSER=qwen3 bash train_all.sh
#MODEL="${MODEL:-google/gemma-4-31b-it}"
MODEL="${MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}"  # made change 4/28/26 at point of first trialchecks, after patient summarization on synth data.
REASONING_PARSER="${REASONING_PARSER:-auto}"

# Skip a training step when its output model directory already exists and is
# non-empty. Returns 0 (skip) → the `|| <cmd>` form leaves <cmd> unrun.
skip_if_done() {
    local out="$1"
    local label="${2:-step}"
    if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then
        echo "[skip] $label: $out already exists and is non-empty"
        return 0
    fi
    return 1
}

# pull a JSON file from ctgov
# clinicaltrials.gov API can be challenging to work with, so we just did it manually by going to this link:
# https://clinicaltrials.gov/search?cond=cancer%20OR%20lymphoma%20OR%20carcinoma%20OR%20leukemia%20OR%20sarcoma%20OR%20melanoma%20OR%20myeloma%20OR%20myelodysplastic%20OR%20myeloproliferative&aggFilters=phase:0%201%202%203%204,status:not%20rec,studyType:int
# and then selecting "Download" and a JSON download option.

# to work with the same file we started with, do:
wget -P ../data/no_phi https://huggingface.co/datasets/ksg-dfci/mmai-synthetic/resolve/main/ctgov_interventional_phased_cancer_trials_11-3-25.json


# try to pre-compile relevant vllm configurations

# aggregator=$(cat << EOF
# from vllm import LLM
# llm = LLM(
#         model='${MODEL}',
#         tensor_parallel_size=1,
#         download_dir="~/models",
#         gpu_memory_utilization=0.95,
#         max_model_len=220000,
#     )

# EOF
# )




python 0a_parse_ctgov_json.py 

python 0b_create_trial_spaces.py --input ../data/no_phi/ctgov_trials.csv --gpus 0,1,2,3,4,5,6,7 --gpus-per-instance 1 \
  --model "$MODEL" --reasoning-parser "$REASONING_PARSER" \
  --download-dir ~/models --max-model-len 30000 --max-tokens 20000 --gpu-mem-util 0.90

echo 0 done

python 0c_sample_trial_spaces.py

python 1a_make_synthetic_enrollee_prompts.py

echo 1a done

python 1b_make_synthetic_negative_enrollee_prompts.py

echo 1b done


python 2_make_synthetic_notes_sharded.py \
  --input_csv ../data/no_phi/trial_spaces_with_positive_prompts.csv \
  --out_dir ../data/no_phi/synthetic_notes \
  --model "$MODEL" \
  --reasoning-parser "$REASONING_PARSER" \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --download_dir ~/models \
  --max_model_len 25000 --max_new_tokens 20000 \
  --batch_size 1000 --tp 1 --temperature 0.75 --top_p 0.5 \
  --perturb_prob 0.3

echo 2a done



python 2_make_synthetic_notes_sharded.py \
  --input_csv ../data/no_phi/trial_spaces_with_negative_prompts.csv \
  --out_dir ../data/no_phi/synthetic_negative_notes \
  --model "$MODEL" \
  --reasoning-parser "$REASONING_PARSER" \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --download_dir ~/models \
  --max_model_len 25000 --max_new_tokens 20000 \
  --batch_size 1000 --tp 1 --temperature 0.75 --top_p 0.5 \
  --perturb_prob 0.3

echo 2b done



aggregator=$(cat << 'EOF'
import hashlib
import pandas as pd
from datetime import date, timedelta


def stable_u64(*parts):
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "big", signed=False)


def add_synthetic_dates(df, patient_id_col="pseudo_mrn", date_col="date"):
    seed = 20260519
    min_start = date.fromisoformat("2014-01-01")
    max_start = date.fromisoformat("2020-12-31")
    max_increment_days = 180
    span_days = (max_start - min_start).days
    states = {}
    dates = []

    for _, row in df.iterrows():
        key = str(row[patient_id_col])
        state = states.get(key)
        if state is None:
            offset = stable_u64(seed, key, "start_date") % (span_days + 1)
            current_date = min_start + timedelta(days=offset)
            states[key] = {"note_count": 1, "current_date": current_date}
        else:
            increment = stable_u64(seed, key, state["note_count"], "increment") % (
                max_increment_days + 1
            )
            current_date = state["current_date"] + timedelta(days=increment)
            state["current_date"] = current_date
            state["note_count"] += 1
        dates.append(current_date)

    output = df.copy()
    output[date_col] = dates
    return output

positive_notes = pd.read_parquet("../data/no_phi/synthetic_notes/synthetic_notes.parquet")

print(positive_notes.info())

negative_notes = pd.read_parquet("../data/no_phi/synthetic_negative_notes/synthetic_notes.parquet")

print(negative_notes.info())

spaces = pd.read_csv('../data/no_phi/trial_spaces_with_positive_prompts.csv')

negative_notes['pseudo_mrn'] = negative_notes.pseudo_mrn * 1000000

output = pd.concat([positive_notes, negative_notes], ignore_index=True)

output = pd.merge(output, spaces, on='space_index')
output = add_synthetic_dates(output)

output.to_parquet("../data/no_phi/all_synthetic_notes.parquet")
output.to_parquet("../data/no_phi/all_synthetic_notes_with_dates.parquet")

EOF
)

python -c "$aggregator"


python 6_summarize_patients.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --patient_id_col pseudo_mrn \
  --text_col synthetic_note \
  --output_parquet ../data/no_phi/patient_serial_summaries.parquet \
  --shard_dir ../data/no_phi/summary_shards \
  --model "$MODEL" \
  --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --gpus_per_server 1 \
  --max_model_len 100000 \
  --max_tokens 20000 \
  --base_port 8000 \
  --chunk_size 50000 \
  --chunk_overlap 500 \
  --max_concurrent_requests 25

echo 6 done




aggregator=$(cat << EOF
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





python llm_check_trials.py \
 --input_parquet ../data/no_phi/patient_summaries_with_spaces.parquet \
 --out_dir ../data/no_phi/initial_trialcheck_outputs \
 --final_output space_specific_eligibility_checks.parquet \
 --gpus 0,1,2,3,4,5,6,7 \
 --gpus_per_kernel 1 \
 --prompt_batch_size 2000 \
 --model "$MODEL" \
 --reasoning-parser "$REASONING_PARSER" \
 --download_dir ~/models \
 --max_model_len 50000 \
 --gpu_memory_utilization 0.95

echo 7 done

mv ../data/no_phi/initial_trialcheck_outputs/space_specific_eligibility_checks.parquet ../data/no_phi/space_specific_eligibility_checks.parquet

skip_if_done ../models/pt_trial_summary_perspace_finetuned.model "step 8 embedder" || \
accelerate launch finetune_embedder.py -i ../data/no_phi/space_specific_eligibility_checks.parquet \
-c ~/models/initial_embedder_training -m Qwen/Qwen3-Embedding-0.6B -o ../models/pt_trial_summary_perspace_finetuned.model

echo 8 done

python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model ../models/pt_trial_summary_perspace_finetuned.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 \
  --top_k_patients 40 \
  --encode_batch_size 128 \
  --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet ../data/no_phi/top_cohorts_tocheck_round1.parquet \
  --out_patients_parquet ../data/no_phi/top_patients_tocheck_round1.parquet

echo 9a done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_cohorts_tocheck_round1.parquet \
  --out_dir ../data/no_phi/round1_patientcentric_checks \
  --final_output top_cohorts_checked_round1.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model "$MODEL" \
  --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 \
  --gpu_memory_utilization 0.95

echo 9b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round1.parquet \
  --out_dir ../data/no_phi/round1_trialcentric_checks \
  --final_output top_patients_checked_round1.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model "$MODEL" \
  --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 \
  --gpu_memory_utilization 0.95

echo 9c done

skip_if_done ../models/reranker_round1.model "step 10 reranker_round1" || \
accelerate launch finetune_embedder.py \
   -i ../data/no_phi/round1_trialcentric_checks/top_patients_checked_round1.parquet \
   -i ../data/no_phi/round1_patientcentric_checks/top_cohorts_checked_round1.parquet \
   -c ../models/reranker1_training \
   -m ../models/pt_trial_summary_perspace_finetuned.model \
   -o ../models/reranker_round1.model

echo 10 done

python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model ../models/reranker_round1.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 \
  --top_k_patients 40 \
  --encode_batch_size 128 \
  --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet ../data/no_phi/top_cohorts_tocheck_round2.parquet \
  --out_patients_parquet ../data/no_phi/top_patients_tocheck_round2.parquet

echo 11a done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_cohorts_tocheck_round2.parquet \
  --out_dir ../data/no_phi/round2_patientcentric_checks \
  --final_output top_cohorts_checked_round2.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model "$MODEL" \
  --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 \
  --gpu_memory_utilization 0.95

echo 11b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round2.parquet \
  --out_dir ../data/no_phi/round2_trialcentric_checks \
  --final_output top_patients_checked_round2.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model "$MODEL" \
  --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 \
  --gpu_memory_utilization 0.95

echo 11c done

skip_if_done ../models/reranker_round2.model "step 12 reranker_round2" || \
accelerate launch finetune_embedder.py \
   -i ../data/no_phi/round2_trialcentric_checks/top_patients_checked_round2.parquet \
   -i ../data/no_phi/round2_patientcentric_checks/top_cohorts_checked_round2.parquet \
   -c ../models/reranker2_training \
   -m ../models/reranker_round1.model \
   -o ../models/reranker_round2.model

echo 12 done


python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model ../models/reranker_round2.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 \
  --top_k_patients 40 \
  --encode_batch_size 128 \
  --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet ../data/no_phi/top_cohorts_tocheck_round3.parquet \
  --out_patients_parquet ../data/no_phi/top_patients_tocheck_round3.parquet

echo 13a done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_cohorts_tocheck_round3.parquet \
  --out_dir ../data/no_phi/round3_patientcentric_checks \
  --final_output top_cohorts_checked_round3.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model "$MODEL" \
  --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 \
  --gpu_memory_utilization 0.95

echo 13b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round3.parquet \
  --out_dir ../data/no_phi/round3_trialcentric_checks \
  --final_output top_patients_checked_round3.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model "$MODEL" \
  --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --max_model_len 50000 \
  --gpu_memory_utilization 0.95

echo 13c done

python 14_check_boilerplate.py \
  --model "$MODEL" \
  --reasoning-parser "$REASONING_PARSER" \
  --download_dir ~/models \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 1000 \
  --max_model_len 50000 \
  --max_new_tokens 20000 \
  --gpu_memory_utilization 0.95 \
  --out_dir ../data/no_phi/boilerplate_checks

echo 14 done

skip_if_done ../models/modernbert-trial-checker-regression "step 15 trial checker" || \
accelerate launch --num_processes 8 15_train_modernbert_trial_checker.py

echo 15 done

skip_if_done ../models/boilerplatechecker "step 16 boilerplate checker" || \
accelerate launch --num_processes 8 16_train_modernbert_boilerplate_checker.py

echo 16 done

# Step 17 — per-drug evidence labels and GoodOptionChecker. One local teacher
# pool uses public registry arm metadata to exclude comparator/background drugs
# before drug-only web research, then labels grouped patient-trial examples.
if skip_if_done \
  ../models/goodoptionchecker_four_point \
  "step 17 good option checker"; then
  echo 17 skipped
else
  python train_good_option_checker.py generate \
    --model "$MODEL" \
    --reasoning-parser "$REASONING_PARSER" \
    --download-dir ~/models \
    --gpus 0,1,2,3,4,5,6,7 \
    --gpus-per-server 1 \
    --max-model-len 50000 \
    --gpu-memory-utilization 0.95

  accelerate launch --num_processes 8 train_good_option_checker.py train \
    --patient-validation-fraction 0.20 \
    --trial-validation-fraction 0.20
  echo 17 done
fi
