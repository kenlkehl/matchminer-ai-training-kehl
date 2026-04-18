

python 6_summarize_patients.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --output_parquet ../data/no_phi/patient_serial_summaries.parquet \
  --shard_dir ../data/no_phi/summary_shards \
  --model Qwen/Qwen3.6-35B-A3B \
  --download_dir ~/models \
  --reasoning_marker "</think>" \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --gpus_per_server 1 \
  --max_model_len 30000 \
  --base_port 8000 \
  --chunk_size 50000 \
  --chunk_overlap 500 \
  --max_concurrent_requests 100 \
  --generate_dates \
  --synthetic_start_date 2017-01-01 \
  --synthetic_min_days 0 \
  --synthetic_max_days 180 

echo 6 done

exit


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
 --model openai/gpt-oss-120b \
 --download_dir /data1/ken/models \
 --max_model_len 20000 \
 --gpu_memory_utilization 0.95

echo 7 done

mv ../data/no_phi/initial_trialcheck_outputs/space_specific_eligibility_checks.parquet ../data/no_phi/space_specific_eligibility_checks.parquet

accelerate launch finetune_embedder.py -i ../data/no_phi/space_specific_eligibility_checks.parquet \
-c /data1/ken/models/initial_embedder_training -m Qwen/Qwen3-Embedding-0.6B -o /data1/ken/models/pt_trial_summary_perspace_finetuned.model

echo 8 done

python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model /data1/ken/models/pt_trial_summary_perspace_finetuned.model \
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
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.95

echo 9b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round1.parquet \
  --out_dir ../data/no_phi/round1_trialcentric_checks \
  --final_output top_patients_checked_round1.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.95

echo 9c done

accelerate launch finetune_embedder.py \
   -i ../data/no_phi/round1_trialcentric_checks/top_patients_checked_round1.parquet \
   -i ../data/no_phi/round1_patientcentric_checks/top_cohorts_checked_round1.parquet \
   -c /data1/ken/models/reranker1_training \
   -m /data1/ken/models/pt_trial_summary_perspace_finetuned.model \
   -o /data1/ken/models/reranker_round1.model

echo 10 done

python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model /data1/ken/models/reranker_round1.model \
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
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.95

echo 11b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round2.parquet \
  --out_dir ../data/no_phi/round2_trialcentric_checks \
  --final_output top_patients_checked_round2.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.95

echo 11c done

accelerate launch finetune_embedder.py \
   -i ../data/no_phi/round2_trialcentric_checks/top_patients_checked_round2.parquet \
   -i ../data/no_phi/round2_patientcentric_checks/top_cohorts_checked_round2.parquet \
   -c /data1/ken/models/reranker2_training \
   -m /data1/ken/models/reranker_round1.model \
   -o /data1/ken/models/reranker_round2.model

echo 12 done


python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model /data1/ken/models/reranker_round2.model \
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
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.95

echo 13b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round3.parquet \
  --out_dir ../data/no_phi/round3_trialcentric_checks \
  --final_output top_patients_checked_round3.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.95

echo 13c done

python 14_check_boilerplate.py \
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 1000 \
  --max_model_len 20000 \
  --max_new_tokens 5000 \
  --gpu_memory_utilization 0.95 \
  --out_dir ../data/no_phi/boilerplate_checks

echo 14 done

accelerate launch --num_processes 8 15_train_modernbert_trial_checker.py --categorical

echo 15 done

accelerate launch --num_processes 8 16_train_modernbert_boilerplate_checker.py

echo 16 done





