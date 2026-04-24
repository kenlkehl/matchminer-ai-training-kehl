#!/usr/bin/env python3
"""
Pick a random synthetic patient summary from a parquet file, embed it, embed
trial spaces from a CSV file, and print the top 10 most cosine-similar trial
spaces.

Patient summaries come from patient_summaries.parquet (pseudo_mrn,
patient_summary, patient_boilerplate_text columns).
Trial spaces come from sample_trial_space_lineitems.csv (nct_id, this_space,
trial_boilerplate_text columns).

Usage:
    python eval_random_patient.py /path/to/embedding_model

Examples:
    # Basic usage with the trialspace model
    python eval_random_patient.py ../../../models/trialspace

    # Specify GPU and max sequence length
    python eval_random_patient.py /path/to/model --gpu 1 --max-seq-length 2500

    # Re-score top-10 results with a trial checker model
    python eval_random_patient.py /path/to/model --trial-checker /path/to/trial_checker_model

    # with llm check
    python eval_random_patient.py ../../models/trialspace --trial-checker ../../models/trialchecker --llm-model \
        ../../models/onco_reasoning_lfm/checkpoint-39000 --llm-gpu-mem-util 0.80 --llm-max-model-len 10000
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parents[1] / ".." / "data" / "no_phi"
PATIENT_FILE = DATA_DIR / "patient_summaries.parquet"
TRIAL_SPACE_FILE = DATA_DIR / "sample_trial_space_lineitems.csv"

QUERY_PROMPT = (
    "Instruct: Given a cancer patient summary, retrieve clinical trial options "
    "that are reasonable for that patient; or, given a clinical trial option, "
    "retrieve cancer patients who are reasonable candidates for that trial."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Folder containing the saved embedding model")
    parser.add_argument("--max-seq-length", type=int, default=2500)
    parser.add_argument("--gpu", type=str, default="0", help="GPU id (default: 0)")
    parser.add_argument("--patient-file", type=str, default=str(PATIENT_FILE),
                        help="Path to patient_summaries.parquet")
    parser.add_argument("--trial-space-file", type=str, default=str(TRIAL_SPACE_FILE),
                        help="Path to sample_trial_space_lineitems.csv")
    parser.add_argument("--trial-checker", type=str, default=None,
                        help="Path to a trained trial checker model (ModernBERT). "
                             "If provided, re-scores top-10 trial spaces.")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Path or HF ID of an LLM for trial check and boilerplate "
                             "check inference on top-10 results (uses vLLM in-process).")
    parser.add_argument("--llm-download-dir", type=str,
                        default="/data1/ken/meta/2024/meta_ai",
                        help="Download directory for LLM weights")
    parser.add_argument("--llm-gpu-mem-util", type=float, default=0.90,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--llm-max-model-len", type=int, default=10000,
                        help="Maximum context length for the LLM")
    parser.add_argument("--llm-max-num-seqs", type=int, default=900,
                        help="vLLM max_num_seqs (concurrent request cap).")

    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from vllm_reasoning_utils import add_reasoning_cli_args
    add_reasoning_cli_args(parser)

    return parser.parse_args()


def run_trial_check(patient_summary, trial_texts, llm, reasoning_parser):
    """Run LLM-based eligibility check on each (patient, trial) pair.

    Returns list of (response_text, score) tuples where score is 0-5 or -1 on
    parse failure.
    """
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    prompts = []

    for trial_summary in trial_texts:
        messages = [
            {'role': 'system', 'content': "Reasoning: high"},
            {'role': 'user', 'content': (
                "You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment. "
                "Your job is to evaluate whether a given clinical trial is a reasonable consideration for a patient, "
                "given a clinical trial summary and a patient summary, and then score how targeted the trial is for "
                "this specific patient.\n\n"
                f"Here is a summary of the clinical trial:\n{trial_summary}\n"
                f"Here is a summary of the patient:\n{patient_summary}\n"
                "Base your judgment on whether the patient generally fits the age requirements if any, sex requirements if any, cancer type(s), cancer burden, prior treatment(s), "
                "and biomarker criteria specified for the trial.\n"
                "You do not have to determine if the patient is actually eligible; instead please just evaluate whether it is reasonable "
                "for the trial to be considered further by the patient's oncologist.\n"
                "Biomarker criteria have to be considered carefully. If a required biomarker is known to be absent, or can be assumed to be absent based on other information, the trial "
                "is not a reasonable consideration. For example, if a trial for lung cancer requires an EGFR mutation, documentation that there "
                "is no EGFR mutation indicates the trial is not a reasonable consideration. Similarly, documentation of a KRAS mutation in the "
                "patient indicates the trial is not a reasonable consideration, since, as you know, KRAS and EGFR driver mutations in lung cancer "
                "are mutually exclusive.\n"
                "Many trials describe required washout periods for prior treatments for eligibility. For example, the eligibility criteria might state "
                "that patients may not have received radiation or chemotherapy in the last 14 days or 30 days. It is CRITICAL that you IGNORE these "
                "eligibility criteria when considering prior treatment requirements. Assume that patients could wait for the washout period to enroll. "
                "Also CRITICAL: Ignore your knowledge of today's current date. Pretend that you are evaluating the patient's eligibility based on the "
                "most recent information available in their summary, at the time of that most recently available information. "
                "Do not provide ethical judgments or comment on resource constraints with respect whether the trial is a reasonable clinical "
                "consideration; just evaluate whether it is, given the available information.\n\n"
                "SCORING INSTRUCTIONS:\n"
                "After reasoning step by step, compute a score from 0 to 5 using the following rubric:\n\n"
                "Start with 0 points.\n"
                "1) REASONABLENESS (0 or 1 point): If the trial is at least a reasonable consideration for this patient "
                "(i.e., the patient does not clearly meet an exclusion criterion such as wrong cancer type, wrong age group, "
                "wrong sex, having an excluded biomarker, etc.), award 1 point. If the trial is NOT reasonable, the final score is 0 — "
                "skip the remaining categories.\n"
                "2) CANCER TYPE SPECIFICITY (+1 point): If the trial specifies the patient's cancer type (e.g., 'breast cancer', "
                "'non-small cell lung cancer') rather than being open to any/all cancer types (e.g., 'solid tumors', 'advanced cancers'), "
                "award +1 point.\n"
                "3) CANCER BURDEN/STAGE SPECIFICITY (+1 point): If the trial specifies a particular disease stage or burden "
                "(e.g., 'metastatic', 'locally advanced', 'stage III-IV') that matches the patient's disease status, award +1 point. "
                "If the trial has no stage/burden requirements or is open to any stage, do not award a point.\n"
                "4) PRIOR TREATMENT SPECIFICITY (+1 point): If the trial has specific prior treatment requirements "
                "(e.g., 'must have progressed on platinum-based chemotherapy', 'prior immunotherapy required') "
                "and the patient's treatment history matches those requirements, award +1 point. "
                "If the trial has no specific prior treatment requirements, do not award a point.\n"
                "5) BIOMARKER SPECIFICITY (+1 point): If the trial requires a specific biomarker (e.g., 'EGFR mutation', "
                "'PD-L1 ≥ 50%', 'HER2-positive') AND the patient is known to have that biomarker, award +1 point. "
                "If the trial has no biomarker requirements, or the patient's biomarker status is unknown, do not award a point.\n\n"
                "Your response MUST end with the following line and nothing else after it:\n"
                "Final score: X\n"
                "where X is the total score (an integer from 0 to 5)."
            )}
        ]
        prompt = tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )
        prompts.append(prompt)

    responses = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, top_k=1, max_tokens=5000, repetition_penalty=1.0, skip_special_tokens=False),
    )

    from vllm_reasoning_utils import parse_reasoning_output

    score_pattern = re.compile(r"[Ff]inal\s+[Ss]core\s*:\s*(\d)")
    results = []
    for resp in responses:
        reasoning, txt = parse_reasoning_output(resp.outputs[0].text, reasoning_parser, tokenizer)
        tail = txt[-60:].replace("*", "").replace("\u202f", " ")
        m = score_pattern.search(tail)
        if m:
            score = min(int(m.group(1)), 5)
        else:
            tail_upper = tail.upper()
            fallback_m = re.search(r"SCORE\s*[:\-=]\s*(\d)", tail_upper)
            if fallback_m:
                score = min(int(fallback_m.group(1)), 5)
            elif "NOT REASONABLE" in tail_upper or "NOT A REASONABLE" in tail_upper:
                score = 0
            else:
                score = -1
        results.append((reasoning, txt, score))
    return results


def run_boilerplate_check(patient_boilerplate, trial_boilerplates, llm, reasoning_parser):
    """Run LLM-based boilerplate exclusion check on each (patient, trial) pair.

    Returns list of (response_text, excluded) tuples where excluded is True/False.
    """
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    prompts = []

    for trial_boilerplate in trial_boilerplates:
        messages = [
            {'role': 'system', 'content': "Reasoning: high"},
            {'role': 'user', 'content': (
                "You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment.\n"
                "Your job is to evaluate whether a patient has any underlying medical conditions that would exclude him or her from a specific clinical trial.\n\n"
                f"Here is an extract of the patient's history:\n{patient_boilerplate}\n"
                f"Here are the exclusion criteria for the trial:\n{trial_boilerplate}\n"
                "Note that the extract was generated by prompting an LLM to determine whether the patient meets specific common exclusion criteria, "
                "such as uncontrolled brain metastases, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, "
                "liver dysfunction, and HIV or hepatitis infection, and to present evidence for whether the patient met the criterion.\n"
                "You should therefore not assume that mention of such condition means the patient has the condition; it may represent the LLM reasoning "
                "about whether the patient has the condition.\n"
                "Based on the extract, you should determine whether the patient clearly meets one of the exclusion criteria for this specific trial.\n"
                "Do not evaluate exclusion criteria other than those listed for this trial.\n"
                "Reason through one exclusion criterion at a time. Generate a numbered list of the criteria as you go. For each one, decide whether the patient clearly "
                "meets the exclusion criteron. If it is not completely clear that the patient meets the exclusion criterion, give the patient the benefit of the doubt, "
                "and err on the side of deciding the patient is not excluded. A description in the patient extract that a condition is mild, low-grade, or resolved is even "
                "more of a reason not to exclude the patient based on that condition.\n"
                'Once you have evaluated all exclusion criteria, answer the question "Is this patient clearly excluded from this trial?" with a one-word "Yes!" or "No!" answer, '
                "based on whether the patient clearly met any of the individual exclusion criteria. It is critical that your final word be either \"Yes!\" or \"No!\", verbatim, and case-sensitive.\n"
                "Make sure to include the exclamation point in your final one-word answer.\n"
                "No introductory text or concluding text after that final answer."
            )}
        ]
        prompt = tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )
        prompts.append(prompt)

    responses = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, top_k=1, max_tokens=7500, repetition_penalty=1.0, skip_special_tokens=False),
    )

    from vllm_reasoning_utils import parse_reasoning_output

    results = []
    for resp in responses:
        reasoning, txt = parse_reasoning_output(resp.outputs[0].text, reasoning_parser, tokenizer)
        excluded = ("Yes!" in txt[-10:]) or ("YES!" in txt[-10:])
        results.append((reasoning, txt, excluded))
    return results


def main():
    args = parse_args()

    # --- Load data from files -----------------------------------------------
    print(f"Loading patient summaries from {args.patient_file} ...")
    patients_df = pd.read_parquet(args.patient_file)
    patients_df = patients_df.dropna(subset=["patient_summary"])

    # Pick a random patient
    patient_row = patients_df.sample(n=1).iloc[0]
    pseudo_mrn = patient_row["pseudo_mrn"]
    patient_summary = patient_row["patient_summary"]
    patient_boilerplate = patient_row.get("patient_boilerplate_text")
    if pd.isna(patient_boilerplate):
        patient_boilerplate = None

    print(f"Loading trial spaces from {args.trial_space_file} ...")
    spaces_df = pd.read_csv(args.trial_space_file)
    spaces_df = spaces_df.dropna(subset=["this_space"])

    # De-duplicate trial spaces: keep only the first row per (nct_id, this_space)
    orig_len = len(spaces_df)
    spaces_df = spaces_df.drop_duplicates(subset=["nct_id", "this_space"], keep="first")
    print(f"Fetched {orig_len} trial spaces, "
          f"{len(spaces_df)} unique after de-duplicating by (nct_id, this_space).")

    space_indices = spaces_df["space_index"].tolist()
    nct_ids = spaces_df["nct_id"].tolist()
    space_texts = spaces_df["this_space"].tolist()
    trial_boilerplates = spaces_df["trial_boilerplate_text"].tolist()

    print(f"Loaded 1 patient (pseudo_mrn={pseudo_mrn}) and {len(space_texts)} trial spaces.\n")

    # --- Embedding model ---------------------------------------------------
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model from {args.model_dir} on {device} ...")
    model = SentenceTransformer(args.model_dir, trust_remote_code=True, device=device)
    model.max_seq_length = args.max_seq_length
    model.prompts["query"] = QUERY_PROMPT

    # --- Trial checker model (optional) ------------------------------------
    tc_model = None
    tc_tokenizer = None
    if args.trial_checker:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"Loading trial checker model from {args.trial_checker} on {device} ...")
        tc_tokenizer = AutoTokenizer.from_pretrained(args.trial_checker)
        tc_model = AutoModelForSequenceClassification.from_pretrained(args.trial_checker).to(device)
        tc_model.eval()

    # --- Encode ------------------------------------------------------------
    print("Encoding patient summary ...")
    with torch.no_grad():
        patient_emb = model.encode(
            [patient_summary],
            convert_to_tensor=True,
            normalize_embeddings=True,
            prompt="query",
        )

    print(f"Encoding {len(space_texts)} trial spaces ...")
    with torch.no_grad():
        space_embs = model.encode(
            space_texts,
            batch_size=12,
            convert_to_tensor=True,
            normalize_embeddings=True,
            prompt="query",
            show_progress_bar=True,
        )

    # --- Cosine similarity (embeddings are already L2-normalised) ----------
    similarities = (patient_emb @ space_embs.T).squeeze(0).cpu().numpy()
    top_indices = np.argsort(similarities)[::-1][:10]

    # --- Trial checker re-scoring (optional) ------------------------------
    tc_scores = None
    if tc_model is not None:
        print("Running trial checker on top 10 trial spaces ...")
        tc_texts = [
            space_texts[idx] + "\nNow here is the patient summary:" + patient_summary
            for idx in top_indices
        ]
        inputs = tc_tokenizer(
            tc_texts, truncation=True, padding=True,
            max_length=4096, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = tc_model(**inputs)
            logits = outputs.logits.squeeze(-1)
            tc_scores = torch.sigmoid(logits).cpu().numpy()

        # Re-rank top indices by trial checker score (highest first)
        rerank_order = np.argsort(tc_scores)[::-1]
        top_indices = top_indices[rerank_order]
        tc_scores = tc_scores[rerank_order]

    # --- LLM-based trial check & boilerplate check (optional) ---------------
    tc_llm_results = None
    bp_llm_results = None
    if args.llm_model:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        from vllm import LLM
        from vllm_reasoning_utils import resolve_parser_name
        reasoning_parser = resolve_parser_name(args.llm_model, args.reasoning_parser)

        print(f"\nInitializing LLM from {args.llm_model} ...")
        llm = LLM(
            model=args.llm_model,
            tensor_parallel_size=1,
            download_dir=args.llm_download_dir,
            gpu_memory_utilization=args.llm_gpu_mem_util,
            max_num_seqs=args.llm_max_num_seqs,
            max_model_len=args.llm_max_model_len,
        )

        top_trial_texts = [space_texts[idx] for idx in top_indices]

        print("Running LLM trial check on top 10 ...")
        tc_llm_results = run_trial_check(patient_summary, top_trial_texts, llm, reasoning_parser)

        top_trial_boilerplates = [trial_boilerplates[idx] for idx in top_indices]
        has_boilerplate = (
            patient_boilerplate is not None
            and any(
                tb is not None and not (isinstance(tb, float) and np.isnan(tb))
                for tb in top_trial_boilerplates
            )
        )
        if has_boilerplate:
            print("Running LLM boilerplate check on top 10 ...")
            bp_llm_results = run_boilerplate_check(
                patient_boilerplate,
                [tb if (isinstance(tb, str)) else "" for tb in top_trial_boilerplates],
                llm,
                reasoning_parser,
            )
        else:
            print("Skipping boilerplate check (no boilerplate data available).")

    # --- Print results -----------------------------------------------------
    print("\n" + "=" * 80)
    print("PATIENT SUMMARY")
    print("=" * 80)
    print(patient_summary)

    print("\n" + "=" * 80)
    if tc_scores is not None:
        print("TOP 10 TRIAL SPACES (re-ranked by trial checker)")
    else:
        print("TOP 10 MOST SIMILAR TRIAL SPACES")
    print("=" * 80)
    for rank, idx in enumerate(top_indices, 1):
        line = (f"\n--- Rank {rank} | similarity={similarities[idx]:.4f} | "
                f"nct_id={nct_ids[idx]} | space_index={space_indices[idx]}")
        if tc_scores is not None:
            line += f" | tc_score={tc_scores[rank - 1]:.4f}"
        if tc_llm_results is not None:
            score = tc_llm_results[rank - 1][2]
            line += f" | eligibility={score}"
        if bp_llm_results is not None:
            excluded = bp_llm_results[rank - 1][2]
            line += f" | excluded={'Yes' if excluded else 'No'}"
        line += " ---"
        print(line)
        print(space_texts[idx])

        if tc_llm_results is not None:
            tc_reasoning, tc_answer, tc_score = tc_llm_results[rank - 1]
            print(f"\n  >> TRIAL CHECK REASONING (score={tc_score}):")
            print(tc_reasoning)
            print(f"\n  >> TRIAL CHECK ANSWER:")
            print(tc_answer)
        if bp_llm_results is not None:
            bp_reasoning, bp_answer, bp_excluded = bp_llm_results[rank - 1]
            label = "EXCLUDED" if bp_excluded else "NOT EXCLUDED"
            print(f"\n  >> BOILERPLATE CHECK REASONING ({label}):")
            print(bp_reasoning)
            print(f"\n  >> BOILERPLATE CHECK ANSWER:")
            print(bp_answer)


if __name__ == "__main__":
    main()
