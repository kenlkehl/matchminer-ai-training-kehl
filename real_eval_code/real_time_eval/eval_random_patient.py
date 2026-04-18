#!/usr/bin/env python3
"""
Pick a random patient summary from the database, embed it, embed all trial
spaces, and print the top 10 most cosine-similar trial spaces.

Reads DB credentials from database_secrets.txt (auto-resolved relative to the
script location at ../../data/phi/database_secrets.txt).

Patient summaries come from the activate_info table (patient_summary column).
Trial spaces come from the trial_spaces table (this_cohort column).

Usage:
    python eval_random_patient.py /path/to/embedding_model

Examples:
    # Basic usage with the trialspace model
    python eval_random_patient.py ../../../models/trialspace

    # Specify GPU and max sequence length
    python eval_random_patient.py /path/to/model --gpu 1 --max-seq-length 2500

    # Use an alternate secrets file
    python eval_random_patient.py /path/to/model --secrets /alt/path/to/database_secrets.txt

    # Re-score top-10 results with a trial checker model
    python eval_random_patient.py /path/to/model --trial-checker /path/to/trial_checker_model
"""

import argparse
import configparser
import os
import re
from pathlib import Path

import numpy as np
import psycopg2
import torch
from sentence_transformers import SentenceTransformer

SECRETS_FILE = Path(__file__).resolve().parents[3] / "data" / "phi" / "database_secrets.txt"

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
    parser.add_argument("--secrets", type=str, default=str(SECRETS_FILE),
                        help="Path to database_secrets.txt")
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
    return parser.parse_args()


def get_db_connection(secrets_path: str):
    cfg = configparser.ConfigParser()
    cfg.read(secrets_path)
    def _strip(val: str) -> str:
        return val.strip('"')

    return psycopg2.connect(
        host=_strip(cfg["database"]["server"]),
        port=int(_strip(cfg["database"]["port"])),
        dbname=_strip(cfg["database"]["database"]),
        user=_strip(cfg["user"]["user"]),
        password=_strip(cfg["user"]["password"]),
    )


def run_trial_check(patient_summary, trial_texts, llm):
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
            conversation=messages, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)

    responses = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, top_k=1, max_tokens=5000, repetition_penalty=1.2),
    )

    score_pattern = re.compile(r"[Ff]inal\s+[Ss]core\s*:\s*(\d)")
    results = []
    for resp in responses:
        txt = resp.outputs[0].text
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
        results.append((txt, score))
    return results


def run_boilerplate_check(patient_boilerplate, trial_boilerplates, llm):
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
            conversation=messages, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)

    responses = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, top_k=1, max_tokens=7500, repetition_penalty=1.2),
    )

    results = []
    for resp in responses:
        txt = resp.outputs[0].text
        excluded = ("Yes!" in txt[-10:]) or ("YES!" in txt[-10:])
        results.append((txt, excluded))
    return results


def main():
    args = parse_args()

    # --- Database ----------------------------------------------------------
    conn = get_db_connection(args.secrets)
    cur = conn.cursor()

    # Fetch a random patient summary
    cur.execute(
        "SELECT id, mrn, patient_summary, patient_boilerplate FROM activate_info "
        "WHERE patient_summary IS NOT NULL "
        "ORDER BY random() LIMIT 1"
    )
    patient_id, mrn, patient_summary, patient_boilerplate = cur.fetchone()

    # Fetch all trial spaces
    cur.execute(
        "SELECT id, nct_id, this_cohort, boilerplate_text FROM trial_spaces "
        "WHERE this_cohort IS NOT NULL"
    )
    spaces_rows = cur.fetchall()
    cur.close()
    conn.close()

    # De-duplicate trial spaces: keep only the first row per (nct_id, this_cohort)
    seen = set()
    deduped_rows = []
    for r in spaces_rows:
        key = (r[1], r[2])  # (nct_id, this_cohort)
        if key not in seen:
            seen.add(key)
            deduped_rows.append(r)

    print(f"Fetched {len(spaces_rows)} trial spaces, "
          f"{len(deduped_rows)} unique after de-duplicating by (nct_id, this_cohort).")
    spaces_rows = deduped_rows

    space_ids = [r[0] for r in spaces_rows]
    nct_ids = [r[1] for r in spaces_rows]
    space_texts = [r[2] for r in spaces_rows]
    trial_boilerplates = [r[3] for r in spaces_rows]

    print(f"Loaded 1 patient (id={patient_id}, mrn={mrn}) and {len(space_texts)} trial spaces.\n")

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
        tc_llm_results = run_trial_check(patient_summary, top_trial_texts, llm)

        top_trial_boilerplates = [trial_boilerplates[idx] for idx in top_indices]
        has_boilerplate = (
            patient_boilerplate is not None
            and any(tb is not None for tb in top_trial_boilerplates)
        )
        if has_boilerplate:
            print("Running LLM boilerplate check on top 10 ...")
            bp_llm_results = run_boilerplate_check(
                patient_boilerplate or "",
                [tb or "" for tb in top_trial_boilerplates],
                llm,
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
                f"nct_id={nct_ids[idx]} | space_id={space_ids[idx]}")
        if tc_scores is not None:
            line += f" | tc_score={tc_scores[rank - 1]:.4f}"
        if tc_llm_results is not None:
            score = tc_llm_results[rank - 1][1]
            line += f" | eligibility={score}"
        if bp_llm_results is not None:
            excluded = bp_llm_results[rank - 1][1]
            line += f" | excluded={'Yes' if excluded else 'No'}"
        line += " ---"
        print(line)
        print(space_texts[idx])

        if tc_llm_results is not None:
            print(f"\n  >> TRIAL CHECK REASONING (score={tc_llm_results[rank - 1][1]}):")
            print(tc_llm_results[rank - 1][0])
        if bp_llm_results is not None:
            label = "EXCLUDED" if bp_llm_results[rank - 1][1] else "NOT EXCLUDED"
            print(f"\n  >> BOILERPLATE CHECK REASONING ({label}):")
            print(bp_llm_results[rank - 1][0])


if __name__ == "__main__":
    main()
