#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel trial-space extraction with vLLM.

Usage examples
--------------
# One instance per GPU (model fits on a single GPU):
python 0b_create_trial_spaces.py \
  --input ctgov_trials.csv \
  --start-year 2020 \
  --gpus 2,3 \
  --gpus-per-instance 1 \
  --download-dir ../models


Outputs
-------
- trials_with_spaces.csv
- trial_space_lineitems.csv
"""

from matchminer_ai.llm.reasoning import reasoning_system_prompt
from vllm_reasoning_utils import resolve_chat_template_kwargs

import os
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing as mp
from copy import copy
from ctgov_metadata import REGISTRY_METADATA_COLUMNS
from trial_space_format import has_required_space_fields
from trial_treatment_screening import (
    build_screening_messages, parse_screening_output, screening_record,
    check_screening_manifest, load_screening_results,
)

# vLLM imports happen inside worker after setting CUDA_VISIBLE_DEVICES


# Keep these snapshots identical to inference-kehl trial.user.primer.txt
# and trial.user.question.txt; tests/test_trial_prompt_contract.py checks
# the complete rendered messages against the sibling checkout.
PROMPT_HEADER = (
    'Your overall goal is to review a cancer clinical trial document and extract a list of structured clinical spaces that are eligible for that trial. This is a real use case; you are not being benchmarked.\n'
    "A clinical space is defined as a self-contained a target population for the trial. Components include that population's age range, sex, cancer primary site, histology, disease state, prior treatment requirements (both inclusion and exclusion criteria with respect to prior treatment), and tumor biomarker requirements (both inclusion and exclusion criteria with respect to biomarkers). \n"
    'Once you define the list of trial spaces, you will be asked to synthesize a list of "boilerplate" exclusion criteria that are mentioned by the trial and commonly exclude patients from trials in general rather than being relevant to the scientific target population(s) of the trial.\n'
    'CRITICAL: Concepts must be listed only in their appropriate fields and NEVER elsewhere. Field-specific guidance follows:\n'
    '-Re sex criteria: This should just be "male", "female", or "no restriction." For cancers originating in organs only present in one sex, you must assume the sex criteria even if not stated explicitly. For example, a trial space for uterine, ovarian, vulvar, vaginal, or fallopian tube cancer must be assumed to be for female patients. Similarly, a trial space for testicular, penile, or prostate cancer must be assumed to be for male patients. For all other cancer types (including breast cancer), you should assume the trial is open to both sexes unless the clinical trial document states otherwise.\n'
    'Examples of valid sex components: "male", "female", "no restriction".\n'
    'Examples of invalid sex components: "use of contraception", "child bearing age".\n'
    '-Re cancer primary site: This means cancer site of origin. It must not contain any other information. Think of the SEER primary site classification. It must not be a general multiple cancer primary site category. Spell out cancer site; do not abbreviate it. Do not output a space that contains multiple cancer types and/or histologies. Instead, generate separate spaces for each cancer type/histology combination. VERY IMPORTANT: This means that if a trial is for patients with any solid tumor, or any hematologic malignancy, etc, you must construct an exhaustive list of spaces for any solid tumor cancer type, AND any heme malignancy type, etc, that could be reasonably consistent with the eligibility criteria. This is required normalization, not invention of eligibility criteria. Use your oncology knowledge to enumerate those diagnoses even when the protocol does not name them individually, while respecting all explicit exclusions.\n'
    'Examples of valid cancer primary site components: "lung", "breast", "bone marrow", "lymphatic"\n'
    'Examples of invalid cancer primary site components: "metastatic lung cancer", "triple negative breast cancer", "melanoma", "lymphoma", "solid tumor", "hematologic malignancy", "NSCLC" (this is histology not primary site, and it is abbreviated)\n'
    '-Re histology: This refers to adenocarcinoma, squamous cell carcinoma, and so on. It does not encompass the organ of origin, which goes in a separate cancer primary site field. Biomarkers such as protein expression, receptor status, etc do NOT belong in histology; they belong in biomarkers. Think of the ICD-O-3 morphology classification. Spell out histology; do not abbreviate it. If a trial is open to any histology for a given primary site, construct separate spaces for each reasonably common histology for that cancer type.\n'
    'Examples of valid histology components: "adenocarcinoma", "squamous cell carcinoma", "invasive ductal carcinoma", "acute myeloid leukemia", "non-small cell carcinoma", "diffuse large B cell lymphoma", "melanoma"\n'
    'Examples of invalid histology components: "triple negative breast cancer", "NSCLC", "any", "pancreatic cancer", "adenocarcinoma or squamous cell carcinoma".\n'
    '-Re disease state: The intent here is to capture the time point along the disease\'s natural history that the trial space is targeting. For many solid tumors, this may include a requirement for early stage or for advanced/metastatic disease; for heme malignancies, this may include the presence of active disease (relapse or remission status.) Disease state also encompasses cancer type-specific clinical prognostic scores or risk indices; it does NOT include ECOG performance status, or concepts like "life expectancy at least 6 months.") The only molecular test results that should go in disease state are serum tumor markers used to monitor quantity of cancer at a given time (CEA, etc). Details about measurable disease, target lesions, biopsy targets, etc are NOT part of disease state; those details DO NOT BELONG IN YOUR TRIAL SPACE OUTPUT TEXT AT ALL. If a trial specifies stage IV disease but is actually seeking patients in any palliative-intent treatment (or advanced/metastatic) setting, including recurrent disease, use a description that would encompass both de novo and recurrent advanced disease. Treatment history requirements are NOT part of the disease state section; treatment history requirements go in the treatment history requirements section. Concepts that are NOT PART OF DISEASE STATE include other past medical history, diagnostic workup, social history, or family history. Do not copy lists of specific eligibility criteria here; disease state should be only a concise description of the intended target population\'s position along the disease trajectory. If a trial specifically REQUIRES disease in a given organ (eg brain, liver), include that in the disease state. However, when trials EXCLUDE disease in a given organ, that information is NOT part of disease state.\n'
    'Examples of valid disease state components: "advanced/metastatic", "early stage", "stage II", "localized", "IPSS-R 3", "blast crisis", "high volume", "transformed", "PSA > 10"\n'
    'Examples of invalid disease state components: "previously untreated", "metastatic with measurable disease", "ECOG 0", "refractory to standard treatment", "PET-CT performed", "family history of cancer", "never smoker", "tumor amenable to biopsy", "EGFR mutant" (EGFR mutation is a biomarker, not a disease state component).\n'
    '-Re treatment history: This includes any requirements that patients have received, or not received, a particular treatment or treatment modality. Hence, this comprises both inclusion and exclusion criteria about prior treatments. These requirements may be as broad as surgery, chemotherapy, radiation, etc; or as specific as a certain drug class or specific drug. Critically, trials often specify that a particular treatment is excluded only if it was given within a short period of time, for example 14 days, one month, etc, prior to trial start. This is called a washout period. Do NOT include this type of time-specific treatment washout eligibility criteria in your output at all. Similarly, trials may require recovery from prior treatment toxicities; do NOT include these either, and do NOT include requirements about whether prior toxicities occurred, since these go in the boilerplate exclusions field. Spell out treatment history components and drug names; do not abbreviate them. Trials often call for the absence of standard treatment alternatives; this is vague and poorly defined, so omit these qualifications.\n'
    'Examples of valid treatment history components: "at least two lines of therapy in the metastatic setting", "prior cisplatin", "no prior cisplatin", "prior neoadjuvant chemoradiation", "lobectomy", "allogeneic stem cell transplanation", "platinum-resistant", "prior pembrolizumab and no prior cisplatin".\n'
    'Examples of invalid treatment history components: "at least 14 days since prior checkpoint inhibitor", "SCT", "no standard therapy available," "refractory to standard therapy", "recovered to at least grade 1", "no prior pneumonitis".\n'
    '-Re biomarkers: This includes germline or somatic gene mutations or alterations, protein expression on tumor, or other molecular test results used to predict benefit of treatment (ie, predictive biomarkers). This includes both inclusion criteria about biomarkers (eg requirements that they be present/positive) and exclusion criteria about biomarkers (eg requirements that they be absent). A requirement for biomarker status to be known, tested, assessed, or available is not part of a trial space. Include only requirements for positive or negative biomarker results, including specific alterations, expression levels, or thresholds. If any result is allowed and only knowledge of the status is required, omit that testing/status requirement and output NA when no other biomarker result restriction applies. Biomarkers do not include tumor markers, such as PSA, CA-19-9, etc, used to quantify burden of cancer--these belong in disease state. If the trial requires a particular biomarker result and specifies whether it will be assessed during screening, note that.\n'
    'Examples of valid biomarker components: "classical EGFR mutant", "PD-L1 TPS > 50%", "ER+ or PR+", "KRAS G12D mutant", "HER2 positive", "no MTAP deletion", "no KRAS mutation".\n'
    'Examples of invalid biomarker components: "AFP > 5", "ECOG > 2"\n'
    'Some trials have only one space, while others have many. \n'
    'CRITICAL: Each trial space must contain all information necessary to define that space on its own. It MUST be self-contained and understandable to a downstream agent that does NOT have access to the original trial text or the full list of spaces for the trial. Therefore, trial space text must NOT refer to the trial protocol\'s cohorts or arms, which the user of the trial space text will not have access to. YOU MAY NOT include text describing a given space that refers to a previous space; eg, "Same as above"-style output is not allowed!!\n'
    'Structure your output like this, as a list of spaces, with spaces separated by newlines, as below. STRICTLY adhere to the formatting, including the serial numbering, with the space number at the beginning of each new space.\n'
    '1. Age range allowed: <age_range_allowed>. Sex allowed: <sex_allowed>. Cancer primary site allowed: <cancer_primary_site_allowed>. Histology allowed: <histology_allowed>. Disease state allowed: <cancer_burden_allowed>. Prior treatment requirements: <prior_treatment_requirements>. Biomarker requirements: <biomarker_requirements>.\n'
    '2. Age range allowed: <age_range_allowed>. Sex allowed: <sex_allowed>. ...\n'
    'If a concept is not relevant, such as if there are no prior treatments required, simply output NA for that concept.\n'
    'CRITICAL: Anytime you provide a list for a particular concept, you must be completely clear on whether "or" versus "and" logic applies to the list. For example, do not output "EGFR L858R mutant, TP53 mutant"; if both are required, output "EGFR L858R mutant and TP53 mutant". Similarly, do not output "ER+, PR+"; if the patient can have either an ER or a PR positive tumor or one that is positive for both ER and PR, output "ER+ or PR+".\n'
    'Create separate spaces when they define meaningfully different populations for treatment decision-making, and retain all applicable shared criteria in every space. For example, if a trial allows patients with either (1) EGFR-mutant non-small cell lung cancer or (2) ALK-rearranged non-small cell lung cancer, that should be output as two separate spaces, one for the EGFR-mutant non-small cell lung cancer and one for the ALK-rearranged non-small cell lung cancer, even if all other criteria are the same for both spaces.\n'
    'NEVER put a newline within a single trial space.\n'
    'After you output the trial spaces, output a newline, then the text "Boilerplate exclusions:" VERBATIM, then another newline.\n'
    'Then, list exclusion criteria described in the trial text that are unrelated to the trial space definitions. Such exclusions tend to be common to clinical trials in general. Common such "boilerplate" exclusion criteria include pregnancy or contraception requirements (unless pregnancy is a core inclusion criterion for the trial), vascular access issues, a history of pneumonitis, heart failure, renal dysfunction, liver dysfunction, uncontrolled brain metastases, HIV or hepatitis, prior unrelated cancer diagnoses, and poor performance status. Do not duplicate criteria that are part of the core trial space definition in the boilerplate section. Make sure your boilerplate exclusions are clearly phrased as exclusion criteria, not inclusion criteria. For example, if a trial requires ECOG 0 or 1 for inclusion, do NOT write "ECOG 0 or 1" in the boilerplate exclusions; instead, write "ECOG > 1" or similar language that clearly indicates this is an exclusion criterion. If a trial specifies INCLUSION criteria in a boilerplate category using an \'and\' conjunction that patients must meet to enroll on the trial (eg, "ECOG 0 and Hct > 30"), then logically, you should phrase your boilerplate EXCLUSION line using \'or\' conjunctions (eg, "ECOG > 0 or Hct < 30" would exclude the patient). Similarly, if an INCLUSION criterion in a boilerplate category is specified with an \'or\' conjunction (eg, Hct > 30 or Hgb > 10), then the EXCLUSION criterion you list would be "Hct <= 30 and Hgb <= 10".\n'
    'ALWAYS output plain text only. NEVER output unicode, Markdown, or tables.\n'
)

PROMPT_SUFFIX = (
    'Now, generate your list of the trial space(s), followed by any boilerplate exclusions, formatted as above.\n'
    'Do not provide any introductory, explanatory, concluding, or disclaimer text.\n'
    'Reminder: Treatment history is an important component of trial space definitions, but treatment history "washout" requirements that are described as applying only in a given period of time prior to trial treatment MUST BE IGNORED.\n'
    'CRITICAL: A given trial space MUST NEVER refer to another previously defined space. You must NEVER output text like "same as #1" or "same criteria as above." Instead, you MUST REPEAT all relevant criteria for each new space SO THAT IT STANDS ON ITS OWN. A user who later looks at the text for one space will not have access to text for other spaces, and so output like "Same criteria as #1..." renders a space useless!\n'
)

BOILERPLATE_MARKER = "Boilerplate"


def build_messages(trial_text: str, model_name: str = ""):
    """Chat messages for a single trial."""
    return [
        {"role": "system", "content": reasoning_system_prompt(model_name)},
        {
            "role": "user",
            "content": (
                PROMPT_HEADER
                + "Here is a clinical trial document: \n"
                + str(trial_text)
                + "\n"
                + PROMPT_SUFFIX
            ),
        },
    ]


def postprocess_outputs(raw_texts, reasoning_parser: str, tokenizer, *, already_parsed=False):
    """
    Split out (1) full raw text (reasoning+final), (2) final only,
    (3) space_text (before boilerplate line), and (4) boilerplate text.

    The boilerplate split is performed on the ENTIRE line containing
    the marker, removing that line from both resultant parts.
    """
    from vllm_reasoning_utils import parse_reasoning_output

    no_reasoning = []
    space_only = []
    boiler_only = []

    for t in raw_texts:
        # 1. Use vLLM's reasoning parser to strip reasoning + trailing sentinels
        if already_parsed:
            final_t = t.strip()
        else:
            _, final_t = parse_reasoning_output(t, reasoning_parser, tokenizer)

        # 2. Handle Boilerplate Split (Line-based)
        if BOILERPLATE_MARKER in final_t:
            # Split into lines, keeping newlines to preserve formatting
            lines = final_t.splitlines(keepends=True)
            
            # Find the index of the line containing the marker
            split_idx = -1
            for i, line in enumerate(lines):
                if BOILERPLATE_MARKER in line:
                    split_idx = i
                    break
            
            if split_idx != -1:
                # Join lines before the marker line
                space_part = "".join(lines[:split_idx])
                # Join lines after the marker line
                boiler_part = "".join(lines[split_idx+1:])
                
                space_only.append(space_part)
                boiler_only.append(boiler_part)
            else:
                # Fallback if marker was in string but not found in line iteration (unlikely edge case)
                space_only.append(final_t)
                boiler_only.append(final_t)
        else:
            space_only.append(final_t)
            boiler_only.append(final_t)

        no_reasoning.append(final_t)

    return no_reasoning, space_only, boiler_only


def worker_process(
    shard_in_path: str,
    shard_out_path: str,
    gpu_group: list,
    model: str,
    download_dir: str,
    max_model_len: int,
    max_num_seqs: int,
    gpu_mem_util: float,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    max_tokens: int,
    batch_size: int,
    reasoning_parser: str,
    screening: bool = False,
    extraction_method: str = "hierarchical",
    hierarchy_max_nodes: int = 4096,
    hierarchy_max_calls: int = 20000,
    max_attempts: int = 4,
    hierarchy_always_think: bool = False,
):
    """
    Worker: loads its own vLLM on the specified GPU group (via CUDA_VISIBLE_DEVICES),
    processes its shard, and writes the shard output to `shard_out_path` (parquet).
    """
    # Bind this process to its GPU set
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)

    # Import vLLM after the environment is set
    from vllm import LLM, SamplingParams

    tp_size = len(gpu_group)

    # Load shard
    shard_df = pd.read_parquet(shard_in_path)

    # Start model
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        download_dir=download_dir,
        gpu_memory_utilization=gpu_mem_util,
        #max_num_seqs=max_num_seqs, # needed for qwen 3.5 style models
        max_model_len=max_model_len,
        language_model_only=True,
    )
    tokenizer = llm.get_tokenizer()

    if not screening and extraction_method == "hierarchical":
        from types import SimpleNamespace
        from trial_hierarchy_runner import local_extract

        sampling = SamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            presence_penalty=presence_penalty, repetition_penalty=repetition_penalty,
            max_tokens=max_tokens, skip_special_tokens=False,
        )
        local_extract(llm, tokenizer, sampling, shard_df, SimpleNamespace(
            hierarchy_max_nodes=hierarchy_max_nodes, hierarchy_max_calls=hierarchy_max_calls,
            max_attempts=max_attempts, batch_size=batch_size, max_tokens=max_tokens,
            max_model_len=max_model_len, hierarchy_always_think=hierarchy_always_think,
        ), reasoning_parser, shard_out_path)
        return

    # Build prompts
    trial_texts = shard_df["trial_text"].astype(str).tolist()
    prompts = [
        tokenizer.apply_chat_template(
            conversation=(build_screening_messages(t) if screening else build_messages(t, model)),
            add_generation_prompt=True,
            tokenize=False,
            **resolve_chat_template_kwargs(model)
        )
        for t in trial_texts
    ]

    # Generate in batches
    raw_texts = []
    screen_records = []
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        skip_special_tokens=False,
    )

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        results = llm.generate(batch, sampling)
        for r in results:
            output = r.outputs[0]
            raw_texts.append(output.text)
            if screening:
                from vllm_reasoning_utils import parse_reasoning_output
                reasoning, final = parse_reasoning_output(output.text, reasoning_parser, tokenizer)
                screen_records.append(screening_record(reasoning, final, output.finish_reason))

    if screening:
        out_df = pd.concat([shard_df.reset_index(drop=True), pd.DataFrame(screen_records)], axis=1)
        out_df["treatment_screen_model"] = model
        temporary = Path(shard_out_path).with_suffix(".tmp")
        out_df.to_parquet(temporary, index=False)
        temporary.replace(shard_out_path)
        return

    # Post-process
    no_reasoning, space_only, boiler_only = postprocess_outputs(raw_texts, reasoning_parser, tokenizer)

    # Attach outputs
    out_df = shard_df.copy()
    out_df["space_reasoning_and_output"] = raw_texts
    out_df["space_output_no_reasoning"] = no_reasoning
    out_df["space_text"] = space_only
    out_df["trial_boilerplate_text"] = boiler_only

    # Persist this shard
    out_df.to_parquet(shard_out_path, index=False)


def make_trial_completion_work_fn(completion_fn, tokenizer, work_dir, *, screening=False,
                                  max_model_len=None, always_reasons=False, model_name=""):
    """Give remote retries validation feedback, with per-item thinking fallback.

    The pool owns the attempt budget and requeues the same payload. Retain only
    the latest rejected answer in that payload; write full failures separately.
    Transport failures leave this state alone and retain normal pool backoff.
    """
    from remote_vllm_pool import OutputValidationError

    audit_path = Path(work_dir) / "validation_failures.jsonl"

    def audit(payload, **record):
        with audit_path.open("a") as stream:
            stream.write(json.dumps({"time": time.time(), "nct_id": payload["nct_id"],
                                     **record}) + "\n")

    async def work_fn(client, payload):
        failures = payload.get("validation_failures", 0)
        thinking = payload.get("enable_thinking", True)
        request = dict(payload)
        if max_model_len is not None:
            # Recount every attempt because validation feedback changes the prompt.
            # Reserve a small margin for server/template differences; never truncate
            # the source trial to make a request fit.
            prompt_tokens = len(tokenizer.encode(payload["prompt"], add_special_tokens=False))
            available = max_model_len - prompt_tokens - 256
            if available <= 0:
                raise OutputValidationError(
                    f"Trial {payload['nct_id']} needs {prompt_tokens} prompt tokens; "
                    f"context limit {max_model_len} leaves no output budget"
                )
            request["max_tokens"] = min(int(payload["max_tokens"]), available)
        reasoning, final, metadata = await completion_fn(
            client, {**request, "enable_thinking": thinking,
                     "include_completion_metadata": True, "include_raw_text": True}
        )
        try:
            if screening:
                result = screening_record(reasoning, final, metadata["finish_reason"])
            else:
                if metadata["finish_reason"] in {"length", "abort", "error"}:
                    raise ValueError(f"Incomplete trial-space response: {metadata['finish_reason']}")
                if not final.strip() or "Boilerplate exclusions:" not in final:
                    raise OutputValidationError(
                        "Trial-space completion is empty or missing the literal 'Boilerplate exclusions:' delimiter"
                    )
                result = reasoning, final
        except ValueError as exc:
            failures += 1
            next_thinking = not thinking if failures % 2 == 0 else thinking
            audit(payload, event="validation_failure", failure_number=failures,
                  enable_thinking=thinking, next_enable_thinking=next_thinking,
                  **({**resolve_chat_template_kwargs("beam", enable_thinking=thinking),
                      "next_reasoning_effort": resolve_chat_template_kwargs("beam", enable_thinking=next_thinking)["reasoning_effort"]}
                     if always_reasons else {}),
                  error=str(exc), finish_reason=metadata["finish_reason"],
                  reasoning=reasoning, final=final, raw_text=metadata.get("raw_text"))
            messages = (build_screening_messages(payload["trial_text"]) if screening
                        else build_messages(payload["trial_text"], model_name or getattr(tokenizer, "name_or_path", "")))
            if not screening and not always_reasons and not next_thinking:
                messages[0] = {"role": "system", "content": "Return the requested final answer directly."}
            # Bounded feedback prevents large extraction responses from making
            # successive retries exceed the original context budget.
            rejected = final[:12000] or "[Empty final answer]"
            feedback = (
                f"Your previous answer failed validation: {exc}\n"
                "Correct this error and return a complete replacement answer using the original trial document "
                "and instructions. The rejected answer is not authoritative evidence. "
                "Do not explain the correction or include Markdown fences. "
            )
            if screening:
                feedback += ('Return exactly one JSON object with a boolean is_cancer_treatment_trial '
                             'and a nonempty double-quoted rationale string; escape quotes inside the string.')
            else:
                feedback += "Return the complete numbered trial spaces followed by 'Boilerplate exclusions:' and the exclusions."
            messages += [{"role": "assistant", "content": rejected},
                         {"role": "user", "content": feedback}]
            template_kwargs = resolve_chat_template_kwargs(
                "beam" if always_reasons else "", enable_thinking=next_thinking)
            if "messages" in payload:
                payload.update(messages=messages, chat_template_kwargs=template_kwargs)
            payload.update(
                validation_failures=failures, enable_thinking=next_thinking,
                prompt=tokenizer.apply_chat_template(
                    conversation=messages, add_generation_prompt=True,
                    tokenize=False, **template_kwargs,
                ),
            )
            raise
        if failures:
            audit(payload, event="recovered", failure_number=failures,
                  **(resolve_chat_template_kwargs("beam", enable_thinking=thinking)
                     if always_reasons else {}),
                  enable_thinking=thinking, finish_reason=metadata["finish_reason"])
        return result

    return work_fn


def remote_main(args, trials: pd.DataFrame, base_cols, reasoning_parser: str, *, screening=False):
    """Run trial-space extraction against a remote vLLM pool. Writes one
    or more parquet shards into args.work_dir and returns the list of
    paths for the caller's existing merge code."""
    if not screening and getattr(args, "extraction_method", "hierarchical") == "hierarchical":
        from trial_hierarchy_runner import remote_main as run_hierarchy
        return run_hierarchy(args, trials, reasoning_parser)
    import asyncio
    import glob
    import re
    from vllm_reasoning_utils import load_tokenizer
    from remote_vllm_pool import (
        CompletionSampling,
        make_completion_work_fn,
        run_pool,
        build_registry_from_args,
    )

    print(f"[remote] loading tokenizer...")
    tokenizer = load_tokenizer(
        getattr(args, "tokenizer_model", None) or args.model,
        cache_dir=args.download_dir or None,
        trust_remote_code=True,
    )

    # Build per-trial prompts upfront
    trials = trials.copy().reset_index(drop=True)
    trials["_work_id"] = np.arange(len(trials), dtype=np.int64)
    trial_texts = trials["trial_text"].astype(str).tolist()
    print(f"[remote] building {len(trial_texts)} prompts...")
    # The shared native Beam parser handles raw completions locally, matching
    # event/note generation and retaining the complete raw response.
    beam = reasoning_parser == "beam"
    template_kwargs = resolve_chat_template_kwargs("beam" if beam else "")
    prompts = [
        tokenizer.apply_chat_template(
            conversation=(build_screening_messages(t) if screening else build_messages(t, getattr(args, "tokenizer_model", None) or args.model)),
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        for t in trial_texts
    ]

    # Resume: skip _work_ids already covered by prior remote shards
    remote_pat = re.compile(r"remote_output_shard_(\d+)\.parquet$")
    done_ids = set()
    next_idx = 0
    for f in sorted(os.listdir(args.work_dir)):
        m = remote_pat.match(f)
        if not m:
            continue
        next_idx = max(next_idx, int(m.group(1)) + 1)
        try:
            df_prev = pd.read_parquet(os.path.join(args.work_dir, f))
            if screening:
                df_prev = df_prev.loc[df_prev.treatment_screen_status.eq("ok")]
                valid_ids = []
                for row in df_prev.itertuples():
                    try:
                        parse_screening_output(row.treatment_screen_output, row.treatment_screen_finish_reason)
                    except ValueError:
                        continue
                    valid_ids.append(row.nct_id)
                df_prev = df_prev.loc[df_prev.nct_id.isin(valid_ids)]
            done_ids.update(int(x) for x in df_prev["_work_id"].tolist())
        except Exception:
            pass
    if done_ids:
        print(f"[remote] skipping {len(done_ids)} trials already in remote shards")

    work_items = [
        (int(trials.iloc[i]["_work_id"]),
         {"prompt": prompts[i], "max_tokens": int(args.max_tokens),
          "trial_text": trial_texts[i], "nct_id": str(trials.iloc[i]["nct_id"]),
          **({"messages": (build_screening_messages(trial_texts[i]) if screening
                           else build_messages(trial_texts[i], getattr(args, "tokenizer_model", None) or args.model)),
              "chat_template_kwargs": template_kwargs} if beam else {})})
        for i in range(len(trials))
        if int(trials.iloc[i]["_work_id"]) not in done_ids
    ]
    if not work_items:
        print("[remote] all trials already complete; just merging existing shards.")
        return sorted(glob.glob(os.path.join(args.work_dir, "remote_output_shard_*.parquet")))

    trials_indexed = trials.set_index("_work_id")

    sampling = CompletionSampling(
        model=args.model,
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        min_p=float(args.min_p),
        presence_penalty=float(args.presence_penalty),
        repetition_penalty=float(args.repetition_penalty),
        request_timeout=float(getattr(args, "request_timeout", 600.0)),
    )
    completion_fn = make_completion_work_fn(sampling, reasoning_parser, tokenizer)

    work_fn = make_trial_completion_work_fn(
        completion_fn, tokenizer, args.work_dir, screening=screening,
        max_model_len=args.max_model_len, always_reasons=beam,
        model_name=getattr(args, "tokenizer_model", None) or args.model,
    )
    registry = build_registry_from_args(args)

    def shard_writer(payload, shard_idx):
        if screening:
            rows = []
            for work_id, result in payload:
                orig = trials_indexed.loc[work_id].to_dict()
                if isinstance(result, str) and result.startswith("ERROR:"):
                    result = {"treatment_screen_status": "error", "treatment_screen_output": result}
                rows.append({**orig, "_work_id": work_id, **result,
                             "treatment_screen_model": args.model})
            path = Path(args.work_dir) / f"remote_output_shard_{shard_idx:06d}.parquet"
            temporary = path.with_suffix(".tmp")
            pd.DataFrame(rows).to_parquet(temporary, index=False)
            temporary.replace(path)
            print(f"[screen] wrote {path} ({len(rows)} rows)")
            return
        rows = []
        raw_texts = []
        # We need post-processing in the same form as the local worker.
        for work_id, result in payload:
            if isinstance(result, str) and result.startswith("ERROR:"):
                reasoning, final = "", result
            else:
                reasoning, final = result
            try:
                orig = trials_indexed.loc[work_id].to_dict()
            except KeyError:
                orig = {}
            # raw_text reconstruction: combine reasoning + final so downstream
            # consumers of space_reasoning_and_output keep working.
            raw_text = reasoning + "\n" + final if reasoning else final
            row_payload = {**{c: orig.get(c) for c in base_cols}}
            row_payload["_work_id"] = work_id
            row_payload["_raw_text"] = raw_text
            row_payload["_final_text"] = final
            rows.append(row_payload)
            raw_texts.append(raw_text)

        # Use the existing post-processor for parity with local mode.
        no_reasoning, space_only, boiler_only = postprocess_outputs(
            [row["_final_text"] for row in rows], reasoning_parser, tokenizer,
            already_parsed=True,
        )
        for i, r in enumerate(rows):
            r["space_reasoning_and_output"] = raw_texts[i]
            r["space_output_no_reasoning"] = no_reasoning[i]
            r["space_text"] = space_only[i]
            r["trial_boilerplate_text"] = boiler_only[i]
            r.pop("_raw_text", None)
            r.pop("_final_text", None)

        out_path = os.path.join(args.work_dir, f"remote_output_shard_{shard_idx:06d}.parquet")
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"[remote] wrote {out_path} ({len(rows)} rows)")

    async def run_stage():
        try:
            await run_pool(
                work_items=work_items,
                work_fn=work_fn,
                registry=registry,
                shard_writer=shard_writer,
                results_per_shard=int(getattr(args, "results_per_shard", 200)),
                starting_shard_idx=next_idx,
                max_attempts=int(getattr(args, "max_attempts", 200)),
            )
        finally:
            # Each stage owns its registry and event loop; close HTTP clients
            # and the dynamic endpoint watcher before starting the next stage.
            await registry.stop()

    asyncio.run(run_stage())

    return sorted(glob.glob(os.path.join(args.work_dir, "remote_output_shard_*.parquet")))


def split_into_groups(items, group_size):
    """Split a list into fixed-size groups; drop any remainder smaller than group_size."""
    groups = []
    for i in range(0, len(items), group_size):
        g = items[i : i + group_size]
        if len(g) == group_size:
            groups.append(g)
    return groups


def select_trials_since_year(trials: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Keep trials first posted in or after start_year, using registry dates only."""
    if not 1 <= start_year <= 9999:
        raise ValueError("--start-year must be a calendar year between 1 and 9999")
    if "study_first_post_date" not in trials.columns:
        raise ValueError(
            "Missing study_first_post_date. Rerun 0a_parse_ctgov_json.py on the original export."
        )
    dates = pd.to_datetime(
        trials["study_first_post_date"].astype("string").str.strip(),
        format="ISO8601", errors="coerce",
    )
    return trials.loc[dates.dt.year.ge(start_year)].copy()


def select_trials(trials: pd.DataFrame, trial_filter: str) -> pd.DataFrame:
    """Filter on registry metadata, never phase words in eligibility text."""
    if trial_filter == "all":
        return trials.copy()
    if trial_filter != "interventional-phase1-3":
        raise ValueError(f"Unknown trial filter: {trial_filter}")
    missing = {"study_type", "phases"} - set(trials.columns)
    if missing:
        raise ValueError(
            f"Missing structured trial metadata: {sorted(missing)}. "
            "Rerun 0a_parse_ctgov_json.py on the original export. "
            "Use --trial-filter all only to bypass study-type/phase filtering."
        )

    def eligible_phase(value):
        if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            return False
        if isinstance(value, str):
            if not value.strip():
                return False
            try:
                value = json.loads(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("phases must be a JSON array from 0a") from exc
        if not isinstance(value, list) or any(not isinstance(p, str) for p in value):
            raise ValueError("phases must be a JSON array of strings from 0a")
        phases = set(value)
        return bool(phases) and phases <= {"EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3"}

    mask = trials.study_type.eq("INTERVENTIONAL") & trials.phases.map(eligible_phase).astype(bool)
    return trials.loc[mask].copy()


def check_extraction_manifest(args, reasoning_parser):
    """Keep extraction prompt/configuration changes out of existing shards."""
    path = Path(args.work_dir) / "extraction_manifest.json"
    messages = json.dumps(build_messages(""), ensure_ascii=False)
    identity = {
        "messages_sha256": hashlib.sha256(messages.encode()).hexdigest(),
        "model": args.model, "reasoning_parser": reasoning_parser,
        "enable_thinking": True,
        "max_model_len": args.max_model_len,
        "transport": "completions",
        "sampling": {key: getattr(args, key) for key in (
            "temperature", "top_p", "top_k", "min_p", "presence_penalty",
            "repetition_penalty", "max_tokens")},
    }
    if getattr(args, "extraction_method", "hierarchical") == "hierarchical":
        from trial_hierarchy_runner import prompt_bundle
        from trial_hierarchy import VERSION
        identity.update(
            extraction_method=VERSION,
            messages_sha256=hashlib.sha256(json.dumps(prompt_bundle(), sort_keys=True).encode()).hexdigest(),
            hierarchy_max_nodes=getattr(args, "hierarchy_max_nodes", 4096),
            hierarchy_max_calls=getattr(args, "hierarchy_max_calls", 20000),
            hierarchy_trial_batch_size=getattr(args, "hierarchy_trial_batch_size", 32),
        )
        if getattr(args, "hierarchy_always_think", False):
            identity["hierarchy_always_think"] = True
    if getattr(args, "tokenizer_model", None):
        identity["tokenizer_model"] = args.tokenizer_model
    if reasoning_parser == "beam":
        identity["chat_template_kwargs"] = resolve_chat_template_kwargs("beam")
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(
                "Trial-space extraction prompt/configuration changed; use fresh extraction checkpoints. "
                "Compatible treatment-screening checkpoints may be retained."
            )
    elif (any(Path(args.work_dir).glob("*output_shard_*.parquet"))
          or any((Path(args.work_dir) / "hierarchy_nodes").rglob("*.json"))):
        raise ValueError("Extraction shards lack a prompt manifest; use fresh extraction checkpoints.")
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity, indent=2))
        temporary.replace(path)


def check_trial_selection_manifest(trials, work_dir, trial_filter, seed, start_year=2020):
    """Prevent row-position checkpoints being reused after selection changes."""
    path = Path(work_dir) / "trial_selection.json"
    # Fingerprint selected content as well as the policy: _work_id is positional.
    digest = hashlib.sha256()
    for offset in range(0, len(trials), 1000):
        content = trials.iloc[offset:offset + 1000].to_json(orient="records", force_ascii=False)
        digest.update(content.encode())
    identity = {"trial_filter": trial_filter, "start_year": start_year,
                "seed": seed, "rows": len(trials),
                "selected_sha256": digest.hexdigest()}
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Trial selection/input changed; use a new --work-dir.")
    elif any(Path(work_dir).glob("*output_shard_*.parquet")):
        raise ValueError("Existing shards have no trial-selection manifest; use a new --work-dir.")
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity, indent=2))
        temporary.replace(path)


def local_main(args, trials, base_cols, reasoning_parser, *, screening=False):
    # ---- Local mode ----
    if not args.gpus:
        raise SystemExit("--gpus is required for local mode (or use --server_urls / --server_urls_file).")

    # Determine GPU groups / instances
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if len(gpu_list) == 0:
        raise ValueError("No GPUs provided.")
    if args.gpus_per_instance < 1:
        raise ValueError("--gpus-per-instance must be >= 1")

    gpu_groups = split_into_groups(gpu_list, args.gpus_per_instance)
    if len(gpu_groups) == 0:
        raise ValueError(
            f"Not enough GPUs ({len(gpu_list)}) to form at least one instance with --gpus-per-instance={args.gpus_per_instance}"
        )

    n_instances = len(gpu_groups)
    shards = [trials.iloc[idx] for idx in np.array_split(range(len(trials)), n_instances)]

    # Persist shard inputs and schedule workers
    shard_in_paths = []
    shard_out_paths = []
    for i, shard_df in enumerate(shards):
        in_path = os.path.join(args.work_dir, f"input_shard_{i:03d}.parquet")
        out_path = os.path.join(args.work_dir, f"output_shard_{i:03d}.parquet")
        # Ensure column order preserved
        shard_df = shard_df[base_cols]
        shard_df.to_parquet(in_path, index=False)
        shard_in_paths.append(in_path)
        shard_out_paths.append(out_path)

    # Launch one process per instance
    procs = []
    ctx = mp.get_context("spawn")
    for i, gpu_group in enumerate(gpu_groups):
        if shards[i].empty:
            continue
        if screening and Path(shard_out_paths[i]).exists():
            previous = pd.read_parquet(shard_out_paths[i])
            if previous[base_cols].equals(shards[i][base_cols].reset_index(drop=True)):
                try:
                    for row in previous.itertuples():
                        parse_screening_output(row.treatment_screen_output, row.treatment_screen_finish_reason)
                except ValueError:
                    pass
                else:
                    continue
        p = ctx.Process(
            target=worker_process,
            kwargs=dict(
                shard_in_path=shard_in_paths[i],
                shard_out_path=shard_out_paths[i],
                gpu_group=gpu_group,
                model=args.model,
                download_dir=args.download_dir,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_mem_util=args.gpu_mem_util,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
                max_tokens=args.max_tokens,
                screening=screening,
                batch_size=args.batch_size,
                reasoning_parser=reasoning_parser,
                extraction_method=getattr(args, "extraction_method", "hierarchical"),
                hierarchy_max_nodes=getattr(args, "hierarchy_max_nodes", 4096),
                hierarchy_max_calls=getattr(args, "hierarchy_max_calls", 20000),
                max_attempts=getattr(args, "max_attempts", 4),
                hierarchy_always_think=getattr(args, "hierarchy_always_think", False),
            ),
            name=f"vllm_worker_{i}",
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Join all
    exit_codes = []
    for p in procs:
        p.join()
        exit_codes.append(p.exitcode)

    if any(ec != 0 for ec in exit_codes):
        bad = [i for i, ec in enumerate(exit_codes) if ec != 0]
        raise RuntimeError(f"One or more workers failed: {bad}")
    return [path for path, shard in zip(shard_out_paths, shards) if not shard.empty]


def screen_trials(args, trials, reasoning_parser):
    """Finish the entire screening pass before dispatching any space extraction."""
    screen_args = copy(args)
    screen_args.work_dir = str(Path(args.work_dir) / "treatment_screening")
    Path(screen_args.work_dir).mkdir(parents=True, exist_ok=True)
    check_screening_manifest(args, reasoning_parser, screen_args.work_dir)
    screen_args.max_tokens = args.screening_max_tokens
    if args.server_urls or args.server_urls_file:
        paths = remote_main(screen_args, trials, trials.columns.tolist(), reasoning_parser, screening=True)
    else:
        # Local screening loads the same model on the same GPU groups. All workers
        # finish before a separate extraction pass starts.
        paths = local_main(screen_args, trials, trials.columns.tolist(), reasoning_parser, screening=True)
    audit = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    audit = audit.drop_duplicates("nct_id", keep="last")
    # The checked screening manifest binds all checkpoints to this model, including
    # older shards saved before per-row model provenance was added.
    audit["treatment_screen_model"] = args.model
    audit_path = Path(args.output_screening) if args.output_screening else Path(args.output_trials).with_name("trial_treatment_screening.csv")
    temporary = audit_path.with_suffix(".tmp")
    audit.drop(columns=["_work_id"], errors="ignore").to_csv(temporary, index=False)
    temporary.replace(audit_path)
    if set(audit.nct_id) != set(trials.nct_id) or not audit.treatment_screen_status.eq("ok").all():
        raise RuntimeError(f"Treatment screening incomplete; see {audit_path}. Resume with the same arguments to retry errors. No spaces were extracted.")
    # Validate persisted outputs too; malformed cached decisions cannot silently
    # exclude a trial or enter the extraction queue.
    decisions = [parse_screening_output(row.treatment_screen_output, row.treatment_screen_finish_reason)
                 for row in audit.itertuples()]
    kept_ids = {nct for nct, decision in zip(audit.nct_id, decisions)
                if decision["is_cancer_treatment_trial"]}
    selected = trials.loc[trials.nct_id.isin(kept_ids)].copy()
    print(f"[screen] retained {len(selected)}/{len(trials)} direct anti-cancer treatment trials; audit: {audit_path}")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV of trials (ctgov_cancer_trials_9-2025.csv)")
    parser.add_argument(
        "--start-year", type=int, default=2020,
        help="Inclusive earliest initial-posting year (default: 2020), using study_first_post_date. "
             "Missing or invalid dates are excluded; applies with every --trial-filter.",
    )
    parser.add_argument(
        "--trial-filter", choices=["interventional-phase1-3", "all"],
        default="interventional-phase1-3",
        help="Default: interventional early phase 1 and phase 1/2/3, including combined phases. "
             "Excludes phase 4, missing and N/A phases.",
    )
    parser.add_argument("--output-trials", default="../data/no_phi/trials_with_spaces.csv")
    parser.add_argument("--output-screening", default=None,
                        help="All treatment-screen decisions/rationales; default trial_treatment_screening.csv beside --output-trials")
    screening_mode = parser.add_mutually_exclusive_group()
    screening_mode.add_argument("--screening-only", action="store_true",
                                help="Save treatment screening results and stop before trial-space extraction")
    screening_mode.add_argument("--screening-input", default=None,
                                help="Reuse a complete screening CSV for identical selected inputs; "
                                     "--model and endpoint settings apply only to extraction")
    parser.add_argument("--screening-max-tokens", type=int, default=8192,
                        help="Reasoning plus final-output token budget for the separate treatment screen")
    parser.add_argument("--output-spaces", default="../data/no_phi/trial_space_lineitems.csv")
    parser.add_argument("--extraction-method", choices=["hierarchical", "one-shot"], default="hierarchical",
                        help="Default: one field per parent branch, plus independent boilerplate extraction")
    parser.add_argument("--hierarchy-max-nodes", type=int, default=4096,
                        help="Fail rather than truncate if one trial exceeds this number of branches")
    parser.add_argument("--hierarchy-max-calls", type=int, default=20000,
                        help="Maximum logical field/boilerplate requests per trial, excluding retries")
    parser.add_argument("--hierarchy-trial-batch-size", type=int, default=32,
                        help="Minimum rolling active-trial window; enlarged to endpoint concurrency")
    parser.add_argument("--hierarchy-always-think", action="store_true",
                        help="Keep thinking enabled on every hierarchical call, including JSON repair retries")
    parser.add_argument("--work-dir", default="../data/no_phi/trial_space_shards", help="Directory for shard inputs/outputs")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs, e.g. 0,1,2,3 (required for local mode; omit when using --server_urls / --server_urls_file)")
    parser.add_argument("--gpus-per-instance", type=int, default=1, help="Tensor-parallel GPUs per instance (set >1 for very large models)")
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--tokenizer-model", default=None,
                        help="Tokenizer/model-family ID when --model is a served alias")
    parser.add_argument("--download-dir", default="/data1/ken/models")
    parser.add_argument("--max-model-len", type=int, default=100000,
                        help="Local model context or remote request context cap, including output tokens")
    parser.add_argument("--max-num-seqs", type=int, default=900, help="vLLM max_num_seqs (concurrent request cap).")
    parser.add_argument("--gpu-mem-util", type=float, default=0.94)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=50000,
                        help="Maximum reasoning plus final-output tokens per extraction request")
    parser.add_argument("--batch-size", type=int, default=1000, help="Prompts per generate() call per instance")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split assignment")
    parser.add_argument(
        "--preserve-input-splits", action="store_true",
        help="Keep existing train/val/test assignments when re-extracting an already sampled cohort; "
             "requires a valid, consistent split for every selected NCT ID.",
    )

    from vllm_reasoning_utils import add_reasoning_cli_args, resolve_parser_name
    add_reasoning_cli_args(parser)

    from remote_vllm_pool import add_remote_cli_args
    add_remote_cli_args(parser)

    args = parser.parse_args()
    if not 1 <= args.start_year <= 9999:
        parser.error("--start-year must be a calendar year between 1 and 9999")
    from vllm_reasoning_utils import resolve_sampling_defaults
    legacy = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                  presence_penalty=1.5, repetition_penalty=1.0)
    sampling = resolve_sampling_defaults(
        args.tokenizer_model or args.model, fallback=legacy,
        **{key: getattr(args, key) for key in legacy}
    )
    for key, value in sampling.items():
        setattr(args, key, value)
    print(f"[main] sampling: {sampling}")
    reasoning_parser = resolve_parser_name(args.tokenizer_model or args.model, args.reasoning_parser)

    if args.screening_max_tokens < 1:
        raise ValueError("--screening-max-tokens must be positive")
    if min(args.hierarchy_max_nodes, args.hierarchy_max_calls, args.hierarchy_trial_batch_size,
           args.max_attempts, args.batch_size) < 1:
        raise ValueError("Hierarchy limits, max attempts and batch size must be positive")
    os.makedirs(args.work_dir, exist_ok=True)
    if any(Path(args.work_dir).glob("*output_shard_*.parquet")):
        raise ValueError("Legacy extraction shards predate treatment screening; use a new --work-dir.")

    # Load and pre-process trials
    trials = pd.read_csv(args.input)
    original_count = len(trials)
    trials = select_trials_since_year(trials, args.start_year)
    print(f"[main] first posted in {args.start_year} or later: kept {len(trials)}/{original_count} trials "
          "(missing or invalid first-posted dates excluded)")
    date_filtered_count = len(trials)
    trials = select_trials(trials, args.trial_filter)
    print(f"[main] {args.trial_filter}: kept {len(trials)}/{date_filtered_count} trials")
    if trials.empty:
        raise ValueError("No trials match the requested start year and trial filter; no inference was started.")
    trials = trials.copy()
    if args.preserve_input_splits:
        if "split" not in trials or not trials["split"].isin(["train", "val", "test"]).all():
            raise ValueError("--preserve-input-splits requires train/val/test in every selected row")
        if trials.groupby("nct_id")["split"].nunique().gt(1).any():
            raise ValueError("--preserve-input-splits requires a consistent split per NCT ID")
    else:
        # Deterministic split assignment for a new cohort.
        rng = np.random.default_rng(args.seed)
        trials["split"] = rng.choice(["train", "val", "test"], size=len(trials), p=[0.8, 0.1, 0.1])

    # Deduplicate by NCT ID like your original
    trials = trials.groupby("nct_id").first().reset_index()
    check_trial_selection_manifest(trials, args.work_dir, args.trial_filter, args.seed, args.start_year)

    if args.screening_input:
        trials = load_screening_results(trials, args.screening_input, args.work_dir)
    else:
        trials = screen_trials(args, trials, reasoning_parser)
    if args.screening_only:
        print("[main] screening complete; --screening-only requested, so trial-space extraction was not started.")
        return
    args.work_dir = str(Path(args.work_dir) / "space_extraction")
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    check_extraction_manifest(args, reasoning_parser)
    check_trial_selection_manifest(trials, args.work_dir, args.trial_filter, args.seed, args.start_year)

    # We only generate from trial_text, but keep the rest for merging later
    # Keep the original column set so we can write the same outputs
    base_cols = trials.columns.tolist()
    print(trials.info())
    print(trials.columns)
    print(base_cols)

    # ---- Remote-mode dispatch ----
    if trials.empty:
        shard_out_paths = []
    elif args.server_urls or args.server_urls_file:
        print("[main] dispatching to remote vLLM pool")
        shard_out_paths = remote_main(args, trials, base_cols, reasoning_parser)
    else:
        shard_out_paths = local_main(args, trials, base_cols, reasoning_parser)

    # Share the bounded exporter across local/remote and one-shot/hierarchical
    # extraction. Full reasoning stays once per trial, with NCT-linked audit
    # references on individual spaces instead of repeating entire trial trees.
    from trial_space_export import write_trial_exports
    counts = write_trial_exports(shard_out_paths, args.output_trials, args.output_spaces, base_cols)
    print(f"[export] wrote {counts['trials']} trials and {counts['spaces']} individual spaces")


if __name__ == "__main__":
    main()
