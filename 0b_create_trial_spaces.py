#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel trial-space extraction with vLLM.

Usage examples
--------------
# One instance per GPU (model fits on a single GPU):
python 0b_create_trial_spaces.py \
  --input ctgov_trials.csv \
  --gpus 2,3 \
  --gpus-per-instance 1 \
  --download-dir ../models


Outputs
-------
- trials_with_spaces.csv
- trial_space_lineitems.csv
"""

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
from trial_treatment_screening import (
    build_screening_messages, parse_screening_output, screening_record,
    check_screening_manifest,
)

# vLLM imports happen inside worker after setting CUDA_VISIBLE_DEVICES


# Keep these snapshots identical to inference-kehl trial.user.primer.txt
# and trial.user.question.txt; tests/test_trial_prompt_contract.py checks
# the complete rendered messages against the sibling checkout.
PROMPT_HEADER = (
    'You are an expert clinical oncologist with a broad and deep knowledge of cancer and its treatments.\n'
    'Your job is to review a clinical trial document and extract a list of structured clinical spaces that are eligible for that trial.\n'
    'A clinical space is defined as a unique combination of patient age range, sex (if any sex criteria), cancer primary site, histology, cancer burden, which treatments a patient must have received, which treatments a patient must not have received, tumor biomarkers (such as germline or somatic gene mutations or alterations, or protein expression on tumor), that a patient must have or must not have to be eligible for the trial.\n'
    '-Re sex criteria: For cancers originating in organs only present in one sex, you must assume the sex criteria even if not stated explicitly. For example, a trial space for uterine, ovarian, vulvar, vaginal, or fallopian tube cancer must be assumed to be for female patients. Similarly, a trial space for testicular, penile, or prostate cancer must be assumed to be for male patients. For all other cancer types (including breast cancer), you should assume the trial is open to both sexes unless the clinical trial document states otherwise.\n'
    '-Re cancer primary site: This simply means lung cancer, vs breast cancer, vs prostate cancer, and so on.\n'
    '-Re histology: This refers to adenocarcinoma, squamous cell carcinoma, and so on. Biomarkers such as protein expression, receptor status, etc do NOT belong in histology; they belong in biomarkers.\n'
    '-Re cancer burden: The intent here is to specify the conceptual type or category of patient targeted by this trial space. For many solid tumors, this may include presence of early stage vs advanced/metastatic disease; for heme malignancies, this may include relapsed, refractory, or remission status. Cancer burden also encompasses cancer type-specific prognostic scores, risk indices, or categories; it does NOT include ECOG performance status, or concepts like "life expectancy at least 6 months.") Do NOT include information the trial incorporates to evaluate eligibility for its specific research plan, as opposed to its conceptual target population. Specifically, details about measurable disease, target lesions, biopsy targets, etc are NOT part of cancer burden; those details DO NOT BELONG IN YOUR TRIAL SPACE OUTPUT TEXT AT ALL. If a trial specifies stage IV disease but is actually seeking patients in any palliative-intent treatment (or advanced/metastatic) setting, including recurrent disease, use a description that would encompass both de novo and recurrent advanced disease, such as terms relating to advanced/metastatic or metastatic. Treatment history requirements are NOT part of cancer burden; treatment history requirements go in the treatment history requirements section. \n'
    '-Re treatment history: This includes any requirements that patients have received, or not received, a particular treatment or treatment modality. These requirements may be as broad as surgery, chemotherapy, radiation, etc; or as specific as a certain drug class or specific drug. Critically, trials often specify that a particular treatment is excluded only if it was given within a short period of time, for example 14 days, one month, etc, prior to trial start. This is called a washout period. Do not include this type of time-specific treatment washout eligibility criteria in your output at all.\n'
    '-Re biomarkers: This includes germline or somatic gene mutations or alterations, or protein expression on tumor, or other molecular test results used to predict benefit of treatment. It does not include tumor markers, often serum markers, used to assess cancer burden. If the trial specifies whether the biomarker will be assessed during screening, note that.\n'
    'Some trials have only one space, while others have many. Do not output a space that contains multiple cancer types and/or histologies. Instead, generate separate spaces for each cancer type/histology combination. VERY IMPORTANT: This means that if a trial is for patients with any solid tumor, or any hematologic malignancy, etc, you must construct an exhaustive list of spaces for any solid tumor cancer type, AND any heme malignancy type, etc, that could be reasonably consistent with the eligibility criteria. This is required normalization, not invention of eligibility criteria. Use your oncology knowledge to enumerate those diagnoses even when the protocol does not name them individually, while respecting all explicit exclusions. Only then can you end with spaces for solid tumors and/or or hematologic malignancies in general. So, general solid tumor or heme malignancy spaces must only come at the end of a long list of cancer type-specific spaces. \n'
    'CRITICAL: Each trial space must contain all information necessary to define that space on its own. It may not refer to other previously defined spaces for the same trial, since for later use, the spaces will be extracted and separated from each other. Similarly, it must not refer to the trial protocol\'s cohorts or arms, which the user of the trial space text will not have immediate access to. YOU MAY NOT include text describing a given space that refers to a previous space; eg, "Same as above"-style output is not allowed!\n'
    'Spell out cancer types; do not abbreviate them. For example, write "non-small cell lung cancer" rather than "NSCLC".\n'
    'Structure your output like this, as a list of spaces, with spaces separated by newlines, as below. STRICTLY adhere to the formatting.\n'
    '1. Age range allowed: <age_range_allowed>. Sex allowed: <sex_allowed>. Cancer type allowed: <cancer_type_allowed>. Histology allowed: <histology_allowed>. Cancer burden allowed: <cancer_burden_allowed>. Prior treatment required: <prior_treatments_requred>. Prior treatment excluded: <prior_treatments_excluded>. Biomarkers required: <biomarkers_required>. Biomarkers excluded: <biomarkers_excluded>.\n'
    '2. Age range allowed: <age_range_allowed>. Sex allowed: <sex_allowed>. ...\n'
    'If a concept is not relevant, such as if there are no prior treatments required, simply output NA for that concept.\n'
    'CRITICAL: Anytime you provide a list for a particular concept, you must be completely clear on whether "or" versus "and" logic applies to the list. For example, do not output "EGFR L858R mutant, TP53 mutant"; if both are required, output "EGFR L858R mutant and TP53 mutant". As another example, do not output "ER+, PR+"; if the patient can have either an ER or a PR positive tumor, output "ER+ or PR+".\n'
    'Alternative prognostic or risk features within the same disease and treatment setting should generally remain in one space with explicit OR logic. Create separate spaces when they define meaningfully different populations for treatment decision-making, and retain all applicable shared criteria in every space. For example, if a trial allows patients with either (1) EGFR-mutant non-small cell lung cancer or (2) ALK-rearranged non-small cell lung cancer, that should be output as two separate spaces, one for the EGFR-mutant non-small cell lung cancer and one for the ALK-rearranged non-small cell lung cancer, even if all other criteria are the same for both spaces.\n'
    'NEVER put a newline within a single trial space.\n'
    'After you output the trial spaces, output a newline, then the text "Boilerplate exclusions:" VERBATIM, then another newline.\n'
    'Then, list exclusion criteria described in the trial text that are unrelated to the trial space definitions. Such exclusions tend to be common to clinical trials in general.\n'
    'Common boilerplate exclusion criteria include a history of pneumonitis, heart failure, renal dysfunction, liver dysfunction, uncontrolled brain metastases, HIV or hepatitis, prior unrelated cancer diagnoses, and poor performance status.\n'
    'Make sure your boilerplate exclusions are clearly phrased as exclusion criteria, not as requirements for exclusion. For example, if a trial requires ECOG 0 or 1 for eligibility, do NOT write "ECOG 0 or 1" in the boilerplate exclusions. Instead, write "Poor performance status (eg ECOG >1)" or similar language that clearly indicates this is an exclusion criterion.\n'
    'ALWAYS output plain text only. NEVER output unicode, Markdown, or tables.'
)

PROMPT_SUFFIX = (
    'Now, generate your list of the trial space(s), followed by any boilerplate exclusions, formatted as above.\n'
    'Do not provide any introductory, explanatory, concluding, or disclaimer text.\n'
    'Reminder: Treatment history is an important component of trial space definitions, but treatment history "washout" requirements that are described as applying only in a given period of time prior to trial treatment MUST BE IGNORED.\n'
    'CRITICAL: A given trial space MUST NEVER refer to another previously defined space. You must NEVER output text like "same as #1" or "same criteria as above." Instead, you MUST REPEAT all relevant criteria for each new space SO THAT IT STANDS ON ITS OWN. A user who later looks at the text for one space will not have access to text for other spaces, and so output like "Same criteria as #1..." renders a space useless!\n'
)

BOILERPLATE_MARKER = "Boilerplate"


def build_messages(trial_text: str):
    """Chat messages for a single trial."""
    return [
        {"role": "system", "content": "\nReasoning: high.\n"},
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

    # Build prompts
    trial_texts = shard_df["trial_text"].astype(str).tolist()
    prompts = [
        tokenizer.apply_chat_template(
            conversation=(build_screening_messages(t) if screening else build_messages(t)),
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True
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


def make_trial_completion_work_fn(completion_fn, tokenizer, work_dir, *, screening=False):
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
        reasoning, final, metadata = await completion_fn(
            client, {**payload, "enable_thinking": thinking,
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
                  error=str(exc), finish_reason=metadata["finish_reason"],
                  reasoning=reasoning, final=final, raw_text=metadata.get("raw_text"))
            messages = (build_screening_messages(payload["trial_text"]) if screening
                        else build_messages(payload["trial_text"]))
            if not next_thinking and messages[0]["content"].strip() == "Reasoning: high.":
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
            payload.update(
                validation_failures=failures, enable_thinking=next_thinking,
                prompt=tokenizer.apply_chat_template(
                    conversation=messages, add_generation_prompt=True,
                    tokenize=False, enable_thinking=next_thinking,
                ),
            )
            raise
        if failures:
            audit(payload, event="recovered", failure_number=failures,
                  enable_thinking=thinking, finish_reason=metadata["finish_reason"])
        return result

    return work_fn


def remote_main(args, trials: pd.DataFrame, base_cols, reasoning_parser: str, *, screening=False):
    """Run trial-space extraction against a remote vLLM pool. Writes one
    or more parquet shards into args.work_dir and returns the list of
    paths for the caller's existing merge code."""
    import asyncio
    import glob
    import re
    from transformers import AutoTokenizer
    from remote_vllm_pool import (
        CompletionSampling,
        make_completion_work_fn,
        run_pool,
        build_registry_from_args,
    )

    print(f"[remote] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        getattr(args, "tokenizer_model", None) or args.model,
        cache_dir=args.download_dir or None,
        trust_remote_code=True,
    )

    # Build per-trial prompts upfront
    trials = trials.copy().reset_index(drop=True)
    trials["_work_id"] = np.arange(len(trials), dtype=np.int64)
    trial_texts = trials["trial_text"].astype(str).tolist()
    print(f"[remote] building {len(trial_texts)} prompts...")
    prompts = [
        tokenizer.apply_chat_template(
            conversation=(build_screening_messages(t) if screening else build_messages(t)),
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
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
          "trial_text": trial_texts[i], "nct_id": str(trials.iloc[i]["nct_id"])})
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
    )
    registry = build_registry_from_args(args)

    def shard_writer(payload, shard_idx):
        if screening:
            rows = []
            for work_id, result in payload:
                orig = trials_indexed.loc[work_id].to_dict()
                if isinstance(result, str) and result.startswith("ERROR:"):
                    result = {"treatment_screen_status": "error", "treatment_screen_output": result}
                rows.append({**orig, "_work_id": work_id, **result})
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
            "Use --trial-filter all only to intentionally process every trial."
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

    mask = trials.study_type.eq("INTERVENTIONAL") & trials.phases.map(eligible_phase)
    return trials.loc[mask].copy()


def check_extraction_manifest(args, reasoning_parser):
    """Keep extraction prompt/configuration changes out of existing shards."""
    path = Path(args.work_dir) / "extraction_manifest.json"
    messages = json.dumps(build_messages(""), ensure_ascii=False)
    identity = {
        "messages_sha256": hashlib.sha256(messages.encode()).hexdigest(),
        "model": args.model, "reasoning_parser": reasoning_parser,
        "enable_thinking": True,
        "sampling": {key: getattr(args, key) for key in (
            "temperature", "top_p", "top_k", "min_p", "presence_penalty",
            "repetition_penalty", "max_tokens")},
    }
    if getattr(args, "tokenizer_model", None):
        identity["tokenizer_model"] = args.tokenizer_model
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(
                "Trial-space extraction prompt/configuration changed; use fresh extraction checkpoints. "
                "Compatible treatment-screening checkpoints may be retained."
            )
    elif any(Path(args.work_dir).glob("*output_shard_*.parquet")):
        raise ValueError("Extraction shards lack a prompt manifest; use fresh extraction checkpoints.")
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity, indent=2))
        temporary.replace(path)


def check_trial_selection_manifest(trials, work_dir, trial_filter, seed):
    """Prevent row-position checkpoints being reused after selection changes."""
    path = Path(work_dir) / "trial_selection.json"
    # Fingerprint selected content as well as the policy: _work_id is positional.
    digest = hashlib.sha256()
    for offset in range(0, len(trials), 1000):
        content = trials.iloc[offset:offset + 1000].to_json(orient="records", force_ascii=False)
        digest.update(content.encode())
    identity = {"trial_filter": trial_filter, "seed": seed, "rows": len(trials),
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
        "--trial-filter", choices=["interventional-phase1-3", "all"],
        default="interventional-phase1-3",
        help="Default: interventional early phase 1 and phase 1/2/3, including combined phases. "
             "Excludes phase 4, missing and N/A phases.",
    )
    parser.add_argument("--output-trials", default="../data/no_phi/trials_with_spaces.csv")
    parser.add_argument("--output-screening", default=None,
                        help="All treatment-screen decisions/rationales; default trial_treatment_screening.csv beside --output-trials")
    parser.add_argument("--screening-max-tokens", type=int, default=8192,
                        help="Reasoning plus final-output token budget for the separate treatment screen")
    parser.add_argument("--output-spaces", default="../data/no_phi/trial_space_lineitems.csv")
    parser.add_argument("--work-dir", default="../data/no_phi/trial_space_shards", help="Directory for shard inputs/outputs")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs, e.g. 0,1,2,3 (required for local mode; omit when using --server_urls / --server_urls_file)")
    parser.add_argument("--gpus-per-instance", type=int, default=1, help="Tensor-parallel GPUs per instance (set >1 for very large models)")
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--tokenizer-model", default=None,
                        help="Tokenizer/model-family ID when --model is a served alias")
    parser.add_argument("--download-dir", default="/data1/ken/models")
    parser.add_argument("--max-model-len", type=int, default=50000)
    parser.add_argument("--max-num-seqs", type=int, default=900, help="vLLM max_num_seqs (concurrent request cap).")
    parser.add_argument("--gpu-mem-util", type=float, default=0.94)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=45000)
    parser.add_argument("--batch-size", type=int, default=1000, help="Prompts per generate() call per instance")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split assignment")

    from vllm_reasoning_utils import add_reasoning_cli_args, resolve_parser_name
    add_reasoning_cli_args(parser)

    from remote_vllm_pool import add_remote_cli_args
    add_remote_cli_args(parser)

    args = parser.parse_args()
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
    os.makedirs(args.work_dir, exist_ok=True)
    if any(Path(args.work_dir).glob("*output_shard_*.parquet")):
        raise ValueError("Legacy extraction shards predate treatment screening; use a new --work-dir.")

    # Load and pre-process trials
    trials = pd.read_csv(args.input)
    original_count = len(trials)
    trials = select_trials(trials, args.trial_filter)
    print(f"[main] {args.trial_filter}: kept {len(trials)}/{original_count} trials")
    if trials.empty:
        raise ValueError("No trials match the requested trial filter; no inference was started.")
    # Deterministic split assignment
    rng = np.random.default_rng(args.seed)
    trials = trials.copy()
    trials["split"] = rng.choice(["train", "val", "test"], size=len(trials), p=[0.8, 0.1, 0.1])

    # Deduplicate by NCT ID like your original
    trials = trials.groupby("nct_id").first().reset_index()
    check_trial_selection_manifest(trials, args.work_dir, args.trial_filter, args.seed)

    trials = screen_trials(args, trials, reasoning_parser)
    args.work_dir = str(Path(args.work_dir) / "space_extraction")
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    check_extraction_manifest(args, reasoning_parser)
    check_trial_selection_manifest(trials, args.work_dir, args.trial_filter, args.seed)

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

    # Merge shard outputs
    out_parts = [pd.read_parquet(pth) for pth in shard_out_paths]
    trials_with_spaces = (pd.concat(out_parts, axis=0, ignore_index=True) if out_parts else
                          pd.DataFrame(columns=base_cols + ["space_reasoning_and_output",
                              "space_output_no_reasoning", "space_text", "trial_boilerplate_text"]))

    # Keep and order columns as in your original final selection
    final_cols = [
        "nct_id",
        "study_type",
        "phases",
        "split",
        "title",
        "brief_summary",
        "eligibility_criteria",
        "trial_text",
        "space_reasoning_and_output",
        "space_output_no_reasoning",
        "space_text",
        "trial_boilerplate_text",
    ]
    # Some inputs may not have every column (defensive handling)
    final_cols = [c for c in final_cols if c in trials_with_spaces.columns]
    trials_with_spaces = trials_with_spaces[final_cols]

    # Write trials_with_spaces.csv
    trials_with_spaces.to_csv(args.output_trials, index=False)

    # Build line-items dataframe
    tws = trials_with_spaces.copy()
    tws["space_text"] = tws["space_text"].astype(str)

    frames = []
    for i in range(tws.shape[0]):
        row = tws.iloc[[i]]  # 1-row DataFrame
        cohorts = pd.Series(str(row.iloc[0]["space_text"]).split("\n"))
        cohorts = cohorts[~(cohorts.isna() | (cohorts == "\n") | (cohorts == ""))].reset_index(drop=True)
        if len(cohorts) == 0:
            continue
        frame = pd.DataFrame(np.repeat(row.values, len(cohorts), axis=0), columns=row.columns)
        frame["this_space"] = cohorts.values

        # quality check - require all space components
        search_list = ["Age", "Sex", "Cancer type", "Histology", "Cancer burden", "Prior treatment", "Prior treatment", "Biomarkers required", "Biomarkers excluded"]
        frame = frame[frame.this_space.apply(lambda x: all(term in x for term in search_list))]
        
        # 1-based space numbering within this trial
        frame["space_number"] = np.arange(1, frame.shape[0] + 1)
        frames.append(frame)

    if frames:
        cohort_level_trials = pd.concat(frames, axis=0, ignore_index=True)
        # Keep only lines that look like numbered spaces ("1.", "2.", ...)
        keep_mask = cohort_level_trials["this_space"].str.strip().str[0].isin(list("123456789"))
        cohort_level_trials = cohort_level_trials[keep_mask]
        # Remove numbers from space names now
        cohort_level_trials["this_space"] = cohort_level_trials['this_space'].str.replace(r'^\s*\d+\.', '', regex=True).str.strip()   

        # Order columns similar to your original (dropping the accidental duplicate nct_id)
        ordered_cols = [
            "nct_id",
            "study_type",
            "phases",
            "split",
            "title",
            "eligibility_criteria",
            "trial_text",
            "space_reasoning_and_output",
            "space_output_no_reasoning",
            "space_text",
            "trial_boilerplate_text",
            "this_space",
            "space_number",
        ]
        ordered_cols = [c for c in ordered_cols if c in cohort_level_trials.columns]
        cohort_level_trials = cohort_level_trials[ordered_cols]
        cohort_level_trials.to_csv(args.output_spaces, index=False)
    else:
        # Still emit an empty file with headers if no frames
        pd.DataFrame(
            columns=[
                "nct_id",
                "study_type",
                "phases",
                "split",
                "title",
                "eligibility_criteria",
                "trial_text",
                "space_reasoning_and_output",
                "space_output_no_reasoning",
                "space_text",
                "trial_boilerplate_text",
                "this_space",
                "space_number",
            ]
        ).to_csv(args.output_spaces, index=False)


if __name__ == "__main__":
    main()
