"""Label outpatient medical oncology notes against PRISSMM curation rules with a vLLM server.

Default input is the synthetic clinical-note parquet at
../../data/no_phi/synthetic_clinical.parquet (relative to this file). The
labeled output is written next to it as synthetic_clinical_vllm_labeled.jsonl
plus a parquet copy synthetic_clinical_vllm_labeled.parquet.

Example: connect to an already-running vLLM OpenAI-compatible server and label
the default synthetic clinical-note parquet.

    python label_oncology_notes.py \
        --model Qwen/Qwen2.5-72B-Instruct \
        --server-url http://127.0.0.1:8000 \
        --workers 16

Example: launch a tensor-parallel vLLM server on GPUs 0-3 and label the default
input, shutting the server down on exit.

    python label_oncology_notes.py \
        --model Qwen/Qwen2.5-72B-Instruct \
        --start-vllm --gpus 0-3 --gpus-per-server 4 \
        --workers 32

Override the input or output paths if needed:

    python label_oncology_notes.py \
        --input some_other_notes.parquet \
        --output some_other_labels.jsonl \
        --model Qwen/Qwen2.5-72B-Instruct \
        --server-url http://127.0.0.1:8000

Pass extra flags through to the vLLM server with --vllm-arg (repeat as needed,
or pack multiple flags into one shell-quoted string):

    python label_oncology_notes.py \
        --model Qwen/Qwen3-32B \
        --start-vllm --gpus 0-3 --gpus-per-server 4 \
        --vllm-arg "--max-model-len 32768" \
        --vllm-arg "--gpu-memory-utilization 0.92" \
        --vllm-arg "--dtype bfloat16"

Example: use the GCP-orchestrated dynamic vLLM pool from the repository root.

    python label_oncology_notes.py \
        --input ../../data/no_phi/synthetic_clinical.parquet \
        --model Qwen/Qwen2.5-72B-Instruct \
        --server_urls_file /tmp/mmai_gcp_servers.json \
        --max_concurrent_per_server 50 \
        --results_per_shard 200

Inputs may be Parquet, CSV, TSV, JSONL, JSON, a single TXT file, or a directory
of TXT files. Use --resume to skip record_ids already present in the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_labeling_common import Record, add_common_args, metadata_for_prompt, run_labeling


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent.parent / "data" / "no_phi"
DEFAULT_INPUT = str(DATA_DIR / "synthetic_clinical.parquet")
DEFAULT_OUTPUT = str(DATA_DIR / "synthetic_clinical_vllm_labeled.jsonl")
DEFAULT_TEXT_FIELD = "synthetic_note"
DEFAULT_ID_FIELDS = ["pseudo_mrn", "row_id"]
DEFAULT_METADATA_FIELDS = ["pseudo_mrn", "patient_id", "event_type", "row_id", "split"]


SYSTEM_PROMPT = """You are an oncology data curator labeling medical oncology notes using PRISSMM-style curation rules.
Return exactly one valid JSON object. Do not return markdown, prose outside JSON, or comments.
Use null when a value cannot be determined from the allowed sections or metadata.
Quote short evidence snippets from the note for important labels."""


GUIDANCE = """
Task: label one outpatient medical oncology assessment note according to the PRISSMM Medical Oncology Assessment guidance.

Note selection and allowed text:
- Eligible notes are internal or external outpatient medical oncology provider notes by MDs, nurse practitioners, or physician assistants actively following the patient for the cancer of interest. Televisits from March 2020 onward can be used.
- Do not curate radiation oncology, surgery/surgical oncology, inpatient, primary care, or unrelated specialty notes. Neuro-oncology or other oncology specialty notes may be eligible only if they are the active oncology follow-up for the cancer of interest and primary medical oncology notes are unavailable.
- For ovarian, fallopian tube, or primary peritoneal cancer, gynecologic oncology notes are eligible if GynOnc administered antineoplastic drug therapy.
- Use only the Impression/Plan/Assessment/Recommendations/Summary/Problem List section and reason for visit for cancer evidence and cancer status. If there are no section headers, use text after the physical exam.
- For ECOG and Karnofsky fields only, use explicitly stated current performance status from Physical Exam, Review of Systems, Impression/Plan, or metadata.
- Do not infer ECOG or Karnofsky from symptoms, function, or general condition.

Fields and values:
- md_onc_visit_date: outpatient oncology visit date, normalized to YYYY-MM-DD if possible. Use visit date, not signature/upload date.
- md_inst: 1 Internal institution, 2 External institution, null unknown.
- md_type_ca_cur: cancer of interest. Use metadata if provided; otherwise infer from note when clear. Include code when obvious from the PRISSMM list; otherwise code null and label text.
- md_ecog: 0, 1, 2, 3, 4, or 9 Not documented in note. Only record explicitly stated current ECOG.
- md_karnof: if ECOG is not documented, record explicitly stated current KPS as 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, or 9 Not documented. If ECOG is documented, set md_karnof code null because the PRISSMM field is only curated when ECOG is unavailable.
- md_pca_status: prostate cancer only. 1 Hormone/castrate/androgen sensitive, 2 Hormone/castrate/androgen resistant or independent, 99 Not stated, null not prostate/unknown applicability.
- md_psa_status: prostate cancer only. 1 PSA nadir or undetectable, 2 PSA stable, 3 PSA increasing, 4 PSA decreasing, 99 PSA change not mentioned, null not prostate/unknown applicability.
- md_ca: 1 Yes evidence of cancer, 2 No evidence of cancer, 3 Uncertain/indeterminate/equivocal, 4 Impression/Plan does not mention cancer.
- md_ca_status: only when md_ca is 1. Values: 1 Improving/Responding, 2 Stable/No change, 3 Mixed, 4 Progressing/Worsening/Enlarging, 5 Not stated/Indeterminate.

Evidence of cancer rules:
- If Impression/Plan says Stage IV cancer, metastatic cancer, known active metastases, on treatment for active cancer, or similar definite active disease language, choose yes.
- "History of cancer" alone does not necessarily mean current evidence of cancer.
- For metastatic disease, choose no evidence only if the provider explicitly says complete response/remission, NED/no evidence of disease, no active cancer, or cancer is no longer present.
- "No measurable disease" does not necessarily mean no evidence of cancer.
- Stage I, II, or III disease alone does not necessarily imply evidence of cancer.
- Neoadjuvant chemotherapy or upcoming planned cancer resection/surgery implies yes evidence of cancer.
- For ovarian, fallopian tube, or primary peritoneal cancer, adjuvant chemotherapy after definitive resection or optimal debulking, including no gross residual disease or residual nodules 1 cm or less, should be no evidence of cancer unless otherwise stated.

Ambiguous term rules:
- Treat as yes evidence when cancer is described as concerning for, compatible with, consistent with, favors, likely/most likely, presumed, probable, suggests, suspected, suspicious for, typical of, worrisome for, or other wording indicating more than 50% chance.
- Treat as uncertain when terms include bears watching with mention of cancer, cannot be ruled out, follow-up recommended with mention of cancer, equivocal, possible, potentially, questionable, rule out, may/could with mention of cancer, or other wording indicating 50% chance or lower.
- Treat as no evidence when terms include bears watching with no mention of cancer or follow-up recommended with no mention of cancer.

Cancer status rules:
- Prefer the provider's general/summary statement or overall impression whenever possible.
- Imaging recaps can be used only if the provider gives no own overall impression.
- Tumor marker changes should not determine status unless the provider explicitly uses them as disease status, e.g. "decrease in tumor markers indicates response."
- Copied-forward status statements still count if present in the current visit note.
- Choose the response reflecting status on the visit date, not historical progression before current treatment.
- Do not use general well-being or treatment tolerance as cancer status. "Doing well on chemo" is not response. "Clinically improving" is not response unless directly tied to antineoplastic treatment or cancer response.
- If statements conflict, choose the option matching the favored/overall impression.
- Responding: decreased tumor burden, responding to therapy, all sites decreasing/resolved, or some decreasing/resolved and some stable. If complete response, md_ca should be no evidence rather than yes.
- Stable: provider says stable disease, unchanged cancer burden, clinically stable to continue treatment/receive chemo, stable response, or good disease control.
- Mixed: mixed response or some sites decreasing/resolved and some new/increasing.
- Progressing: recurrence, relapse, progression/progressive disease, increased tumor burden, not responding, all sites increasing, some increasing and some stable, or a new cancer site.
- Not stated/indeterminate: no general cancer status, no imaging/status assessment described, provider says status cannot be determined, or baseline/recent diagnosis before treatment or definitive local therapy.
"""


SCHEMA = """
Return this JSON shape:
{
  "record_id": string,
  "should_curate": {"value": true|false|null, "reason": string, "evidence": string|null},
  "md_onc_visit_date": {"value": string|null, "evidence": string|null},
  "md_inst": {"code": 1|2|null, "label": string|null, "evidence": string|null},
  "md_type_ca_cur": {"code": integer|null, "label": string|null, "evidence": string|null},
  "md_ecog": {"code": 0|1|2|3|4|9|null, "label": string|null, "evidence": string|null},
  "md_karnof": {"code": 100|90|80|70|60|50|40|30|20|10|9|null, "label": string|null, "evidence": string|null},
  "md_pca_status": {"code": 1|2|99|null, "label": string|null, "evidence": string|null},
  "md_psa_status": {"code": 1|2|3|4|99|null, "label": string|null, "evidence": string|null},
  "md_ca": {"code": 1|2|3|4|null, "label": string|null, "evidence": string|null},
  "md_ca_status": {"code": 1|2|3|4|5|null, "label": string|null, "evidence": string|null},
  "allowed_sections_used": [string],
  "uncertainties": [string]
}
"""


def build_messages(record: Record, args: argparse.Namespace) -> list[dict[str, str]]:
    metadata_fields = [field.strip() for field in (args.metadata_fields or "").split(",") if field.strip()] or None
    metadata = metadata_for_prompt(record, args.text_field, metadata_fields=metadata_fields)
    user_prompt = f"""{GUIDANCE}

{SCHEMA}

Record id: {record.record_id}

Available metadata:
{metadata}

Note text:
\"\"\"
{record.text}
\"\"\"
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Label medical oncology notes with a vLLM OpenAI-compatible server.")
    add_common_args(
        parser,
        default_text_field=DEFAULT_TEXT_FIELD,
        default_output=DEFAULT_OUTPUT,
        default_input=DEFAULT_INPUT,
        default_id_fields=DEFAULT_ID_FIELDS,
        default_metadata_fields=DEFAULT_METADATA_FIELDS,
    )
    args = parser.parse_args()
    run_labeling(args=args, build_messages=build_messages)


if __name__ == "__main__":
    main()
