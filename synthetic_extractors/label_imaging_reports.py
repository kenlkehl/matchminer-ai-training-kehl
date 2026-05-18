"""Label imaging/radiology reports against PRISSMM curation rules with a vLLM server.

Default input is the synthetic imaging parquet at ../../data/no_phi/synthetic_imaging.parquet
(relative to this file). The labeled output is written next to it as
synthetic_imaging_vllm_labeled.jsonl plus a parquet copy
synthetic_imaging_vllm_labeled.parquet.

Example: connect to an already-running vLLM OpenAI-compatible server and label
the default synthetic imaging parquet.

    python label_imaging_reports.py \
        --model Qwen/Qwen2.5-72B-Instruct \
        --server-url http://127.0.0.1:8000 \
        --workers 16

Example: launch a tensor-parallel vLLM server on GPUs 0-3 and label the default
input, shutting the server down on exit.

    python label_imaging_reports.py \
        --model Qwen/Qwen2.5-72B-Instruct \
        --start-vllm --gpus 0-3 --gpus-per-server 4 \
        --workers 32

Override the input or output paths if needed:

    python label_imaging_reports.py \
        --input some_other_reports.parquet \
        --output some_other_labels.jsonl \
        --model Qwen/Qwen2.5-72B-Instruct \
        --server-url http://127.0.0.1:8000

Pass extra flags through to the vLLM server with --vllm-arg (repeat as needed,
or pack multiple flags into one shell-quoted string):

    python label_imaging_reports.py \
        --model Qwen/Qwen3-32B \
        --start-vllm --gpus 0-3 --gpus-per-server 4 \
        --vllm-arg "--max-model-len 32768" \
        --vllm-arg "--gpu-memory-utilization 0.92" \
        --vllm-arg "--dtype bfloat16"

Example: use the GCP-orchestrated dynamic vLLM pool from the repository root.

    python label_imaging_reports.py \
        --input ../../data/no_phi/synthetic_imaging.parquet \
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
DEFAULT_INPUT = str(DATA_DIR / "synthetic_imaging.parquet")
DEFAULT_OUTPUT = str(DATA_DIR / "synthetic_imaging_vllm_labeled.jsonl")
DEFAULT_TEXT_FIELD = "synthetic_note"
DEFAULT_ID_FIELDS = ["pseudo_mrn", "row_id"]
DEFAULT_METADATA_FIELDS = ["pseudo_mrn", "patient_id", "event_type", "row_id", "split"]


SYSTEM_PROMPT = """You are an oncology data curator labeling radiology/imaging reports using PRISSMM-style curation rules.
Return exactly one valid JSON object. Do not return markdown, prose outside JSON, or comments.
Use null when a value cannot be determined from the allowed sections or metadata.
Quote short evidence snippets from the report for important labels."""


GUIDANCE = """
Task: label one imaging/radiology report according to the PRISSMM Radiology/Imaging guidance.

Report selection and allowed text:
- Curate bone scans, CTs, MRIs, PET/PET-CTs, PET MRIs, nuclear medicine scans, breast mammograms for breast cancer, and other cancer-specific scans.
- Do not curate ultrasounds, x-rays, fluoroscopy, IR biopsy/procedure images, MUGA, HIDA, bone density, or scans not used to monitor cancer status.
- Use only the Indication/Reason for Exam and Impression/Conclusion/Opinion/Integrated Summary sections to determine cancer evidence, status, sites, and reference scan. Do not use Findings/body text to resolve ambiguity.
- Prior scan context can be used if it is included in metadata or in the allowed report sections. If a prior site was cancer, assume it remains cancer until stated otherwise; absence of mention does not mean resolution.

Fields and values:
- image_inst_performed: 1 Internal institution, 2 External institution, null unknown.
- image_inst_interpreted: 1 Internal institution, 2 External institution, null unknown. Prefer internal reinterpretation when both exist.
- image_scan_date: scan/performed date, normalized to YYYY-MM-DD if possible.
- image_report_date: report/filed date, normalized to YYYY-MM-DD if possible; null if same as scan date or absent.
- image_scan_type: 1 CT, 3 MRI, 5 PET or PET-CT, 13 PET MRI, 7 Bone Scan, 9 Other Nuclear Medicine Scan, 11 Mammogram - Breast Cancer only, 20 Other CA-specific scan, null unknown/ineligible.
- image_scan_type_other: free text only when image_scan_type is 20.
- image_scan_sites: multi-select from Brain/Head, Spine, Neck, Chest, Abdomen, Pelvis, Extremity, Full Body. Use Full Body for PET/PET-CT and bone scan rather than listing each body part.
- image_ca: 1 Yes evidence of cancer, 2 No evidence of cancer, 3 Uncertain/indeterminate/equivocal, 4 Impression does not mention cancer.
- image_overall: only when image_ca is 1. Values: 1 Improving/Responding, 2 Stable/No change, 3 Mixed, 4 Progressing/Worsening/Enlarging, 5 Not stated/Indeterminate.
- image_cancer_sites: up to 15 sites of cancer mentioned on this report. Use the ICD-O-3 site/topography guide below for codes. Do not invent codes, and do not use ICD-10 secondary/metastatic codes such as C78._ or C79._. If no listed code fits confidently, set icdo_topography_code to null. Do not include resolved or indeterminate sites.
- image_ref_scan_date: comparison/reference scan date, normalized to YYYY-MM-DD if possible. If multiple dates are in the Impression, use the Impression date; otherwise use the most recent comparison date. Do not enter x-ray comparisons.

Evidence of cancer rules:
- Explicit evidence includes tumor or neoplasm unless stated benign/indeterminate; a mass, lesion, nodule, opacity, adenopathy, or enlarged lymph node described as malignant/metastatic/cancer; a non-lymph-node nodule/lesion/mass >= 1 cm unless benign; lymph node >= 1 cm followed for size; recurrence, progression, disease/tumor burden; PET FDG-avid/uptake/increased activity/hypermetabolic activity.
- Implicit evidence includes definite indication context such as known liver metastases, Stage IV cancer on treatment, treatment effect, or response to therapy. "History of cancer" or "restaging" alone does not necessarily mean current evidence of cancer.
- A new abnormality with no other information is no evidence of cancer. If a benign cause is favored and cancer is only a possibility, choose uncertain. If cancer is favored/likely and benign cause is possible, choose yes.
- Previously cancerous or suspicious sites remain cancer until stated otherwise. Resolved sites are no evidence and should not be listed as cancer sites.
- Post-resection changes alone are no evidence unless residual disease is noted. Post-treatment/radiation changes with explicit no residual disease/no recurrence are no evidence; residual abnormality attributed to treatment without a no-disease statement is indeterminate; residual abnormality without benign/treatment attribution is evidence of cancer.
- Pleural effusion, pericardial effusion, ascites, and other fluid collections require explicit malignant description or prior implicit evidence. Exception: abdominopelvic ascites in ovarian, fallopian tube, or primary peritoneal cancer is evidence of cancer.
- Sclerotic bone requires explicit or prior implicit malignancy. Likely treated bone metastases are evidence of cancer but image_overall should usually be 5 unless response is explicit.
- Mammogram BI-RADS 4C, 5, or 6 indicates cancer.

Ambiguous term rules:
- Treat as yes evidence when cancer is described as concerning for, comparable with, compatible with, consistent with, favors, likely/most likely, presumed, probable, suggests, suspected, suspicious for, typical of, worrisome for, or other wording indicating more than 50% chance.
- Treat as uncertain when terms include bears watching with mention of cancer, cannot be ruled out, follow-up recommended with mention of cancer, equivocal, possible, potentially, questionable, rule out, may/could with mention of cancer, or other wording indicating 50% chance or lower.
- Treat as no evidence when terms include bears watching with no mention of cancer or follow-up recommended with no mention of cancer.

Cancer status rules:
- Prefer an overall/summary statement. If absent, use cancer site changes in the Impression.
- Do not use effusion/ascites/fluid changes to determine response even if malignant.
- Use only cancer status or size terms, not visualization terms such as more prominent/conspicuous or increased avidity.
- Responding: all sites decreasing/resolved, or some stable and some decreasing/resolved; treatment effect indicates response.
- Stable: no change mentioned for any cancer site.
- Mixed: some sites new/increasing/progressing and some decreasing/responding/resolved.
- Progressing: all sites increasing, or some stable and some increasing, new disease site, recurrence after NED/no evidence of disease, or increasing tumor burden.
- Not stated/indeterminate: no status for any site, entire scan indeterminate, or treated bone metastases without explicit response/progression.
"""


ICDO_TOPOGRAPHY_GUIDE = """
ICD-O-3 site/topography guide for image_cancer_sites:
- Use the anatomic site code itself. ICD-O topography does not use ICD-10 secondary/metastatic site codes such as C78._ or C79._.
- Prefer the most specific subsite documented in the Impression. If the organ is known but no subsite is stated, use the NOS code for that organ.
- For lymph nodes, use the lymph node region codes C77.0-C77.9.
- For bone involvement, use C40._ for limb bones and C41._ for skull, spine, ribs/sternum/clavicle, pelvis, or bone NOS. Spine/vertebra maps to C41.2.
- For chest wall/axilla soft-tissue involvement, use C49.3 unless the report clearly describes lymph nodes, breast, lung, pleura, or bone instead.
- For malignant fluid collections, code the involved anatomic site only when this codebook supplies a suitable topography code, e.g. pleura C38.4 or peritoneum C48.2. If the local project uses separate fluid-specific values not listed here, keep the site_text and set icdo_topography_code to null.
- If the site is cancer but no code below fits confidently, set icdo_topography_code to null and explain the uncertainty.

PRISSMM appendix cancer-type site codes:
C67.0 Trigone of bladder; C67.1 Dome of bladder; C67.2 Lateral wall of bladder; C67.3 Anterior wall of bladder; C67.4 Posterior wall of bladder; C67.5 Bladder neck; C67.6 Ureteric orifice; C67.7 Urachus; C67.8 Overlapping lesion of bladder; C67.9 Bladder NOS.
C50.0 Nipple; C50.1 Central portion of breast; C50.2 Upper-inner quadrant of breast; C50.3 Lower-inner quadrant of breast; C50.4 Upper-outer quadrant of breast; C50.5 Lower-outer quadrant of breast; C50.6 Axillary tail of breast; C50.8 Overlapping lesion of breast; C50.9 Breast NOS.
C18.0 Cecum; C18.1 Appendix; C18.2 Ascending colon; C18.3 Hepatic flexure of colon; C18.4 Transverse colon; C18.5 Splenic flexure of colon; C18.6 Descending colon; C18.7 Sigmoid colon; C18.8 Overlapping lesion of colon; C18.9 Colon NOS; C19.9 Rectosigmoid junction; C20.9 Rectum NOS.
C34.0 Main bronchus; C34.1 Upper lobe lung; C34.2 Middle lobe lung; C34.3 Lower lobe lung; C34.8 Overlapping lesion of lung; C34.9 Lung NOS.
C25.0 Head of pancreas; C25.1 Body of pancreas; C25.2 Tail of pancreas; C25.3 Pancreatic duct; C25.7 Other specified parts of pancreas; C25.8 Overlapping lesion of pancreas; C25.9 Pancreas NOS.
C61.9 Prostate gland; C64.9 Kidney NOS; C56.9 Ovary; C57.0 Fallopian tube; C48.1 Specified parts of peritoneum; C48.2 Peritoneum NOS; C48.8 Overlapping lesion of retroperitoneum and peritoneum.

Common imaging involved-site codes:
Head, neck, CNS, and senses:
C00.9 Lip NOS; C01.9 Base of tongue; C02.9 Tongue NOS; C03.9 Gum NOS; C04.9 Floor of mouth NOS; C05.9 Palate NOS; C06.9 Mouth NOS; C07.9 Parotid gland; C08.9 Major salivary gland NOS; C09.9 Tonsil NOS; C10.9 Oropharynx NOS; C11.9 Nasopharynx NOS; C12.9 Pyriform sinus; C13.9 Hypopharynx NOS; C14.0 Pharynx NOS; C14.2 Waldeyer ring; C14.8 Overlapping lesion of lip/oral cavity/pharynx.
C30.0 Nasal cavity; C30.1 Middle ear; C31.0 Maxillary sinus; C31.1 Ethmoid sinus; C31.2 Frontal sinus; C31.3 Sphenoid sinus; C31.8 Overlapping lesion of accessory sinuses; C31.9 Accessory sinus NOS.
C69.0 Conjunctiva; C69.1 Cornea; C69.2 Retina; C69.3 Choroid; C69.4 Ciliary body; C69.5 Lacrimal gland; C69.6 Orbit NOS; C69.8 Overlapping lesion of eye/adnexa; C69.9 Eye NOS.
C70.0 Cerebral meninges; C70.1 Spinal meninges; C70.9 Meninges NOS.
C71.0 Cerebrum; C71.1 Frontal lobe; C71.2 Temporal lobe; C71.3 Parietal lobe; C71.4 Occipital lobe; C71.5 Ventricle NOS; C71.6 Cerebellum NOS; C71.7 Brain stem; C71.8 Overlapping lesion of brain; C71.9 Brain NOS.
C72.0 Spinal cord; C72.1 Cauda equina; C72.2 Olfactory nerve; C72.3 Optic nerve; C72.4 Acoustic nerve; C72.5 Cranial nerve NOS; C72.8 Overlapping lesion of brain/CNS; C72.9 Nervous system NOS.

Thorax and respiratory:
C32.9 Larynx NOS; C33.9 Trachea; C34.0 Main bronchus; C34.1 Upper lobe lung; C34.2 Middle lobe lung; C34.3 Lower lobe lung; C34.8 Overlapping lesion of lung; C34.9 Lung NOS.
C37.9 Thymus; C38.0 Heart; C38.1 Anterior mediastinum; C38.2 Posterior mediastinum; C38.3 Mediastinum NOS; C38.4 Pleura NOS; C38.8 Overlapping lesion of heart/mediastinum/pleura; C39.9 Respiratory/intrathoracic organ NOS.

Digestive abdomen and pelvis:
C15.9 Esophagus NOS; C16.9 Stomach NOS; C17.0 Duodenum; C17.1 Jejunum; C17.2 Ileum; C17.9 Small intestine NOS.
C18.0 Cecum; C18.1 Appendix; C18.2 Ascending colon; C18.3 Hepatic flexure; C18.4 Transverse colon; C18.5 Splenic flexure; C18.6 Descending colon; C18.7 Sigmoid colon; C18.8 Overlapping lesion of colon; C18.9 Colon NOS; C19.9 Rectosigmoid junction; C20.9 Rectum NOS.
C21.0 Anus NOS; C21.1 Anal canal; C21.2 Cloacogenic zone; C21.8 Overlapping lesion of rectum/anus/anal canal.
C22.0 Liver; C22.1 Intrahepatic bile duct; C23.9 Gallbladder; C24.0 Extrahepatic bile duct; C24.1 Ampulla of Vater; C24.8 Overlapping lesion of biliary tract; C24.9 Biliary tract NOS.
C25.0 Head of pancreas; C25.1 Body of pancreas; C25.2 Tail of pancreas; C25.3 Pancreatic duct; C25.4 Islets of Langerhans; C25.7 Other specified parts of pancreas; C25.8 Overlapping lesion of pancreas; C25.9 Pancreas NOS.
C26.0 Intestinal tract NOS; C26.8 Overlapping lesion of digestive system; C26.9 Gastrointestinal tract NOS.

Breast and female genital:
C50.0 Nipple; C50.1 Central portion of breast; C50.2 Upper-inner quadrant of breast; C50.3 Lower-inner quadrant of breast; C50.4 Upper-outer quadrant of breast; C50.5 Lower-outer quadrant of breast; C50.6 Axillary tail of breast; C50.8 Overlapping lesion of breast; C50.9 Breast NOS.
C51.9 Vulva NOS; C52.9 Vagina NOS; C53.9 Cervix uteri; C54.1 Endometrium; C54.2 Myometrium; C54.3 Fundus uteri; C54.9 Corpus uteri; C55.9 Uterus NOS; C56.9 Ovary.
C57.0 Fallopian tube; C57.1 Broad ligament; C57.2 Round ligament; C57.3 Parametrium; C57.4 Uterine adnexa; C57.7 Other specified female genital organs; C57.8 Overlapping lesion of female genital organs; C57.9 Female genital tract NOS; C58.9 Placenta.

Male genital and urinary:
C60.9 Penis NOS; C61.9 Prostate gland; C62.9 Testis NOS; C63.2 Scrotum NOS; C63.9 Male genital organ NOS.
C64.9 Kidney NOS; C65.9 Renal pelvis; C66.9 Ureter; C67.0 Trigone of bladder; C67.1 Dome of bladder; C67.2 Lateral wall of bladder; C67.3 Anterior wall of bladder; C67.4 Posterior wall of bladder; C67.5 Bladder neck; C67.6 Ureteric orifice; C67.7 Urachus; C67.8 Overlapping lesion of bladder; C67.9 Bladder NOS.
C68.0 Urethra; C68.1 Paraurethral gland; C68.8 Overlapping lesion of urinary organs; C68.9 Urinary system NOS.

Bone, soft tissue, skin, hematopoietic, endocrine, lymph nodes, and ill-defined sites:
C40.0 Long bones of upper limb/shoulder; C40.1 Short bones of upper limb; C40.2 Long bones of lower limb/hip; C40.3 Short bones of lower limb; C40.8 Overlapping lesion of bones/joints/articular cartilage of limbs; C40.9 Bone of limb NOS.
C41.0 Skull and face bones; C41.1 Mandible; C41.2 Vertebral column; C41.3 Rib/sternum/clavicle; C41.4 Pelvic bones/sacrum/coccyx; C41.8 Overlapping lesion of bones/joints/articular cartilage; C41.9 Bone NOS.
C42.0 Blood; C42.1 Bone marrow; C42.2 Spleen; C42.3 Reticuloendothelial system NOS; C42.4 Hematopoietic system NOS.
C44.0 Skin of lip; C44.1 Eyelid; C44.2 External ear; C44.3 Skin of other/unspecified parts of face; C44.4 Skin of scalp and neck; C44.5 Skin of trunk; C44.6 Skin of upper limb/shoulder; C44.7 Skin of lower limb/hip; C44.8 Overlapping lesion of skin; C44.9 Skin NOS.
C47.0 Peripheral nerves/autonomic nervous system of head/face/neck; C47.1 Upper limb/shoulder peripheral nerves; C47.2 Lower limb/hip peripheral nerves; C47.3 Thorax peripheral nerves; C47.4 Abdomen peripheral nerves; C47.5 Pelvis peripheral nerves; C47.6 Trunk NOS peripheral nerves; C47.8 Overlapping peripheral nerves; C47.9 Peripheral/autonomic nervous system NOS.
C48.0 Retroperitoneum; C48.1 Specified parts of peritoneum; C48.2 Peritoneum NOS; C48.8 Overlapping lesion of retroperitoneum and peritoneum.
C49.0 Connective/subcutaneous/soft tissue of head/face/neck; C49.1 Upper limb/shoulder soft tissue; C49.2 Lower limb/hip soft tissue; C49.3 Thorax/chest wall/axilla soft tissue; C49.4 Abdomen soft tissue; C49.5 Pelvis soft tissue; C49.6 Trunk NOS soft tissue; C49.8 Overlapping soft tissue; C49.9 Soft tissue NOS.
C74.0 Cortex of adrenal gland; C74.1 Medulla of adrenal gland; C74.9 Adrenal gland NOS.
C75.0 Parathyroid gland; C75.1 Pituitary gland; C75.2 Craniopharyngeal duct; C75.3 Pineal gland; C75.4 Carotid body; C75.5 Aortic body; C75.8 Overlapping lesion of endocrine glands; C75.9 Endocrine gland NOS.
C76.0 Head/face/neck NOS; C76.1 Thorax NOS; C76.2 Abdomen NOS; C76.3 Pelvis NOS; C76.4 Upper limb NOS; C76.5 Lower limb NOS; C76.7 Other ill-defined sites; C76.8 Overlapping lesion of ill-defined sites.
C77.0 Lymph nodes of head/face/neck; C77.1 Intrathoracic lymph nodes; C77.2 Intra-abdominal lymph nodes; C77.3 Lymph nodes of axilla or arm; C77.4 Lymph nodes of inguinal region or leg; C77.5 Pelvic lymph nodes; C77.8 Lymph nodes of multiple regions; C77.9 Lymph node NOS.
C80.9 Unknown primary site.
"""


SCHEMA = """
Return this JSON shape:
{
  "record_id": string,
  "should_curate": {"value": true|false|null, "reason": string, "evidence": string|null},
  "image_inst_performed": {"code": 1|2|null, "label": string|null, "evidence": string|null},
  "image_inst_interpreted": {"code": 1|2|null, "label": string|null, "evidence": string|null},
  "image_scan_date": {"value": string|null, "evidence": string|null},
  "image_report_date": {"value": string|null, "evidence": string|null},
  "image_scan_type": {"code": 1|3|5|13|7|9|11|20|null, "label": string|null, "evidence": string|null},
  "image_scan_type_other": {"value": string|null, "evidence": string|null},
  "image_scan_sites": [{"label": "Brain/Head|Spine|Neck|Chest|Abdomen|Pelvis|Extremity|Full Body", "evidence": string|null}],
  "image_ca": {"code": 1|2|3|4|null, "label": string|null, "evidence": string|null},
  "image_overall": {"code": 1|2|3|4|5|null, "label": string|null, "evidence": string|null},
  "image_cancer_sites": [{"site_text": string, "icdo_topography_code": string|null, "icdo_topography_label": string|null, "basis": string, "evidence": string}],
  "image_ref_scan_date": {"value": string|null, "evidence": string|null},
  "allowed_sections_used": [string],
  "uncertainties": [string]
}
"""


def build_messages(record: Record, args: argparse.Namespace) -> list[dict[str, str]]:
    metadata_fields = [field.strip() for field in (args.metadata_fields or "").split(",") if field.strip()] or None
    metadata = metadata_for_prompt(record, args.text_field, metadata_fields=metadata_fields)
    user_prompt = f"""{GUIDANCE}

{ICDO_TOPOGRAPHY_GUIDE}

{SCHEMA}

Record id: {record.record_id}

Available metadata:
{metadata}

Report text:
\"\"\"
{record.text}
\"\"\"
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Label imaging/radiology reports with a vLLM OpenAI-compatible server.")
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
