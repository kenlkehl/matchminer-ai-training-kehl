#!/usr/bin/env python3
"""Build single-task SFT data for iterative patient summarization.

The source data is the final parquet produced by ``6_summarize_patients.py``.
That file contains the prior summary, model reasoning, and updated summary for
each patient chunk. The clinical chunk text is recovered from the matching
``prepared_chunks.parquet`` cache created by the same summarization run.

Each training row is rendered with the Gemma 4 chat format and contains:

  user prior summary + next clinical record segment
  assistant <|channel>thought reasoning trace <channel|> final updated summary

The output parquet has one column, ``text``. The script can also tokenize that
parquet into a Hugging Face dataset with prompt-masked labels for SFT.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets.arrow_writer import ArrowWriter
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_TOKENIZER = "google/gemma-4-E2B-it"
DEFAULT_REASONING_PARSER = "gemma4"
DEFAULT_INPUT = "../data/no_phi/patient_serial_summaries.parquet"
DEFAULT_CHUNKS = "../data/no_phi/summary_shards/prepared_chunks.parquet"
DEFAULT_OUTPUT_DIR = "../data/no_phi/patient_summarization_training_data"
DEFAULT_NUM_WORKERS = min(os.cpu_count() or 1, 32)
GEMMA4_THINKING_START = "<|channel>thought\n"
GEMMA4_THINKING_END = "<channel|>"
GEMMA4_TURN_END = "<turn|>\n"


def clean_scalar(value: Any) -> Any | None:
    """Return None for pandas/Arrow missing values or empty strings."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if str(value).strip() == "":
        return None
    return value


def token_len(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def truncate_field(text: str, max_tokens: int, tokenizer) -> str:
    """Token-truncate text with a head+tail strategy."""
    toks = tokenizer(text, add_special_tokens=False).input_ids
    if len(toks) <= max_tokens:
        return text
    half = max(1, max_tokens // 2)
    return tokenizer.decode(toks[:half]) + " ... " + tokenizer.decode(toks[-half:])


def require_columns(path: str, schema_names: set[str], columns: list[str]) -> None:
    missing = [col for col in columns if col not in schema_names]
    if missing:
        available = ", ".join(sorted(schema_names))
        raise ValueError(
            f"{path} is missing required columns: {missing}. Available columns: {available}"
        )


def build_user_content(
    prior_summary: str | None,
    first_date: str,
    last_date: str,
    chunk_text: str,
) -> str:
    """Build the current patient-summary prompt from ``6_summarize_patients.py``."""
    prior_summary_text = (
        prior_summary if prior_summary else "None - this is the first segment for this patient"
    )

    return f"""You are an experienced clinical oncology history summarization bot.

You are maintaining a running summary of the history of a patient's active cancer(s) in their electronic health record.
You will be given:
1. A PRIOR SUMMARY of the patient's history (may be empty for the first segment)
2. THE NEXT SEGMENT of the patient's clinical record (may contain multiple notes with dates)

Your task:
- Update the summary to incorporate any new relevant information from this segment of the clinical record
- If the segment contains no information that would change the summary, output the prior summary exactly as-is
- The patient may not yet have a cancer diagnosis. If not, state "No cancer diagnosis documented as of [date]" and summarize relevant medical history that might be relevant to a future oncology workup.

Document the following sections, and ONLY the following sections:
--(start of sections)
Age: (patient's most recent age)
Sex: (patient's sex)
Cancer type: (patient's cancer type/primary site (eg breast cancer, lung cancer, etc))
Histology: (patient's histology (eg adenocarcinoma, squamous carcinoma, etc))
Current extent: (patient's current extent (localized, advanced, metastatic, etc); this is also where tumor markers for following disease status, such as CEA or PSA, should be documented if relevant. Don't list every such marker the patient has had checked over time, though, because these can get lengthy; just list the most recent value and trend if relevant to disease status.)
Biomarkers: (genomic results, protein expression, etc, relevant for informing treatment selection. Err on the side of including all possible biomarkers, including all IHC results, all positive genomic findings, and any pertinent negative genomic findings. However, critically, standard lab values (eg CBC, CMP, LFTs, etc) MUST NOT be included in this section - only tumor biomarkers relevant to cancer treatment selection should be included. Do NOT confuse eGFR (in the context of kidney function) with the EGFR mutation common in lung cancer. Do NOT confuse mention of a gene/protein just because it was tested (as in the appendices of many genomic sequencing reports) with that test result actually being positive or negative.)
Treatment history: (surgery, radiation, chemotherapy/targeted therapy/immunotherapy, etc, including start and stop dates, and best response if noted. Treatment history should be provided chronologically. For cancer drug names, use generic names whenever you know them. Expand abbreviations where possible ,(eg "carbo" -> "carboplatin", "pembro" -> "pembrolizumab", "AC/T" -> "doxorubicin + cyclophosphamide followed by paclitaxel", etc)

Boilerplate conditions:
(any history of conditions that might meet common "boilerplate" exclusion criteria for clinical trials, such as uncontrolled brain metastases, poor performance status, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, liver dysfunction, HIV or hepatitis infection, prior unrelated cancer diagnoses, etc.)

Clearly separate the "boilerplate" section by adding a newline after the patient history; then the "Boilerplate conditions:' text VERBATIM; then another newline; and then the boilerplate condition output text.
--(end of sections)

Do not consider localized basal cell or squamous carcinomas of the skin, or colon polyps, to be cancers for your purposes.
Do not include the patient's name, but do include relevant dates whenever documented.
If a patient has more than one active cancer, document the active cancers one at a time. List the most active cancer first, followed by any other active cancers. Within each active cancer, events should be in chronological order. Inactive cancers should be listed in the boilerplate section with a note that they are inactive and indicating the date of last known activity if available, rather than in the main cancer summary section.
CRITICAL: Format your response as free text ONLY. Do NOT output markdown, Unicode, or tables.

Here is an example of the desired output format:

Age: 70
Sex: Male
Cancer type: Lung cancer
Histology: Adenocarcinoma
Current extent: Metastatic
Biomarkers: PD-L1 75%, KRAS G12C mutant
Treatment history:
# 1/5/2020-2/5/2021: carboplatin/pemetrexed/pembrolizumab; best response stable disease
# 1/2021: Palliative radiation for progressive spinal metastases
# 3/2021-present: docetaxel; achieved partial response, ongoing as of last note

Boilerplate conditions:
ECOG 1. Remote history of prostate cancer (inactive).

Reference: common systemic therapy regimen abbreviations (use this list to expand abbreviations into generic drug names whenever they appear in the clinical record):
- AC: doxorubicin + cyclophosphamide
- AC-T / AC followed by T: doxorubicin + cyclophosphamide followed by paclitaxel
- ddAC-T: dose-dense doxorubicin + cyclophosphamide followed by paclitaxel
- TC: docetaxel + cyclophosphamide
- TCH: docetaxel + carboplatin + trastuzumab
- TCHP: docetaxel + carboplatin + trastuzumab + pertuzumab
- THP: paclitaxel + trastuzumab + pertuzumab
- HP: trastuzumab + pertuzumab
- T-DM1: ado-trastuzumab emtansine
- T-DXd: trastuzumab deruxtecan
- CMF: cyclophosphamide + methotrexate + 5-fluorouracil
- CAF / FAC: cyclophosphamide + doxorubicin + 5-fluorouracil
- FEC: 5-fluorouracil + epirubicin + cyclophosphamide
- CDK4/6i: CDK4/6 inhibitor (e.g., palbociclib, ribociclib, abemaciclib)
- AI: aromatase inhibitor (e.g., anastrozole, letrozole, exemestane); note this abbreviation can also mean doxorubicin + ifosfamide in sarcoma contexts - disambiguate by cancer type
- FOLFOX: 5-fluorouracil + leucovorin + oxaliplatin
- FOLFIRI: 5-fluorouracil + leucovorin + irinotecan
- FOLFOXIRI / FOLFIRINOX: 5-fluorouracil + leucovorin + oxaliplatin + irinotecan
- mFOLFIRINOX: modified FOLFIRINOX (reduced doses of 5-fluorouracil + leucovorin + oxaliplatin + irinotecan)
- CAPOX / XELOX: capecitabine + oxaliplatin
- CAPIRI / XELIRI: capecitabine + irinotecan
- DCF: docetaxel + cisplatin + 5-fluorouracil
- FLOT: 5-fluorouracil + leucovorin + oxaliplatin + docetaxel
- ECF: epirubicin + cisplatin + 5-fluorouracil
- ECX: epirubicin + cisplatin + capecitabine
- Gem/Cis: gemcitabine + cisplatin
- Gem/Carbo: gemcitabine + carboplatin
- Gem/Abraxane / Gem/nab-pac: gemcitabine + nab-paclitaxel
- GemOx: gemcitabine + oxaliplatin
- Carbo/Tax: carboplatin + paclitaxel
- EP / PE: cisplatin + etoposide
- CE: carboplatin + etoposide
- BEP / PEB: bleomycin + etoposide + cisplatin
- VIP: etoposide + ifosfamide + cisplatin
- TIP: paclitaxel + ifosfamide + cisplatin
- MVAC / ddMVAC: methotrexate + vinblastine + doxorubicin + cisplatin (dose-dense variant)
- GC: gemcitabine + cisplatin (or gemcitabine + carboplatin in bladder cancer)
- EV: enfortumab vedotin
- EV+P: enfortumab vedotin + pembrolizumab
- CHOP: cyclophosphamide + doxorubicin + vincristine + prednisone
- R-CHOP: rituximab + cyclophosphamide + doxorubicin + vincristine + prednisone
- EPOCH / R-EPOCH: etoposide + prednisone + vincristine + cyclophosphamide + doxorubicin (+/- rituximab)
- DA-EPOCH-R: dose-adjusted EPOCH + rituximab
- ABVD: doxorubicin + bleomycin + vinblastine + dacarbazine
- BEACOPP: bleomycin + etoposide + doxorubicin + cyclophosphamide + vincristine + procarbazine + prednisone
- BV-AVD: brentuximab vedotin + doxorubicin + vinblastine + dacarbazine
- ICE / R-ICE: ifosfamide + carboplatin + etoposide (+/- rituximab)
- DHAP / R-DHAP: dexamethasone + high-dose cytarabine + cisplatin (+/- rituximab)
- ESHAP: etoposide + methylprednisolone + cytarabine + cisplatin
- GDP: gemcitabine + dexamethasone + cisplatin
- BR: bendamustine + rituximab
- HyperCVAD: cyclophosphamide + vincristine + doxorubicin + dexamethasone, alternating with high-dose methotrexate + cytarabine
- 7+3: cytarabine (7 days) + daunorubicin or idarubicin (3 days), induction for AML
- HiDAC: high-dose cytarabine
- VRd / RVd: bortezomib + lenalidomide + dexamethasone
- KRd: carfilzomib + lenalidomide + dexamethasone
- DRd: daratumumab + lenalidomide + dexamethasone
- DVd: daratumumab + bortezomib + dexamethasone
- D-VRd: daratumumab + bortezomib + lenalidomide + dexamethasone
- VAD: vincristine + doxorubicin + dexamethasone
- MAP: methotrexate + doxorubicin + cisplatin (osteosarcoma)
- VAC: vincristine + actinomycin-D + cyclophosphamide
- VDC/IE: vincristine + doxorubicin + cyclophosphamide alternating with ifosfamide + etoposide (Ewing sarcoma)
- AI: doxorubicin + ifosfamide (sarcoma)
- Common single-agent abbreviations: pembro = pembrolizumab; nivo = nivolumab; ipi = ipilimumab; atezo = atezolizumab; durva = durvalumab; cemi = cemiplimab; dostarlimab; cetux = cetuximab; pani = panitumumab; bev = bevacizumab; ram = ramucirumab; trastuzumab = Herceptin; pertuzumab = Perjeta; carbo = carboplatin; cis = cisplatin; tax / pac = paclitaxel; doce = docetaxel; gem = gemcitabine; cape = capecitabine; 5-FU = fluorouracil; oxali = oxaliplatin; iri = irinotecan; etop = etoposide; doxo / adria = doxorubicin; cyclo / CTX = cyclophosphamide; ifos = ifosfamide; vinc / VCR = vincristine; len = lenalidomide; pom = pomalidomide; bort / Velcade = bortezomib; carfilzomib = Kyprolis; dara = daratumumab; ven = venetoclax.
- Ipi/Nivo: ipilimumab + nivolumab
- Chemo-IO: chemotherapy combined with immune checkpoint inhibitor (specify the agents based on context)

If an abbreviation in the record is not on this list and you are not confident of its expansion, write the abbreviation as-is rather than guessing.

The following are the patient's data.
---
PRIOR SUMMARY:
{prior_summary_text}

NEXT CLINICAL RECORD SEGMENT (covering {first_date} to {last_date}):
{chunk_text}
---
Now, write your updated summary, or if there is no new relevant information, output the prior summary exactly as it was.
If any information is still relevant but is unchanged, just restate it in the updated summary, but do NOT state "no change" or similar - just produce the updated summary text as if you were writing it fresh, incorporating any new information but keeping relevant old information, without calling out what changed vs what stayed the same from the prior summary.
You may update the old summary content in your output if the new information demonstrates that there was an error in the old output.
You may sometimes encounter contradictory information across notes (eg different biomarker results, or different cancer stage descriptions) - in that case, use your best judgment to determine which information is most likely to be correct based on the dates and context, and update the summary accordingly to reflect the most likely current state of the patient.
Do not add preceding text before the abstraction, and do not add commentary afterwards."""


def build_prompt_messages(
    prior_summary: str | None,
    first_date: str,
    last_date: str,
    chunk_text: str,
    system_content: str,
) -> list[dict[str, str]]:
    user_content = build_user_content(
        prior_summary=prior_summary,
        first_date=first_date,
        last_date=last_date,
        chunk_text=chunk_text,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def format_assistant_content(reasoning: str, summary: str, reasoning_parser: str) -> str:
    """Format the supervised assistant turn for the selected reasoning parser."""
    reasoning = reasoning.strip()
    summary = summary.strip()
    if reasoning_parser == "gemma4":
        thinking = f"{reasoning}\n" if reasoning else ""
        return (
            f"{GEMMA4_THINKING_START}{thinking}"
            f"{GEMMA4_THINKING_END}{summary}{GEMMA4_TURN_END}"
        )
    return f"<think>\n{reasoning}\n</think>\n{summary}"


def render_gemma4_prompt(messages: list[dict[str, str]], enable_thinking: bool) -> str:
    """Render the simple text-only Gemma 4 prompt when no HF chat template exists."""
    pieces = ["<bos>"]
    start_idx = 0
    system_content = ""
    if messages and messages[0]["role"] in {"system", "developer"}:
        system_content = messages[0].get("content", "").strip()
        start_idx = 1

    if enable_thinking or system_content:
        pieces.append("<|turn>system\n")
        if enable_thinking:
            pieces.append("<|think|>\n")
        if system_content:
            pieces.append(system_content)
        pieces.append(GEMMA4_TURN_END)

    for message in messages[start_idx:]:
        role = "model" if message["role"] == "assistant" else message["role"]
        pieces.append(
            f"<|turn>{role}\n"
            f"{message.get('content', '').strip()}{GEMMA4_TURN_END}"
        )

    pieces.append("<|turn>model\n")
    return "".join(pieces)


def render_prompt_text(
    tokenizer,
    messages: list[dict[str, str]],
    reasoning_parser: str,
) -> str:
    if reasoning_parser == "gemma4" and not getattr(tokenizer, "chat_template", None):
        return render_gemma4_prompt(messages, enable_thinking=True)
    return tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
    )


def render_full_training_text(
    tokenizer,
    messages: list[dict[str, str]],
    reasoning: str,
    summary: str,
    reasoning_parser: str,
) -> str:
    assistant_content = format_assistant_content(
        reasoning=reasoning,
        summary=summary,
        reasoning_parser=reasoning_parser,
    )
    if reasoning_parser == "gemma4":
        return (
            render_prompt_text(tokenizer, messages, reasoning_parser)
            + assistant_content
        )

    return tokenizer.apply_chat_template(
        conversation=messages + [{"role": "assistant", "content": assistant_content}],
        tokenize=False,
        enable_thinking=True,
    )


def render_training_text(
    tokenizer,
    prior_summary: str | None,
    first_date: str,
    last_date: str,
    chunk_text: str,
    reasoning: str,
    summary: str,
    system_content: str,
    max_seq_length: int,
    reasoning_parser: str,
) -> tuple[str | None, bool]:
    """Render one example, truncating only the source chunk if needed."""
    original_chunk = chunk_text
    working_chunk = chunk_text
    was_truncated = False

    for _ in range(4):
        messages = build_prompt_messages(
            prior_summary=prior_summary,
            first_date=first_date,
            last_date=last_date,
            chunk_text=working_chunk,
            system_content=system_content,
        )
        text = render_full_training_text(
            tokenizer=tokenizer,
            messages=messages,
            reasoning=reasoning,
            summary=summary,
            reasoning_parser=reasoning_parser,
        )
        total = token_len(text, tokenizer)
        if total <= max_seq_length:
            return text, was_truncated

        chunk_tokens = token_len(working_chunk, tokenizer)
        overhead = total - chunk_tokens
        budget = max_seq_length - overhead - 16
        if budget <= 0:
            return None, was_truncated

        working_chunk = truncate_field(original_chunk, budget, tokenizer)
        was_truncated = True

    return None, was_truncated


def load_joined_training_frame(args: argparse.Namespace) -> pd.DataFrame:
    summary_pf = pq.ParquetFile(args.input_parquet)
    chunk_pf = pq.ParquetFile(args.chunks_parquet)
    summary_schema = set(summary_pf.schema_arrow.names)
    chunk_schema = set(chunk_pf.schema_arrow.names)

    summary_cols = [
        args.patient_id_col,
        args.chunk_index_col,
        args.first_date_col,
        args.last_date_col,
        args.prior_summary_col,
        args.reasoning_col,
        args.summary_col,
    ]
    chunk_cols = [
        args.chunk_patient_id_col,
        args.chunk_local_idx_col,
        args.first_date_col,
        args.last_date_col,
        args.chunk_text_col,
    ]

    require_columns(args.input_parquet, summary_schema, summary_cols)
    require_columns(args.chunks_parquet, chunk_schema, chunk_cols)

    print(f"Reading {args.input_parquet}")
    print(f"Summary rows: {summary_pf.metadata.num_rows}")
    summaries = pd.read_parquet(args.input_parquet, columns=summary_cols)

    print(f"Reading {args.chunks_parquet}")
    print(f"Chunk rows: {chunk_pf.metadata.num_rows}")
    chunks = pd.read_parquet(args.chunks_parquet, columns=chunk_cols).rename(
        columns={
            args.chunk_patient_id_col: args.patient_id_col,
            args.chunk_local_idx_col: args.chunk_index_col,
        }
    )

    key_cols = [
        args.patient_id_col,
        args.chunk_index_col,
        args.first_date_col,
        args.last_date_col,
    ]

    summaries[args.patient_id_col] = summaries[args.patient_id_col].astype(str)
    chunks[args.patient_id_col] = chunks[args.patient_id_col].astype(str)
    summaries[args.chunk_index_col] = pd.to_numeric(
        summaries[args.chunk_index_col], errors="raise"
    ).astype("int64")
    chunks[args.chunk_index_col] = pd.to_numeric(
        chunks[args.chunk_index_col], errors="raise"
    ).astype("int64")

    summary_dupes = int(summaries.duplicated(key_cols).sum())
    chunk_dupes = int(chunks.duplicated(key_cols).sum())
    if summary_dupes or chunk_dupes:
        raise ValueError(
            "Expected one-to-one summary/chunk keys. "
            f"Summary duplicate keys: {summary_dupes}; chunk duplicate keys: {chunk_dupes}."
        )

    joined = summaries.merge(
        chunks[key_cols + [args.chunk_text_col]],
        on=key_cols,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = joined["_merge"].ne("both")
    if missing.any():
        sample = joined.loc[missing, key_cols].head(5).to_dict(orient="records")
        raise ValueError(
            f"Could not find chunk_text for {int(missing.sum())} summary rows. "
            f"Example missing keys: {sample}"
        )

    joined = joined.drop(columns=["_merge"])
    print(f"Joined rows: {len(joined)}")
    return joined


def flush_texts(writer: pq.ParquetWriter, texts: list[str]) -> None:
    table = pa.Table.from_pydict(
        {"text": texts},
        schema=pa.schema([pa.field("text", pa.large_string())]),
    )
    writer.write_table(table)


def build_text_parquet(args: argparse.Namespace, tokenizer) -> tuple[int, int, int]:
    out_path = Path(args.output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_path} exists. Pass --overwrite to replace it.")
        out_path.unlink()

    joined = load_joined_training_frame(args)

    writer = pq.ParquetWriter(
        out_path,
        pa.schema([pa.field("text", pa.large_string())]),
    )

    written = 0
    dropped = 0
    truncated = 0
    pending: list[str] = []

    try:
        for row_number, row in joined.iterrows():
            if args.max_examples is not None and written >= args.max_examples:
                break

            chunk_text = clean_scalar(row.get(args.chunk_text_col))
            reasoning = clean_scalar(row.get(args.reasoning_col))
            summary = clean_scalar(row.get(args.summary_col))
            prior_summary = clean_scalar(row.get(args.prior_summary_col))
            if chunk_text is None or summary is None:
                dropped += 1
                continue
            if not args.keep_error_summaries and str(summary).strip().startswith("ERROR:"):
                dropped += 1
                continue
            if reasoning is None and not args.keep_empty_reasoning:
                dropped += 1
                continue

            text, was_truncated = render_training_text(
                tokenizer=tokenizer,
                prior_summary=None if prior_summary is None else str(prior_summary),
                first_date=str(row.get(args.first_date_col, "unknown date")),
                last_date=str(row.get(args.last_date_col, "unknown date")),
                chunk_text=str(chunk_text),
                reasoning="" if reasoning is None else str(reasoning),
                summary=str(summary),
                system_content=args.system_content,
                max_seq_length=args.max_seq_length,
                reasoning_parser=args.resolved_reasoning_parser,
            )
            if text is None:
                dropped += 1
                continue
            pending.append(text)
            written += 1
            truncated += int(was_truncated)

            if len(pending) >= args.writer_batch_size:
                flush_texts(writer, pending)
                pending.clear()
                print(f"  Wrote {written} examples...")

        if pending:
            flush_texts(writer, pending)
    finally:
        writer.close()

    print(f"Text parquet: {out_path}")
    print(f"Built {written} examples; truncated {truncated}; dropped {dropped}.")
    return written, truncated, dropped


def find_last_subsequence(seq: list[int], subseq: list[int]) -> int:
    if not subseq:
        return len(seq)
    seq_bytes = struct.pack(f"{len(seq)}I", *seq)
    sub_bytes = struct.pack(f"{len(subseq)}I", *subseq)
    pos = seq_bytes.rfind(sub_bytes)
    if pos < 0:
        return -1
    return pos // 4


def assistant_header_ids(tokenizer) -> list[list[int]]:
    headers = [
        "<|turn>model\n",
        "<|im_start|>assistant\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ]
    encoded = []
    for header in headers:
        ids = tokenizer.encode(header, add_special_tokens=False)
        if ids:
            encoded.append(ids)
    return encoded


def mask_prompt(input_ids: list[int], header_options: list[list[int]]) -> tuple[list[int], bool]:
    labels = list(input_ids)
    best_idx = -1
    best_len = 0
    for header_ids in header_options:
        idx = find_last_subsequence(input_ids, header_ids)
        if idx > best_idx:
            best_idx = idx
            best_len = len(header_ids)
    if best_idx < 0:
        return labels, False
    mask_end = best_idx + best_len
    labels[:mask_end] = [-100] * mask_end
    return labels, True


def tokenize_worker(args_tuple):
    (
        source_parquet,
        row_group_indices,
        tokenizer_name,
        max_seq_length,
        arrow_path,
        header_options,
        batch_size,
        worker_idx,
        num_workers,
    ) = args_tuple
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    pf = pq.ParquetFile(source_parquet)
    writer = ArrowWriter(path=arrow_path)
    total = sum(pf.metadata.row_group(i).num_rows for i in row_group_indices)
    rows_done = 0
    masked_count = 0
    unmasked_count = 0
    prefix = f"[Worker {worker_idx + 1}/{num_workers}] "

    for rg_idx in row_group_indices:
        texts = pf.read_row_group(rg_idx, columns=["text"]).column("text").to_pylist()
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]
            tokenized = tokenizer(
                batch_texts,
                max_length=max_seq_length,
                truncation=True,
            )
            labels = []
            for input_ids in tokenized["input_ids"]:
                row_labels, masked = mask_prompt(input_ids, header_options)
                labels.append(row_labels)
                masked_count += int(masked)
                unmasked_count += int(not masked)

            writer.write_batch(
                {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": labels,
                }
            )

            rows_done += len(batch_texts)
            if rows_done % (batch_size * 10) < batch_size:
                print(f"  {prefix}Tokenized {rows_done}/{total} examples...")

    num_examples, num_bytes = writer.finalize()
    print(f"  {prefix}Wrote {num_examples} examples ({num_bytes / 1e6:.1f} MB)")
    return num_examples, num_bytes, masked_count, unmasked_count


def streaming_tokenize(
    source_parquet: str,
    tokenizer,
    max_seq_length: int,
    output_path: str,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> int:
    output = Path(output_path)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(source_parquet)
    total_rows = pf.metadata.num_rows
    num_row_groups = pf.metadata.num_row_groups
    num_workers = max(1, min(num_workers, total_rows, num_row_groups))
    header_options = assistant_header_ids(tokenizer)

    if num_workers <= 1:
        worker_args = [
            (
                source_parquet,
                list(range(num_row_groups)),
                tokenizer.name_or_path,
                max_seq_length,
                str(output / "data-00000-of-00001.arrow"),
                header_options,
                batch_size,
                0,
                1,
            )
        ]
        results = [tokenize_worker(worker_args[0])]
        data_files = [{"filename": "data-00000-of-00001.arrow"}]
    else:
        assignments = [[] for _ in range(num_workers)]
        for rg_idx in range(num_row_groups):
            assignments[rg_idx % num_workers].append(rg_idx)

        worker_args = []
        for i, group in enumerate(assignments):
            if not group:
                continue
            worker_args.append(
                (
                    source_parquet,
                    group,
                    tokenizer.name_or_path,
                    max_seq_length,
                    str(output / f"data-{i:05d}-of-{num_workers:05d}.arrow"),
                    header_options,
                    batch_size,
                    i,
                    num_workers,
                )
            )
        print(f"Launching {len(worker_args)} tokenization workers...")
        with mp.Pool(len(worker_args)) as pool:
            results = pool.map(tokenize_worker, worker_args)
        data_files = [
            {"filename": f"data-{i:05d}-of-{num_workers:05d}.arrow"}
            for i in range(len(worker_args))
        ]

    num_examples = sum(r[0] for r in results)
    num_bytes = sum(r[1] for r in results)
    masked_count = sum(r[2] for r in results)
    unmasked_count = sum(r[3] for r in results)

    with open(output / "state.json", "w") as f:
        json.dump(
            {
                "_data_files": data_files,
                "_fingerprint": "patient_summarization_tokenized",
                "_format_columns": None,
                "_format_kwargs": {},
                "_format_type": None,
                "_output_all_columns": False,
                "_split": None,
            },
            f,
            indent=2,
        )
    with open(output / "dataset_info.json", "w") as f:
        json.dump({}, f)

    print(f"Tokenized dataset: {output}")
    print(f"Rows: {num_examples}; bytes: {num_bytes / 1e6:.1f} MB")
    print(f"Prompt-masked: {masked_count}; unmasked: {unmasked_count}")
    if unmasked_count:
        print("WARNING: some examples did not match a known assistant header.")
    return num_examples


def default_tokenizer_for_model(model_name: str) -> str:
    if model_name == DEFAULT_MODEL:
        return DEFAULT_TOKENIZER
    return model_name


def resolve_reasoning_parser(model_name: str, reasoning_parser: str) -> str:
    if reasoning_parser != "auto":
        return reasoning_parser
    try:
        from vllm_reasoning_utils import resolve_parser_name

        return resolve_parser_name(model_name, reasoning_parser)
    except (ImportError, ValueError):
        if "gemma" in model_name.lower():
            return DEFAULT_REASONING_PARSER
        return "think_tags"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Gemma 4 SFT data for the patient summarization task."
    )
    parser.add_argument("--input-parquet", default=DEFAULT_INPUT)
    parser.add_argument("--chunks-parquet", default=DEFAULT_CHUNKS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-parquet", default=None)
    parser.add_argument("--tokenized-dir", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help=(
            "Tokenizer/chat-template source. Defaults to google/gemma-4-E2B-it "
            "when using the default Gemma 4 E2B-IT model, otherwise --model-name."
        ),
    )
    parser.add_argument(
        "--reasoning-parser",
        default="auto",
        help=(
            "Reasoning format to use in assistant turns. "
            "Default auto resolves Gemma models to the vLLM gemma4 parser format."
        ),
    )
    parser.add_argument("--max-seq-length", type=int, default=50000)
    parser.add_argument("--patient-id-col", default="pseudo_mrn")
    parser.add_argument("--chunk-index-col", default="chunk_index")
    parser.add_argument("--first-date-col", default="first_date")
    parser.add_argument("--last-date-col", default="last_date")
    parser.add_argument("--prior-summary-col", default="prior_summary")
    parser.add_argument("--reasoning-col", default="new_summary_reasoning")
    parser.add_argument("--summary-col", default="new_summary")
    parser.add_argument("--chunk-patient-id-col", default="patient_id")
    parser.add_argument("--chunk-local-idx-col", default="local_idx")
    parser.add_argument("--chunk-text-col", default="chunk_text")
    parser.add_argument("--system-content", default="")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--writer-batch-size", type=int, default=1000)
    parser.add_argument("--tokenize-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--keep-empty-reasoning", action="store_true")
    parser.add_argument("--keep-error-summaries", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    args.output_parquet = args.output_parquet or str(
        output_dir / "all_training_data.parquet"
    )
    args.tokenized_dir = args.tokenized_dir or str(
        output_dir / "tokenized_training_data.dataset"
    )
    args.tokenizer_name = args.tokenizer_name or default_tokenizer_for_model(
        args.model_name
    )
    args.resolved_reasoning_parser = resolve_reasoning_parser(
        args.model_name,
        args.reasoning_parser,
    )

    print(f"Model target: {args.model_name}")
    print(f"Loading tokenizer: {args.tokenizer_name}")
    print(f"Reasoning parser format: {args.resolved_reasoning_parser}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    build_text_parquet(args, tokenizer)

    if not args.skip_tokenize:
        streaming_tokenize(
            source_parquet=args.output_parquet,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            output_path=args.tokenized_dir,
            batch_size=args.tokenize_batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
