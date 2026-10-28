#!/usr/bin/env python3
"""Prepare component requests, distill final answers, then tokenize for training.

No teacher calls occur during prepare/build. Run each stage explicitly; use
fresh output directories when changing any data, tokenizer, or prompt contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oncoreasoning_training import contracts as c
from oncoreasoning_training.teacher import TeacherPool, server_urls, worker_count, sampling_parameters
from oncoreasoning_training.scheduling import run_tasks
from trial_cohort_format import file_stamp


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def check_manifest(directory, name, manifest, *, transport_keys=()):
    path = Path(directory) / name
    if path.exists():
        previous = {key: value for key, value in read_json(path).items() if key not in transport_keys}
        if previous != manifest:
            raise ValueError(f"Incompatible {name}; select a fresh output directory")
    atomic_json(path, manifest)


def source_record(path):
    path = Path(path).expanduser().resolve()
    return {"path": str(path), "stamp": file_stamp(path)}


def preparation_inputs(paths, pipeline_manifest=None, *, required_recorded=()):
    recorded = {}
    if pipeline_manifest:
        recorded = {str(Path(item["path"]).resolve()): item
                    for item in read_json(pipeline_manifest)["inputs"]}
        if any(str(Path(path).expanduser().resolve()) not in recorded for path in required_recorded):
            raise ValueError("Source notes are missing from the recorded pipeline inputs")
    records = []
    for path in paths:
        key = str(Path(path).expanduser().resolve())
        records.append(recorded[key] if key in recorded else source_record(path))
    return records


def split_name(value):
    aliases = {"train": "train", "training": "train", "val": "validation",
               "valid": "validation", "validation": "validation", "test": "test"}
    if value is None or str(value).strip().lower() in ("", "nan", "none"):
        return None
    try:
        return aliases[str(value).strip().lower()]
    except KeyError:
        raise ValueError("Unrecognized input split; normalize to train/validation/test") from None


def required_text(row, *names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() and str(value).lower() not in ("nan", "none"):
            return str(value).strip()
    raise ValueError(f"Missing nonempty column (one of {', '.join(names)})")


def patient_identifier(row, *names):
    for name in names:
        value = row.get(name)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if value is not None and str(value).strip() and str(value).lower() not in ("nan", "none"):
            return str(value).strip()
    raise ValueError("Missing patient identifier")


def deduplicate_notes(notes):
    """Keep the serial summarizer's first-occurrence sentence deduplication."""
    seen, result = set(), []
    for date, text in notes:
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            normalized = sentence.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                kept.append(sentence)
        if kept:
            result.append((date, " ".join(kept)))
    return result


def variable_chunks(notes, tokenizer, *, minimum=10000, maximum=50000, overlap=500, seed=42):
    """Uniformly sample each chunk budget; retain short patients and final tails.

    Token spans retain dates even when a segment starts in the middle of a note.
    Notes must already be sorted. Overlap never creates an extra overlap-only tail.
    """
    if not 0 <= overlap < minimum <= maximum:
        raise ValueError("Require 0 <= overlap < minimum <= maximum")
    tokens, spans = [], []
    for date, text in notes:
        start = len(tokens)
        tokens.extend(tokenizer.encode(f"=== Clinical Note dated {date} ===\n{text}\n\n", add_special_tokens=False))
        spans.append((start, len(tokens), str(date)))
    rng, start = random.Random(seed), 0
    while start < len(tokens):
        target = rng.randint(minimum, maximum)
        end = min(start + target, len(tokens))
        # Decoding a partial word can change its tokenization. Enforce the actual
        # re-encoded size, not just the original token slice size.
        while True:
            text = tokenizer.decode(tokens[start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            actual = len(tokenizer.encode(text, add_special_tokens=False))
            if actual <= target:
                break
            end -= max(1, actual - target)
            if end <= start + overlap:
                raise ValueError("Tokenizer cannot form a chunk within the configured budget")
        dates = [date for lo, hi, date in spans if lo < end and hi > start]
        yield {"chunk_text": text, "first_date": dates[0], "last_date": dates[-1],
               "token_start": start, "token_end": end, "chunk_tokens": actual,
               "target_chunk_tokens": target}
        if end == len(tokens):
            break
        start = end - overlap


def component_tasks(pair, sources, catalog=None, pack_evidence=None):
    """Ignore old aggregate labels/reasoning and create independent fresh questions."""
    base = {"category": "clinical_qa", "patient_id": pair["patient_id"],
            "patient_summary": pair["patient_summary"], "split": pair["split"]}
    for component in c.TRIAL_COMPONENTS:
        task = {**base, "family": "trialchecking", "component": component,
                "trial_id": pair["trial_id"], "trial_summary": pair["trial_summary"]}
        task["messages"] = c.build_question_messages(task, sources)
        yield task
    for criterion in pair.get("exclusion_criteria", []):
        task = {**base, "family": "boilerplatechecking", "component": "exclusion_" + c.digest(criterion)[:16],
                "trial_id": pair["trial_id"], "criterion": criterion,
                "patient_boilerplate": pair.get("patient_boilerplate", "")}
        task["messages"] = c.build_question_messages(task, sources)
        yield task
    if catalog is None:
        return
    status = catalog.trial_status(pair["trial_id"])
    if status == "no_scoreable_drug":
        return
    if status != "ok":
        raise ValueError(f"GoodOption catalog has an unready trial ({status}); repair the catalog before preparing")
    for drug in catalog.scoreable_summaries_for_trial(pair["trial_id"]):
        classes = [replace(item, drug_names=(drug.preferred_name,))
                   for item in catalog.class_evidence_for_trial(pair["trial_id"])
                   if drug.preferred_name in item.drug_names]
        classes.sort(key=lambda item: item.class_id)
        evidence, _ = pack_evidence((drug,), class_evidence=classes)
        for component in c.GOOD_OPTION_COMPONENTS:
            task = {**base, "family": "goodoptionschecking", "component": component,
                    "drug_id": drug.drug_id, "drug_name": drug.preferred_name, "evidence": evidence}
            task["messages"] = c.build_question_messages(task, sources)
            yield task


def prepare(args, tokenizer=None):
    from synthetic_pool import require_all_sources
    require_all_sources(args.splits)
    import pandas as pd
    tokenizer = tokenizer or c.load_chat_tokenizer(args.model_name)
    c.letter_token_ids(tokenizer)
    sources = c.load_prompt_sources(args.inference_repo)
    c.good_option_rules(sources)  # Fail on an incompatible canonical rubric.
    paths = ([args.notes] if "summarization" in args.tasks else []) + (args.candidates if "clinical_qa" in args.tasks else [])
    if not paths or ("clinical_qa" in args.tasks and not args.candidates):
        raise ValueError("Clinical QA requires explicit --candidates parquet paths")
    non_phi_root = c.DEFAULT_DATA_DIR.resolve()
    non_phi = all(Path(path).expanduser().resolve().is_relative_to(non_phi_root) for path in paths)
    if not non_phi and not args.confirm_inputs_are_non_phi:
        raise ValueError("Custom inputs require --confirm-inputs-are-non-phi before preparing teacher requests")
    catalog = pack_evidence = None
    if "clinical_qa" in args.tasks:
        if not args.catalog:
            raise ValueError("Clinical QA requires --catalog with drug AND class evidence")
        sys.path.insert(0, str(Path(args.inference_repo).resolve() / "src"))
        from matchminer_ai.trials import load_good_option_catalog
        from matchminer_ai.matching.good_options import pack_good_option_evidence
        catalog = load_good_option_catalog(args.catalog, validate=True)
        pack_evidence = pack_good_option_evidence

    # Resolve explicit patient splits across both task categories. A missing
    # split inherits the known patient split, otherwise defaults to train.
    patient_splits, pairs, note_groups = {}, [], []
    def register(patient, split):
        if split:
            if patient in patient_splits and patient_splits[patient] != split:
                raise ValueError("Conflicting patient splits across inputs; supply consistently split sources")
            patient_splits[patient] = split

    if "summarization" in args.tasks:
        frame = pd.read_parquet(args.notes)
        for name in (args.patient_id_column, args.date_column, args.note_column):
            if name not in frame or frame[name].isna().any():
                raise ValueError(f"Summary notes need non-null {name}")
        frame[args.patient_id_column] = frame[args.patient_id_column].map(lambda value: patient_identifier({"id": value}, "id"))
        frame["_date"] = pd.to_datetime(frame[args.date_column], errors="raise", utc=True)
        for patient, group in frame.sort_values("_date", kind="stable").groupby(args.patient_id_column, sort=True):
            for split in group.get("split", pd.Series(dtype=str)).dropna().unique():
                register(patient, split_name(split))
            notes = deduplicate_notes([(str(date.date()), str(text)) for date, text in zip(group["_date"], group[args.note_column]) if str(text).strip()])
            if notes:
                note_groups.append((patient, notes))
        if args.max_patients:
            note_groups = note_groups[:args.max_patients]
    for path in args.candidates if "clinical_qa" in args.tasks else []:
        for row in pd.read_parquet(path).to_dict(orient="records"):
            patient = patient_identifier(row, "pseudo_mrn", "patient_id")
            split = split_name(row.get('patient_source_split', row.get("split")))
            register(patient, split)
            pairs.append({"patient_id": patient,
                          "patient_summary": required_text(row, "patient_summary", "cancer_history_summary"),
                          "trial_id": required_text(row, "nct_id", "trial_id").upper(),
                          "trial_summary": required_text(row, "this_space", "clinical_space_summary"),
                          "patient_boilerplate": str(row.get("patient_boilerplate_text") or ""),
                          "trial_boilerplate": str(row.get("trial_boilerplate_text") or "").strip()})
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    boilerplate_sources = {}
    if args.boilerplate_components:
        from oncoreasoning_training.prepare_boilerplate import validate_criteria
        for pair in pairs:
            source = pair["trial_boilerplate"]
            if source and source.lower() != "nan":
                key = c.digest([pair["trial_id"], source])
                record = read_json(Path(args.boilerplate_components) / f"{key}.json")
                if record["source"] != source:
                    raise ValueError("Boilerplate component source mismatch")
                pair["exclusion_criteria"] = validate_criteria(source, record["criteria"])
                boilerplate_sources[key] = c.digest(record)
    manifest = {"format_version": c.FORMAT_VERSION, "student_model": args.model_name,
                "tokenizer_fingerprint": c.tokenizer_fingerprint(tokenizer),
                "contract_sha256": file_digest(c.__file__), "sources": sources,
                "inputs": preparation_inputs(paths, args.pipeline_manifest,
                    required_recorded=[args.notes] if "summarization" in args.tasks else []),
                "inputs_confirmed_non_phi": non_phi or args.confirm_inputs_are_non_phi,
                "catalog_id": catalog.compatibility_id if catalog else None,
                "boilerplate_sources": boilerplate_sources,
                "settings": {key: getattr(args, key) for key in (
                    "tasks", "chunk_min_tokens", "chunk_max_tokens", "chunk_overlap", "seed",
                    "max_patients", "max_pairs", "patient_id_column", "date_column", "note_column", "splits")}}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    check_manifest(output, "prepare_manifest.json", manifest)
    seen, counts = set(), Counter()
    temporary = output / "requests.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        def write(task):
            task["id"] = c.digest(task)
            if task["id"] not in seen:
                seen.add(task["id"])
                stream.write(json.dumps(task, ensure_ascii=False) + "\n")
                counts[task["category"]] += 1
            return task["id"]
        summaries = []
        for patient, notes in note_groups:
            split = patient_splits.get(patient, "train")
            if split not in args.splits:
                continue
            parent = None
            for index, chunk in enumerate(variable_chunks(notes, tokenizer,
                    minimum=args.chunk_min_tokens, maximum=args.chunk_max_tokens,
                    overlap=args.chunk_overlap, seed=int(c.digest([args.seed, patient])[:16], 16))):
                task = {"category": "summarization", "patient_id": patient, "split": split,
                        "chunk_index": index, "parent_id": parent, **chunk}
                task["id"] = c.digest(task)
                parent = task["id"]
                summaries.append(task)
        # Rounds keep each patient's dependency behind its parent, while allowing
        # different patients to use the teacher concurrently.
        for task in sorted(summaries, key=lambda row: (row["chunk_index"], row["patient_id"])):
            task.pop("id")
            write(task)
        for pair in pairs:
            pair["split"] = patient_splits.get(pair["patient_id"], "train")
            if pair["split"] not in args.splits:
                continue
            for task in component_tasks(pair, sources, catalog, pack_evidence):
                write(task)
    if not counts or any(counts[task] == 0 for task in args.tasks):
        temporary.unlink()
        raise ValueError("No examples for a selected category; check inputs and splits")
    temporary.replace(output / "requests.jsonl")
    atomic_json(output / "prepared.json", {"requests_stamp": file_stamp(output / "requests.jsonl"),
                                         "manifest_sha256": c.digest(manifest), "counts": counts})
    print(f"Prepared {dict(counts)}")


def load_prepared(directory):
    directory = Path(directory)
    manifest, prepared = read_json(directory / "prepare_manifest.json"), read_json(directory / "prepared.json")
    if manifest["format_version"] != c.FORMAT_VERSION or prepared["manifest_sha256"] != c.digest(manifest):
        raise ValueError("Prepared manifest is incompatible or changed")
    if manifest["contract_sha256"] != file_digest(c.__file__):
        raise ValueError("Prompt contract changed; prepare a fresh run")
    if prepared["requests_stamp"] != file_stamp(directory / "requests.jsonl"):
        raise ValueError("Prepared requests changed; prepare a fresh run")
    return manifest, prepared


def response_path(response_dir, task_id):
    # Millions of component outputs must not share one filesystem directory.
    return Path(response_dir) / task_id[:2] / f"{task_id}.json"


def task_messages(task, sources, response_dir):
    if task["category"] == "clinical_qa":
        return task["messages"]
    prior = ""
    if task["parent_id"]:
        parent = read_json(response_path(response_dir, task['parent_id']))
        prior = c.validate_answer(parent["answer"], "summarization")
    return c.build_summarization_messages(task, sources, prior)


def invoke_teacher(task, messages, call, *, attempts=3):
    """Keep teacher reasoning in archives; supervise only the validated final.

    Retry the original request, never train on an error-correction conversation.
    Truncated completions are invalid even when their first letter is correct.
    """
    from matchminer_ai.llm.traces import response_record
    traces = []
    for attempt in range(attempts):
        try:
            result = call(messages)
            traces.append(response_record(result, messages=messages))
            choice = result.choices[0]
            if choice.finish_reason != "stop":
                raise ValueError("Teacher completion did not finish normally")
            answer = c.validate_answer(choice.message.content, task["category"])
            return {"id": task["id"], "category": task["category"], "split": task["split"],
                    "messages": messages, "answer": answer,
                    "reasoning": traces[-1]['reasoning'], "teacher_attempts": traces,
                    "trace_version": traces[-1]['trace_version']}
        except Exception:
            if attempt + 1 == attempts:
                # Do not echo server exceptions: they can contain clinical text.
                raise RuntimeError(f"Teacher request {task['id']} failed after {attempts} attempts; no target saved") from None


def generate(args, tokenizer=None, call=None):
    directory = Path(args.output_dir)
    manifest, prepared = load_prepared(directory)
    if not manifest["inputs_confirmed_non_phi"]:
        raise ValueError("Teacher requests must have confirmed non-PHI inputs")
    tokenizer = tokenizer or c.load_chat_tokenizer(args.teacher_tokenizer or args.teacher_model)
    urls = server_urls(url=args.server_url) if not args.server_urls_file else []
    sampling = sampling_parameters(args.teacher_model, thinking=args.teacher_thinking,
                                   temperature=args.temperature)
    config = {"prepared": prepared, "teacher_model": args.teacher_model,
              "teacher_tokenizer": args.teacher_tokenizer or args.teacher_model,
              "tokenizer_fingerprint": c.tokenizer_fingerprint(tokenizer),
              "enable_thinking": args.teacher_thinking, "max_tokens": args.teacher_max_tokens,
              "max_context": args.teacher_context, "temperature": args.temperature,
              "sampling": sampling}
    from matchminer_ai.llm.traces import TRACE_VERSION
    config['trace_version'] = TRACE_VERSION
    check_manifest(directory, "generation_manifest.json", config, transport_keys=("server_urls",))
    atomic_json(directory / "teacher_execution.json", {"server_urls": urls,
        "registry": str(args.server_urls_file) if args.server_urls_file else None,
        "concurrency": args.concurrency, "concurrency_per_server": args.concurrency_per_server,
        "timeout": args.timeout, "recovery_timeout": args.recovery_timeout})
    response_dir = directory / "responses"
    response_dir.mkdir(exist_ok=True)
    if call is None:
        call = TeacherPool(urls, args.teacher_model, max_tokens=args.teacher_max_tokens,
                           thinking=args.teacher_thinking, temperature=args.temperature, timeout=args.timeout,
                           registry=args.server_urls_file, per_server=args.concurrency_per_server,
                           recovery_timeout=args.recovery_timeout, trace_dir=directory / 'teacher_attempts')
    def worker(task):
        messages = task_messages(task, manifest["sources"], response_dir)
        path = response_path(response_dir, task['id'])
        if path.exists():
            record = read_json(path)
            if record["id"] != task["id"] or record["messages"] != messages:
                raise ValueError("Cached response no longer matches its prepared request")
            c.validate_answer(record["answer"], task["category"])
            if record.get('trace_version') != TRACE_VERSION or not record.get('teacher_attempts'):
                raise ValueError('Cached response lacks saved reasoning/raw traces; use a new generation directory')
            return
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               **sampling["extra_body"]["chat_template_kwargs"])
        if len(tokenizer.encode(prompt, add_special_tokens=False)) + args.teacher_max_tokens > args.teacher_context:
            raise ValueError(f"Teacher context exceeded for {task['id']}; no input was truncated")
        record = invoke_teacher(task, messages, call, attempts=args.attempts)
        atomic_json(path, record)
    done = run_tasks(jsonl(directory / "requests.jsonl"), worker,
                     concurrency=worker_count(args.concurrency),
                     progress=lambda n: print(f"Validated {n} teacher responses", flush=True) if n % 100 == 0 else None)
    print(f"Validated {done} teacher responses", flush=True)


def training_rows(directory, tokenizer, max_seq_length):
    manifest, _ = load_prepared(directory)
    directory = Path(directory)
    for task in jsonl(directory / "requests.jsonl"):
        path = response_path(directory / "responses", task['id'])
        if not path.exists():
            raise ValueError(f"Missing teacher response {task['id']}; finish generation before building")
        record = read_json(path)
        messages = task_messages(task, manifest["sources"], directory / "responses")
        if record["id"] != task["id"] or record["messages"] != messages or record["category"] != task["category"]:
            raise ValueError("Response does not match the prepared task")
        answer = c.validate_answer(record["answer"], task["category"])
        tokens = c.tokenize_example(tokenizer, messages, answer, task["category"], max_seq_length)
        yield {**tokens, "id": task["id"], "category": task["category"], "split": task["split"],
               "family": task.get("family", "summarization"), "component": task.get("component", "summary"),
               "patient_id": task["patient_id"]}


def build(args, tokenizer=None):
    from datasets import Dataset, DatasetDict
    directory = Path(args.output_dir)
    manifest, prepared = load_prepared(directory)
    tokenizer = tokenizer or c.load_chat_tokenizer(manifest["student_model"])
    if c.tokenizer_fingerprint(tokenizer) != manifest["tokenizer_fingerprint"]:
        raise ValueError("Student tokenizer changed since preparation")
    if args.max_seq_length > tokenizer.model_max_length:
        raise ValueError("Requested training context exceeds the tokenizer context limit")
    destination = directory / "tokenized_dataset"
    generation = read_json(directory / "generation_manifest.json")
    if generation["prepared"] != prepared:
        raise ValueError("Teacher generation belongs to a different prepared run")
    metadata_path = directory / "training_manifest.json"
    if destination.exists():
        if metadata_path.exists():
            previous = read_json(metadata_path)
            if (previous["prepared"] == prepared and previous["generation"] == generation
                    and previous["max_seq_length"] == args.max_seq_length and previous["shuffle_seed"] == args.seed):
                print("Student dataset already built for this exact run")
                return
            raise ValueError("Existing student dataset belongs to different build settings; use a fresh directory")
        # Dataset installation is atomic. A preemption between installation and
        # manifest creation leaves a recoverable derived output, never source data.
        import shutil
        shutil.rmtree(destination)
    # Arrow-backed generator keeps long-context tensors off the Python heap.
    # A private cache prevents reusing data after any response shard changes.
    with tempfile.TemporaryDirectory(prefix="oncoreasoning-build-", dir=directory) as cache:
        dataset = Dataset.from_generator(training_rows,
            gen_kwargs={"directory": str(directory), "tokenizer": tokenizer, "max_seq_length": args.max_seq_length},
            cache_dir=cache)
        splits = DatasetDict({split: dataset.filter(lambda row: row["split"] == split).shuffle(seed=args.seed)
                              for split in sorted(set(dataset["split"]))})
        if not len(dataset):
            raise ValueError("No prepared examples")
        counts = {split: dict(Counter(data["category"])) for split, data in splits.items()}
        pending_dataset = Path(cache) / "completed_dataset"
        splits.save_to_disk(str(pending_dataset))
        pending_dataset.replace(destination)
        # Close Arrow memory maps before removing their cache on an NFS mount.
        del splits, dataset
        gc.collect()
    metadata = {"format_version": c.FORMAT_VERSION, "student_model": manifest["student_model"],
                "tokenizer_fingerprint": manifest["tokenizer_fingerprint"], "prepared": prepared,
                "generation": generation, "max_seq_length": args.max_seq_length,
                "counts": counts,
                "answer_token_ids": c.letter_token_ids(tokenizer), "enable_thinking": False,
                "shuffle_seed": args.seed}
    tokenizer.save_pretrained(directory / "tokenizer")
    atomic_json(directory / "training_manifest.json", metadata)
    print(f"Saved student dataset: {destination}")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    stages = root.add_subparsers(dest="stage", required=True)
    prepare_parser = stages.add_parser("prepare", help="Prepare fresh serial summary and individual component requests")
    prepare_parser.add_argument("--model-name", default=c.DEFAULT_MODEL_NAME)
    prepare_parser.add_argument("--inference-repo", type=Path, default=c.DEFAULT_INFERENCE_REPO)
    prepare_parser.add_argument("--pipeline-manifest", type=Path,
                                help="Reuse recorded source identities from the integrated run without rehashing them")
    prepare_parser.add_argument("--notes", type=Path, default=c.DEFAULT_DATA_DIR / "all_synthetic_notes.parquet")
    prepare_parser.add_argument("--candidates", type=Path, nargs="+", default=[])
    prepare_parser.add_argument("--catalog", type=Path, default=c.DEFAULT_DATA_DIR / "good_option_catalog_v3")
    prepare_parser.add_argument("--boilerplate-components", type=Path, help="Validated patient-free exclusion extraction directory")
    prepare_parser.add_argument("--tasks", choices=c.TASKS, nargs="+", default=list(c.TASKS))
    prepare_parser.add_argument("--splits", choices=("train", "validation", "test"), nargs="+",
                                default=["train", "validation", "test"])
    prepare_parser.add_argument("--patient-id-column", default="pseudo_mrn")
    prepare_parser.add_argument("--date-column", default="date")
    prepare_parser.add_argument("--note-column", default="synthetic_note")
    prepare_parser.add_argument("--chunk-min-tokens", type=int, default=10000)
    prepare_parser.add_argument("--chunk-max-tokens", type=int, default=50000)
    prepare_parser.add_argument("--chunk-overlap", type=int, default=500)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--max-patients", type=int)
    prepare_parser.add_argument("--max-pairs", type=int)
    prepare_parser.add_argument("--confirm-inputs-are-non-phi", action="store_true")
    generate_parser = stages.add_parser("generate", help="Generate full final teacher outputs, resumably")
    endpoints = generate_parser.add_mutually_exclusive_group(required=True)
    endpoints.add_argument("--server-url", help="Explicit OpenAI-compatible /v1 endpoint")
    endpoints.add_argument("--server-urls-file", type=Path, help="Ready local teacher pool")
    generate_parser.add_argument("--teacher-model", required=True)
    generate_parser.add_argument("--teacher-tokenizer", help="HF tokenizer ID when teacher model is a server alias")
    generate_parser.add_argument("--teacher-thinking", action=argparse.BooleanOptionalAction, default=True)
    generate_parser.add_argument("--teacher-max-tokens", type=int, default=16384)
    generate_parser.add_argument("--teacher-context", type=int, default=131072)
    generate_parser.add_argument("--temperature", type=float, default=None,
                                 help="Default: model-family sampling, or 0.2 for unknown models")
    generate_parser.add_argument("--concurrency", type=int, default=0,
                                 help="Total in-flight ceiling; 0 follows the per-server limits")
    generate_parser.add_argument("--concurrency-per-server", type=int, default=96)
    generate_parser.add_argument("--recovery-timeout", type=float, default=86400)
    generate_parser.add_argument("--attempts", type=int, default=3)
    generate_parser.add_argument("--timeout", type=float, default=7200)
    build_parser = stages.add_parser("build", help="Mask prompts and supervise only complete final answers")
    build_parser.add_argument("--max-seq-length", type=int, default=98304)
    build_parser.add_argument("--seed", type=int, default=42)
    for stage in (prepare_parser, generate_parser, build_parser):
        stage.add_argument("--output-dir", type=Path, default=c.DEFAULT_OUTPUT_DIR)
    return root


def main():
    args = parser().parse_args()
    if args.stage == "generate" and (args.concurrency < 0 or min(args.concurrency_per_server, args.attempts, args.teacher_max_tokens, args.teacher_context, args.timeout, args.recovery_timeout) <= 0):
        raise ValueError("Generation limits must be positive")
    {"prepare": prepare, "generate": generate, "build": build}[args.stage](args)


if __name__ == "__main__":
    main()
