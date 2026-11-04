#!/usr/bin/env python3
"""Local review dashboard: each trial's source document beside its extracted cohorts and their criteria.

    python scripts/cohort_dashboard.py --run ../data/no_phi/<run> [--oncore ../data/no_phi/<oncore run>] [--port 8770]

Reads a run_pipeline.py run's protocol-cohorts and registry-cohorts stages, live while they run
(per-trial extraction.json/merge.json under the protocol output dir, and the registry
checkpoints), so a refresh picks up newly decided trials. The protocol view shows the synthetic
protocol; the registry view shows the listing exactly as the prompts receive it (title, brief
summary and eligibility criteria as one document). Cohorts can be shown as extracted, as merged
for training, or merged with the boilerplate separated (remove_boilerplate_criteria;
deboilerplate.json): each cohort's criteria that are not boilerplate (they seed synthetic patients),
then its boilerplate criteria, with each line's classification reasoning. Clicking a criterion line jumps to the document lines that share the most of its
distinctive words (a navigation aid for spotting paraphrase or unsupported criteria, not a
correctness check). Each --oncore directory (a prepared real-protocol run: inputs.csv plus
trials/<protocol_no>/ from extract_protocol_cohorts, merge_equivalent_cohorts and
remove_boilerplate_criteria) adds a tab
marked confidential. Serves on 127.0.0.1 only and uses no network resources; from another
machine, forward the port (ssh -L 8770:localhost:8770 camus).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import sys
import threading
from urllib.parse import parse_qs, urlparse

csv.field_size_limit(sys.maxsize)
TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
STOPWORDS = set("""a an and any are as at be been before but by can do does for from had has have if in into is it
its may must no not of on or other per such than that the their them then there these this those to under was were
which while who will with within without""".split())


def tokens(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def registry_text(seed):
    # Same layout as matchminer_ai.trials.cohort_workflow.registry_trial_text.
    text = lambda value: value.strip() if isinstance(value, str) else ""
    return (f"Title: {text(seed.get('title'))}\n\nBrief summary:\n{text(seed.get('brief_summary'))}\n\n"
            f"Eligibility criteria:\n{text(seed.get('eligibility_criteria'))}")


def option(command, name):
    return Path(command[command.index(name) + 1]) if name in command else None


def mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


class Collection:
    """One tab of the dashboard: trials with documents and per-source extraction records.

    Subclasses provide ids, titles, the files whose mtimes identify a trial's state, normalized
    records and document text; summaries are cached by those mtimes so Refresh rereads only changes.
    """

    sources = ("protocol",)
    confidential = False
    STATES = ("cohorts", "not_cancer", "failed", "excluded", "pending")

    def __init__(self, name, label):
        self.name, self.label, self.cache, self.lock = name, label, {}, threading.Lock()

    def describe(self):
        return {"name": self.name, "label": self.label, "sources": list(self.sources), "confidential": self.confidential}

    def context(self):
        return {}

    @staticmethod
    def normalized(decision, names, identical, attempts, groups, error, document=None, merge_attempts=None, clean=None):
        if decision is None:
            return {"status": "failed", "error": error or "no validated decision", "attempts": attempts}
        if decision["cancer_treatment_trial"] != "yes":
            return {"status": "not_cancer", "attempts": attempts, "document": document}
        cohorts = decision["cohorts"]
        if groups is None and len(cohorts) == 1:
            groups = [[0]]
        return {"status": "cohorts", "cohorts": cohorts, "names": names or [c["cohort_name"] for c in cohorts],
                "identical": identical or [], "groups": groups, "attempts": attempts, "document": document,
                "merge_attempts": merge_attempts or [], "clean": clean}

    @staticmethod
    def extraction_record(directory, document=None):
        """A protocol decision saved by extract_protocol_cohorts (+ merge_equivalent_cohorts), or None."""
        extraction, merge, clean = (directory / name for name in ("extraction.json", "merge.json", "deboilerplate.json"))
        if not extraction.exists():
            return None
        rec = json.loads(extraction.read_text())
        merging = json.loads(merge.read_text()) if merge.exists() else {}
        return Collection.normalized(rec["decision"], rec.get("cohort_names"), rec.get("merged_cohorts"),
                                     rec.get("attempts") or {}, merging.get("groups"), None,
                                     document=document or rec.get("source_document", {}).get("text_path"),
                                     merge_attempts=merging.get("attempts"),
                                     clean=json.loads(clean.read_text())["cohorts"] if clean.exists() else None)

    @staticmethod
    def registry_record(path, nct):
        """A registry listing's decision from its 0b_extract_trial_cohorts.py checkpoint (with merge and boilerplate)."""
        if not path.exists():
            return {"status": "pending"}
        with gzip.open(path, "rt") as stream:
            rec = json.load(stream)
        meta = rec.get("metadata") or {}
        merging = rec.get("merge") or {}
        return Collection.normalized(rec.get("result"), (meta.get("trial_cohort_names") or {}).get(nct),
                                     (meta.get("trial_merged_cohorts") or {}).get(nct),
                                     (meta.get("trial_attempts") or {}).get(nct) or {},
                                     (merging.get("trial_groups") or {}).get(nct), rec.get("error"),
                                     merge_attempts=(merging.get("trial_attempts") or {}).get(nct),
                                     clean=(rec.get("deboilerplate") or {}).get("cohorts"))

    @staticmethod
    def registry_outline(lines):
        return [{"line": i, "text": line.strip(), "level": 1} for i, line in enumerate(lines) if re.match(
            r"^(Title:|Brief summary:|Eligibility criteria:|\s*(Inclusion|Exclusion) Criteria:?\s*$)", line, re.I)]

    def summary(self, trial, context):
        key = (tuple(mtime(f) for f in self.files(trial)), self.context_key(trial, context))
        cached = self.cache.get(trial)
        if cached and cached[0] == key:
            return cached[1]
        entry = {"id": trial, **self.info(trial)}
        for source in self.sources:
            record = self.record(trial, source, context)
            entry[source] = {"status": record["status"],
                             "extracted": len(record["cohorts"]) if record["status"] == "cohorts" else 0,
                             "merged": len(record["groups"]) if record.get("groups") is not None else None}
        self.cache[trial] = (key, entry)
        return entry

    def context_key(self, trial, context):
        return None

    def index(self):
        with self.lock:
            context = self.context()
            with ThreadPoolExecutor(16) as pool:
                entries = list(pool.map(lambda trial: self.summary(trial, context), self.ids()))
        stats = {}
        for source in self.sources:
            states = [e[source] for e in entries]
            stats[source] = {s: sum(x["status"] == s for x in states) for s in self.STATES}
            stats[source]["extracted"] = sum(x["extracted"] for x in states)
            stats[source]["merged"] = sum(x["merged"] or 0 for x in states)
            stats[source]["merge_pending"] = sum(x["status"] == "cohorts" and x["merged"] is None for x in states)
        return {**self.describe(), "trials": entries, "stats": stats}

    def outline(self, lines, source):
        return [{"line": i, "text": re.sub(r"^#+\s*", "", line).strip(), "level": len(line) - len(line.lstrip("#"))}
                for i, line in enumerate(lines) if re.match(r"^#{1,4}\s+\S", line)]

    def detail(self, trial, source):
        record = self.record(trial, source, self.context())
        text = self.document(trial, source, record)
        found = matcher(text)
        lines = found.lines
        steps = []
        attempts = record.get("attempts") or {}
        for step in ("screen", "list"):
            for a in attempts.get(step) or []:
                steps.append(self.attempt(step, None, a))
        names = record.get("names") or []
        for key, log in sorted((attempts.get("criteria") or {}).items(), key=lambda kv: int(kv[0])):
            for a in log:
                steps.append(self.attempt("criteria", names[int(key)] if int(key) < len(names) else key, a))
        for a in record.get("merge_attempts") or []:
            if "answer" in a:  # merges saved before 2026-11-04 kept only the grouping
                steps.append(self.attempt("merge", None, a))
        cohorts = [{"number": number, "name": cohort["cohort_name"], "criteria": self.items(cohort["eligibility_criteria"], found)}
                   for number, cohort in enumerate(record.get("cohorts") or [])]
        clean = self.clean(record, found, steps)
        return {"id": trial, "source": source, **self.info(trial), "status": record["status"],
                "error": record.get("error"), "lines": lines, "outline": self.outline(lines, source)[:600],
                "cohorts": cohorts, "names": names, "identical": record.get("identical") or [],
                "groups": record.get("groups"), "clean": clean, "steps": steps, "characters": len(text)}

    @staticmethod
    def items(criteria, found):
        items = []
        for line in criteria.split("\n"):
            if not line.strip():
                continue
            heading = bool(re.match(r"^\s*(inclusion|exclusion)( criteria)?\s*:?\s*$", line, re.I)) or line.rstrip().endswith(":") and len(line) < 80
            items.append({"text": line, "heading": heading, "matches": [] if heading else found.find(line)})
        return items

    LABEL = re.compile(r"\s*(inclusion|exclusion)\s+criteria\s*:?\s*", re.I)  # as cohort_boilerplate.split_criteria
    FOLD = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2212": "-", "\u2013": "-",
                          "\u2014": "-", "\u00a0": " "})

    @classmethod
    def placed(cls, output):
        """Whether a line of the merged criteria survives in the step's output: the same line apart from quotes,
        dashes and spacing, or at least 90% of its words in output lines that are pieces of it (a split criterion)."""
        fold = lambda line: re.sub(r"\s+", " ", line.translate(cls.FOLD)).strip().casefold()
        lines = [line for line in output.split("\n") if line.strip()]
        same, words = {fold(line) for line in lines}, [set(tokens(line)) for line in lines]

        def check(line):
            want = set(tokens(line))
            if fold(line) in same or not want or re.match(r"^\s*(inclusion|exclusion)( criteria)?\s*:?\s*$", line, re.I):
                return True  # a label may be left out when nothing remains under it
            pieces = set().union(*(w for w in words if w and len(w & want) >= 0.6 * len(w)))
            return len(want & pieces) >= 0.9 * len(want)
        return check

    def clean(self, record, found, steps):
        """The merged cohorts with boilerplate removed (deboilerplate.json), or None when that step has not run.

        A saved result counts only when it was made from the current merged cohort's name and
        criteria (its input_sha256). Each cohort has its criteria that are not boilerplate and its
        boilerplate criteria, both matched to the document. Since schema v3 every line was classified
        in its own request and sorted in code: each line carries the step of its classification
        ("step") and lines whose answers never validated are listed ("unclassified"). Schema v2
        answers rewrote the criteria, so the lines of the merged criteria found in neither list are
        flagged ("unplaced"); schema v1 results list the removed or changed lines instead.
        """
        if record.get("clean") is None or record["status"] != "cohorts" or record.get("groups") is None:
            return None
        saved = {e["cohort_number"]: e for e in record["clean"]}
        norm = lambda line: re.sub(r"\s+", " ", line).strip().casefold()
        clean = []
        for number, group in enumerate(record["groups"]):
            name = "; ".join(record["cohorts"][m]["cohort_name"] for m in group)
            original = record["cohorts"][group[0]]["eligibility_criteria"]
            entry = saved.get(number) or {}
            current = entry.get("input_sha256") == hashlib.sha256(json.dumps([name, original]).encode()).hexdigest()
            text = entry.get("eligibility_criteria") if current else None
            boilerplate = entry.get("boilerplate_criteria") if current else None
            classified = entry.get("items") if current else None
            if classified is not None:  # schema v3: nothing to compare, the lines are the merged criteria's
                left = None
            elif boilerplate is None:  # schema v1: the lines removed or changed
                kept = {norm(line) for line in (text or "").split("\n")}
                left = [line for line in original.split("\n") if line.strip() and norm(line) not in kept] if text else []
            else:
                placed = self.placed(text + "\n" + boilerplate)
                left = [line for line in original.split("\n") if line.strip() and not placed(line)]
            lists = {"criteria": self.items(text, found) if text else [],
                     "boilerplate": self.items(boilerplate, found) if boilerplate is not None and text else None}
            error = None if text else entry.get("error") or ("made from other criteria" if entry else "not run yet")
            if classified is not None:
                for item in classified:
                    for a in item.get("attempts", []):
                        steps.append(self.attempt("boilerplate", f"{name} · line {item['number']}", a))
                    item["step"] = len(steps) - 1 if item.get("attempts") else None
                for key, flag in (("criteria", False), ("boilerplate", True)):
                    sequence = iter([i for i in classified if bool(i.get("boilerplate")) == flag])
                    for line in lists[key] or []:  # code sorted the lines in order under their labels
                        if not self.LABEL.fullmatch(line["text"]):
                            line["step"] = (next(sequence, None) or {}).get("step")
                if not text and not entry.get("error"):
                    total = sum(1 for line in original.split("\n") if line.strip() and not self.LABEL.fullmatch(line))
                    error = f"in progress: {sum('boilerplate' in i for i in classified)} of {total} lines classified"
            else:
                for a in entry.get("attempts", []):
                    steps.append(self.attempt("boilerplate", name, a))
            clean.append({"number": number, "name": name, "group": group, **lists,
                          "removed": left if boilerplate is None and classified is None else None,
                          "unplaced": left if boilerplate is not None else None,
                          "unclassified": entry.get("unclassified_items") if text else None, "error": error})
        return clean

    @staticmethod
    def attempt(step, cohort, a):
        return {"step": step, "cohort": cohort, "attempt": a.get("attempt"), "finish": a.get("finish_reason"),
                "error": a.get("validation_error"), "answer": a.get("answer"), "reasoning": a.get("reasoning") or ""}


class PipelineRun(Collection):
    """A run_pipeline.py run: synthetic protocols (protocol-cohorts) and their registry listings (registry-cohorts)."""

    sources = ("protocol", "registry")

    def __init__(self, run_dir, batch_dir=None, protocol_dir=None, registry_dir=None, name="synthetic", label=None):
        super().__init__(name, label or f"Synthetic protocols ({run_dir.name})")
        self.root = run_dir
        requests = run_dir / "upstream" / "requests"
        protocol = json.loads((requests / "protocol-cohorts.json").read_text())["execution_command"] \
            if (requests / "protocol-cohorts.json").exists() else []
        registry = json.loads((requests / "registry-cohorts.json").read_text())["execution_command"] \
            if (requests / "registry-cohorts.json").exists() else []
        self.batch = batch_dir or option(protocol, "--batch-dir") or (option(registry, "--trials-jsonl") or Path()).parent
        self.protocol_dir = protocol_dir or option(protocol, "--output-dir") or self.batch / "cohort_extraction"
        self.registry_dir = registry_dir or option(registry, "--work-dir") or run_dir / "upstream" / "registry_extraction"
        self.seeds, self.failed = {}, (None, {})
        for line in (self.batch / "seeds.jsonl").read_text().splitlines():
            if line.strip():
                seed = json.loads(line)
                self.seeds[seed["nct_id"]] = seed

    def ids(self):
        return list(self.seeds)

    def info(self, nct):
        return {"title": self.seeds[nct].get("title") or "", "sub": ""}

    def files(self, nct):
        d = self.protocol_dir / "trials" / nct
        return (d / "extraction.json", d / "merge.json", d / "deboilerplate.json",
                self.registry_dir / "checkpoints" / f"{nct}.json.gz")

    def context(self):
        """Failed protocols from the export's trials.csv, read again only when the file changes."""
        path = self.protocol_dir / "trials.csv"
        stamp = mtime(path)
        if stamp is None:
            return {}
        if self.failed[0] != stamp:
            with path.open(newline="") as stream:
                self.failed = (stamp, {row["trial_id"]: row.get("extraction_error") or "failed every attempt"
                                       for row in csv.DictReader(stream) if row.get("extraction_status") == "failed"})
        return self.failed[1]

    def context_key(self, nct, failed):
        return nct in failed

    def record(self, nct, source, failed):
        if source == "protocol":
            record = self.extraction_record(self.protocol_dir / "trials" / nct)
            if record is None:
                error = failed.get(nct)
                return {"status": "failed" if error else "pending", "error": error}
            return record
        return self.registry_record(self.files(nct)[3], nct)

    def document(self, nct, source, record):
        if source == "registry":
            return registry_text(self.seeds[nct])
        path = Path(record.get("document") or self.batch / "protocols" / nct / "protocol.md")
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")

    def outline(self, lines, source):
        return super().outline(lines, source) if source == "protocol" else self.registry_outline(lines)


class ProtocolSet(Collection):
    """Protocol documents prepared into inputs.csv and extracted into trials/<id>/ (e.g. the DFCI OnCore runs).

    A registry_cohorts/ folder beside them (trials.jsonl of the trials' ClinicalTrials.gov listings and a
    0b_extract_trial_cohorts.py --trials-jsonl work dir, work/) adds the Registry listing version, matched by nct_id.
    """

    PAGE = re.compile(r"^── page \d+ ──$")
    SECTION = re.compile(r"^\s{0,4}(\d{1,2}(\.\d{1,2}){0,3}\.?\s+[A-Z][^.]{2,100}|APPENDIX\s+\S.{0,90}|[A-Z][A-Z0-9 ,/&()'\-]{6,90})\s*$")

    def __init__(self, directory, name, label, confidential=False):
        super().__init__(name, label)
        self.root, self.confidential = directory, confidential
        with (directory / "inputs.csv").open(newline="") as stream:
            self.rows = {row["trial_id"]: row for row in csv.DictReader(stream)}
        registry = directory / "registry_cohorts"
        self.listings = {}
        if (registry / "trials.jsonl").exists():
            for line in (registry / "trials.jsonl").read_text().splitlines():
                if line.strip():
                    listing = json.loads(line)
                    self.listings[listing["nct_id"]] = listing
            self.registry_dir, self.sources = registry / "work", ("protocol", "registry")

    def ids(self):
        return list(self.rows)

    def info(self, trial):
        row = self.rows[trial]
        sub = " · ".join(x for x in (row.get("nct_id"), row.get("phase"), row.get("status")) if x)
        return {"title": row.get("short_title") or "", "sub": sub}

    def files(self, trial):
        d = self.root / "trials" / trial
        files = (d / "extraction.json", d / "merge.json", d / "deboilerplate.json", self.root / "run_metadata.json")
        nct = self.rows[trial].get("nct_id")
        return files + ((self.registry_dir / "checkpoints" / f"{nct}.json.gz",) if self.listings and nct else ())

    def context(self):
        path = self.root / "run_metadata.json"
        return json.loads(path.read_text()).get("trial_errors") or {} if path.exists() else {}

    def context_key(self, trial, errors):
        return trial in errors

    def record(self, trial, source, errors):
        row = self.rows[trial]
        if source == "registry":
            if row.get("nct_id") not in self.listings:
                return {"status": "excluded", "error": "no ClinicalTrials.gov listing for this protocol"}
            return self.registry_record(self.files(trial)[4], row["nct_id"])
        if row.get("excluded_reason"):
            return {"status": "excluded", "error": row["excluded_reason"]}
        record = self.extraction_record(self.root / "trials" / trial, document=row.get("protocol_path"))
        if record is None:
            error = errors.get(trial)
            return {"status": "failed" if error else "pending", "error": error, "document": row.get("protocol_path")}
        return record

    def document(self, trial, source, record):
        if source == "registry":
            listing = self.listings.get(self.rows[trial].get("nct_id"))
            return registry_text(listing) if listing else ""
        path = record.get("document") or self.rows[trial].get("protocol_path")
        if not path:
            return ""
        text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\x00", "")
        # PDF text keeps form feeds between pages; show them as page markers.
        return "\n".join(f"── page {i} ──\n{page.strip(chr(10))}" for i, page in enumerate(text.split("\f"), 1))

    def outline(self, lines, source):
        if source == "registry":
            return self.registry_outline(lines)
        return [{"line": i, "text": line.strip()[:100], "level": 1 if self.PAGE.match(line) is None else 2}
                for i, line in enumerate(lines)
                if len(line) <= 110 and self.SECTION.match(line) and not re.search(r"\.{4,}|\s\d+\s*$", line)]


class Matcher:
    """Runs of one to three consecutive nonblank document lines sharing the most IDF-weighted words of a criterion.

    Score: the share of the criterion's word weight found in the run. Ties prefer shorter runs, then runs with
    fewer other words, so a verbatim line beats a longer passage that merely contains the same words.
    Results are cached per word set: cohorts of one protocol repeat most of their criteria.
    """

    def __init__(self, lines):
        self.lines = lines
        self.rows = [i for i, line in enumerate(lines) if line.strip()]
        self.sets = [set(tokens(lines[i])) for i in self.rows]
        df = {}
        for s in self.sets:
            for t in s:
                df[t] = df.get(t, 0) + 1
        n = len(self.rows)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.postings = {}
        for k, s in enumerate(self.sets):
            for t in s:
                self.postings.setdefault(t, []).append(k)
        self.cache = {}

    def find(self, criterion, top=3):
        words = frozenset(tokens(criterion))
        if (words, top) not in self.cache:
            self.cache[words, top] = self.search(words, top) if words else []
        return self.cache[words, top]

    def pick(self, scored, top):
        """The best runs in score order that share no row."""
        picked, used = [], set()
        for entry in sorted(scored, key=lambda x: x[:4]):
            rows = range(entry[3], min(entry[3] + entry[1], len(self.rows)))
            if not used.isdisjoint(rows):
                continue
            used.update(rows)
            picked.append(entry)
            if len(picked) == top:
                break
        return picked

    def search(self, words, top):
        weight = {t: self.idf.get(t, math.log(len(self.rows) + 1) + 1) for t in words}
        total = sum(weight.values())
        # A run scores at most the sum of its rows' shares, so runs are scored in order of that bound
        # until no unscored run can round to the third pick's score; the picks equal scoring every run.
        share = {}
        for t in words:
            for k in self.postings.get(t, ()):
                share[k] = share.get(k, 0.0) + weight[t] / total
        bounds = sorted(((min(1.0, sum(share.get(j, 0.0) for j in range(k, k + span))), span, k)
                         for k in share for span in (1, 2, 3)), reverse=True)
        scored, picked = [], []
        for at, (bound, span, k) in enumerate(bounds):
            if at % 64 == 0 and at:
                picked = self.pick(scored, top)
                if len(picked) == top and round(bound, 2) < -picked[-1][0]:
                    break
            window = set().union(*self.sets[k:k + span])
            found = window & words
            score = sum(weight[t] for t in found) / total
            precision = sum(weight[t] for t in found) / max(1e-9, sum(self.idf[t] for t in window))
            scored.append((-round(score, 2), span, -round(precision, 3), k, score, found))
        else:
            picked = self.pick(scored, top)
        results = []
        for _, span, _, k, score, found in picked:
            first, last = self.rows[k], self.rows[min(k + span, len(self.rows)) - 1]
            marks = {j: [[m.start(), m.end()] for m in TOKEN.finditer(self.lines[j].lower()) if m.group() in found]
                     for j in range(first, last + 1)}
            results.append({"line": first, "end": last, "score": round(score, 3), "marks": marks})
        return results


@lru_cache(maxsize=16)
def matcher(text):
    """One Matcher per document text, kept across requests so reopening a trial reuses its matches."""
    return Matcher(text.split("\n"))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Cohort extraction review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.viz-root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,0.10);--hl:rgba(42,120,214,0.14);--mark:rgba(250,178,25,0.45);
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;
 --good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;
 --surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,0.10);
 --hl:rgba(57,135,229,0.22);--mark:rgba(250,178,25,0.35);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;
 --s6:#008300;--s7:#9085e9;--s8:#e66767}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
 --muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,0.10);--hl:rgba(57,135,229,0.22);
 --mark:rgba(250,178,25,0.35);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
.viz-root{height:100vh;display:grid;grid-template-rows:auto auto 1fr;background:var(--page);color:var(--ink);
 font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;align-items:center;gap:16px;padding:10px 16px;border-bottom:1px solid var(--grid);background:var(--surface);flex-wrap:wrap}
header h1{font-size:15px;margin:0 8px 0 0;font-weight:600}
.tiles{display:flex;gap:8px;flex-wrap:wrap}
.tile{border:1px solid var(--ring);border-radius:6px;padding:4px 10px;background:var(--page);min-width:118px}
.tile .label{color:var(--ink2);font-size:11px}.tile .value{font-size:16px;font-weight:600}
.tile .sub{color:var(--muted);font-size:11px}
.spacer{flex:1}
button,select,input{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--axis);border-radius:5px;padding:3px 8px}
button{cursor:pointer}button:hover{background:var(--hl)}
button.on{background:var(--hl);border-color:var(--s1);font-weight:600}
main{display:grid;grid-template-columns:300px 1fr;min-height:0}
aside{border-right:1px solid var(--grid);display:flex;flex-direction:column;min-height:0;background:var(--surface)}
.filters{padding:8px;display:grid;gap:6px;border-bottom:1px solid var(--grid)}
.filters .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.filters input[type=search]{width:100%}
#count{color:var(--muted);font-size:11px}
#list{overflow:auto;flex:1}
.item{padding:6px 10px;border-bottom:1px solid var(--grid);cursor:pointer}
.item:hover{background:var(--hl)}.item.sel{background:var(--hl);box-shadow:inset 3px 0 var(--s1)}
.item .id{font-weight:600;font-variant-numeric:tabular-nums}
.item .t{color:var(--ink2);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;font-size:12px}
.badges{display:flex;gap:6px;margin-top:2px;font-size:11px;color:var(--ink2)}
.st{display:inline-flex;align-items:center;gap:4px;white-space:nowrap}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.d-cohorts{background:var(--good)}.d-not_cancer{background:var(--warning)}.d-failed{background:var(--critical)}
.d-pending{background:transparent;border:1.5px solid var(--muted)}.d-excluded{background:var(--muted)}
.tabs{display:inline-flex;gap:0}.tabs button{border-radius:0}.tabs button:first-child{border-radius:5px 0 0 5px}
.tabs button:last-child{border-radius:0 5px 5px 0}.tabs button+button{border-left:none}.tabs button:only-child{border-radius:5px}
.banner{grid-column:1/-1;padding:4px 16px;font-size:12px;color:var(--ink);background:var(--page);border-bottom:1px solid var(--grid)}
.banner .dot{background:var(--critical)}
.item .s{color:var(--muted);font-size:11px}
#view{display:grid;grid-template-rows:auto 1fr;min-height:0}
.trialhead{padding:8px 14px;border-bottom:1px solid var(--grid);background:var(--surface);display:grid;gap:6px}
.trialhead .top{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.trialhead .nct{font-weight:700;font-size:15px}.trialhead .title{color:var(--ink2)}
.controls{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.seg{display:inline-flex}.seg button{border-radius:0}.seg button:first-child{border-radius:5px 0 0 5px}
.seg button:last-child{border-radius:0 5px 5px 0}.seg button+button{border-left:none}
.panes{display:grid;grid-template-columns:minmax(0,55fr) minmax(0,45fr);min-height:0}
.pane{display:flex;flex-direction:column;min-height:0}
.pane+.pane{border-left:1px solid var(--grid)}
.panebar{display:flex;gap:6px;align-items:center;padding:6px 10px;border-bottom:1px solid var(--grid);background:var(--surface);flex-wrap:wrap}
.panebar .grow{flex:1;min-width:120px}
#doc{overflow:auto;flex:1;background:var(--surface);font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.ln{display:grid;grid-template-columns:52px 1fr;padding:0 10px 0 0}
.ln .n{color:var(--muted);text-align:right;padding-right:10px;user-select:none;font-variant-numeric:tabular-nums}
.ln .x{white-space:pre-wrap;word-break:break-word}
.ln.h{font-weight:700}.ln.hit{background:var(--hl);box-shadow:inset 3px 0 var(--s1)}
mark{background:var(--mark);color:inherit;border-radius:2px}
mark.cur{outline:2px solid var(--s1)}
#cohorts{overflow:auto;flex:1;padding:10px;display:grid;gap:10px;align-content:start}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:8px 10px}
.card h3{margin:0 0 4px;font-size:13px;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.card h3 .num{color:var(--muted);font-variant-numeric:tabular-nums}
.card.picked{animation:picked 1.5s ease-out}
@keyframes picked{from{box-shadow:0 0 0 3px var(--s1)}to{box-shadow:0 0 0 3px transparent}}
#cjump{max-width:260px}
.chip{font-size:11px;font-weight:600;border-radius:10px;padding:0 8px;color:#fff;white-space:nowrap}
.chip.g4{color:#0b0b0b}.chip.gx{background:var(--muted)}
.cov{font-size:11px;color:var(--ink2);margin-bottom:4px}
details.removed{margin-top:6px;font-size:12px;color:var(--ink2)}details.removed summary{cursor:pointer}
details.removed li{padding:1px 4px}
.boiler{margin-top:8px;padding-top:6px;border-top:1px dashed var(--ring)}
.boiler h4{margin:0 0 2px;font-size:12px;color:var(--ink2)}.boiler ul.crit{color:var(--ink2)}
details.why{display:inline}details.why summary{display:inline;cursor:pointer;color:var(--muted);font-size:11px}
details.why[open]{display:block}details.why pre{white-space:pre-wrap;font-size:11px;margin:4px 0;max-height:260px;overflow:auto}
.unplaced{margin-top:6px;font-size:12px;border-left:3px solid var(--warning);padding:2px 8px}
ul.crit{list-style:none;margin:0;padding:0}
ul.crit li{display:grid;grid-template-columns:62px 1fr;gap:6px;padding:2px 4px;border-radius:4px;cursor:pointer}
ul.crit li:hover{background:var(--hl)}ul.crit li.active{background:var(--hl);box-shadow:inset 3px 0 var(--s1)}
ul.crit li.head{grid-template-columns:1fr;font-weight:600;cursor:default;margin-top:4px}
ul.crit li.head:hover{background:none}
.sc{font-size:11px;color:var(--ink2);white-space:nowrap;font-variant-numeric:tabular-nums}
.notice{padding:10px 14px;color:var(--ink2)}
.note{font-size:12px;color:var(--ink2);background:var(--page);border:1px solid var(--ring);border-radius:6px;padding:6px 8px}
details.steps{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:6px 10px}
details.steps summary{cursor:pointer;font-weight:600}
.step{border-top:1px solid var(--grid);padding:6px 0}.step:first-of-type{border-top:none}
.step .meta{font-size:11px;color:var(--ink2)}
.step pre{white-space:pre-wrap;word-break:break-word;max-height:320px;overflow:auto;background:var(--page);
 border:1px solid var(--ring);border-radius:5px;padding:6px;font-size:11.5px;margin:4px 0 0}
#view.loading .panes,#view.loading .trialhead{opacity:.45;transition:opacity .15s}
#busy{color:var(--ink2);font-size:12px}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--axis);border-radius:3px;padding:0 4px;color:var(--ink2)}
.help{color:var(--muted);font-size:11px}
</style></head>
<body><div class="viz-root">
<header>
 <h1>Cohort extraction review</h1>
 <span class="tabs" id="tabs"></span>
 <div class="tiles" id="tiles"></div>
 <span class="spacer"></span>
 <span class="help"><kbd>j</kbd>/<kbd>k</kbd> next/previous trial · <kbd>/</kbd> search trials · <kbd>p</kbd>/<kbd>r</kbd> protocol/registry</span>
 <span id="busy" hidden></span>
 <button id="refresh" title="Re-read the run's files (new decisions appear while the run is going)">Refresh</button>
 <select id="theme" title="Theme"><option value="">Auto</option><option value="light">Light</option><option value="dark">Dark</option></select>
</header>
<div class="banner" id="banner" hidden></div>
<main>
 <aside>
  <div class="filters">
   <input type="search" id="q" placeholder="Search NCT ID or title">
   <div class="row">
    <select id="fsource" title="Which version the status filter applies to"></select>
    <select id="fstatus"><option value="">any status</option><option value="cohorts">cohorts</option>
     <option value="not_cancer">not cancer treatment</option><option value="failed">failed</option>
     <option value="excluded">not extracted</option><option value="pending">pending</option></select>
   </div>
   <div class="row">
    <select id="fshape"><option value="">all trials</option><option value="multi">2+ extracted cohorts</option>
     <option value="merged">merge combined cohorts</option><option value="differ">protocol and registry counts differ</option>
     <option value="both">decided in both versions</option></select>
    <select id="sort"><option value="id">sort: NCT ID</option><option value="cohorts">sort: most cohorts</option>
     <option value="title">sort: title</option></select>
   </div>
   <div id="count"></div>
  </div>
  <div id="list"></div>
 </aside>
 <section id="view"><div class="notice">Loading the run index…</div></section>
</main>
</div>
<script>
var state = {request: 0, collections: [], collection: null, index: null, filtered: [], id: null, source: "protocol", mode: "extracted", detail: null,
             searchHits: [], searchAt: -1, activeCrit: null, matchAt: 0};
var STATUS = {cohorts: "cohorts", not_cancer: "not cancer treatment", failed: "failed", excluded: "not extracted", pending: "pending"};
var SOURCE = {protocol: "Protocol", registry: "Registry"};
function $(id) { return document.getElementById(id); }
function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }
function fmt(n) { return n == null ? "–" : n.toLocaleString(); }
function getJSON(url) { return fetch(url).then(function (r) { if (!r.ok) throw new Error(r.status + " " + url); return r.json(); }); }

function applyTheme(v) { if (v) document.documentElement.setAttribute("data-theme", v); else document.documentElement.removeAttribute("data-theme"); }
$("theme").value = localStorage.getItem("cohortTheme") || "";
applyTheme($("theme").value);
$("theme").onchange = function () { localStorage.setItem("cohortTheme", this.value); applyTheme(this.value); };

function collection() { return state.collections.filter(function (c) { return c.name === state.collection; })[0]; }
function renderTabs() {
  $("tabs").innerHTML = state.collections.map(function (c) {
    return '<button data-c="' + c.name + '"' + (c.name === state.collection ? ' class="on"' : "") + ">" + esc(c.label) + "</button>"; }).join("");
  Array.prototype.forEach.call($("tabs").querySelectorAll("button"), function (b) { b.onclick = function () { openCollection(b.dataset.c); }; });
  var c = collection();
  $("fsource").innerHTML = c.sources.map(function (s) { return '<option value="' + s + '">' + SOURCE[s] + "</option>"; }).join("");
  $("banner").hidden = !c.confidential;
  $("banner").innerHTML = '<span class="st"><span class="dot"></span><strong>Confidential</strong></span> DFCI OnCore protocol documents, ' +
    "for KSG members working on MatchMiner only (oncore-protocols CLAUDE.md §10a): do not copy them outside DFCI systems. " +
    "Extractions are research abstractions for review, not eligibility determinations; OnCore holds the current protocol.";
}
function openCollection(name, trial, source) {
  state.collection = name; state.id = null; state.index = null; state.source = source || collection().sources[0];
  renderTabs(); $("list").innerHTML = ""; $("tiles").innerHTML = "";
  $("view").innerHTML = '<div class="notice">Loading the index…</div>';
  return loadIndex(trial);
}
function loadIndex(trial) {
  $("refresh").disabled = true;
  return getJSON("/api/index?c=" + encodeURIComponent(state.collection)).then(function (idx) {
    if (idx.name !== state.collection) return;
    state.index = idx; renderTiles(); filter(); $("refresh").disabled = false;
    if (!state.id && trial && state.index.trials.some(function (t) { return t.id === trial; })) select(trial);
    else if (!state.id && state.filtered.length) select(state.filtered[0].id);
  }).catch(function (e) { $("refresh").disabled = false; $("view").innerHTML = '<div class="notice">' + esc(e) + "</div>"; });
}
function tile(label, value, sub) {
  return '<div class="tile"><div class="label">' + esc(label) + '</div><div class="value">' + value + '</div><div class="sub">' + sub + "</div></div>";
}
function renderTiles() {
  var s = state.index.stats, n = state.index.trials.length, html = "";
  state.index.sources.forEach(function (src) {
    var x = s[src], decided = x.cohorts + x.not_cancer;
    html += tile(src === "protocol" ? "Protocols decided" : "Registry listings decided", fmt(decided) + " / " + fmt(n),
      fmt(x.not_cancer) + " not cancer treatment · " + fmt(x.failed) + " failed" + (x.excluded ? " · " + fmt(x.excluded) + " not extracted" : ""));
    html += tile(src === "protocol" ? "Protocol cohorts" : "Registry cohorts",
      fmt(x.extracted) + (x.merge_pending ? "" : " → " + fmt(x.merged)),
      x.merge_pending ? "extracted · " + fmt(x.merge_pending) + " multi-cohort trials not merged yet" : "extracted → for training");
  });
  $("tiles").innerHTML = html;
}
function statusBadge(label, x) {
  var counts = x.status === "cohorts" ? " " + x.extracted + (x.merged == null ? " → ?" : x.merged !== x.extracted ? " → " + x.merged : "") : "";
  return '<span class="st" title="' + label + ": " + STATUS[x.status] + '"><span class="dot d-' + x.status + '"></span>' + label + counts +
         (x.status === "cohorts" ? "" : " " + STATUS[x.status]) + "</span>";
}
function filter() {
  if (!state.index) return;
  var q = $("q").value.trim().toLowerCase(), src = $("fsource").value, st = $("fstatus").value, shape = $("fshape").value;
  var rows = state.index.trials.filter(function (t) {
    if (q && t.id.toLowerCase().indexOf(q) < 0 && t.title.toLowerCase().indexOf(q) < 0) return false;
    if (st && t[src].status !== st) return false;
    if (shape === "multi" && t[src].extracted < 2) return false;
    if (shape === "merged" && !(t[src].merged != null && t[src].merged < t[src].extracted)) return false;
    if (shape === "both" && !(t.protocol.status !== "pending" && t.registry.status !== "pending")) return false;
    if (shape === "differ" && !(t.protocol.status === "cohorts" && t.registry.status === "cohorts" && t.protocol.extracted !== t.registry.extracted)) return false;
    return true;
  });
  var key = $("sort").value;
  rows.sort(function (a, b) {
    if (key === "cohorts") return (b[src].extracted - a[src].extracted) || (a.id < b.id ? -1 : 1);
    if (key === "title") return a.title.localeCompare(b.title);
    return a.id < b.id ? -1 : 1;
  });
  state.filtered = rows;
  $("count").textContent = rows.length.toLocaleString() + " of " + state.index.trials.length.toLocaleString() + " trials";
  var html = "";
  rows.slice(0, 3000).forEach(function (t) {
    html += '<div class="item' + (t.id === state.id ? " sel" : "") + '" data-id="' + t.id + '"><div class="id">' + t.id +
      '</div>' + (t.sub ? '<div class="s">' + esc(t.sub) + "</div>" : "") + '<div class="t">' + esc(t.title) + '</div><div class="badges">' +
      state.index.sources.map(function (s) { return statusBadge(SOURCE[s], t[s]); }).join("") + "</div></div>";
  });
  if (rows.length > 3000) html += '<div class="notice">Showing the first 3,000; narrow the search.</div>';
  $("list").innerHTML = html;
}
["q", "fsource", "fstatus", "fshape", "sort"].forEach(function (id) { $(id).addEventListener("input", filter); });
$("list").onclick = function (e) { var item = e.target.closest(".item"); if (item) select(item.dataset.id); };
$("refresh").onclick = function () { loadIndex().then(function () { if (state.id) select(state.id, true); }); };

function select(id, keepScroll) {
  state.id = id;
  Array.prototype.forEach.call(document.querySelectorAll(".item"), function (el) { el.classList.toggle("sel", el.dataset.id === id); });
  var el = document.querySelector('.item[data-id="' + id + '"]');
  if (el && !keepScroll) el.scrollIntoView({block: "nearest"});
  var scroll = keepScroll && $("doc") ? $("doc").scrollTop : 0, request = ++state.request;
  $("view").classList.add("loading"); $("busy").hidden = false; $("busy").textContent = "Loading " + id + "…";
  function done() { $("view").classList.remove("loading"); $("busy").hidden = true; }
  getJSON("/api/trial?c=" + encodeURIComponent(state.collection) + "&id=" + encodeURIComponent(id) + "&source=" + state.source).then(function (d) {
    if (request !== state.request) return;  // a newer selection is loading
    state.detail = d; state.activeCrit = null; state.searchHits = []; state.searchAt = -1;
    history.replaceState(null, "", "#" + state.collection + "/" + d.id + "/" + d.source);
    renderDetail(); done(); if (scroll) $("doc").scrollTop = scroll;
  }).catch(function (e) { if (request !== state.request) return; done(); $("view").innerHTML = '<div class="notice">' + esc(e) + "</div>"; });
}
function setSource(src) {
  if (state.source !== src && collection().sources.indexOf(src) >= 0) { state.source = src; if (state.id) select(state.id); }
}
function setMode(mode) { state.mode = mode; renderCohorts(); }

function lineHTML(i, text, marks) {
  var cls = "ln" + (/^#{1,4}\s|^── page \d+ ──$/.test(text) ? " h" : "");
  var body = "";
  if (marks && marks.length) {
    var at = 0;
    marks.forEach(function (m) { body += esc(text.slice(at, m[0])) + "<mark>" + esc(text.slice(m[0], m[1])) + "</mark>"; at = m[1]; });
    body += esc(text.slice(at));
  } else body = esc(text);
  return '<div class="' + cls + '" id="L' + i + '"><span class="n">' + (i + 1) + '</span><span class="x">' + (body || " ") + "</span></div>";
}
function renderDoc() {
  var d = state.detail, html = "";
  for (var i = 0; i < d.lines.length; i++) html += lineHTML(i, d.lines[i]);
  $("doc").innerHTML = html;
}
function renderDetail() {
  var d = state.detail, t = (state.index.trials.filter(function (x) { return x.id === d.id; })[0]) || {};
  var outline = '<option value="">Jump to section…</option>' + d.outline.map(function (o) {
    return '<option value="' + o.line + '">' + " ".repeat(Math.max(0, (o.level || 1) - 1) * 2) + esc(o.text.slice(0, 90)) + "</option>"; }).join("");
  $("view").innerHTML =
    '<div class="trialhead"><div class="top"><span class="nct">' + esc(d.id) + "</span>" + (d.sub ? '<span class="help">' + esc(d.sub) + "</span>" : "") +
    '<span class="title">' + esc(d.title || "") + "</span></div>" +
    '<div class="controls">' + (state.index.sources.length > 1 ? '<span class="seg"><button id="srcP">Protocol</button><button id="srcR">Registry listing</button></span>' : "") +
    '<span class="seg"><button id="modeE">As extracted</button><button id="modeM">Merged</button>' +
    '<button id="modeC" title="Merged, then the boilerplate step: criteria that are not boilerplate (these seed synthetic patients) and, separately, the boilerplate">Boilerplate separated</button></span>' +
    (t.protocol ? state.index.sources.map(function (s) { return statusBadge(SOURCE[s], t[s]); }).join("") : "") + "</div></div>" +
    '<div class="panes"><div class="pane"><div class="panebar"><strong>' + (d.source === "registry" ? "Registry listing as the prompts see it" :
      state.index.confidential ? "Protocol text (converted from the PDF)" : "Synthetic protocol") +
    '</strong><span class="help">' + fmt(d.lines.length) + " lines · " + fmt(d.characters) + ' characters</span>' +
    '<input type="search" class="grow" id="dq" placeholder="Find in document (regex)"><button id="dprev">↑</button><button id="dnext">↓</button>' +
    '<span class="help" id="dcount"></span><select id="outline">' + outline + "</select></div><div id=\"doc\"></div></div>" +
    '<div class="pane"><div class="panebar"><strong>Cohorts</strong><select id="cjump" hidden title="Jump to a cohort\'s criteria"></select>' +
    '<span class="help" id="chelp"></span></div><div id="cohorts"></div></div></div>';
  if ($("srcP")) {
    $("srcP").classList.toggle("on", d.source === "protocol"); $("srcR").classList.toggle("on", d.source === "registry");
    $("srcP").onclick = function () { setSource("protocol"); }; $("srcR").onclick = function () { setSource("registry"); };
  }
  $("modeE").onclick = function () { setMode("extracted"); }; $("modeM").onclick = function () { setMode("merged"); };
  $("modeC").onclick = function () { setMode("clean"); };
  $("outline").onchange = function () { if (this.value !== "") jumpTo(+this.value, +this.value); this.value = ""; };
  $("cjump").onchange = function () { if (this.value !== "") showCard(+this.value); this.value = ""; };
  var typing = null;
  $("dq").addEventListener("input", function () { var q = this.value; clearTimeout(typing); typing = setTimeout(function () { docSearch(q); }, 250); });
  $("dq").addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); stepSearch(e.shiftKey ? -1 : 1); } });
  $("dnext").onclick = function () { stepSearch(1); }; $("dprev").onclick = function () { stepSearch(-1); };
  renderDoc(); renderCohorts();
}
function groupChip(g, members) {
  var cls = g < 8 ? "" : " gx", style = g < 8 ? ' style="background:var(--s' + (g + 1) + ')"' : "";
  return '<span class="chip' + cls + (g === 3 ? " g4" : "") + '"' + style + ">G" + (g + 1) + " · extracted " +
         members.map(function (m) { return m + 1; }).join(" + ") + "</span>";
}
function critList(items, ci, kind) {
  var html = '<ul class="crit">';
  items.forEach(function (it, k) {
    if (it.heading) { html += '<li class="head">' + esc(it.text) + "</li>"; return; }
    var best = it.matches.length ? it.matches[0].score : 0;
    var label = best >= 0.8 ? "found" : best >= 0.5 ? "partial" : "weak";
    var color = best >= 0.8 ? "good" : best >= 0.5 ? "warning" : "serious";
    var icon = best >= 0.8 ? "✓" : best >= 0.5 ? "≈" : "?";
    html += '<li data-c="' + ci + '" data-k="' + k + '" data-kind="' + (kind || "") + '" title="Best document match covers ' + Math.round(best * 100) +
      '% of this line\'s distinctive words; click to jump, click again for the next match"><span class="sc"><span class="dot" style="background:var(--' +
      color + ')"></span> ' + icon + " " + label + "</span><span>" + esc(it.text) + (it.step == null ? "" :
      ' <details class="why" data-s="' + it.step + '" data-f="reasoning" title="The boilerplate step\'s reasoning for this line"><summary>why</summary><pre></pre></details>') +
      "</span></li>";
  });
  return html + "</ul>";
}
function coverage(items) {
  var lines = items.filter(function (i) { return !i.heading; });
  var found = lines.filter(function (i) { return i.matches.length && i.matches[0].score >= 0.8; }).length;
  var weak = lines.filter(function (i) { return !i.matches.length || i.matches[0].score < 0.5; }).length;
  return lines.length + " criterion lines · " + found + " found nearly verbatim · " + weak + " weak";
}
function renderCohorts() {
  var d = state.detail, box = $("cohorts");
  if (!box) return;
  $("modeE").classList.toggle("on", state.mode === "extracted"); $("modeM").classList.toggle("on", state.mode === "merged");
  $("modeC").classList.toggle("on", state.mode === "clean");
  var html = "", cards = [];
  if (d.status !== "cohorts") {
    html = '<div class="note"><span class="st"><span class="dot d-' + d.status + '"></span><strong>' + STATUS[d.status] + "</strong></span>" +
      (d.error ? "<br>" + esc(d.error) : d.status === "not_cancer" ? "<br>The screen answered that this trial does not evaluate a cancer treatment." :
       d.status === "pending" ? "<br>No decision yet; Refresh picks it up when the run saves it." : "") + "</div>";
    if (!d.lines.length || (d.lines.length === 1 && !d.lines[0])) html += '<div class="note">No document text is available for this trial.</div>';
  } else {
    var groupOf = {};
    (d.groups || []).forEach(function (g, gi) { g.forEach(function (m) { groupOf[m] = gi; }); });
    if (d.identical.length) html += '<div class="note">The list step named ' + d.names.length + " cohorts; " + d.identical.map(function (g) {
      return g.map(function (i) { return "“" + esc(d.names[i]) + "”"; }).join(" and "); }).join("; ") +
      " had identical criteria and were combined in code.</div>";
    if (state.mode === "clean") {
      if (!d.clean) html += '<div class="note">The boilerplate step has not run for this trial yet (it follows the merge); showing cohorts ' +
        (d.groups ? "as merged" : "as extracted") + ".</div>";
      else html += '<div class="note">' + d.clean.length + " merged cohort" + (d.clean.length === 1 ? "" : "s") +
        " after the boilerplate step: each card lists the criteria that are not boilerplate (these seed synthetic patients), then the boilerplate criteria." +
        (d.clean.some(function (c) { return c.unplaced; }) ? " These answers rewrote the criteria; merged-criteria lines found in neither list are flagged." :
         " Each line was classified in its own request and sorted in code; “why” beside a line shows the model's reasoning.") + "</div>";
    }
    if (state.mode === "merged") {
      if (!d.groups) html += '<div class="note">The merge step has not run for this trial yet (protocols are merged after all are extracted); showing cohorts as extracted.</div>';
      else html += '<div class="note">' + d.groups.length + " cohort" + (d.groups.length === 1 ? "" : "s") + " for training from " + d.cohorts.length +
        " extracted. A merged cohort keeps its first member's criteria; its name joins the members' names.</div>";
    }
    cards = (state.mode === "clean" && d.clean) ? d.clean.map(function (c, i) {
      return {title: c.name, ci: i, kind: "clean", items: c.criteria, num: i + 1, removed: c.removed, error: c.error,
              boilerplate: c.boilerplate, unplaced: c.unplaced, unclassified: c.unclassified, chip: c.group.length > 1 ? groupChip(i, c.group) : ""};
    }) : (state.mode !== "extracted" && d.groups) ? d.groups.map(function (g, gi) {
      var first = d.cohorts[g[0]];
      return {title: g.map(function (m) { return d.cohorts[m].name; }).join("; "), ci: g[0], items: first.criteria, num: gi + 1,
              chip: g.length > 1 ? groupChip(gi, g) : ""};
    }) : d.cohorts.map(function (c, ci) {
      var gi = groupOf[ci], g = gi == null ? null : d.groups[gi];
      return {title: c.name, ci: ci, items: c.criteria, num: ci + 1, chip: g && g.length > 1 ? groupChip(gi, g) : ""};
    });
    cards.forEach(function (c, i) {
      html += '<div class="card" id="card' + i + '"><h3><span class="num">' + c.num + "</span><span>" + esc(c.title) + "</span>" + c.chip + "</h3>" +
        (c.error ? '<div class="note">Boilerplate step: ' + esc(c.error) + "</div>" :
         '<div class="cov">' + coverage(c.items) + "</div>" + critList(c.items, c.ci, c.kind)) +
        (c.boilerplate && !c.error ? '<div class="boiler"><h4>Boilerplate criteria</h4>' + (c.boilerplate.length ?
           '<div class="cov">' + coverage(c.boilerplate) + "</div>" + critList(c.boilerplate, c.ci, "boiler") : '<div class="cov">None.</div>') + "</div>" : "") +
        (c.unclassified && c.unclassified.length ? '<div class="unplaced"><strong>Line' + (c.unclassified.length === 1 ? " " : "s ") +
           c.unclassified.join(", ") + "</strong> never got a valid answer and " + (c.unclassified.length === 1 ? "is" : "are") +
           " kept as not boilerplate.</div>" : "") +
        (c.unplaced && c.unplaced.length ? '<div class="unplaced"><strong>' + c.unplaced.length + " line" + (c.unplaced.length === 1 ? "" : "s") +
           " of the merged criteria not found in either list</strong> (dropped or substantially reworded by the boilerplate step):<ul class=\"crit\">" +
           c.unplaced.map(function (line) { return "<li>" + esc(line) + "</li>"; }).join("") + "</ul></div>" : "") +
        (!c.removed || c.error ? "" : !c.removed.length ? '<div class="cov">No line removed or changed by the boilerplate step.</div>' :
         '<details class="removed"><summary>Removed or changed by the boilerplate step (' + c.removed.length + " line" + (c.removed.length === 1 ? "" : "s") +
          ")</summary><ul class=\"crit\">" + c.removed.map(function (line) { return "<li>" + esc(line) + "</li>"; }).join("") + "</ul></details>") + "</div>";
    });
  }
  html += stepsHTML(d);
  box.innerHTML = html;
  $("cjump").hidden = cards.length < 2;
  $("cjump").innerHTML = '<option value="">Jump to cohort (' + cards.length + ")…</option>" + cards.map(function (c, i) {
    return '<option value="' + i + '">' + c.num + " · " + esc(c.title.length > 120 ? c.title.slice(0, 119) + "…" : c.title) + "</option>"; }).join("");
  if (!box.dataset.lazy) { box.dataset.lazy = "1"; box.addEventListener("toggle", function (e) { fillStep(e.target); }, true); }
  $("chelp").textContent = d.status === "cohorts" ? "click a criterion to find it in the document" : "";
  Array.prototype.forEach.call(box.querySelectorAll("ul.crit li[data-k]"), function (li) {
    li.onclick = function (e) { if (!e.target.closest("details")) showMatch(+li.dataset.c, +li.dataset.k, li); };
  });
}
function showCard(i) {
  var box = $("cohorts"), card = $("card" + i);
  if (!card) return;
  box.scrollTop += card.getBoundingClientRect().top - box.getBoundingClientRect().top - 10;
  card.classList.remove("picked"); void card.offsetWidth; card.classList.add("picked");  // restart the highlight
}
function stepsHTML(d) {
  if (!d.steps.length) return "";
  var html = '<details class="steps"><summary>Model requests (' + d.steps.length + ")</summary>";
  d.steps.forEach(function (s, i) {
    html += '<div class="step"><div class="meta"><strong>' + esc(s.step) + "</strong>" + (s.cohort ? " · " + esc(s.cohort) : "") +
      " · attempt " + esc(s.attempt) + " · finish " + esc(s.finish) + (s.error ? " · validation error: " + esc(s.error) : "") +
      '</div><details data-s="' + i + '" data-f="answer"><summary class="meta">answer</summary><pre></pre></details>' +
      '<details data-s="' + i + '" data-f="reasoning"><summary class="meta">reasoning (' + fmt(s.reasoning.length) + " characters)</summary><pre></pre></details></div>";
  });
  return html + "</details>";
}
function fillStep(el) {  // toggle events do not bubble; capture them on the cohorts pane
  if (el.dataset && el.dataset.s != null && el.open && !el.dataset.filled) {
    el.dataset.filled = "1"; el.querySelector("pre").textContent = state.detail.steps[+el.dataset.s][el.dataset.f] || "";
  }
}
function clearHits() {
  Array.prototype.forEach.call(document.querySelectorAll(".ln.hit"), function (el) {
    el.classList.remove("hit"); var i = +el.id.slice(1); el.outerHTML = lineHTML(i, state.detail.lines[i]);
  });
}
function jumpTo(line, end, marks) {
  clearHits();
  for (var j = line; j <= end; j++) {
    var el = $("L" + j); if (!el) continue;
    var html = lineHTML(j, state.detail.lines[j], marks ? marks[j] : null);
    el.outerHTML = html; $("L" + j).classList.add("hit");
  }
  var target = $("L" + line);
  if (target) target.scrollIntoView({block: "center"});
}
function showMatch(ci, k, li) {
  var d = state.detail, kind = li.dataset.kind || "";
  var it = (kind === "clean" ? d.clean[ci].criteria : kind === "boiler" ? d.clean[ci].boilerplate : d.cohorts[ci].criteria)[k];
  var same = state.activeCrit && state.activeCrit[0] === kind && state.activeCrit[1] === ci && state.activeCrit[2] === k;
  state.matchAt = same ? (state.matchAt + 1) % Math.max(1, it.matches.length) : 0;
  state.activeCrit = [kind, ci, k];
  Array.prototype.forEach.call(document.querySelectorAll("ul.crit li.active"), function (el) { el.classList.remove("active"); });
  li.classList.add("active");
  if (!it.matches.length) { clearHits(); $("dcount").textContent = "no shared words in the document"; return; }
  var m = it.matches[state.matchAt];
  jumpTo(m.line, m.end, m.marks);
  $("dcount").textContent = "match " + (state.matchAt + 1) + "/" + it.matches.length + " · covers " + Math.round(m.score * 100) + "% of the line's distinctive words";
}
var EAGER_LINES = 1500;
function docSearch(q) {
  var d = state.detail; state.searchHits = []; state.searchAt = -1; state.searchMarks = {};
  clearHits();
  Array.prototype.forEach.call(document.querySelectorAll("#doc .ln.found"), function (el) { var i = +el.id.slice(1); el.outerHTML = lineHTML(i, d.lines[i]); });
  if (!q) { $("dcount").textContent = ""; return; }
  var re; try { re = new RegExp(q, "gi"); } catch (e) { $("dcount").textContent = "invalid pattern"; return; }
  var lines = 0;
  for (var i = 0; i < d.lines.length; i++) {
    var marks = [], m; re.lastIndex = 0;
    while ((m = re.exec(d.lines[i])) && marks.length < 200) { if (!m[0].length) { re.lastIndex++; continue; } marks.push([m.index, m.index + m[0].length]); }
    if (marks.length) {
      state.searchMarks[i] = marks; marks.forEach(function (_, k) { state.searchHits.push([i, k]); });
      if (++lines <= EAGER_LINES) markLine(i);
    }
  }
  $("dcount").textContent = state.searchHits.length ? state.searchHits.length + " matches" : "no matches";
  if (state.searchHits.length) stepSearch(1);
}
function markLine(i) {
  var el = $("L" + i);
  if (el && !el.classList.contains("found")) { el.outerHTML = lineHTML(i, state.detail.lines[i], state.searchMarks[i]); $("L" + i).classList.add("found"); }
}
function stepSearch(dir) {
  if (!state.searchHits.length) return;
  var cur = document.querySelector("#doc mark.cur"); if (cur) cur.classList.remove("cur");
  state.searchAt = (state.searchAt + dir + state.searchHits.length) % state.searchHits.length;
  var hit = state.searchHits[state.searchAt]; markLine(hit[0]);
  var mark = $("L" + hit[0]).querySelectorAll("mark")[hit[1]];
  if (mark) { mark.classList.add("cur"); mark.scrollIntoView({block: "center"}); }
  $("dcount").textContent = (state.searchAt + 1) + "/" + state.searchHits.length;
}
document.addEventListener("keydown", function (e) {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") { if (e.key === "Escape") e.target.blur(); return; }
  if (e.key === "/") { e.preventDefault(); $("q").focus(); return; }
  if (e.key === "p") { setSource("protocol"); return; }
  if (e.key === "r") { setSource("registry"); return; }
  if (e.key === "j" || e.key === "k") {
    var at = state.filtered.findIndex(function (t) { return t.id === state.id; });
    var next = state.filtered[at + (e.key === "j" ? 1 : -1)];
    if (next) select(next.id);
  }
});
getJSON("/api/collections").then(function (list) {
  state.collections = list;
  var hash = decodeURIComponent(location.hash.slice(1)).split("/");  // #<tab>/<trial>/<protocol|registry>
  var start = list.some(function (c) { return c.name === hash[0]; }) ? hash[0] : list[0].name;
  state.collection = start;
  openCollection(start, hash[0] === start ? hash[1] : null, hash[0] === start && collection().sources.indexOf(hash[2]) >= 0 ? hash[2] : null);
});
</script></body></html>
"""


def make_handler(collections):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, kind="application/json", status=200):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self.send(PAGE, "text/html")
                elif url.path == "/api/collections":
                    self.send(json.dumps([c.describe() for c in collections.values()]))
                elif url.path in ("/api/index", "/api/trial"):
                    collection = collections.get(query.get("c", ""))
                    trial, source = query.get("id", ""), query.get("source", "protocol")
                    if collection is None:
                        self.send(json.dumps({"error": "unknown collection"}), status=404)
                    elif url.path == "/api/index":
                        self.send(json.dumps(collection.index()))
                    elif trial not in collection.ids() or source not in collection.sources:
                        self.send(json.dumps({"error": "unknown trial or source"}), status=404)
                    else:
                        self.send(json.dumps(collection.detail(trial, source)))
                else:
                    self.send(json.dumps({"error": "not found"}), status=404)
            except Exception as exc:  # shown in the page; the server keeps running
                self.send(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), status=500)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, help="A run_pipeline.py --run-dir (with upstream/requests/)")
    parser.add_argument("--batch-dir", type=Path, help="Synthetic protocol batch; default from the protocol-cohorts request")
    parser.add_argument("--protocol-cohort-dir", type=Path, help="Default from the protocol-cohorts request")
    parser.add_argument("--registry-work-dir", type=Path, help="Default from the registry-cohorts request")
    parser.add_argument("--oncore", type=Path, action="append", default=[],
                        help="A prepared DFCI OnCore protocol run (inputs.csv, trials/); one tab each; confidential")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    collections = {}
    if args.run:
        run = PipelineRun(args.run.resolve(), args.batch_dir, args.protocol_cohort_dir, args.registry_work_dir)
        collections[run.name] = run
        print(f"{run.label}: {len(run.seeds)} trials; protocols {run.protocol_dir}; registry {run.registry_dir}", flush=True)
    for i, directory in enumerate(args.oncore):
        name = "dfci" if i == 0 else f"dfci{i + 1}"
        collections[name] = ProtocolSet(directory.resolve(), name, f"DFCI protocols ({directory.resolve().name})", confidential=True)
        print(f"{collections[name].label}: {len(collections[name].rows)} protocols", flush=True)
    if not collections:
        parser.error("give --run and/or --oncore")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(collections))
    print(f"Serving on http://127.0.0.1:{args.port}/ (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
