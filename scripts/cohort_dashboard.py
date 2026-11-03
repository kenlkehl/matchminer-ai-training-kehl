#!/usr/bin/env python3
"""Local review dashboard: each trial's source document beside its extracted cohorts and their criteria.

    python scripts/cohort_dashboard.py --run ../data/no_phi/<run> [--port 8770]

Reads a run_pipeline.py run's protocol-cohorts and registry-cohorts stages, live while they run
(per-trial extraction.json/merge.json under the protocol output dir, and the registry
checkpoints), so a refresh picks up newly decided trials. The protocol view shows the synthetic
protocol; the registry view shows the listing exactly as the prompts receive it (title, brief
summary and eligibility criteria as one document). Cohorts can be shown as extracted or as merged
for training. Clicking a criterion line jumps to the document lines that share the most of its
distinctive words (a navigation aid for spotting paraphrase or unsupported criteria, not a
correctness check). Serves on 127.0.0.1 only and uses no network resources; from another
machine, forward the port (ssh -L 8770:localhost:8770 camus).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
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


class Run:
    """Paths of one run_pipeline.py run and a cache of per-trial summaries keyed by file mtimes."""

    def __init__(self, run_dir, batch_dir=None, protocol_dir=None, registry_dir=None):
        self.root = run_dir
        requests = run_dir / "upstream" / "requests"
        protocol = json.loads((requests / "protocol-cohorts.json").read_text())["execution_command"] \
            if (requests / "protocol-cohorts.json").exists() else []
        registry = json.loads((requests / "registry-cohorts.json").read_text())["execution_command"] \
            if (requests / "registry-cohorts.json").exists() else []
        self.batch = batch_dir or option(protocol, "--batch-dir") or (option(registry, "--trials-jsonl") or Path()).parent
        self.protocol_dir = protocol_dir or option(protocol, "--output-dir") or self.batch / "cohort_extraction"
        self.registry_dir = registry_dir or option(registry, "--work-dir") or run_dir / "upstream" / "registry_extraction"
        self.seeds = {}
        for line in (self.batch / "seeds.jsonl").read_text().splitlines():
            if line.strip():
                seed = json.loads(line)
                self.seeds[seed["nct_id"]] = seed
        self.cache, self.lock, self.documents = {}, threading.Lock(), {}

    def protocol_files(self, nct):
        d = self.protocol_dir / "trials" / nct
        return d / "extraction.json", d / "merge.json"

    def registry_file(self, nct):
        return self.registry_dir / "checkpoints" / f"{nct}.json.gz"

    def failed_protocols(self):
        path = self.protocol_dir / "trials.csv"
        if not path.exists():
            return {}
        with path.open(newline="") as stream:
            return {row["trial_id"]: row.get("extraction_error") or "failed every attempt"
                    for row in csv.DictReader(stream) if row.get("extraction_status") == "failed"}

    # -- normalized records -------------------------------------------------------------------
    def protocol_record(self, nct, failed=None):
        extraction, merge = self.protocol_files(nct)
        if not extraction.exists():
            error = (failed or {}).get(nct)
            return {"status": "failed" if error else "pending", "error": error}
        rec = json.loads(extraction.read_text())
        groups = json.loads(merge.read_text())["groups"] if merge.exists() else None
        return self.normalized(rec["decision"], rec.get("cohort_names"), rec.get("merged_cohorts"),
                               rec.get("attempts") or {}, groups, None,
                               document=rec.get("source_document", {}).get("text_path"))

    def registry_record(self, nct):
        path = self.registry_file(nct)
        if not path.exists():
            return {"status": "pending"}
        with gzip.open(path, "rt") as stream:
            rec = json.load(stream)
        meta = rec.get("metadata") or {}
        groups = ((rec.get("merge") or {}).get("trial_groups") or {}).get(nct)
        return self.normalized(rec.get("result"), (meta.get("trial_cohort_names") or {}).get(nct),
                               (meta.get("trial_merged_cohorts") or {}).get(nct),
                               (meta.get("trial_attempts") or {}).get(nct) or {}, groups, rec.get("error"))

    @staticmethod
    def normalized(decision, names, identical, attempts, groups, error, document=None):
        if decision is None:
            return {"status": "failed", "error": error or "no validated decision", "attempts": attempts}
        if decision["cancer_treatment_trial"] != "yes":
            return {"status": "not_cancer", "attempts": attempts, "document": document}
        cohorts = decision["cohorts"]
        if groups is None and len(cohorts) == 1:
            groups = [[0]]
        return {"status": "cohorts", "cohorts": cohorts, "names": names or [c["cohort_name"] for c in cohorts],
                "identical": identical or [], "groups": groups, "attempts": attempts, "document": document}

    # -- index --------------------------------------------------------------------------------
    def summary(self, nct, failed):
        key = (mtime(self.protocol_files(nct)[0]), mtime(self.protocol_files(nct)[1]), mtime(self.registry_file(nct)),
               nct in failed)
        cached = self.cache.get(nct)
        if cached and cached[0] == key:
            return cached[1]
        entry = {"id": nct, "title": self.seeds[nct].get("title") or ""}
        for source, record in (("protocol", self.protocol_record(nct, failed)), ("registry", self.registry_record(nct))):
            entry[source] = {"status": record["status"],
                             "extracted": len(record["cohorts"]) if record["status"] == "cohorts" else 0,
                             "merged": len(record["groups"]) if record.get("groups") is not None else None}
        self.cache[nct] = (key, entry)
        return entry

    def index(self):
        with self.lock:
            failed = self.failed_protocols()
            with ThreadPoolExecutor(16) as pool:
                entries = list(pool.map(lambda nct: self.summary(nct, failed), self.seeds))
        stats = {}
        for source in ("protocol", "registry"):
            states = [e[source] for e in entries]
            stats[source] = {s: sum(x["status"] == s for x in states) for s in ("cohorts", "not_cancer", "failed", "pending")}
            stats[source]["extracted"] = sum(x["extracted"] for x in states)
            stats[source]["merged"] = sum(x["merged"] or 0 for x in states)
            stats[source]["merge_pending"] = sum(x["status"] == "cohorts" and x["merged"] is None for x in states)
        return {"run": str(self.root), "batch": str(self.batch), "trials": entries, "stats": stats}

    # -- detail -------------------------------------------------------------------------------
    def document(self, nct, source, record):
        if source == "registry":
            return registry_text(self.seeds[nct])
        path = Path(record.get("document") or self.batch / "protocols" / nct / "protocol.md")
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")

    def detail(self, nct, source):
        record = self.registry_record(nct) if source == "registry" else self.protocol_record(nct, self.failed_protocols())
        text = self.document(nct, source, record)
        lines = text.split("\n")
        matcher = Matcher(lines)
        outline = [{"line": i, "text": re.sub(r"^#+\s*", "", line).strip(), "level": len(line) - len(line.lstrip("#"))}
                   for i, line in enumerate(lines)
                   if re.match(r"^#{1,4}\s+\S", line) or (source == "registry" and re.match(
                       r"^(Title:|Brief summary:|Eligibility criteria:|\s*(Inclusion|Exclusion) Criteria:?\s*$)", line, re.I))]
        steps = []
        attempts = record.get("attempts") or {}
        for step in ("screen", "list"):
            for a in attempts.get(step) or []:
                steps.append(self.attempt(step, None, a))
        names = record.get("names") or []
        for key, log in sorted((attempts.get("criteria") or {}).items(), key=lambda kv: int(kv[0])):
            for a in log:
                steps.append(self.attempt("criteria", names[int(key)] if int(key) < len(names) else key, a))
        cohorts = []
        for number, cohort in enumerate(record.get("cohorts") or []):
            items = []
            for line in cohort["eligibility_criteria"].split("\n"):
                if not line.strip():
                    continue
                heading = bool(re.match(r"^\s*(inclusion|exclusion)( criteria)?\s*:?\s*$", line, re.I)) or line.rstrip().endswith(":") and len(line) < 80
                items.append({"text": line, "heading": heading, "matches": [] if heading else matcher.find(line)})
            cohorts.append({"number": number, "name": cohort["cohort_name"], "criteria": items})
        return {"id": nct, "source": source, "title": self.seeds[nct].get("title"), "status": record["status"],
                "error": record.get("error"), "lines": lines, "outline": outline, "cohorts": cohorts,
                "names": names, "identical": record.get("identical") or [], "groups": record.get("groups"),
                "steps": steps, "characters": len(text)}

    @staticmethod
    def attempt(step, cohort, a):
        return {"step": step, "cohort": cohort, "attempt": a.get("attempt"), "finish": a.get("finish_reason"),
                "error": a.get("validation_error"), "answer": a.get("answer"), "reasoning": a.get("reasoning") or ""}


class Matcher:
    """Runs of one to three consecutive nonblank document lines sharing the most IDF-weighted words of a criterion.

    Score: the share of the criterion's word weight found in the run. Ties prefer shorter runs, then runs with
    fewer other words, so a verbatim line beats a longer passage that merely contains the same words.
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

    def find(self, criterion, top=3):
        words = set(tokens(criterion))
        if not words:
            return []
        weight = {t: self.idf.get(t, math.log(len(self.rows) + 1) + 1) for t in words}
        total = sum(weight.values())
        candidates = {k for t in words for k in self.postings.get(t, ())}
        scored = []
        for k in candidates:
            for span in (1, 2, 3):
                window = set().union(*self.sets[k:k + span])
                found = window & words
                score = sum(weight[t] for t in found) / total
                precision = sum(weight[t] for t in found) / max(1e-9, sum(self.idf[t] for t in window))
                scored.append((-round(score, 2), span, -round(precision, 3), k, score, found))
        scored.sort(key=lambda x: x[:4])
        results, used = [], set()
        for _, span, _, k, score, found in scored:
            rows = range(k, min(k + span, len(self.rows)))
            if any(r in used for r in rows):
                continue
            used.update(rows)
            first, last = self.rows[rows[0]], self.rows[rows[-1]]
            marks = {j: [[m.start(), m.end()] for m in TOKEN.finditer(self.lines[j].lower()) if m.group() in found]
                     for j in range(first, last + 1)}
            results.append({"line": first, "end": last, "score": round(score, 3), "marks": marks})
            if len(results) == top:
                break
        return results


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
.viz-root{height:100vh;display:grid;grid-template-rows:auto 1fr;background:var(--page);color:var(--ink);
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
.d-pending{background:transparent;border:1.5px solid var(--muted)}
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
.chip{font-size:11px;font-weight:600;border-radius:10px;padding:0 8px;color:#fff;white-space:nowrap}
.chip.g4{color:#0b0b0b}.chip.gx{background:var(--muted)}
.cov{font-size:11px;color:var(--ink2);margin-bottom:4px}
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
kbd{font:11px ui-monospace,monospace;border:1px solid var(--axis);border-radius:3px;padding:0 4px;color:var(--ink2)}
.help{color:var(--muted);font-size:11px}
</style></head>
<body><div class="viz-root">
<header>
 <h1>Cohort extraction review</h1>
 <div class="tiles" id="tiles"></div>
 <span class="spacer"></span>
 <span class="help"><kbd>j</kbd>/<kbd>k</kbd> next/previous trial · <kbd>/</kbd> search trials · <kbd>p</kbd>/<kbd>r</kbd> protocol/registry</span>
 <button id="refresh" title="Re-read the run's files (new decisions appear while the run is going)">Refresh</button>
 <select id="theme" title="Theme"><option value="">Auto</option><option value="light">Light</option><option value="dark">Dark</option></select>
</header>
<main>
 <aside>
  <div class="filters">
   <input type="search" id="q" placeholder="Search NCT ID or title">
   <div class="row">
    <select id="fsource" title="Which version the status filter applies to"><option value="protocol">Protocol</option><option value="registry">Registry</option></select>
    <select id="fstatus"><option value="">any status</option><option value="cohorts">cohorts</option>
     <option value="not_cancer">not cancer treatment</option><option value="failed">failed</option><option value="pending">pending</option></select>
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
var state = {request: 0, index: null, filtered: [], id: null, source: "protocol", mode: "extracted", detail: null,
             searchHits: [], searchAt: -1, activeCrit: null, matchAt: 0};
var STATUS = {cohorts: "cohorts", not_cancer: "not cancer treatment", failed: "failed", pending: "pending"};
function $(id) { return document.getElementById(id); }
function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }
function fmt(n) { return n == null ? "–" : n.toLocaleString(); }
function getJSON(url) { return fetch(url).then(function (r) { if (!r.ok) throw new Error(r.status + " " + url); return r.json(); }); }

function applyTheme(v) { if (v) document.documentElement.setAttribute("data-theme", v); else document.documentElement.removeAttribute("data-theme"); }
$("theme").value = localStorage.getItem("cohortTheme") || "";
applyTheme($("theme").value);
$("theme").onchange = function () { localStorage.setItem("cohortTheme", this.value); applyTheme(this.value); };

function loadIndex() {
  $("refresh").disabled = true;
  return getJSON("/api/index").then(function (idx) {
    state.index = idx; renderTiles(); filter(); $("refresh").disabled = false;
    var hash = location.hash.slice(1).split("/");  // #NCT01234567/registry opens that trial and version
    if (hash[1] === "registry" || hash[1] === "protocol") state.source = hash[1];
    if (!state.id && hash[0] && state.index.trials.some(function (t) { return t.id === hash[0]; })) select(hash[0]);
    else if (!state.id && state.filtered.length) select(state.filtered[0].id);
  }).catch(function (e) { $("refresh").disabled = false; $("view").innerHTML = '<div class="notice">' + esc(e) + "</div>"; });
}
function tile(label, value, sub) {
  return '<div class="tile"><div class="label">' + esc(label) + '</div><div class="value">' + value + '</div><div class="sub">' + sub + "</div></div>";
}
function renderTiles() {
  var s = state.index.stats, n = state.index.trials.length, html = "";
  ["protocol", "registry"].forEach(function (src) {
    var x = s[src], decided = x.cohorts + x.not_cancer;
    html += tile(src === "protocol" ? "Protocols decided" : "Registry listings decided", fmt(decided) + " / " + fmt(n),
      fmt(x.not_cancer) + " not cancer treatment · " + fmt(x.failed) + " failed");
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
      '</div><div class="t">' + esc(t.title) + '</div><div class="badges">' + statusBadge("Protocol", t.protocol) +
      statusBadge("Registry", t.registry) + "</div></div>";
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
  getJSON("/api/trial?id=" + encodeURIComponent(id) + "&source=" + state.source).then(function (d) {
    if (request !== state.request) return;  // a newer selection is loading
    state.detail = d; state.activeCrit = null; state.searchHits = []; state.searchAt = -1;
    history.replaceState(null, "", "#" + d.id + "/" + d.source);
    renderDetail(); if (scroll) $("doc").scrollTop = scroll;
  }).catch(function (e) { $("view").innerHTML = '<div class="notice">' + esc(e) + "</div>"; });
}
function setSource(src) { if (state.source !== src) { state.source = src; if (state.id) select(state.id); } }
function setMode(mode) { state.mode = mode; renderCohorts(); }

function lineHTML(i, text, marks) {
  var cls = "ln" + (/^#{1,4}\s/.test(text) ? " h" : "");
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
    '<div class="trialhead"><div class="top"><span class="nct">' + esc(d.id) + '</span><span class="title">' + esc(d.title || "") + "</span></div>" +
    '<div class="controls"><span class="seg"><button id="srcP">Protocol</button><button id="srcR">Registry listing</button></span>' +
    '<span class="seg"><button id="modeE">As extracted</button><button id="modeM">Merged for training</button></span>' +
    (t.protocol ? statusBadge("Protocol", t.protocol) + statusBadge("Registry", t.registry) : "") + "</div></div>" +
    '<div class="panes"><div class="pane"><div class="panebar"><strong>' + (d.source === "protocol" ? "Synthetic protocol" : "Registry listing as the prompts see it") +
    '</strong><span class="help">' + fmt(d.lines.length) + " lines · " + fmt(d.characters) + ' characters</span>' +
    '<input type="search" class="grow" id="dq" placeholder="Find in document (regex)"><button id="dprev">↑</button><button id="dnext">↓</button>' +
    '<span class="help" id="dcount"></span><select id="outline">' + outline + "</select></div><div id=\"doc\"></div></div>" +
    '<div class="pane"><div class="panebar"><strong>Cohorts</strong><span class="help" id="chelp"></span></div><div id="cohorts"></div></div></div>';
  $("srcP").classList.toggle("on", d.source === "protocol"); $("srcR").classList.toggle("on", d.source === "registry");
  $("srcP").onclick = function () { setSource("protocol"); }; $("srcR").onclick = function () { setSource("registry"); };
  $("modeE").onclick = function () { setMode("extracted"); }; $("modeM").onclick = function () { setMode("merged"); };
  $("outline").onchange = function () { if (this.value !== "") jumpTo(+this.value, +this.value); this.value = ""; };
  $("dq").addEventListener("input", function () { docSearch(this.value); });
  $("dq").addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); stepSearch(e.shiftKey ? -1 : 1); } });
  $("dnext").onclick = function () { stepSearch(1); }; $("dprev").onclick = function () { stepSearch(-1); };
  renderDoc(); renderCohorts();
}
function groupChip(g, members) {
  var cls = g < 8 ? "" : " gx", style = g < 8 ? ' style="background:var(--s' + (g + 1) + ')"' : "";
  return '<span class="chip' + cls + (g === 3 ? " g4" : "") + '"' + style + ">G" + (g + 1) + " · extracted " +
         members.map(function (m) { return m + 1; }).join(" + ") + "</span>";
}
function critList(items, ci) {
  var html = '<ul class="crit">';
  items.forEach(function (it, k) {
    if (it.heading) { html += '<li class="head">' + esc(it.text) + "</li>"; return; }
    var best = it.matches.length ? it.matches[0].score : 0;
    var label = best >= 0.8 ? "found" : best >= 0.5 ? "partial" : "weak";
    var color = best >= 0.8 ? "good" : best >= 0.5 ? "warning" : "serious";
    var icon = best >= 0.8 ? "✓" : best >= 0.5 ? "≈" : "?";
    html += '<li data-c="' + ci + '" data-k="' + k + '" title="Best document match covers ' + Math.round(best * 100) +
      '% of this line\'s distinctive words; click to jump, click again for the next match"><span class="sc"><span class="dot" style="background:var(--' +
      color + ')"></span> ' + icon + " " + label + "</span><span>" + esc(it.text) + "</span></li>";
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
  var html = "";
  if (d.status !== "cohorts") {
    html = '<div class="note"><span class="st"><span class="dot d-' + d.status + '"></span><strong>' + STATUS[d.status] + "</strong></span>" +
      (d.error ? "<br>" + esc(d.error) : d.status === "not_cancer" ? "<br>The screen answered that this trial does not evaluate a cancer treatment." :
       d.status === "pending" ? "<br>No decision yet; Refresh picks it up when the run saves it." : "") + "</div>";
  } else {
    var groupOf = {};
    (d.groups || []).forEach(function (g, gi) { g.forEach(function (m) { groupOf[m] = gi; }); });
    if (d.identical.length) html += '<div class="note">The list step named ' + d.names.length + " cohorts; " + d.identical.map(function (g) {
      return g.map(function (i) { return "“" + esc(d.names[i]) + "”"; }).join(" and "); }).join("; ") +
      " had identical criteria and were combined in code.</div>";
    if (state.mode === "merged") {
      if (!d.groups) html += '<div class="note">The merge step has not run for this trial yet (protocols are merged after all are extracted); showing cohorts as extracted.</div>';
      else html += '<div class="note">' + d.groups.length + " cohort" + (d.groups.length === 1 ? "" : "s") + " for training from " + d.cohorts.length +
        " extracted. A merged cohort keeps its first member's criteria; its name joins the members' names.</div>";
    }
    var cards = (state.mode === "merged" && d.groups) ? d.groups.map(function (g, gi) {
      var first = d.cohorts[g[0]];
      return {title: g.map(function (m) { return d.cohorts[m].name; }).join("; "), ci: g[0], items: first.criteria, num: gi + 1,
              chip: g.length > 1 ? groupChip(gi, g) : ""};
    }) : d.cohorts.map(function (c, ci) {
      var gi = groupOf[ci], g = gi == null ? null : d.groups[gi];
      return {title: c.name, ci: ci, items: c.criteria, num: ci + 1, chip: g && g.length > 1 ? groupChip(gi, g) : ""};
    });
    cards.forEach(function (c) {
      html += '<div class="card"><h3><span class="num">' + c.num + "</span><span>" + esc(c.title) + "</span>" + c.chip +
        '</h3><div class="cov">' + coverage(c.items) + "</div>" + critList(c.items, c.ci) + "</div>";
    });
  }
  html += stepsHTML(d);
  box.innerHTML = html;
  $("chelp").textContent = d.status === "cohorts" ? "click a criterion to find it in the document" : "";
  Array.prototype.forEach.call(box.querySelectorAll("ul.crit li[data-k]"), function (li) {
    li.onclick = function () { showMatch(+li.dataset.c, +li.dataset.k, li); };
  });
}
function stepsHTML(d) {
  if (!d.steps.length) return "";
  var html = '<details class="steps"><summary>Model requests (' + d.steps.length + ")</summary>";
  d.steps.forEach(function (s) {
    html += '<div class="step"><div class="meta"><strong>' + esc(s.step) + "</strong>" + (s.cohort ? " · " + esc(s.cohort) : "") +
      " · attempt " + esc(s.attempt) + " · finish " + esc(s.finish) + (s.error ? " · validation error: " + esc(s.error) : "") +
      "</div><details><summary class=\"meta\">answer</summary><pre>" + esc(s.answer || "") + "</pre></details>" +
      "<details><summary class=\"meta\">reasoning (" + fmt(s.reasoning.length) + " characters)</summary><pre>" + esc(s.reasoning) + "</pre></details></div>";
  });
  return html + "</details>";
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
  var it = state.detail.cohorts[ci].criteria[k];
  var same = state.activeCrit && state.activeCrit[0] === ci && state.activeCrit[1] === k;
  state.matchAt = same ? (state.matchAt + 1) % Math.max(1, it.matches.length) : 0;
  state.activeCrit = [ci, k];
  Array.prototype.forEach.call(document.querySelectorAll("ul.crit li.active"), function (el) { el.classList.remove("active"); });
  li.classList.add("active");
  if (!it.matches.length) { clearHits(); $("dcount").textContent = "no shared words in the document"; return; }
  var m = it.matches[state.matchAt];
  jumpTo(m.line, m.end, m.marks);
  $("dcount").textContent = "match " + (state.matchAt + 1) + "/" + it.matches.length + " · covers " + Math.round(m.score * 100) + "% of the line's distinctive words";
}
function docSearch(q) {
  var d = state.detail; state.searchHits = []; state.searchAt = -1;
  clearHits();
  Array.prototype.forEach.call(document.querySelectorAll("#doc mark"), function (m) { var el = m.closest(".ln"); var i = +el.id.slice(1); el.outerHTML = lineHTML(i, d.lines[i]); });
  if (!q) { $("dcount").textContent = ""; return; }
  var re; try { re = new RegExp(q, "gi"); } catch (e) { $("dcount").textContent = "invalid pattern"; return; }
  for (var i = 0; i < d.lines.length; i++) {
    var marks = [], m; re.lastIndex = 0;
    while ((m = re.exec(d.lines[i])) && marks.length < 200) { if (!m[0].length) { re.lastIndex++; continue; } marks.push([m.index, m.index + m[0].length]); }
    if (marks.length) { $("L" + i).outerHTML = lineHTML(i, d.lines[i], marks); marks.forEach(function (_, k) { state.searchHits.push([i, k]); }); }
  }
  $("dcount").textContent = state.searchHits.length ? state.searchHits.length + " matches" : "no matches";
  if (state.searchHits.length) stepSearch(1);
}
function stepSearch(dir) {
  if (!state.searchHits.length) return;
  var cur = document.querySelector("#doc mark.cur"); if (cur) cur.classList.remove("cur");
  state.searchAt = (state.searchAt + dir + state.searchHits.length) % state.searchHits.length;
  var hit = state.searchHits[state.searchAt], mark = $("L" + hit[0]).querySelectorAll("mark")[hit[1]];
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
loadIndex();
</script></body></html>
"""


def make_handler(run):
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
                elif url.path == "/api/index":
                    self.send(json.dumps(run.index()))
                elif url.path == "/api/trial":
                    nct, source = query.get("id", ""), query.get("source", "protocol")
                    if nct not in run.seeds or source not in ("protocol", "registry"):
                        self.send(json.dumps({"error": "unknown trial or source"}), status=404)
                    else:
                        self.send(json.dumps(run.detail(nct, source)))
                else:
                    self.send(json.dumps({"error": "not found"}), status=404)
            except Exception as exc:  # shown in the page; the server keeps running
                self.send(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), status=500)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="A run_pipeline.py --run-dir (with upstream/requests/)")
    parser.add_argument("--batch-dir", type=Path, help="Synthetic protocol batch; default from the protocol-cohorts request")
    parser.add_argument("--protocol-cohort-dir", type=Path, help="Default from the protocol-cohorts request")
    parser.add_argument("--registry-work-dir", type=Path, help="Default from the registry-cohorts request")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    run = Run(args.run.resolve(), args.batch_dir, args.protocol_cohort_dir, args.registry_work_dir)
    print(f"{len(run.seeds)} trials; protocols {run.protocol_dir}; registry {run.registry_dir}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(run))
    print(f"Serving on http://127.0.0.1:{args.port}/ (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
