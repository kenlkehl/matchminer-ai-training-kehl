import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  AlertTriangle,
  Brain,
  Database,
  Download,
  ExternalLink,
  FileText,
  Loader2,
  Lock,
  Play,
  RefreshCcw,
  Settings,
  Table2,
  Trash2,
  Upload
} from "lucide-react";
import type { MatchResult, ModelSettings, PatientDocument, PromptKey, StatusMessage, TrialSpaceRecord } from "./types";
import { DEFAULT_PROMPTS } from "./data/defaultPrompts";
import { DEFAULT_MODEL_SETTINGS } from "./data/defaultSettings";
import { buildExtractiveFallbackSummary, buildRecordSegment, chunkTextByCharacters, fillPrompt, splitBoilerplate } from "./lib/text";
import { hashTextEmbedding } from "./lib/hashEmbedding";
import { parseCsvPatientFile } from "./services/csvIngest";
import { parsePdfPatientFile, type PdfProgress } from "./services/pdfIngest";
import { clearPatientSideData, loadModelSettings, loadPrompts, resetPrompt, saveModelSettings, savePrompt, saveTrialIndex } from "./services/storage";
import { embedText, generateText, isWebGpuAvailable, warmModel } from "./services/modelRuntime";
import { ensureTrialEmbeddings, fetchCtGovCancerTrials, loadOrFetchTrialIndex } from "./services/trialIndex";
import { fetchEmbeddedTrialIndex, parseEmbeddedTrialIndexFile } from "./services/trialImport";
import { retrieveByEmbedding, scoreAndRankMatches } from "./services/matching";

const promptOrder: PromptKey[] = ["patientSummary", "trialSpaceExtraction", "trialDeepScreen", "boilerplateDeepScreen"];

interface TrialProgress {
  phase: "download" | "extract" | "embed" | "import";
  current: number;
  total?: number;
  detail?: string;
  percent?: number;
}

export default function App() {
  const [webGpu, setWebGpu] = useState<boolean | null>(null);
  const [patientDocument, setPatientDocument] = useState<PatientDocument | null>(null);
  const [summary, setSummary] = useState("");
  const [patientBoilerplate, setPatientBoilerplate] = useState("");
  const [trialIndex, setTrialIndex] = useState<TrialSpaceRecord[]>([]);
  const [matches, setMatches] = useState<MatchResult[]>([]);
  const [settings, setSettings] = useState<ModelSettings>(DEFAULT_MODEL_SETTINGS);
  const [prompts, setPrompts] = useState<Record<PromptKey, string>>(
    Object.fromEntries(promptOrder.map((key) => [key, DEFAULT_PROMPTS[key].defaultValue])) as Record<PromptKey, string>
  );
  const [activePrompt, setActivePrompt] = useState<PromptKey>("patientSummary");
  const [showInputs, setShowInputs] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [status, setStatus] = useState<StatusMessage[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [pdfProgress, setPdfProgress] = useState<PdfProgress | null>(null);
  const [trialProgress, setTrialProgress] = useState<TrialProgress | null>(null);
  const [embeddedTrialUrl, setEmbeddedTrialUrl] = useState("");

  useEffect(() => {
    void Promise.all([isWebGpuAvailable(), loadModelSettings(), loadPrompts()]).then(([gpu, savedSettings, savedPrompts]) => {
      setWebGpu(gpu);
      setSettings(savedSettings);
      setPrompts(savedPrompts);
      addStatus(gpu ? "success" : "warning", gpu ? "WebGPU is available." : "WebGPU is not available in this runtime.");
    });
  }, []);

  const sortedNotes = patientDocument?.notes ?? [];
  const summaryParts = useMemo(() => splitBoilerplate(summary), [summary]);

  async function handleFile(file: File) {
    setBusy("Reading records");
    setPdfProgress(null);
    setMatches([]);
    setSummary("");
    try {
      const lower = file.name.toLowerCase();
      const doc = lower.endsWith(".csv") ? await parseCsvPatientFile(file) : await parsePdfPatientFile(file, setPdfProgress, { ocrMode: settings.pdfOcrMode });
      setPatientDocument(doc);
      sessionStorage.setItem("matchminer-current-patient", JSON.stringify({ fileName: doc.fileName, createdAt: doc.createdAt }));
      addStatus("success", `Loaded ${doc.source.toUpperCase()} with ${doc.notes.length} record${doc.notes.length === 1 ? "" : "s"}.`);
    } catch (error) {
      addStatus("error", errorMessage(error));
    } finally {
      setBusy(null);
      setPdfProgress(null);
    }
  }

  async function summarize() {
    if (!patientDocument) return;
    setBusy("Summarizing");
    setMatches([]);
    const chunks = chunkTextByCharacters(buildRecordSegment(patientDocument.notes));
    let prior = "";
    try {
      for (let index = 0; index < chunks.length; index += 1) {
        const prompt = fillPrompt(prompts.patientSummary, {
          prior_summary: prior || "None - this is the first segment for this patient",
          first_date: patientDocument.notes[0]?.isoDate ?? "unknown date",
          last_date: patientDocument.notes.at(-1)?.isoDate ?? "unknown date",
          record_segment: chunks[index]
        });
        prior = await generateText(settings.llmModelId, prompt, {
          dtype: settings.llmDtype,
          maxNewTokens: settings.maxSummaryTokens
        });
      }
      setSummary(prior);
      setPatientBoilerplate(splitBoilerplate(prior).patientBoilerplate);
      addStatus("success", "Patient summary generated locally.");
    } catch (error) {
      const fallback = buildExtractiveFallbackSummary(patientDocument.notes);
      setSummary(fallback);
      setPatientBoilerplate(splitBoilerplate(fallback).patientBoilerplate);
      addStatus("warning", `LLM summarization failed; local extractive summary was used. ${errorMessage(error)}`);
    } finally {
      setBusy(null);
    }
  }

  async function prepareTrialIndex() {
    setBusy("Preparing trials");
    setTrialProgress(null);
    try {
      const records = await loadOrFetchTrialIndex();
      setTrialIndex(records);
      addStatus("success", `Loaded ${records.length} trial spaces.`);
    } catch (error) {
      addStatus("error", errorMessage(error));
    } finally {
      setBusy(null);
    }
  }

  async function refreshCtGov() {
    setBusy("Downloading ClinicalTrials.gov");
    setTrialProgress({ phase: "download", current: 0, detail: "starting" });
    try {
      const records = await fetchCtGovCancerTrials({
        pageSize: 1000,
        onProgress: ({ records: downloaded, totalCount, page }) => {
          const total = totalCount ? ` of ${totalCount}` : "";
          setTrialProgress({
            phase: "download",
            current: downloaded,
            total: totalCount,
            detail: `page ${page}`,
            percent: percentComplete(downloaded, totalCount)
          });
          addStatus("info", `Downloaded ${downloaded}${total} open phase I-III interventional cancer trial records from ClinicalTrials.gov across ${page} page${page === 1 ? "" : "s"}.`);
        }
      });
      addStatus("success", `Downloaded ${records.length} public phase I-III interventional cancer trial records from ClinicalTrials.gov.`);
      setBusy("Extracting trial spaces");
      setTrialProgress({ phase: "extract", current: 0, total: records.length, detail: "warming local LLM", percent: 0 });
      const spaces = await extractTrialSpaces(records);
      await saveTrialIndex(spaces);
      setTrialIndex(spaces);
      addStatus("success", `Extracted ${spaces.length} trial spaces from ${records.length} ClinicalTrials.gov trials.`);
    } catch (error) {
      addStatus("error", errorMessage(error));
    } finally {
      setBusy(null);
      setTrialProgress(null);
    }
  }

  async function loadEmbeddedTrialFile(file: File) {
    setBusy("Loading embedded trial index");
    setTrialProgress({ phase: "import", current: 0, detail: file.name });
    try {
      const records = await parseEmbeddedTrialIndexFile(file);
      await persistEmbeddedTrialIndex(records, file.name);
    } catch (error) {
      addStatus("error", errorMessage(error));
    } finally {
      setBusy(null);
      setTrialProgress(null);
    }
  }

  async function loadEmbeddedTrialUrl() {
    const url = embeddedTrialUrl.trim();
    if (!url) {
      addStatus("warning", "Enter a URL for the embedded trial index.");
      return;
    }
    setBusy("Loading embedded trial index");
    setTrialProgress({ phase: "import", current: 0, detail: "fetching URL" });
    try {
      const records = await fetchEmbeddedTrialIndex(url);
      await persistEmbeddedTrialIndex(records, url);
    } catch (error) {
      addStatus("error", errorMessage(error));
    } finally {
      setBusy(null);
      setTrialProgress(null);
    }
  }

  async function persistEmbeddedTrialIndex(records: TrialSpaceRecord[], sourceLabel: string) {
    if (!records.length) throw new Error("Embedded trial index contains no trial spaces");
    const embeddingDim = records[0].embedding?.length ?? 0;
    setTrialProgress({ phase: "import", current: records.length, total: records.length, detail: "saving", percent: 100 });
    await saveTrialIndex(records);
    setTrialIndex(records);
    setMatches([]);
    addStatus("success", `Loaded ${records.length} pre-embedded trial spaces${embeddingDim ? ` (dim=${embeddingDim})` : ""} from ${sourceLabel}.`);
  }

  async function extractTrialSpaces(records: TrialSpaceRecord[]): Promise<TrialSpaceRecord[]> {
    const spaces: TrialSpaceRecord[] = [];
    let fallbackCount = 0;
    addStatus("info", "Extracting trial spaces with the local LLM. This can take a long time for a full CT.gov refresh.");
    await warmModel(settings.llmModelId, "text-generation", settings.llmDtype);
    for (let index = 0; index < records.length; index += 1) {
      const record = records[index];
      setTrialProgress({
        phase: "extract",
        current: index + 1,
        total: records.length,
        detail: `${spaces.length} trial spaces found`,
        percent: percentComplete(index, records.length)
      });
      try {
        const prompt = fillPrompt(prompts.trialSpaceExtraction, {
          trial_text: trialDocumentForExtraction(record)
        });
        const response = await generateText(settings.llmModelId, prompt, {
          dtype: settings.llmDtype,
          maxNewTokens: 1800
        });
        const extracted = parseExtractedTrialSpaces(record, response);
        if (extracted.length) {
          spaces.push(...extracted);
        } else {
          fallbackCount += 1;
          spaces.push(record);
        }
      } catch (error) {
        fallbackCount += 1;
        spaces.push(record);
        if (fallbackCount <= 3) {
          addStatus("warning", `Trial-space extraction fell back to one heuristic space for ${record.nctId}. ${errorMessage(error)}`);
        }
      }
      if ((index + 1) % 25 === 0 || index + 1 === records.length) {
        addStatus("info", `Extracted spaces from ${index + 1} of ${records.length} trials; current spaces=${spaces.length}.`);
      }
    }
    setTrialProgress({
      phase: "extract",
      current: records.length,
      total: records.length,
      detail: `${spaces.length} trial spaces found`,
      percent: 100
    });
    if (fallbackCount) {
      addStatus("warning", `${fallbackCount} trial${fallbackCount === 1 ? "" : "s"} used one heuristic fallback space.`);
    }
    return spaces;
  }

  async function cacheModels() {
    setBusy("Caching models");
    try {
      await warmModel(settings.llmModelId, "text-generation", settings.llmDtype);
      await warmModel(settings.trialSpaceModelId, "feature-extraction", settings.classifierDtype);
      await warmModel(settings.trialCheckerModelId, "text-classification", settings.classifierDtype);
      await warmModel(settings.boilerplateCheckerModelId, "text-classification", settings.classifierDtype);
      addStatus("success", "Model cache warmup complete.");
    } catch (error) {
      addStatus("warning", `Model cache warmup stopped: ${errorMessage(error)}`);
    } finally {
      setBusy(null);
    }
  }

  async function runMatching() {
    const trimmedSummary = summary.trim();
    if (!trimmedSummary) {
      addStatus("warning", "A patient summary is required before matching.");
      return;
    }
    setBusy("Matching trials");
    setTrialProgress(null);
    try {
      let records = trialIndex.length ? trialIndex : await loadOrFetchTrialIndex();
      let patientEmbedding: number[];
      try {
        setTrialProgress({ phase: "embed", current: 0, total: records.length, detail: "preparing embeddings", percent: 0 });
        records = await ensureTrialEmbeddings(records, settings.trialSpaceModelId, (done, total) => {
          setTrialProgress({
            phase: "embed",
            current: done,
            total,
            percent: percentComplete(done, total)
          });
          if (done === total || done % 10 === 0) addStatus("info", `Embedded ${done} of ${total} trial spaces.`);
        });
        setTrialProgress({ phase: "embed", current: records.length, total: records.length, detail: "embedding patient summary", percent: 100 });
        patientEmbedding = await embedText(settings.trialSpaceModelId, trimmedSummary, settings.classifierDtype);
      } catch (error) {
        addStatus("warning", `TrialSpace model unavailable; using local lexical fallback. ${errorMessage(error)}`);
        records = records.map((record) => ({
          ...record,
          embedding: record.embedding?.length ? record.embedding : hashTextEmbedding(record.trialSpaceText)
        }));
        patientEmbedding = hashTextEmbedding(trimmedSummary);
      }
      setTrialIndex(records);
      const candidates = retrieveByEmbedding(patientEmbedding, records, settings.retrievalCount);
      const ranked = await scoreAndRankMatches({
        patientSummary: trimmedSummary,
        patientBoilerplate: patientBoilerplate || summaryParts.patientBoilerplate,
        candidates,
        trialCheckerModelId: settings.trialCheckerModelId,
        boilerplateCheckerModelId: settings.boilerplateCheckerModelId,
        dtype: settings.classifierDtype,
        displayCount: settings.displayCount
      });
      const deepScreened = settings.runDeepScreen ? await runDeepScreen(ranked, trimmedSummary) : ranked;
      setMatches(deepScreened.map((match, index) => ({ ...match, rank: index + 1 })));
      addStatus("success", `Ranked ${deepScreened.length} trial options.`);
    } catch (error) {
      addStatus("error", errorMessage(error));
    } finally {
      setBusy(null);
      setTrialProgress(null);
    }
  }

  async function runDeepScreen(ranked: MatchResult[], patientSummary: string): Promise<MatchResult[]> {
    const screened: MatchResult[] = [];
    for (const match of ranked) {
      try {
        const trialPrompt = fillPrompt(prompts.trialDeepScreen, {
          patient_summary: patientSummary,
          trial_summary: match.trial.trialSpaceText
        });
        const trialResponse = await generateText(settings.llmModelId, trialPrompt, {
          dtype: settings.llmDtype,
          maxNewTokens: 700
        });
        const boilerplatePrompt = fillPrompt(prompts.boilerplateDeepScreen, {
          patient_boilerplate: patientBoilerplate || summaryParts.patientBoilerplate,
          trial_boilerplate: match.trial.boilerplateText
        });
        const boilerplateResponse = await generateText(settings.llmModelId, boilerplatePrompt, {
          dtype: settings.llmDtype,
          maxNewTokens: 600
        });
        screened.push({
          ...match,
          llmTrialCheckScore: parseFinalScore(trialResponse),
          llmTrialCheckReasoning: trialResponse,
          llmBoilerplateExcluded: parseYesNo(boilerplateResponse),
          llmBoilerplateReasoning: boilerplateResponse
        });
      } catch (error) {
        screened.push({ ...match, warnings: [...match.warnings, `Deep screen unavailable: ${errorMessage(error)}`] });
      }
    }
    return screened.sort((a, b) => (b.llmTrialCheckScore ?? b.trialCheckerScore ?? 0) - (a.llmTrialCheckScore ?? a.trialCheckerScore ?? 0));
  }

  async function persistPrompt(key: PromptKey, value: string) {
    const next = { ...prompts, [key]: value };
    setPrompts(next);
    await savePrompt(key, value);
  }

  async function persistSettings(next: ModelSettings) {
    setSettings(next);
    await saveModelSettings(next);
  }

  async function deleteLocalPatientData() {
    setPatientDocument(null);
    setSummary("");
    setPatientBoilerplate("");
    setMatches([]);
    await clearPatientSideData();
    addStatus("success", "Local patient workspace cleared.");
  }

  function addStatus(kind: StatusMessage["kind"], text: string) {
    setStatus((prior) => [{ kind, text }, ...prior].slice(0, 7));
  }

  const busyNow = Boolean(busy);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Brain size={22} /></div>
          <div>
            <h1>MatchMiner-AI</h1>
            <p>Local clinical trial matching</p>
          </div>
        </div>
        <div className="top-actions">
          <button className="icon-button" onClick={() => setShowSettings(true)} title="Settings" type="button">
            <Settings size={18} />
          </button>
          <button className="text-button danger" onClick={deleteLocalPatientData} type="button">
            <Trash2 size={16} /> Clear patient
          </button>
        </div>
      </header>

      <section className="notice">
        <Lock size={18} />
        <span>
          Runs locally on this device. Beta software only. It does not replace medical advice, may be wrong, must not make autonomous decisions, does not guarantee eligibility, and does not show whether a trial has slots.
        </span>
      </section>

      <main className="workspace">
        <section className="left-column">
          <Panel title="Records" icon={<Upload size={18} />}>
            <label className="dropzone">
              <input
                type="file"
                accept=".csv,application/pdf,.pdf,text/csv"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void handleFile(file);
                  event.currentTarget.value = "";
                }}
              />
              <FileText size={34} />
              <span>{patientDocument ? patientDocument.fileName : "PDF or CSV"}</span>
            </label>
            <div className="compact-row">
              <button className="text-button" disabled={!patientDocument} onClick={() => setShowInputs((value) => !value)} type="button">
                <Table2 size={16} /> {showInputs ? "Hide inputs" : "View inputs"}
              </button>
              <span className="muted">{sortedNotes.length ? `${sortedNotes.length} dated record${sortedNotes.length === 1 ? "" : "s"}` : "No records loaded"}</span>
            </div>
            {pdfProgress && <ProgressLine label={formatPdfProgress(pdfProgress)} />}
            {showInputs && patientDocument && <InputViewer document={patientDocument} />}
          </Panel>

          <Panel title="Models and trials" icon={<Database size={18} />}>
            <div className="readiness-grid">
              <Readiness label="WebGPU" value={webGpu === null ? "checking" : webGpu ? "ready" : "unavailable"} />
              <Readiness label="Trial spaces" value={trialIndex.length ? `${trialIndex.length}` : "not loaded"} />
              <Readiness label="Deep screen" value={settings.runDeepScreen ? "on" : "off"} />
            </div>
            <div className="button-grid">
              <button className="text-button" disabled={busyNow} onClick={cacheModels} type="button">
                <Download size={16} /> Cache models
              </button>
              <button className="text-button" disabled={busyNow} onClick={prepareTrialIndex} type="button">
                <Database size={16} /> Load index
              </button>
              <button className="text-button" disabled={busyNow} onClick={refreshCtGov} type="button">
                <RefreshCcw size={16} /> CT.gov refresh
              </button>
            </div>
            <div className="trial-import">
              <label className={`text-button file-loader ${busyNow ? "disabled-control" : ""}`}>
                <Upload size={16} /> Load embedded file
                <input
                  type="file"
                  accept=".json,.jsonl,.ndjson,.csv,application/json,text/csv"
                  disabled={busyNow}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void loadEmbeddedTrialFile(file);
                    event.currentTarget.value = "";
                  }}
                />
              </label>
              <div className="url-load-row">
                <input
                  aria-label="Embedded trial index URL"
                  disabled={busyNow}
                  onChange={(event) => setEmbeddedTrialUrl(event.target.value)}
                  placeholder="https://huggingface.co/.../resolve/main/trials.json"
                  value={embeddedTrialUrl}
                />
                <button className="text-button" disabled={busyNow || !embeddedTrialUrl.trim()} onClick={() => void loadEmbeddedTrialUrl()} type="button">
                  <Download size={16} /> Load URL
                </button>
              </div>
            </div>
            {trialProgress && <ProgressLine label={formatTrialProgress(trialProgress)} />}
          </Panel>
        </section>

        <section className="center-column">
          <Panel title="Patient summary" icon={<FileText size={18} />}>
            <div className="toolbar">
              <button className="primary-button" disabled={!patientDocument || busyNow} onClick={summarize} type="button">
                {busy === "Summarizing" ? <Loader2 className="spin" size={17} /> : <Brain size={17} />} Summarize
              </button>
              <button className="primary-button" disabled={!summary.trim() || busyNow} onClick={runMatching} type="button">
                {busy === "Matching trials" ? <Loader2 className="spin" size={17} /> : <Play size={17} />} Match trials
              </button>
            </div>
            <textarea
              className="summary-box"
              value={summary}
              onChange={(event) => {
                setSummary(event.target.value);
                setPatientBoilerplate(splitBoilerplate(event.target.value).patientBoilerplate);
              }}
              placeholder="Patient summary"
            />
          </Panel>

          <Panel title="Prompt controls" icon={<Settings size={18} />}>
            <div className="tabs">
              {promptOrder.map((key) => (
                <button key={key} className={activePrompt === key ? "tab active" : "tab"} onClick={() => setActivePrompt(key)} type="button">
                  {DEFAULT_PROMPTS[key].label}
                </button>
              ))}
            </div>
            <textarea
              className="prompt-box"
              value={prompts[activePrompt]}
              onChange={(event) => void persistPrompt(activePrompt, event.target.value)}
            />
            <div className="compact-row end">
              <span className="muted">{DEFAULT_PROMPTS[activePrompt].variables.map((v) => `{${v}}`).join(" ")}</span>
              <button
                className="text-button"
                onClick={async () => {
                  const next = await resetPrompt(activePrompt);
                  setPrompts(next);
                }}
                type="button"
              >
                <RefreshCcw size={15} /> Reset
              </button>
            </div>
          </Panel>
        </section>

        <section className="right-column">
          <Panel title="Retrieved trial spaces" icon={<Brain size={18} />}>
            {matches.length === 0 ? (
              <div className="empty-state">
                <AlertTriangle size={22} />
                <span>No ranked trials yet.</span>
              </div>
            ) : (
              <div className="results-list">
                {matches.map((match) => (
                  <ResultCard key={match.id} match={match} />
                ))}
              </div>
            )}
          </Panel>

          <Panel title="Run log" icon={<AlertTriangle size={18} />}>
            {trialProgress ? <ProgressLine label={formatTrialProgress(trialProgress)} /> : busy && <ProgressLine label={busy} />}
            <div className="status-list">
              {status.map((item, index) => (
                <div className={`status ${item.kind}`} key={`${item.text}-${index}`}>
                  {item.text}
                </div>
              ))}
            </div>
          </Panel>
        </section>
      </main>

      {showSettings && (
        <SettingsDialog
          settings={settings}
          onClose={() => setShowSettings(false)}
          onChange={(next) => void persistSettings(next)}
        />
      )}
    </div>
  );
}

function Panel({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        {icon}
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}

function InputViewer({ document }: { document: PatientDocument }) {
  if (document.source === "pdf") {
    return <textarea className="input-text" value={document.rawText} readOnly />;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Text</th>
          </tr>
        </thead>
        <tbody>
          {document.notes.map((note) => (
            <tr key={note.id}>
              <td>{note.isoDate}</td>
              <td>{note.text}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Readiness({ label, value }: { label: string; value: string }) {
  return (
    <div className="readiness">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ProgressLine({ label }: { label: string }) {
  return (
    <div className="progress-line">
      <Loader2 className="spin" size={15} />
      <span>{label}</span>
    </div>
  );
}

function formatPdfProgress(progress: PdfProgress): string {
  if (progress.phase === "docling") {
    if (progress.current <= 0) return "Loading Granite Docling WebGPU";
    const percent = typeof progress.percent === "number" ? ` (${progress.percent}%)` : "";
    return `Processing PDF page ${progress.current} of ${progress.total} with Granite Docling${percent}`;
  }
  if (progress.phase === "local-ocr") {
    const detail = progress.detail ? ` with ${progress.detail}` : "";
    return `Processing PDF${detail}`;
  }
  if (progress.phase === "ocr") {
    const percent = typeof progress.percent === "number" ? ` (${progress.percent}%)` : "";
    return `Processing PDF page ${progress.current} of ${progress.total}${percent}`;
  }
  const phase = progress.phase.toUpperCase();
  const detail = progress.detail ?? `${progress.current}/${progress.total}`;
  const percent = typeof progress.percent === "number" ? ` (${progress.percent}%)` : "";
  return `${phase} ${detail}${percent}`;
}

function formatTrialProgress(progress: TrialProgress): string {
  const current = formatCount(progress.current);
  const total = typeof progress.total === "number" ? ` of ${formatCount(progress.total)}` : "";
  const percent = typeof progress.percent === "number" ? ` (${progress.percent}%)` : "";
  const detail = progress.detail ? ` - ${progress.detail}` : "";
  if (progress.phase === "download") return `Downloading trial records ${current}${total}${percent}${detail}`;
  if (progress.phase === "extract") return `Processing trial ${current}${total}${percent}${detail}`;
  if (progress.phase === "import") return `Loading embedded trial index${total ? ` ${current}${total}${percent}` : ""}${detail}`;
  return `Embedding trial space ${current}${total}${percent}${detail}`;
}

function percentComplete(current: number, total: number | undefined): number | undefined {
  if (!total || total <= 0) return undefined;
  return Math.max(0, Math.min(100, Math.round((current / total) * 100)));
}

function formatCount(value: number): string {
  return value.toLocaleString();
}

function ResultCard({ match }: { match: MatchResult }) {
  const boiler = match.boilerplateScore;
  return (
    <article className="result-card">
      <div className="result-head">
        <span className="rank">#{match.rank}</span>
        <div>
          <h3>{match.trial.title}</h3>
          <p>{match.trial.nctId} {match.trial.overallStatus ? `- ${match.trial.overallStatus}` : ""}</p>
        </div>
      </div>
      <div className="score-row">
        <Score label="Match" value={match.trialCheckerScore} />
        <Score label="Retrieval" value={match.cosineSimilarity} />
        <Score label="Exclusion risk" value={boiler} invert />
        {typeof match.llmTrialCheckScore === "number" && <Score label="LLM" value={match.llmTrialCheckScore / 5} />}
      </div>
      <details>
        <summary>Trial space</summary>
        <p>{match.trial.trialSpaceText}</p>
      </details>
      <details>
        <summary>Boilerplate exclusions</summary>
        <p>{match.trial.boilerplateText || "None extracted."}</p>
      </details>
      {match.llmTrialCheckReasoning && (
        <details>
          <summary>Deep screen</summary>
          <p>{match.llmTrialCheckReasoning}</p>
          {match.llmBoilerplateReasoning && <p>{match.llmBoilerplateReasoning}</p>}
        </details>
      )}
      <div className="result-foot">
        <span>{match.trial.locations?.slice(0, 3).join(" | ")}</span>
        {match.trial.url && (
          <a href={match.trial.url} rel="noreferrer" target="_blank">
            CT.gov <ExternalLink size={14} />
          </a>
        )}
      </div>
      {match.warnings.map((warning) => (
        <div className="status warning" key={warning}>{warning}</div>
      ))}
    </article>
  );
}

function Score({ label, value, invert = false }: { label: string; value: number | null | undefined; invert?: boolean }) {
  const display = typeof value === "number" && Number.isFinite(value) ? value : null;
  const percent = display === null ? null : Math.max(0, Math.min(100, display * 100));
  const tone = percent === null ? "unknown" : invert ? (percent > 50 ? "bad" : "good") : percent > 50 ? "good" : "mixed";
  return (
    <div className={`score ${tone}`}>
      <span>{label}</span>
      <strong>{percent === null ? "n/a" : `${percent.toFixed(0)}%`}</strong>
    </div>
  );
}

function SettingsDialog({ settings, onChange, onClose }: { settings: ModelSettings; onChange: (settings: ModelSettings) => void; onClose: () => void }) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true">
      <div className="modal">
        <div className="modal-head">
          <h2>Settings</h2>
          <button className="icon-button" onClick={onClose} type="button">x</button>
        </div>
        <label>
          LLM model
          <input value={settings.llmModelId} onChange={(event) => onChange({ ...settings, llmModelId: event.target.value })} />
        </label>
        <label>
          TrialSpace model
          <input value={settings.trialSpaceModelId} onChange={(event) => onChange({ ...settings, trialSpaceModelId: event.target.value })} />
        </label>
        <label>
          TrialChecker model
          <input value={settings.trialCheckerModelId} onChange={(event) => onChange({ ...settings, trialCheckerModelId: event.target.value })} />
        </label>
        <label>
          BoilerplateChecker model
          <input value={settings.boilerplateCheckerModelId} onChange={(event) => onChange({ ...settings, boilerplateCheckerModelId: event.target.value })} />
        </label>
        <label>
          PDF OCR
          <select value={settings.pdfOcrMode} onChange={(event) => onChange({ ...settings, pdfOcrMode: event.target.value as ModelSettings["pdfOcrMode"] })}>
            <option value="auto">Auto: Granite WebGPU, then browser fallback</option>
            <option value="granite">Granite WebGPU only</option>
            <option value="browser">Browser only: Tesseract.js</option>
            <option value="local">Local CLI only: Docling/OCRmyPDF</option>
          </select>
        </label>
        <div className="settings-grid">
          <label>
            Retrieval count
            <input type="number" min={1} max={200} value={settings.retrievalCount} onChange={(event) => onChange({ ...settings, retrievalCount: Number(event.target.value) })} />
          </label>
          <label>
            Display count
            <input type="number" min={1} max={50} value={settings.displayCount} onChange={(event) => onChange({ ...settings, displayCount: Number(event.target.value) })} />
          </label>
          <label>
            Summary tokens
            <input type="number" min={100} max={4000} value={settings.maxSummaryTokens} onChange={(event) => onChange({ ...settings, maxSummaryTokens: Number(event.target.value) })} />
          </label>
          <label className="checkbox-label">
            <input type="checkbox" checked={settings.runDeepScreen} onChange={(event) => onChange({ ...settings, runDeepScreen: event.target.checked })} />
            Deep screen
          </label>
        </div>
      </div>
    </div>
  );
}

function trialDocumentForExtraction(record: TrialSpaceRecord): string {
  return [
    record.title,
    record.overallStatus ? `Status: ${record.overallStatus}` : "",
    record.phases?.length ? `Phase: ${record.phases.join(", ")}` : "",
    record.conditions?.length ? `Conditions: ${record.conditions.join(", ")}` : "",
    record.trialSpaceText
  ].filter(Boolean).join("\n\n");
}

function parseExtractedTrialSpaces(record: TrialSpaceRecord, response: string): TrialSpaceRecord[] {
  const boilerplateText = extractTrialBoilerplate(response) || record.boilerplateText;
  const spaceText = response.split(/^\s*Boilerplate exclusions\s*:/im)[0] ?? response;
  const matches = Array.from(spaceText.matchAll(/(?:^|\n)\s*\d+[\.)]\s*(Age range allowed:.*?)(?=\n\s*\d+[\.)]|\n\s*Boilerplate exclusions\s*:|$)/gis));
  return matches.map((match, index) => ({
    ...record,
    spaceId: `${record.nctId}-llm-space-${index + 1}`,
    trialSpaceText: normalizeExtractedSpace(match[1]),
    boilerplateText,
    embedding: undefined
  }));
}

function extractTrialBoilerplate(response: string): string {
  const marker = response.match(/^\s*Boilerplate exclusions\s*:/im);
  if (!marker || marker.index === undefined) return "";
  return response
    .slice(marker.index + marker[0].length)
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*[-*]\s*/, "").trim())
    .filter(Boolean)
    .join("\n");
}

function normalizeExtractedSpace(text: string): string {
  return text
    .replace(/\s+/g, " ")
    .replace(/^\s*[-*]\s*/, "")
    .trim();
}

function parseFinalScore(text: string): number | null {
  const match = text.match(/final\s+score\s*:\s*(\d)/i);
  return match ? Math.min(5, Number(match[1])) : null;
}

function parseYesNo(text: string): boolean | null {
  const tail = text.trim().slice(-20).toLowerCase();
  if (tail.includes("yes!")) return true;
  if (tail.includes("no!")) return false;
  return null;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
