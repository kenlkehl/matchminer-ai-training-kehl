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
import { retrieveByEmbedding, scoreAndRankMatches } from "./services/matching";

const promptOrder: PromptKey[] = ["patientSummary", "trialSpaceExtraction", "trialDeepScreen", "boilerplateDeepScreen"];

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
      const doc = lower.endsWith(".csv") ? await parseCsvPatientFile(file) : await parsePdfPatientFile(file, setPdfProgress);
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
    try {
      const records = await fetchCtGovCancerTrials({ pageSize: 100, maxPages: 3 });
      await saveTrialIndex(records);
      setTrialIndex(records);
      addStatus("success", `Downloaded ${records.length} public trial records from ClinicalTrials.gov.`);
    } catch (error) {
      addStatus("error", errorMessage(error));
    } finally {
      setBusy(null);
    }
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
    try {
      let records = trialIndex.length ? trialIndex : await loadOrFetchTrialIndex();
      let patientEmbedding: number[];
      try {
        records = await ensureTrialEmbeddings(records, settings.trialSpaceModelId, (done, total) => {
          if (done === total || done % 10 === 0) addStatus("info", `Embedded ${done} of ${total} trial spaces.`);
        });
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
            {pdfProgress && <ProgressLine label={`${pdfProgress.phase.toUpperCase()} ${pdfProgress.current}/${pdfProgress.total}`} />}
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
            {busy && <ProgressLine label={busy} />}
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
