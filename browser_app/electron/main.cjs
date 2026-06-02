const { app, BrowserWindow, ipcMain, net, protocol, shell } = require("electron");
const { spawn } = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { ensureDefaultArtifacts } = require("./runtime/artifacts.cjs");
const { LlamaRuntime } = require("./runtime/llama-runtime.cjs");
const { OnnxRuntime } = require("./runtime/onnx-runtime.cjs");

const APP_SCHEME = "matchminer";
const DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL ?? "http://127.0.0.1:5173";
const LOCAL_OCR_TIMEOUT_MS = Number(process.env.MATCHMINER_LOCAL_OCR_TIMEOUT_MS ?? 10 * 60 * 1000);
const isDev = !app.isPackaged;
let appProtocolRegistered = false;
const llamaRuntime = new LlamaRuntime(app);
const onnxRuntime = new OnnxRuntime(app);

protocol.registerSchemesAsPrivileged([
  {
    scheme: APP_SCHEME,
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true
    }
  }
]);

if (process.env.MATCHMINER_DISABLE_GPU_FLAGS !== "1") {
  app.commandLine.appendSwitch("enable-unsafe-webgpu");
  app.commandLine.appendSwitch("ignore-gpu-blocklist");
  if (process.platform === "linux") {
    app.commandLine.appendSwitch("enable-features", "Vulkan");
  }
}

app.setName("MatchMiner AI");

ipcMain.handle("matchminer:local-pdf-ocr", async (_event, request) => runLocalPdfOcr(request));
ipcMain.handle("matchminer:runtime-status", async () => ({
  llama: llamaRuntime.status(),
  onnx: onnxRuntime.status()
}));
ipcMain.handle("matchminer:prepare-runtime-artifacts", async (event, request) =>
  ensureDefaultArtifacts(app, { onProgress: createProgressEmitter(event, request?.progressToken) })
);
ipcMain.handle("matchminer:warm-runtime", async (event, request) =>
  warmRuntime(request, createProgressEmitter(event, request?.progressToken))
);
ipcMain.handle("matchminer:generate-text", async (_event, request) => llamaRuntime.generate(request));
ipcMain.handle("matchminer:tokenize-text", async (_event, request) => llamaRuntime.tokenize(request));
ipcMain.handle("matchminer:detokenize-tokens", async (_event, request) => llamaRuntime.detokenize(request));
ipcMain.handle("matchminer:embed-trial-space-texts", async (_event, request) =>
  onnxRuntime.embedTexts(request?.modelId, request?.texts ?? [])
);
ipcMain.handle("matchminer:score-trial-checker-texts", async (_event, request) =>
  onnxRuntime.scoreTrialChecker(request?.modelId, request?.texts ?? [])
);
ipcMain.handle("matchminer:score-boilerplate-checker-texts", async (_event, request) =>
  onnxRuntime.scoreBoilerplateChecker(request?.modelId, request?.texts ?? [])
);

async function createWindow() {
  await registerAppProtocol();

  const mainWindow = new BrowserWindow({
    width: 1600,
    height: 900,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: "#f5f7fb",
    title: "MatchMiner AI",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true
    }
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    openAllowedExternalUrl(url);
    return { action: "deny" };
  });

  mainWindow.webContents.on("will-navigate", (event, url) => {
    const currentUrl = mainWindow.webContents.getURL();
    if (url !== currentUrl && isExternalUrl(url)) {
      event.preventDefault();
      openAllowedExternalUrl(url);
    }
  });

  if (isDev) {
    await mainWindow.loadURL(DEV_SERVER_URL);
    if (process.env.MATCHMINER_OPEN_DEVTOOLS === "1") {
      mainWindow.webContents.openDevTools({ mode: "detach" });
    }
  } else {
    await mainWindow.loadURL(`${APP_SCHEME}://app/index.html`);
  }
}

async function registerAppProtocol() {
  if (isDev || appProtocolRegistered) return;

  const appRoot = app.getAppPath();
  const distRoot = path.join(appRoot, "dist");

  protocol.handle(APP_SCHEME, async (request) => {
    const requestUrl = new URL(request.url);
    if (requestUrl.host !== "app") {
      return new Response("Unknown MatchMiner resource host", { status: 404 });
    }

    const relativePath = safeRelativePath(requestUrl.pathname);
    const filePath = path.join(distRoot, relativePath || "index.html");

    if (!isPathInside(filePath, distRoot)) {
      return new Response("Blocked path traversal", { status: 403 });
    }

    try {
      const response = await net.fetch(pathToFileURL(filePath).toString());
      return withSecurityHeaders(response);
    } catch {
      return new Response("Resource not found", { status: 404 });
    }
  });
  appProtocolRegistered = true;
}

function safeRelativePath(pathname) {
  const decoded = decodeURIComponent(pathname);
  const withoutLeadingSlash = decoded.replace(/^\/+/, "");
  const normalized = path.normalize(withoutLeadingSlash);
  if (normalized === "." || normalized === "") return "index.html";
  return normalized;
}

function isPathInside(filePath, rootPath) {
  const relative = path.relative(rootPath, filePath);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function isExternalUrl(url) {
  return url.startsWith("http://") || url.startsWith("https://");
}

function openAllowedExternalUrl(url) {
  if (!isExternalUrl(url)) return;
  const { hostname, protocol } = new URL(url);
  const allowedHosts = new Set(["clinicaltrials.gov", "www.clinicaltrials.gov", "huggingface.co"]);
  if (protocol === "https:" && allowedHosts.has(hostname)) {
    void shell.openExternal(url);
  }
}

function withSecurityHeaders(response) {
  const headers = new Headers(response.headers);
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  headers.set("Cross-Origin-Resource-Policy", "same-origin");
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers
  });
}

async function warmRuntime(request, onProgress) {
  const task = request?.task;
  if (task === "text-generation") {
    return llamaRuntime.warm({
      contextTokens: request?.contextTokens,
      repo: request?.llamaModelRepo,
      file: request?.llamaModelFile,
      onProgress
    });
  }
  if (task === "feature-extraction" || task === "text-classification") {
    return onnxRuntime.warm(request?.modelId, task, { onProgress });
  }
  throw new Error(`Unsupported runtime warmup task: ${task}`);
}

function createProgressEmitter(event, progressToken) {
  if (!progressToken) return undefined;
  return (progress) => {
    if (event.sender.isDestroyed()) return;
    event.sender.send("matchminer:runtime-progress", { ...progress, progressToken });
  };
}

async function runLocalPdfOcr(request) {
  const fileName = typeof request?.fileName === "string" ? request.fileName : "uploaded.pdf";
  const bytes = request?.bytes;
  if (!(bytes instanceof ArrayBuffer) && !ArrayBuffer.isView(bytes)) {
    throw new Error("Local OCR request did not include PDF bytes");
  }
  const buffer = Buffer.from(bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : bytes);
  if (!buffer.length) throw new Error("Local OCR request included an empty PDF");

  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "matchminer-ocr-"));
  const inputPath = path.join(tempRoot, safeTempPdfName(fileName));
  await fs.writeFile(inputPath, buffer);
  try {
    const attempts = [];
    attempts.push(await runDoclingOcr(inputPath).catch((error) => ({ ok: false, engine: "Docling", error })));
    if (attempts.at(-1).ok) return attempts.at(-1);

    attempts.push(await runOcrMyPdf(inputPath, true).catch((error) => ({ ok: false, engine: "OCRmyPDF clean", error })));
    if (attempts.at(-1).ok) return attempts.at(-1);

    attempts.push(await runOcrMyPdf(inputPath, false).catch((error) => ({ ok: false, engine: "OCRmyPDF", error })));
    if (attempts.at(-1).ok) return attempts.at(-1);

    const details = attempts.map((attempt) => `${attempt.engine}: ${errorMessage(attempt.error)}`).join("; ");
    throw new Error(`No local OCR backend succeeded. ${details}`);
  } finally {
    await fs.rm(tempRoot, { recursive: true, force: true });
  }
}

async function runDoclingOcr(inputPath) {
  const pythonScript = `
import sys
from docling.document_converter import DocumentConverter

source = sys.argv[1]
converter = DocumentConverter()
document = converter.convert(source).document
print(document.export_to_markdown())
`;
  const candidates = uniqueValues([process.env.MATCHMINER_DOCLING_PYTHON, "python3", "python"].filter(Boolean));
  const failures = [];
  for (const command of candidates) {
    const result = await runCommand(command, ["-c", pythonScript, inputPath]).catch((error) => {
      failures.push(`${command}: ${errorMessage(error)}`);
      return null;
    });
    if (!result) continue;
    const text = result.stdout.trim();
    if (text) return { ok: true, engine: `Docling (${command})`, text };
    failures.push(`${command}: no text returned`);
  }
  throw new Error(failures.join("; ") || "Docling is unavailable");
}

async function runOcrMyPdf(inputPath, clean) {
  const command = process.env.MATCHMINER_OCRMYPDF_COMMAND || "ocrmypdf";
  const sidecarPath = path.join(path.dirname(inputPath), clean ? "ocr-clean.txt" : "ocr.txt");
  const outputPath = path.join(path.dirname(inputPath), clean ? "ocr-clean.pdf" : "ocr.pdf");
  const args = ["--skip-text", "--rotate-pages", "--deskew", "--sidecar", sidecarPath];
  if (clean) args.splice(3, 0, "--clean");
  args.push(inputPath, outputPath);
  await runCommand(command, args);
  const text = (await fs.readFile(sidecarPath, "utf8")).trim();
  if (!text) throw new Error("OCRmyPDF sidecar text was empty");
  return { ok: true, engine: clean ? "OCRmyPDF clean" : "OCRmyPDF", text };
}

function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      env: process.env,
      shell: false,
      windowsHide: true
    });
    const stdout = [];
    const stderr = [];
    let settled = false;
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish(new Error(`${command} timed out after ${Math.round(LOCAL_OCR_TIMEOUT_MS / 1000)}s`));
    }, LOCAL_OCR_TIMEOUT_MS);

    child.stdout.on("data", (chunk) => stdout.push(chunk));
    child.stderr.on("data", (chunk) => stderr.push(chunk));
    child.on("error", (error) => finish(error));
    child.on("close", (code) => {
      if (code === 0) {
        finish(null, {
          stdout: Buffer.concat(stdout).toString("utf8"),
          stderr: Buffer.concat(stderr).toString("utf8")
        });
      } else {
        const errText = Buffer.concat(stderr).toString("utf8").trim();
        finish(new Error(`${command} exited ${code}${errText ? `: ${errText.slice(0, 2000)}` : ""}`));
      }
    });

    function finish(error, result) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error);
      else resolve(result);
    }
  });
}

function safeTempPdfName(fileName) {
  const base = path.basename(fileName).replace(/[^A-Za-z0-9._-]/g, "_") || "uploaded.pdf";
  const withExtension = /\.pdf$/i.test(base) ? base : `${base}.pdf`;
  return `${crypto.randomUUID()}-${withExtension}`;
}

function uniqueValues(values) {
  return [...new Set(values)];
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

const gotLock = app.requestSingleInstanceLock();

if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    const windows = BrowserWindow.getAllWindows();
    const mainWindow = windows[0];
    if (!mainWindow) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  });

  app.whenReady().then(createWindow).catch((error) => {
    console.error(error);
    app.quit();
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void createWindow();
    }
  });

  app.on("before-quit", () => {
    void llamaRuntime.stop();
    void onnxRuntime.stop();
  });
}

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
