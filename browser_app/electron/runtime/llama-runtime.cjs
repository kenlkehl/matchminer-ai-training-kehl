const { spawn } = require("node:child_process");
const fs = require("node:fs/promises");
const http = require("node:http");
const net = require("node:net");
const path = require("node:path");
const { DEFAULT_LLAMA_FILE, DEFAULT_LLAMA_REPO, downloadFile, ensureLlamaModel, pathExists, requestText } = require("./artifacts.cjs");

const DEFAULT_CONTEXT_TOKENS = 32768;
const SERVER_START_TIMEOUT_MS = 180000;
const GENERATION_SYSTEM_PROMPT = "Reasoning: high";

class LlamaRuntime {
  constructor(app) {
    this.app = app;
    this.child = null;
    this.port = null;
    this.modelPath = null;
    this.contextTokens = null;
    this.starting = null;
    this.stopping = false;
  }

  async warm(options = {}) {
    await this.ensureStarted(options);
    return this.status();
  }

  status() {
    return {
      running: Boolean(this.child && !this.child.killed),
      port: this.port,
      modelPath: this.modelPath,
      contextTokens: this.contextTokens
    };
  }

  async generate({ prompt, maxNewTokens, contextTokens, repo, file, llamaModelRepo, llamaModelFile, systemPrompt }) {
    await this.ensureStarted({ contextTokens, repo: repo || llamaModelRepo, file: file || llamaModelFile });
    const response = await postJson(this.baseUrl("/v1/chat/completions"), {
      model: "matchminer-local",
      messages: [
        { role: "system", content: String(systemPrompt || GENERATION_SYSTEM_PROMPT) },
        { role: "user", content: String(prompt ?? "") }
      ],
      max_tokens: Math.max(1, Math.floor(maxNewTokens ?? 512)),
      temperature: 0.2,
      stream: false
    }, 30 * 60 * 1000);
    const text = response?.choices?.[0]?.message?.content ?? response?.choices?.[0]?.text;
    if (typeof text !== "string") throw new Error("llama.cpp returned no generated text");
    return text;
  }

  async tokenize({ text, contextTokens, repo, file, llamaModelRepo, llamaModelFile }) {
    await this.ensureStarted({ contextTokens, repo: repo || llamaModelRepo, file: file || llamaModelFile });
    const response = await postJson(this.baseUrl("/tokenize"), {
      content: String(text ?? ""),
      add_special: false,
      parse_special: true
    });
    if (!Array.isArray(response?.tokens)) throw new Error("llama.cpp returned no tokens");
    return response.tokens.map((token) => typeof token === "object" ? token.id : token).filter(Number.isFinite);
  }

  async detokenize({ tokens, contextTokens, repo, file, llamaModelRepo, llamaModelFile }) {
    await this.ensureStarted({ contextTokens, repo: repo || llamaModelRepo, file: file || llamaModelFile });
    const response = await postJson(this.baseUrl("/detokenize"), { tokens: Array.isArray(tokens) ? tokens : [] });
    return String(response?.content ?? response?.text ?? "");
  }

  async ensureStarted(options = {}) {
    if (this.starting) return this.starting;

    const contextTokens = normalizeContext(options.contextTokens ?? this.contextTokens ?? DEFAULT_CONTEXT_TOKENS);
    const repo = options.repo || DEFAULT_LLAMA_REPO;
    const file = options.file || DEFAULT_LLAMA_FILE;
    const modelPath = await ensureLlamaModel(this.app, { repo, file });

    if (this.child && this.modelPath === modelPath && this.contextTokens === contextTokens) return;
    await this.stop();

    this.starting = this.startServer({ modelPath, contextTokens }).finally(() => {
      this.starting = null;
    });
    return this.starting;
  }

  async startServer({ modelPath, contextTokens }) {
    const command = await findLlamaServerCommand(this.app);
    const port = await getFreePort();
    const args = [
      "--host", "127.0.0.1",
      "--port", String(port),
      "-m", modelPath,
      "-c", String(contextTokens),
      "-np", "1"
    ];

    const child = spawn(command, args, {
      env: process.env,
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"]
    });

    const stderr = [];
    child.stderr.on("data", (chunk) => {
      stderr.push(chunk);
      if (stderr.length > 64) stderr.shift();
    });
    child.stdout.on("data", () => {});
    child.on("close", () => {
      if (this.child === child) {
        this.child = null;
        this.port = null;
      }
    });

    this.child = child;
    this.port = port;
    this.modelPath = modelPath;
    this.contextTokens = contextTokens;

    try {
      await Promise.race([
        waitForHealth(this.baseUrl("/health"), SERVER_START_TIMEOUT_MS),
        new Promise((_, reject) => child.once("error", reject))
      ]);
    } catch (error) {
      const details = Buffer.concat(stderr).toString("utf8").trim().slice(-2000);
      await this.stop();
      throw new Error(`Could not start llama.cpp server with ${command}.${details ? ` ${details}` : ""} ${error.message}`);
    }
  }

  async stop() {
    if (!this.child) return;
    const child = this.child;
    this.child = null;
    this.port = null;
    this.stopping = true;
    await new Promise((resolve) => {
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        resolve();
      }, 3000);
      child.once("close", () => {
        clearTimeout(timer);
        resolve();
      });
      child.kill("SIGTERM");
    });
    this.stopping = false;
  }

  baseUrl(pathname) {
    if (!this.port) throw new Error("llama.cpp server is not running");
    return `http://127.0.0.1:${this.port}${pathname}`;
  }
}

async function findLlamaServerCommand(app) {
  const exe = process.platform === "win32" ? "llama-server.exe" : "llama-server";
  const candidates = [
    process.env.MATCHMINER_LLAMA_SERVER_COMMAND,
    process.resourcesPath ? path.join(process.resourcesPath, "bin", exe) : null,
    path.join(__dirname, "..", "..", "resources", "bin", exe)
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (await pathExists(candidate)) return candidate;
  }
  const cached = await findFile(path.join(app.getPath("userData"), "bin", "llama.cpp"), exe).catch(() => null);
  if (cached) return cached;
  const downloaded = await ensureDownloadedLlamaServer(app, exe).catch(() => null);
  if (downloaded) return downloaded;
  return exe;
}

async function ensureDownloadedLlamaServer(app, exe) {
  if (process.platform !== "linux" || !["x64", "arm64"].includes(process.arch)) return null;
  const binDir = path.join(app.getPath("userData"), "bin");
  const installRoot = path.join(binDir, "llama.cpp");
  const cached = await findFile(installRoot, exe).catch(() => null);
  if (cached) return cached;

  await fs.mkdir(binDir, { recursive: true });
  const release = JSON.parse(await requestText("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"));
  const archLabel = process.arch === "x64" ? "x64" : "arm64";
  const asset = (release.assets ?? []).find((item) =>
    item?.browser_download_url &&
    new RegExp(`^llama-b\\d+-bin-ubuntu-${archLabel}\\.tar\\.gz$`).test(item.name ?? "")
  );
  if (!asset) return null;

  const archive = path.join(binDir, asset.name);
  const extractDir = path.join(binDir, `${asset.name}.extract`);
  const installDir = path.join(installRoot, asset.name.replace(/[^A-Za-z0-9._-]/g, "_"));
  await downloadFile(asset.browser_download_url, archive);
  await fs.rm(extractDir, { recursive: true, force: true });
  await fs.rm(installDir, { recursive: true, force: true });
  await fs.mkdir(extractDir, { recursive: true });
  await runProcess("tar", ["-xzf", archive, "-C", extractDir], 120000);
  const extracted = await findFile(extractDir, exe);
  if (!extracted) throw new Error(`Downloaded llama.cpp release did not contain ${exe}`);
  await fs.mkdir(path.dirname(installDir), { recursive: true });
  await fs.cp(path.dirname(extracted), installDir, { recursive: true });
  const target = path.join(installDir, exe);
  await fs.chmod(target, 0o755);
  await fs.rm(extractDir, { recursive: true, force: true });
  return target;
}

async function findFile(root, fileName) {
  const entries = await fs.readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(root, entry.name);
    if (entry.isFile() && entry.name === fileName) return fullPath;
    if (entry.isDirectory()) {
      const found = await findFile(fullPath, fileName);
      if (found) return found;
    }
  }
  return null;
}

function runProcess(command, args, timeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { shell: false, windowsHide: true });
    const stderr = [];
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`${command} timed out`));
    }, timeoutMs);
    child.stderr.on("data", (chunk) => stderr.push(chunk));
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolve();
      else reject(new Error(`${command} exited ${code}: ${Buffer.concat(stderr).toString("utf8").slice(0, 1000)}`));
    });
  });
}

function normalizeContext(value) {
  return Number.isFinite(Number(value)) ? Math.max(1024, Math.floor(Number(value))) : DEFAULT_CONTEXT_TOKENS;
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
    server.on("error", reject);
  });
}

async function waitForHealth(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const response = await getJson(url, 2000);
      if (response) return;
    } catch (error) {
      lastError = error;
    }
    await delay(500);
  }
  throw lastError || new Error("Timed out waiting for llama.cpp health endpoint");
}

function getJson(url, timeoutMs = 10000) {
  return requestJson("GET", url, null, timeoutMs);
}

function postJson(url, body, timeoutMs = 120000) {
  return requestJson("POST", url, body, timeoutMs);
}

function requestJson(method, url, body, timeoutMs) {
  return new Promise((resolve, reject) => {
    const payload = body === null ? null : Buffer.from(JSON.stringify(body));
    const parsed = new URL(url);
    const req = http.request(parsed, {
      method,
      headers: payload ? {
        "Content-Type": "application/json",
        "Content-Length": payload.length
      } : undefined
    }, (res) => {
      const chunks = [];
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => {
        const text = Buffer.concat(chunks).toString("utf8");
        if (res.statusCode < 200 || res.statusCode >= 300) {
          reject(new Error(`llama.cpp ${method} ${parsed.pathname} failed ${res.statusCode}: ${text.slice(0, 1000)}`));
          return;
        }
        try {
          resolve(text ? JSON.parse(text) : {});
        } catch (error) {
          reject(new Error(`llama.cpp returned invalid JSON: ${error.message}`));
        }
      });
    });
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`llama.cpp ${method} ${parsed.pathname} timed out`));
    });
    req.on("error", reject);
    if (payload) req.write(payload);
    req.end();
  });
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

module.exports = {
  LlamaRuntime
};
