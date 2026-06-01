const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const http = require("node:http");
const https = require("node:https");
const path = require("node:path");
const { pipeline } = require("node:stream/promises");

const DEFAULT_LLAMA_REPO = "LiquidAI/LFM2.5-1.2B-Thinking-GGUF";
const DEFAULT_LLAMA_FILE = "LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
const DEFAULT_ONNX_MODELS = [
  "ksg-dfci/TrialSpace-0526-ONNX",
  "ksg-dfci/TrialChecker-0526-ONNX",
  "ksg-dfci/BoilerplateChecker-0526-ONNX"
];

function safeModelName(modelId) {
  return String(modelId).replace(/[^A-Za-z0-9._-]/g, "--");
}

function encodePathSegmented(value) {
  return String(value).split("/").map(encodeURIComponent).join("/");
}

function modelCacheRoot(app) {
  return path.join(app.getPath("userData"), "models");
}

function modelCacheDir(app, modelId) {
  return path.join(modelCacheRoot(app), safeModelName(modelId));
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function ensureLlamaModel(app, options = {}) {
  const repo = options.repo || DEFAULT_LLAMA_REPO;
  const file = options.file || DEFAULT_LLAMA_FILE;
  const dir = modelCacheDir(app, repo);
  const target = path.join(dir, file);
  if (await pathExists(target)) return target;

  await fs.mkdir(dir, { recursive: true });
  const url = `https://huggingface.co/${repo}/resolve/main/${encodePathSegmented(file)}?download=true`;
  await downloadFile(url, target, { sha256: options.sha256 });
  return target;
}

async function ensureOnnxModel(app, modelId) {
  if (path.isAbsolute(modelId) || modelId.startsWith(".") || modelId.includes(path.sep)) {
    const resolved = path.resolve(modelId);
    if (await pathExists(resolved)) return resolved;
  }

  const dir = modelCacheDir(app, modelId);
  const marker = path.join(dir, ".complete.json");
  if (await pathExists(marker)) return dir;

  await fs.mkdir(dir, { recursive: true });
  const files = await listHuggingFaceFiles(modelId);
  for (const file of files) {
    if (!shouldDownloadRepoFile(file.path)) continue;
    const target = path.join(dir, ...file.path.split("/"));
    if (await pathExists(target)) {
      const stat = await fs.stat(target);
      if (!file.size || stat.size === file.size) continue;
    }
    const url = `https://huggingface.co/${modelId}/resolve/main/${encodePathSegmented(file.path)}?download=true`;
    await downloadFile(url, target);
  }
  await fs.writeFile(marker, JSON.stringify({ modelId, completedAt: new Date().toISOString() }, null, 2) + "\n", "utf8");
  return dir;
}

async function ensureDefaultArtifacts(app) {
  const llamaPath = await ensureLlamaModel(app);
  const onnxDirs = {};
  for (const modelId of DEFAULT_ONNX_MODELS) {
    onnxDirs[modelId] = await ensureOnnxModel(app, modelId);
  }
  return { llamaPath, onnxDirs };
}

function shouldDownloadRepoFile(filePath) {
  const base = path.posix.basename(filePath);
  if (base === ".gitattributes") return false;
  return true;
}

async function listHuggingFaceFiles(repo) {
  const url = `https://huggingface.co/api/models/${repo}/tree/main?recursive=1`;
  const body = await requestText(url);
  const parsed = JSON.parse(body);
  if (!Array.isArray(parsed)) throw new Error(`Unexpected Hugging Face tree response for ${repo}`);
  return parsed
    .filter((item) => item && item.type === "file" && typeof item.path === "string")
    .map((item) => ({ path: item.path, size: Number.isFinite(item.size) ? item.size : null }));
}

async function requestText(url, redirects = 0) {
  const response = await request(url, { method: "GET" }, redirects);
  if (response.statusCode < 200 || response.statusCode >= 300) {
    throw new Error(`Request failed ${response.statusCode} for ${url}`);
  }
  return response.body.toString("utf8");
}

async function downloadFile(url, target, options = {}, redirects = 0) {
  await fs.mkdir(path.dirname(target), { recursive: true });
  const tmp = `${target}.${process.pid}.${Date.now()}.tmp`;
  const response = await downloadToFile(url, tmp, redirects);
  if (response.statusCode < 200 || response.statusCode >= 300) {
    await fs.rm(tmp, { force: true }).catch(() => {});
    throw new Error(`Download failed ${response.statusCode} for ${url}`);
  }
  if (options.sha256) {
    const actual = await sha256File(tmp);
    if (actual !== options.sha256) {
      await fs.rm(tmp, { force: true }).catch(() => {});
      throw new Error(`Checksum mismatch for ${path.basename(target)}: expected ${options.sha256}, got ${actual}`);
    }
  }
  await fs.rename(tmp, target);
}

function downloadToFile(url, target, redirects = 0) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const client = parsed.protocol === "http:" ? http : https;
    const req = client.request(parsed, { method: "GET", headers: { "User-Agent": "MatchMiner-AI" } }, async (res) => {
      const location = res.headers.location;
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && location) {
        res.resume();
        if (redirects >= 8) {
          reject(new Error(`Too many redirects for ${url}`));
          return;
        }
        const nextUrl = new URL(location, parsed).toString();
        resolve(downloadToFile(nextUrl, target, redirects + 1));
        return;
      }
      try {
        await pipeline(res, require("node:fs").createWriteStream(target));
        resolve({ statusCode: res.statusCode, headers: res.headers });
      } catch (error) {
        reject(error);
      }
    });
    req.on("error", reject);
    req.end();
  });
}

function request(url, options = {}, redirects = 0) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const client = parsed.protocol === "http:" ? http : https;
    const req = client.request(parsed, { method: options.method || "GET", headers: { "User-Agent": "MatchMiner-AI" } }, (res) => {
      const location = res.headers.location;
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && location) {
        res.resume();
        if (redirects >= 8) {
          reject(new Error(`Too many redirects for ${url}`));
          return;
        }
        const nextUrl = new URL(location, parsed).toString();
        resolve(request(nextUrl, options, redirects + 1));
        return;
      }

      const chunks = [];
      res.on("data", (chunk) => {
        chunks.push(chunk);
      });
      res.on("end", () => resolve({ statusCode: res.statusCode, headers: res.headers, body: Buffer.concat(chunks) }));
      res.on("error", reject);
    });
    req.on("error", reject);
    req.end();
  });
}

async function sha256File(filePath) {
  const hash = crypto.createHash("sha256");
  const handle = await fs.open(filePath, "r");
  try {
    for await (const chunk of handle.createReadStream()) hash.update(chunk);
  } finally {
    await handle.close();
  }
  return hash.digest("hex");
}

module.exports = {
  DEFAULT_LLAMA_FILE,
  DEFAULT_LLAMA_REPO,
  DEFAULT_ONNX_MODELS,
  downloadFile,
  ensureDefaultArtifacts,
  ensureLlamaModel,
  ensureOnnxModel,
  modelCacheDir,
  modelCacheRoot,
  pathExists,
  requestText
};
