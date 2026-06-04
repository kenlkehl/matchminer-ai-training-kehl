const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const http = require("node:http");
const https = require("node:https");
const path = require("node:path");
const { Transform } = require("node:stream");
const { pipeline } = require("node:stream/promises");

const DEFAULT_LLAMA_REPO = "ksg-dfci/OncoReasoning-0526-GGUF";
const DEFAULT_LLAMA_FILE = "OncoReasoning-0526.Q4_K_M.gguf";
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
  if (await pathExists(target)) {
    const stat = await fs.stat(target);
    emitProgress(options.onProgress, {
      status: "cached",
      modelId: repo,
      file,
      loadedBytes: stat.size,
      totalBytes: stat.size,
      progress: 100,
      detail: "already cached"
    });
    return target;
  }

  await fs.mkdir(dir, { recursive: true });
  const url = `https://huggingface.co/${repo}/resolve/main/${encodePathSegmented(file)}?download=true`;
  emitProgress(options.onProgress, {
    status: "download",
    modelId: repo,
    file,
    progress: 0,
    detail: "starting download"
  });
  await downloadFile(url, target, {
    sha256: options.sha256,
    onProgress: (progress) => emitProgress(options.onProgress, {
      status: "progress",
      modelId: repo,
      file,
      loadedBytes: progress.loaded,
      totalBytes: progress.total,
      progress: progress.percent,
      detail: "downloading"
    })
  });
  const stat = await fs.stat(target);
  emitProgress(options.onProgress, {
    status: "done",
    modelId: repo,
    file,
    loadedBytes: stat.size,
    totalBytes: stat.size,
    progress: 100,
    detail: "downloaded"
  });
  return target;
}

async function ensureOnnxModel(app, modelId, options = {}) {
  if (path.isAbsolute(modelId) || modelId.startsWith(".") || modelId.includes(path.sep)) {
    const resolved = path.resolve(modelId);
    if (await pathExists(resolved)) {
      emitProgress(options.onProgress, {
        status: "cached",
        modelId,
        progress: 100,
        detail: "using local path"
      });
      return resolved;
    }
  }

  const dir = modelCacheDir(app, modelId);
  const marker = path.join(dir, ".complete.json");
  if (await pathExists(marker)) {
    emitProgress(options.onProgress, {
      status: "cached",
      modelId,
      progress: 100,
      detail: "already cached"
    });
    return dir;
  }

  await fs.mkdir(dir, { recursive: true });
  emitProgress(options.onProgress, {
    status: "checking",
    modelId,
    progress: 0,
    detail: "listing repository files"
  });
  const files = await listHuggingFaceFiles(modelId);
  const downloadableFiles = files.filter((file) => shouldDownloadRepoFile(file.path));
  const loadedByFile = new Map();
  const totalByFile = new Map(downloadableFiles.map((file) => [file.path, file.size || 0]));
  const totalFiles = downloadableFiles.length;

  function reportFile(file, fileIndex, status, loaded, total, detail) {
    loadedByFile.set(file.path, loaded);
    if (total > 0) totalByFile.set(file.path, total);
    const loadedBytes = Array.from(loadedByFile.values()).reduce((sum, value) => sum + value, 0);
    const knownTotalBytes = Array.from(totalByFile.values()).reduce((sum, value) => sum + value, 0);
    const fileProgress = total > 0 ? loaded / total : status === "done" || status === "cached" ? 1 : 0;
    const progress = knownTotalBytes > 0
      ? Math.min(100, (loadedBytes / knownTotalBytes) * 100)
      : totalFiles > 0
        ? Math.min(100, ((fileIndex + fileProgress) / totalFiles) * 100)
        : 100;
    emitProgress(options.onProgress, {
      status,
      modelId,
      file: file.path,
      current: Math.min(fileIndex + 1, totalFiles),
      total: totalFiles,
      loadedBytes,
      totalBytes: knownTotalBytes || undefined,
      fileLoadedBytes: loaded,
      fileTotalBytes: total || undefined,
      progress,
      detail
    });
  }

  for (let index = 0; index < downloadableFiles.length; index += 1) {
    const file = downloadableFiles[index];
    const target = path.join(dir, ...file.path.split("/"));
    if (await pathExists(target)) {
      const stat = await fs.stat(target);
      if (!file.size || stat.size === file.size) {
        reportFile(file, index, "cached", file.size || stat.size, file.size || stat.size, "file already cached");
        continue;
      }
    }
    const url = `https://huggingface.co/${modelId}/resolve/main/${encodePathSegmented(file.path)}?download=true`;
    reportFile(file, index, "download", 0, file.size || 0, "starting download");
    await downloadFile(url, target, {
      onProgress: (progress) => reportFile(
        file,
        index,
        "progress",
        progress.loaded,
        file.size || progress.total || 0,
        "downloading"
      )
    });
    const stat = await fs.stat(target);
    reportFile(file, index, "done", file.size || stat.size, file.size || stat.size, "downloaded");
  }
  await fs.writeFile(marker, JSON.stringify({ modelId, completedAt: new Date().toISOString() }, null, 2) + "\n", "utf8");
  emitProgress(options.onProgress, {
    status: "done",
    modelId,
    progress: 100,
    detail: "model cache complete"
  });
  return dir;
}

async function ensureDefaultArtifacts(app, options = {}) {
  const artifacts = [
    { kind: "llama", modelId: DEFAULT_LLAMA_REPO },
    ...DEFAULT_ONNX_MODELS.map((modelId) => ({ kind: "onnx", modelId }))
  ];
  function scopedProgress(index) {
    return (progress) => emitProgress(options.onProgress, {
      ...progress,
      current: index + 1,
      total: artifacts.length,
      overallPercent: steppedProgress(index, progress.progress, artifacts.length)
    });
  }

  const llamaPath = await ensureLlamaModel(app, { onProgress: scopedProgress(0) });
  const onnxDirs = {};
  for (let index = 0; index < DEFAULT_ONNX_MODELS.length; index += 1) {
    const modelId = DEFAULT_ONNX_MODELS[index];
    onnxDirs[modelId] = await ensureOnnxModel(app, modelId, { onProgress: scopedProgress(index + 1) });
  }
  emitProgress(options.onProgress, {
    status: "done",
    current: artifacts.length,
    total: artifacts.length,
    progress: 100,
    overallPercent: 100,
    detail: "all runtime artifacts are cached"
  });
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
  const response = await downloadToFile(url, tmp, {
    redirects,
    onProgress: options.onProgress
  });
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

function downloadToFile(url, target, options = {}) {
  return new Promise((resolve, reject) => {
    const redirects = options.redirects ?? 0;
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
        resolve(downloadToFile(nextUrl, target, { ...options, redirects: redirects + 1 }));
        return;
      }
      try {
        const total = Number(res.headers["content-length"]) || undefined;
        let loaded = 0;
        const progressStream = new Transform({
          transform(chunk, _encoding, callback) {
            loaded += chunk.length;
            emitProgress(options.onProgress, {
              loaded,
              total,
              percent: total ? Math.min(100, (loaded / total) * 100) : undefined
            });
            callback(null, chunk);
          }
        });
        await pipeline(res, progressStream, require("node:fs").createWriteStream(target));
        resolve({ statusCode: res.statusCode, headers: res.headers });
      } catch (error) {
        reject(error);
      }
    });
    req.on("error", reject);
    req.end();
  });
}

function emitProgress(onProgress, progress) {
  if (typeof onProgress !== "function") return;
  try {
    onProgress(progress);
  } catch {
    // Progress reporting must never break artifact preparation.
  }
}

function steppedProgress(index, nestedPercent, total) {
  const safeTotal = Math.max(1, total || 1);
  const safePercent = Number.isFinite(nestedPercent) ? Math.max(0, Math.min(100, nestedPercent)) : 0;
  return Math.max(0, Math.min(100, ((index + safePercent / 100) / safeTotal) * 100));
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
