const fsSync = require("node:fs");
const fs = require("node:fs/promises");
const path = require("node:path");
const { parentPort } = require("node:worker_threads");

const DEFAULT_TRIALSPACE_MAX_LENGTH = 2500;
const TRIAL_CHECKER_MAX_LENGTH = 4096;
const BOILERPLATE_CHECKER_MAX_LENGTH = 3192;
const MODEL_QUERY_PROMPT =
  "Instruct: Given a cancer patient summary, retrieve clinical trial options that are reasonable for that patient; or, given a clinical trial option, retrieve cancer patients who are reasonable candidates for that trial. ";

const loadedModels = new Map();
let TokenizerClass = null;
let ortModule = null;

parentPort.on("message", async (message) => {
  try {
    const result = await handleMessage(message);
    parentPort.postMessage({ id: message.id, ok: true, result });
  } catch (error) {
    parentPort.postMessage({
      id: message.id,
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

async function handleMessage(message) {
  if (message.type === "load") {
    await loadModel(message.modelId, message.task, message.modelDir);
    return { loaded: true };
  }
  if (message.type === "embed") {
    const model = await loadModel(message.modelId, "feature-extraction", message.modelDir);
    return embedTexts(model, message.texts ?? []);
  }
  if (message.type === "score") {
    const model = await loadModel(message.modelId, "text-classification", message.modelDir);
    return scoreTexts(model, message.texts ?? [], message.transform);
  }
  throw new Error(`Unknown ONNX worker message type: ${message.type}`);
}

async function loadModel(modelId, task, modelDir) {
  const key = `${task}:${modelId}:${modelDir}`;
  if (loadedModels.has(key)) return loadedModels.get(key);

  const ort = loadOnnxRuntime();
  const tokenizer = await loadTokenizer(modelDir);
  const modelPath = await findOnnxModel(modelDir);
  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"]
  });
  const model = { modelId, task, modelDir, tokenizer, session };
  loadedModels.set(key, model);
  return model;
}

async function loadTokenizer(modelDir) {
  if (!TokenizerClass) {
    const module = await import("@huggingface/tokenizers");
    TokenizerClass = module.Tokenizer;
  }
  const tokenizerJson = JSON.parse(await fs.readFile(path.join(modelDir, "tokenizer.json"), "utf8"));
  const tokenizerConfigPath = path.join(modelDir, "tokenizer_config.json");
  const tokenizerConfig = await readJsonIfExists(tokenizerConfigPath, {});
  return {
    tokenizer: new TokenizerClass(tokenizerJson, tokenizerConfig),
    config: tokenizerConfig
  };
}

async function readJsonIfExists(filePath, fallback) {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

async function findOnnxModel(modelDir) {
  const files = await walk(modelDir);
  const onnxFiles = files.filter((file) => file.endsWith(".onnx"));
  if (!onnxFiles.length) throw new Error(`No ONNX model found in ${modelDir}`);
  const quantized = onnxFiles.find((file) => /quant/i.test(path.basename(file)));
  const namedModel = onnxFiles.find((file) => path.basename(file) === "model.onnx");
  return quantized || namedModel || onnxFiles[0];
}

async function walk(root) {
  const out = [];
  const entries = await fs.readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) out.push(...await walk(fullPath));
    else out.push(fullPath);
  }
  return out;
}

async function embedTexts(model, texts) {
  if (!texts.length) return [];
  const encoded = encodeBatch(model.tokenizer, texts.map((text) => MODEL_QUERY_PROMPT + String(text ?? "")), DEFAULT_TRIALSPACE_MAX_LENGTH);
  const result = await runSession(model.session, encoded);
  const output = firstOutput(result);
  return meanPoolNormalize(output, encoded.attentionMask);
}

async function scoreTexts(model, texts, transform) {
  if (!texts.length) return [];
  const maxLength = transform === "sigmoid" ? TRIAL_CHECKER_MAX_LENGTH : BOILERPLATE_CHECKER_MAX_LENGTH;
  const encoded = encodeBatch(model.tokenizer, texts.map((text) => String(text ?? "")), maxLength);
  const result = await runSession(model.session, encoded);
  const output = firstOutput(result);
  const logits = tensorRows(output);
  return logits.map((row) => transform === "sigmoid" ? sigmoid(row[0] ?? 0) : softmaxPositive(row));
}

function encodeBatch(tokenizerBundle, texts, maxLength) {
  const rows = texts.map((text) => encodeOne(tokenizerBundle, text, maxLength));
  const width = Math.max(1, ...rows.map((row) => row.ids.length));
  const batch = rows.length || 1;
  const inputIds = new BigInt64Array(batch * width);
  const attentionMask = new BigInt64Array(batch * width);
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    for (let col = 0; col < row.ids.length; col += 1) {
      inputIds[rowIndex * width + col] = BigInt(row.ids[col]);
      attentionMask[rowIndex * width + col] = 1n;
    }
  }
  return {
    inputIds,
    attentionMask,
    attentionMaskRows: rows.map((row) => row.ids.map(() => 1)),
    dims: [batch, width]
  };
}

function encodeOne(tokenizerBundle, text, maxLength) {
  const encoding = tokenizerBundle.tokenizer.encode(text, { add_special_tokens: true });
  let ids = Array.from(encoding.ids ?? []);
  if (ids.length > maxLength) {
    ids = ids.slice(0, maxLength);
    const sepToken = tokenizerBundle.config?.sep_token;
    const sepId = sepToken ? tokenizerBundle.tokenizer.token_to_id(sepToken) : undefined;
    if (Number.isFinite(sepId)) ids[ids.length - 1] = sepId;
  }
  return { ids };
}

async function runSession(session, encoded) {
  const ort = loadOnnxRuntime();
  const feeds = {};
  if (session.inputNames.includes("input_ids")) {
    feeds.input_ids = new ort.Tensor("int64", encoded.inputIds, encoded.dims);
  }
  if (session.inputNames.includes("attention_mask")) {
    feeds.attention_mask = new ort.Tensor("int64", encoded.attentionMask, encoded.dims);
  }
  if (session.inputNames.includes("token_type_ids")) {
    feeds.token_type_ids = new ort.Tensor("int64", new BigInt64Array(encoded.inputIds.length), encoded.dims);
  }
  return session.run(feeds);
}

function loadOnnxRuntime() {
  if (ortModule) return ortModule;

  const candidates = [
    "onnxruntime-node-native",
    process.resourcesPath ? path.join(process.resourcesPath, "app.asar.unpacked", "node_modules", "onnxruntime-node-native") : null,
    process.resourcesPath ? path.join(process.resourcesPath, "app", "node_modules", "onnxruntime-node-native") : null,
    path.join(__dirname, "..", "..", "node_modules", "onnxruntime-node-native")
  ].filter(Boolean);
  const errors = [];

  for (const candidate of candidates) {
    try {
      if (path.isAbsolute(candidate) && !fsSync.existsSync(path.join(candidate, "package.json"))) continue;
      ortModule = require(candidate);
      return ortModule;
    } catch (error) {
      errors.push(`${candidate}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  throw new Error(
    "Native ONNX Runtime is not installed in this Electron runtime. Run `npm install` from browser_app, then restart `npm run electron:dev`; for packaged builds rerun `npm run electron:pack` or `npm run electron:dist`. " +
    `Tried ${candidates.join(", ")}. ${errors.join(" | ")}`
  );
}

function firstOutput(result) {
  const output = Object.values(result)[0];
  if (!output) throw new Error("ONNX model returned no outputs");
  return output;
}

function tensorRows(tensor) {
  const dims = tensor.dims ?? [];
  const data = Array.from(tensor.data ?? []);
  if (dims.length === 1) return [data];
  const rowWidth = dims.slice(1).reduce((product, value) => product * value, 1);
  const rows = [];
  for (let offset = 0; offset < data.length; offset += rowWidth) {
    rows.push(data.slice(offset, offset + rowWidth));
  }
  return rows;
}

function meanPoolNormalize(tensor, attentionMaskRows) {
  const dims = tensor.dims ?? [];
  const data = Array.from(tensor.data ?? []);
  if (dims.length === 2) {
    const rows = tensorRows(tensor);
    return rows.map(normalizeVector);
  }
  if (dims.length !== 3) throw new Error(`Unexpected embedding tensor shape: ${dims.join("x")}`);
  const [batch, seq, hidden] = dims;
  const embeddings = [];
  for (let b = 0; b < batch; b += 1) {
    const vector = new Array(hidden).fill(0);
    const valid = Math.max(1, attentionMaskRows[b]?.length ?? seq);
    for (let s = 0; s < Math.min(seq, valid); s += 1) {
      const tokenOffset = (b * seq + s) * hidden;
      for (let h = 0; h < hidden; h += 1) {
        vector[h] += Number(data[tokenOffset + h] ?? 0);
      }
    }
    embeddings.push(normalizeVector(vector.map((value) => value / valid)));
  }
  return embeddings;
}

function normalizeVector(vector) {
  const mag = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (!mag) return vector.map(() => 0);
  return vector.map((value) => value / mag);
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function softmaxPositive(values) {
  if (!values.length) return 0;
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return (exps[1] ?? exps[0] ?? 0) / total;
}
