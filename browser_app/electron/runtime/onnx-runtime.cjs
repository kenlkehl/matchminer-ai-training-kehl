const path = require("node:path");
const { Worker } = require("node:worker_threads");
const { ensureOnnxModel } = require("./artifacts.cjs");

class OnnxRuntime {
  constructor(app) {
    this.app = app;
    this.worker = null;
    this.nextId = 1;
    this.pending = new Map();
  }

  async warm(modelId, task) {
    if (!modelId) throw new Error("ONNX model id is required");
    const modelDir = await ensureOnnxModel(this.app, modelId);
    await this.send({ type: "load", modelId, task, modelDir });
    return { loaded: true, modelId, task, modelDir };
  }

  async embedTexts(modelId, texts) {
    if (!modelId) throw new Error("TrialSpace model id is required");
    const modelDir = await ensureOnnxModel(this.app, modelId);
    return this.send({ type: "embed", modelId, modelDir, texts });
  }

  async scoreTrialChecker(modelId, texts) {
    if (!modelId) throw new Error("TrialChecker model id is required");
    const modelDir = await ensureOnnxModel(this.app, modelId);
    return this.send({ type: "score", modelId, modelDir, texts, transform: "sigmoid" });
  }

  async scoreBoilerplateChecker(modelId, texts) {
    if (!modelId) throw new Error("BoilerplateChecker model id is required");
    const modelDir = await ensureOnnxModel(this.app, modelId);
    return this.send({ type: "score", modelId, modelDir, texts, transform: "softmax_positive" });
  }

  status() {
    return {
      workerRunning: Boolean(this.worker),
      pendingJobs: this.pending.size
    };
  }

  async stop() {
    if (!this.worker) return;
    const worker = this.worker;
    this.worker = null;
    for (const request of this.pending.values()) request.reject(new Error("ONNX runtime stopped"));
    this.pending.clear();
    await worker.terminate();
  }

  send(payload) {
    const worker = this.ensureWorker();
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      worker.postMessage({ ...payload, id });
    });
  }

  ensureWorker() {
    if (this.worker) return this.worker;
    this.worker = new Worker(path.join(__dirname, "onnx-worker.cjs"));
    this.worker.on("message", (message) => {
      const request = this.pending.get(message.id);
      if (!request) return;
      this.pending.delete(message.id);
      if (message.ok) request.resolve(message.result);
      else request.reject(new Error(message.error || "ONNX worker failed"));
    });
    this.worker.on("error", (error) => {
      for (const request of this.pending.values()) request.reject(error);
      this.pending.clear();
      this.worker = null;
    });
    this.worker.on("exit", (code) => {
      if (code !== 0) {
        const error = new Error(`ONNX worker exited with code ${code}`);
        for (const request of this.pending.values()) request.reject(error);
        this.pending.clear();
      }
      this.worker = null;
    });
    return this.worker;
  }
}

module.exports = {
  OnnxRuntime
};
