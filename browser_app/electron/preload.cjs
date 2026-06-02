const { contextBridge, ipcRenderer } = require("electron");

function invokeWithProgress(channel, request, onProgress) {
  if (typeof onProgress !== "function") return ipcRenderer.invoke(channel, request);
  const progressToken = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const listener = (_event, progress) => {
    if (progress?.progressToken !== progressToken) return;
    const { progressToken: _progressToken, ...payload } = progress;
    onProgress(payload);
  };
  ipcRenderer.on("matchminer:runtime-progress", listener);
  return ipcRenderer.invoke(channel, { ...request, progressToken }).finally(() => {
    ipcRenderer.removeListener("matchminer:runtime-progress", listener);
  });
}

contextBridge.exposeInMainWorld("matchminerElectron", {
  parsePdfWithLocalOcr: (request) => ipcRenderer.invoke("matchminer:local-pdf-ocr", request),
  runtimeStatus: () => ipcRenderer.invoke("matchminer:runtime-status"),
  prepareRuntimeArtifacts: (onProgress) => invokeWithProgress("matchminer:prepare-runtime-artifacts", {}, onProgress),
  warmRuntime: (request, onProgress) => invokeWithProgress("matchminer:warm-runtime", request, onProgress),
  generateText: (request) => ipcRenderer.invoke("matchminer:generate-text", request),
  tokenizeText: (request) => ipcRenderer.invoke("matchminer:tokenize-text", request),
  detokenizeTokens: (request) => ipcRenderer.invoke("matchminer:detokenize-tokens", request),
  embedTrialSpaceTexts: (request) => ipcRenderer.invoke("matchminer:embed-trial-space-texts", request),
  scoreTrialCheckerTexts: (request) => ipcRenderer.invoke("matchminer:score-trial-checker-texts", request),
  scoreBoilerplateCheckerTexts: (request) => ipcRenderer.invoke("matchminer:score-boilerplate-checker-texts", request)
});
