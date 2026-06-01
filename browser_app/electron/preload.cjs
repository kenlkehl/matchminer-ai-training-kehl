const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("matchminerElectron", {
  parsePdfWithLocalOcr: (request) => ipcRenderer.invoke("matchminer:local-pdf-ocr", request),
  runtimeStatus: () => ipcRenderer.invoke("matchminer:runtime-status"),
  prepareRuntimeArtifacts: () => ipcRenderer.invoke("matchminer:prepare-runtime-artifacts"),
  warmRuntime: (request) => ipcRenderer.invoke("matchminer:warm-runtime", request),
  generateText: (request) => ipcRenderer.invoke("matchminer:generate-text", request),
  tokenizeText: (request) => ipcRenderer.invoke("matchminer:tokenize-text", request),
  detokenizeTokens: (request) => ipcRenderer.invoke("matchminer:detokenize-tokens", request),
  embedTrialSpaceTexts: (request) => ipcRenderer.invoke("matchminer:embed-trial-space-texts", request),
  scoreTrialCheckerTexts: (request) => ipcRenderer.invoke("matchminer:score-trial-checker-texts", request),
  scoreBoilerplateCheckerTexts: (request) => ipcRenderer.invoke("matchminer:score-boilerplate-checker-texts", request)
});
