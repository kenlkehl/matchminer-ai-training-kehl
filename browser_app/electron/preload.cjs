const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("matchminerElectron", {
  parsePdfWithLocalOcr: (request) => ipcRenderer.invoke("matchminer:local-pdf-ocr", request)
});
