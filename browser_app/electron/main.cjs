const { app, BrowserWindow, net, protocol, shell } = require("electron");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const APP_SCHEME = "matchminer";
const DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL ?? "http://127.0.0.1:5173";
const isDev = !app.isPackaged;
let appProtocolRegistered = false;

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

async function createWindow() {
  await registerAppProtocol();

  const mainWindow = new BrowserWindow({
    width: 1500,
    height: 980,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: "#f5f7fb",
    title: "MatchMiner AI",
    webPreferences: {
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
}

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
