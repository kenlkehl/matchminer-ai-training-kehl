import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const extraAllowedHosts = (process.env.VITE_ALLOWED_HOSTS ?? "")
  .split(",")
  .map((host) => host.trim())
  .filter(Boolean);

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ["@huggingface/transformers", "onnxruntime-web", "pdfjs-dist", "tesseract.js"]
  },
  worker: {
    format: "es"
  },
  server: {
    allowedHosts: ["kehl-lab.dfci.harvard.edu", ...extraAllowedHosts],
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp"
    }
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp"
    }
  }
});
