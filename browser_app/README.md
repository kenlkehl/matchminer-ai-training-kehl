# MatchMiner-AI Browser App

Static browser app for running a patient-centric MatchMiner-AI workflow locally on the user's machine.

Patient data is parsed and processed in the browser. Network requests are limited to user-triggered downloads of public model artifacts and public ClinicalTrials.gov/trial-index data.

## Development

```bash
npm install
npm run dev
```

Open `http://localhost:5173` when the browser is on the same machine.

For a browser on another machine, prefer an SSH tunnel so WebGPU still sees a
local secure origin:

```bash
ssh -L 5173:localhost:5173 user@kehl-lab.dfci.harvard.edu
```

Then open `http://localhost:5173` in the local browser. Direct access via
`http://kehl-lab.dfci.harvard.edu:5173` is allowed for development, but WebGPU
may still be unavailable because remote HTTP origins are not secure contexts.
Additional Vite development hosts can be allowed with:

```bash
VITE_ALLOWED_HOSTS=other-host.example.edu npm run dev
```

On Linux Chrome/Chromium, also check `chrome://gpu`. Some installations require
enabling Vulkan/WebGPU flags for local development.

## Desktop App

For normal users, build the Electron desktop app instead of running a browser
dev server:

```bash
npm install
npm run electron:dist:linux
```

Linux installers are written to `release/`:

- `MatchMiner AI-0.1.0.AppImage`
- `matchminer-browser-app_0.1.0_amd64.deb`

Run the AppImage directly:

```bash
chmod +x "release/MatchMiner AI-0.1.0.AppImage"
"release/MatchMiner AI-0.1.0.AppImage"
```

Or install the Debian package:

```bash
sudo apt install ./release/matchminer-browser-app_0.1.0_amd64.deb
```

For development inside Electron:

```bash
npm run electron:dev
```

The desktop app loads the built renderer from a local secure
`matchminer://app` protocol with Node integration disabled. Patient files are
still processed locally in the renderer. The app only makes network requests
when the user downloads public model artifacts or public ClinicalTrials.gov
trial data.

## Native Local Runtime

Electron builds default to native local inference:

- text generation runs through `llama.cpp` `llama-server`;
- TrialSpace, TrialChecker, and BoilerplateChecker run through native ONNX Runtime
  in an Electron worker.

Model artifacts are downloaded on first warmup into the app user-data directory.
The default LLM artifact is:

- `LiquidAI/LFM2.5-1.2B-Thinking-GGUF`
- `LFM2.5-1.2B-Thinking-Q4_K_M.gguf`

For development, install `llama-server` on `PATH`, set
`MATCHMINER_LLAMA_SERVER_COMMAND=/path/to/llama-server`, or place a platform
binary under `resources/bin/llama-server` (`llama-server.exe` on Windows). The
packaged app copies `resources/bin` into the app resources directory. On Linux
x64/arm64, if no bundled or PATH binary is found, the app attempts a first-run
download of the official CPU `llama.cpp` release archive and extracts it with
system `tar`.

Settings can switch generation or ONNX tasks back to the browser WebGPU
runtime for comparison/debugging.

If Electron dev mode reports that `onnxruntime-node-native` cannot be found,
run `npm install` from this `browser_app` directory and restart
`npm run electron:dev`. Packaged builds need to be rebuilt after dependency
changes with `npm run electron:pack` or `npm run electron:dist`.

Electron is launched with WebGPU development flags by default. If those flags
cause a GPU-driver issue on a particular machine, start it with:

```bash
MATCHMINER_DISABLE_GPU_FLAGS=1 npm run electron:dev
```

## Model Artifacts

The default browser settings expect ONNX versions of:

- `ksg-dfci/TrialSpace-0526`
- `ksg-dfci/TrialChecker-0526`
- `ksg-dfci/BoilerplateChecker-0526`

Generate local ONNX folders:

```bash
python -m pip install torch transformers huggingface_hub onnx onnxruntime safetensors
npm run convert:models
npm run validate:onnx
```

On a multi-GPU machine, regenerate the three independent exports concurrently:

```bash
GPUS=5,6,7 bash scripts/convert_matchminer_models_parallel.sh
```

The conversion writes to `artifacts/models`, which is ignored by git. Publish those folders to Hugging Face or serve them locally, then update `public/models.manifest.json` and the in-app settings.

The temporary LLM default is `onnx-community/gemma-4-E2B-it-ONNX`; it can be replaced by a compatible custom LFM ONNX model ID in Settings.

## Trial Index

Normal use should consume a prebuilt trial-space embedding index. The app also includes a ClinicalTrials.gov v2 refresh path that downloads current trials and runs the local LLM trial-space extraction prompt in the browser. This can take a long time for the full CT.gov result set.

Build a public heuristic JSON index from Node:

```bash
npm run build:trial-index -- --output public/ctgov_trial_index.json
```

The ClinicalTrials.gov refresh uses the training search definition: cancer,
lymphoma, carcinoma, leukemia, sarcoma, melanoma, myeloma, myelodysplastic, or
myeloproliferative; open interventional trials; Early Phase 1 through Phase 3.
Pass `--max-pages N` only when you intentionally want a smaller development
sample.

For production, run trial-space extraction offline using the MatchMiner model/prompt, pre-embed spaces with TrialSpace, and publish the resulting JSON or sharded JSON files.

The browser app can import a pre-embedded trial index from a local JSON/JSONL/CSV file or from a URL. The importer accepts the browser-native fields (`spaceId`, `nctId`, `trialSpaceText`, `boilerplateText`, `embedding`) and the real-time eval pre-embed column names (`id`, `nct_id`, `this_cohort`, `boilerplate_text`, `embedding`). Parquet files should be exported to JSON first; `real_eval_code/real_time_eval/embed_trial_spaces.py` supports:

```bash
python ../real_eval_code/real_time_eval/embed_trial_spaces.py /path/to/model \
  --browser-json-output trial_space_embeddings.browser.json
```

## Patient Summarization

The Summarize button follows the training pipeline's serial summarization pattern: dated notes are concatenated, tokenized with the selected LLM tokenizer, split into overlapping token chunks, and fed through a running summary one chunk at a time. Settings expose the per-chunk input token target and overlap. The browser app also checks the full prompt token count before each local generation call; if the prompt would exceed the LLM budget, that segment is split into smaller serial subsegments before retrying, without truncating the source record. If generation still fails with a retryable context, length, memory, or no-visible-summary error, the app restarts summarization from the original notes with smaller base chunks before falling back to the local extractive summary.

## PDF OCR

PDF upload first extracts embedded text with PDF.js. For scanned or image-only PDFs, the default OCR setting is `Auto`: the renderer tries `onnx-community/granite-docling-258M-ONNX` with Transformers.js/WebGPU, then falls back to browser Tesseract.js. Use Settings -> PDF OCR to force `Granite WebGPU only`, `Browser only`, or the advanced `Local CLI only` mode.

The local CLI mode is optional and exists for development or workstation-specific installs. Install one or both local OCR backends on the machine running Electron:

```bash
python3 -m pip install docling
# optional alternative/fallback; install system tesseract/ghostscript as required by OCRmyPDF
python3 -m pip install ocrmypdf
```

Environment overrides:

- `MATCHMINER_DOCLING_PYTHON=/path/to/python`
- `MATCHMINER_OCRMYPDF_COMMAND=/path/to/ocrmypdf`
- `MATCHMINER_LOCAL_OCR_TIMEOUT_MS=900000`
