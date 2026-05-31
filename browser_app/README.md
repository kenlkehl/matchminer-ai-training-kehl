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
