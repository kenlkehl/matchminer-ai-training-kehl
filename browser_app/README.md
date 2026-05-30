# MatchMiner-AI Browser App

Static browser app for running a patient-centric MatchMiner-AI workflow locally on the user's machine.

Patient data is parsed and processed in the browser. Network requests are limited to user-triggered downloads of public model artifacts and public ClinicalTrials.gov/trial-index data.

## Development

```bash
npm install
npm run dev
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

Normal use should consume a prebuilt trial-space embedding index. The app also includes a ClinicalTrials.gov v2 refresh path that creates a heuristic index in the browser.

Build a public JSON index from Node:

```bash
npm run build:trial-index -- --output public/ctgov_trial_index.json --max-pages 10
```

For production, run trial-space extraction offline using the MatchMiner model/prompt, pre-embed spaces with TrialSpace, and publish the resulting JSON or sharded JSON files.
