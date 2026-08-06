# Astro Analysis Pipeline

Local browser GUI plus SQLite registry for successful imaging experiments and latest analysis outputs.

## Start

```powershell
conda env create -f environment.yml
conda activate astro-analysis-pipeline
python -m app.web.start
```

The app binds to `127.0.0.1:8765` by default and opens the browser automatically.

## Configure

Edit `config.example.yml` or copy it elsewhere and set:

```powershell
$env:ASTRO_PIPELINE_CONFIG="Z:\Adam-Lab-Shared\Code\Miki\astro-analysis-pipeline\config.example.yml"
```

The active SQLite file and all project-created outputs live under `astro-analysis-pipeline/runtime` by default.

## Normal Workflow

1. Paste or browse to a final `z<zplane>-x<zoom>` experiment folder on the main page.
2. The app validates raw TIFF, `pipeline_results`, Suite2p ROI files, PB-corrected astro movie, behavior CSV, and parsed metadata.
3. Valid experiments are registered, neurons are discovered from Suite2p ROI IDs, and local spike/behavior runs are launched.
4. When both spike and behavior runs complete, the app automatically launches the real sparse STA algorithm.
5. Open an experiment, then a neuron.
6. Review the latest run status, parameters, logs, metrics, and artifact manifests.

## Test Mode

Test-mode UI is currently hidden while the app is simplified around latest production runs.

## Safety

The source repository and experiment folders are treated as read-only. Removing an experiment deletes its SQLite records and project-owned output folders, then records the source path so future scans do not re-add it.

## Current Limitations

- Spike and behavior adapters preserve the discovered source calculations and write outputs under project run folders.
- The STA adapter submits the repo-local `cluster-code\run_sta_maps.sh` through `/usr/bin/sbatch` on the cluster using Paramiko SSH settings from `config.example.yml` or `ASTRO_CLUSTER_*` environment variables.
- Browser folder browsing is available from the experiment path field.

## Tests

```powershell
conda activate astro-analysis-pipeline
pytest
```
