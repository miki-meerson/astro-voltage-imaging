# Astro Analysis Pipeline

Pipeline of the Adam lab (HUJI) for two-photon GEVI imaging of astrocytes and neurons in the same field of view: preprocessing and spike detection, spike-triggered average (STA) movies of the astro channel with a permutation test, and a local browser app (Flask + SQLite) that registers experiments, launches each step (heavy ones on the Slurm cluster) and shows the results.

## Quick start

```powershell
conda env create -f environment.yml
conda activate astro-analysis-pipeline
copy secrets.example.yml secrets.yml     # then fill in the cluster login; secrets.yml is git-ignored
python -m app.web.start
```

The app opens at `http://127.0.0.1:8765`. Settings are in `config.example.yml` (or the file named in the environment variable `ASTRO_PIPELINE_CONFIG`). The database and all outputs created by the app live in `runtime/`.

Typical use: on the main page paste or browse to an experiment folder (`...\Hyp<n>\<side>\<date>-<state>\FOV<k>\z-<plane>-x<zoom>`), review preprocessing, confirm Suite2p, then spikes, behavior and STA run, and open the experiment and its cells. Source experiment folders are treated as read-only (the one exception, a re-creatable binned-movie cache, is described in the architecture document).

Tests (kept locally, not in git): `pytest tests/`.

## Documentation

Start at [docs/README.md](docs/README.md). Working rules for searching the slow data share are in [CLAUDE.md](CLAUDE.md).
