# Provenance

| Source | Adapted destination | Notes |
| --- | --- | --- |
| `..\automated-spike-analysis.py` | `app/pipelines/spike_adapter.py` | Constants and `run_cell` behavior adapted. Interactive Tk review removed. Outputs redirected to run directories. |
| `..\behavioral-separator.ipynb` | `app/pipelines/behavior_adapter.py` | Cells 0, 1, 2, 4, 6, 7, and 16 converted to a Python CLI module. Plotting is saved to PNG instead of displayed. |
| `..\cluster-code\sta_maps_sparse.py` and `..\cluster-code\sta_maps_job_params.json` | `app/pipelines/sta_adapter.py`, `config.example.yml` | The adapter invokes the real sparse STA algorithm with a generated config. A project-owned mirror redirects script writes away from source experiment folders. |
| `..\utils\db_utils.py` | `docs/schema.md`, CSV export service | Existing dataset fields used to shape the normalized SQLite schema and export columns. |
| `..\utils\paths_utils.py`, `..\utils\suite2p_utils.py` | `app/services/experiments.py` | Path semantics and Suite2p ROI identity logic adapted for validation and neuron discovery. |

No source files were modified.
