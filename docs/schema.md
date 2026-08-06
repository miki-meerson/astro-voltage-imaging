# SQLite Schema

| Table | Purpose |
| --- | --- |
| `experiments` | Successful-experiment registry with original parsed path metadata and validation report. |
| `metadata_overrides` | Corrected metadata values, preserving original parsed value and required note. |
| `neurons` | Suite2p-derived neurons using `(experiment_id, pipeline_neuron_id)` identity. |
| `analysis_runs` | Latest analysis run per experiment and step, with backend, status, parameters, manifests, logs, and provenance. |
| `artifacts` | Relative references to project-created files. |
| `scan_history` | Recursive discovery scan summaries. |
| `app_settings` | Small key-value settings such as last scan time. |

Foreign keys are enabled for every connection. WAL mode is enabled at migration time. Multi-table operations use transactions in `app/db/core.py`.

Existing dataset-builder fields can be reproduced from the latest run artifacts and manifests. Large arrays such as peak arrays and STA movies are stored as files and referenced from manifests.
