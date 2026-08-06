# Acceptance Demonstration Status

The GUI workflows are implemented, but a full real-experiment demonstration was not completed in this coding session.

## Workflow Coverage

| Workflow | Status |
| --- | --- |
| Manual add through latest results | Implemented in GUI. Requires running on a workstation with a selected valid experiment and enough time/dependencies for spike, behavior, and real sparse STA jobs. |
| Scheduled-style scan through latest results | Implemented via `/scan` and the same `scan_parent` service used by APScheduler. Requires a valid unregistered experiment under the scan parent. |
| Rerun and replace | Rerun replaces the previous run for that step and displays only the latest run. |
| Test-mode demonstration | Deferred while the app is simplified around latest production runs. |

## Source Integrity Checks

Project code writes under:

- `astro-analysis-pipeline/runtime/outputs`
- `astro-analysis-pipeline/runtime/test_outputs`
- `astro-analysis-pipeline/runtime/logs`
- the configured SQLite path

The adapters do not write to raw movie files, original `pipeline_results`, existing preprocessing outputs, or original source scripts.

The STA adapter runs `cluster-code\sta_maps_sparse.py` against a run-local mirror so the script's normal `sta-analysis` and binned-cache writes are project-owned.
