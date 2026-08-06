# Discovery Report

Source repository inspected: `Z:\Adam-Lab-Shared\Code\Miki\astro-sta-analysis`

## Existing Pipeline Files

| Area | Source file | Notes |
| --- | --- | --- |
| Spike classification | `automated-spike-analysis.py` | Uses Suite2p traces from `<experiment>/pipeline_results/for_suite2p/suite2p/plane0`, detects and classifies peaks, and writes per-cell NPZ and PNG outputs. Original script writes to `<experiment>/automated-spike-analysis`. |
| Behavioral separation | `behavioral-separator.ipynb` | Notebook logic reads `imaging_<i>.csv`, bad-segment JSON, TIFF shapes, neuron dataset pickle, and Suite2p context. It writes `behavioral_segements.npz` under `<experiment>/sta-analysis`. The spelling `segements` is preserved because STA maps expect it. |
| STA maps | `cluster-code/sta_maps_sparse.py` | CLI accepts optional JSON config. Reads `sta_maps_job_params.json` defaults, automated spike labels, behavior segments, PB-corrected astro movie, Suite2p, and anatomy crop params for anatomy mode. Writes per-cell `astro_sta_cluster_permutation*.npz`, `neuron_sta*.npz`, binned movie caches, and summary CSV under `sta-analysis`. |
| Dataset builder | `build_datasets.py` | Builds normalized pickle datasets from STA outputs. Uses fields defined in `utils/db_utils.py`. |
| Viewer | `sta-maps-viewer.py` | Reads STA cluster NPZ metadata and Suite2p ROI masks for review visualization. |
| Cluster submission | `cluster-code/run_sta_maps.sh` | SLURM script using `sbatch` directives, activates a hard-coded `caiman` Conda environment, and runs `sta_maps_sparse.py`. No complete credential/SSH workflow was found. |

## Existing Database/Dataset Formats

The repo currently produces pandas pickle datasets rather than SQLite databases:

| Dataset | Default file | Index fields | Value fields |
| --- | --- | --- | --- |
| Neuron dataset | `sta-analysis/neuron_dataset.pkl` | `cage`, `mouse_name`, `recording_date`, `behavior`, `recording_name`, `neuron_id` | `all_peak_frames`, `simple_peak_frames`, `complex_peak_frames` |
| Cluster dataset | `sta-analysis/cluster_dataset.pkl` | neuron index fields plus `cluster_id` | `sign`, `is_significant`, `p_value`, `disqualify_latency`, `latency_shared_peak_frame_for_kinetics`, `all_mean_trace`, `simple_mean_trace`, `complex_mean_trace` |
| Params dataset | `sta-analysis/params_dataset.pkl` | cluster index fields | `analysis_window`, `pre_frames`, `post_frames`, `n_permutations`, `rng_seed`, `voxel_alpha`, `cluster_alpha`, `spatial_bin`, `anatomy_radius_binned`, `cluster_connectivity`, `min_cluster_size`, `min_cluster_mass`, `merge_similar_clusters`, `merge_max_distance_px`, `merge_min_trace_corr` |

Large arrays in these datasets are represented in the MVP as artifact references or JSON summaries rather than SQLite blobs.

## ROI Outputs and Neuron IDs

ROI detections are Suite2p outputs in:

`<experiment>/pipeline_results/for_suite2p/suite2p/plane0`

Required files include `iscell.npy`, `stat.npy`, `F.npy`, and often `Fneu.npy`/`ops.npy`.

`utils/suite2p_utils.py` filters `stat.npy` using `iscell[:, 0].astype(bool)`, then assigns zero-based `cell_id` by enumeration of the filtered `stat_cells`. Existing downstream code uses folders named `cell_<cell_id>`. The MVP preserves these zero-based IDs and uses `(experiment_id, pipeline_neuron_id)` as the stable compound identity. If upstream Suite2p files are regenerated, these IDs may shift; the MVP stores ROI count and file checks where practical and documents that risk.

## Required Inputs by Step

| Step | Required inputs discovered | Main outputs |
| --- | --- | --- |
| Validation/neuron discovery | Raw TIFF `Image_scan_1_region_0_0.tif`, `pipeline_results`, Suite2p `iscell.npy`, `stat.npy`, `F.npy`, `ops.npy`, `Experiment.xml`, PB-corrected astro TIFF | Registered experiment, parsed metadata, neurons |
| Spike analysis | `Experiment.xml`, Suite2p plane folder, `F.npy`, `iscell.npy`, `stat.npy`, `ops.npy` | `cell_<id>/simple_complex_peaks_for_analysis.npz`, `simple_complex_spike_labels.npz`, `peak_detection_params.npz`, review PNGs |
| Behavioral separation | `imaging_<i>.csv`, raw TIFF, PB-corrected astro TIFF, optional `pipeline_results/qc/bad_segments.json`, Suite2p, neuron dataset or spike-derived peaks | `behavioral_segements.npz`, behavior summaries/figures |
| STA maps | PB-corrected astro TIFF, Suite2p, spike labels, behavior segments for behavioral mode, `anatomy_crop_params.npz` for anatomy mode | per-cell cluster NPZ files, `neuron_sta*.npz`, summary CSV |

## Parameters

Spike defaults are constants in `automated-spike-analysis.py`: MAD threshold, high-pass cutoff, complex spike thresholds, windows, and output toggles.

Behavior defaults are notebook variables: `speed_threshold=5`, `min_epoch_duration_s=1`, `smoothing_window_samples=50`, and chosen alignment `behavior_to_imaging_end`.

STA defaults are in `cluster-code/sta_maps_job_params.json` and `sta_maps_sparse.py`, including `cell_ids`, `result_types`, `n_permutations`, `spatial_bin`, alpha thresholds, window values, and worker count.

## Local and Cluster Execution

Existing local execution is direct Python script/notebook execution with hard-coded paths in top-level variables or JSON config. The MVP runs copied/adapted modules as isolated subprocesses and writes to project-controlled run directories.

Cluster execution discovery found a SLURM shell script only. Because host, credentials, shared-path mapping, and scheduler availability are machine-specific, the MVP includes the cluster backend interface and configuration validation but keeps it disabled unless configured manually.

## Copied or Adapted Logic

| Source | Destination | Purpose |
| --- | --- | --- |
| `automated-spike-analysis.py` functions/constants | `app/pipelines/spike_adapter.py` | Noninteractive spike detection/classification into project-owned run output. |
| `behavioral-separator.ipynb` cells 0, 1, 2, 4, 6, 7, 9, 14, 16 | `app/pipelines/behavior_adapter.py` | Notebook logic converted to Python module with configurable paths and output directory. |
| `cluster-code/sta_maps_sparse.py` invocation/config behavior | `app/pipelines/sta_adapter.py` | Adapter prepares JSON config and delegates to copied source where dependencies permit, with output rooted in the project. |
| `utils/db_utils.py` field definitions | `app/services/results.py`, `docs/schema.md` | Defines unified SQLite result fields and CSV export columns. |
| `utils/paths_utils.py`, `utils/suite2p_utils.py` path and ROI semantics | `app/services/experiments.py` | Validation and neuron discovery. |

## Incompatibilities and Ambiguities

- Original spike and STA scripts write into source experiment folders; the MVP avoids that by using adapted wrappers and project output roots.
- `behavioral_segements.npz` is misspelled in existing code and must remain so for compatibility.
- STA anatomy mode requires `anatomy_crop_params.npz`, produced by another review script; when absent, anatomy-mode STA is marked invalid/blocked rather than guessed.
- Cluster execution cannot be fully proven without external SLURM access and environment details.
- Existing per-experiment "database" outputs are pickle datasets, not relational databases.
