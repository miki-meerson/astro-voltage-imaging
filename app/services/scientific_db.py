from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np

from app.db import confirmed_repository
from app.services.runs import run_dir
from app.settings import Settings
from utils.clusters_utils import cluster_mask2d_from_cluster, extract_clusters, saved_cluster_args
from utils.db_utils import collect_cluster_result_metadata
from utils.paths_utils import PathKind, path_for, sta_dir_of_run


def _as_json_array(value) -> str:
    arr = np.asarray(value)
    return json.dumps(arr.ravel().tolist())


def _jsonable(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _jsonable(value.item())
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _row_json_array(row, name: str) -> str:
    value = getattr(row, name, None)
    if value is None:
        return "[]"
    try:
        if bool(np.isscalar(value)) and not np.isfinite(value):
            return "[]"
    except TypeError:
        pass
    return json.dumps(_jsonable(value))


def _row_int(row, name: str):
    value = getattr(row, name, None)
    arr = np.asarray(value)
    if arr.shape != ():
        return None
    value = arr.item()
    try:
        if value is None or not np.isfinite(value):
            return None
    except TypeError:
        if value is None:
            return None
    return int(value)


def _row_float(row, name: str):
    value = getattr(row, name, None)
    arr = np.asarray(value)
    if arr.shape != ():
        return None
    value = arr.item()
    try:
        if value is None or not np.isfinite(value):
            return None
    except TypeError:
        if value is None:
            return None
    return float(value)


def _load_npz_arrays(path: Path) -> dict:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as d:
        return {key: np.asarray(d[key]) for key in d.files}


def _copy_source_artifact(src: Path, dest_root: Path) -> str:
    if not src.exists():
        return ""
    dest_root.mkdir(parents=True, exist_ok=True)
    dst = dest_root / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst)


CONFIRMED_PARAMETER_KEYS = frozenset({
    "pre_window",
    "post_window",
    "analysis_window",
    "baseline_window",
    "voxel_alpha",
    "cluster_alpha",
    "spatial_bin",
    "anatomy_spatial_bin",
    "anatomy_radius_binned",
    "anatomy_source_radius_binned",
    "anatomy_source_spatial_bin",
    "anatomy_fullres_radius_px",
    "cluster_connectivity",
    "min_cluster_size",
    "min_cluster_mass",
    "min_spikes_for_cluster",
    "merge_similar_clusters",
    "merge_max_distance_px",
    "merge_min_trace_corr",
    "null_standardization",
    "per_sign_null",
    "cluster_threshold_pos",
    "cluster_threshold_neg",
    "smooth_sigma_px",
    "smooth_sigma_fullres_px",
    "min_cluster_footprint_px",
    "min_cluster_frames",
    "significance_statistic",
    "n_permutations",
    "rng_seed",
    "pre_seconds",
    "post_seconds",
    "baseline_seconds",
    "analysis_half_window_seconds",
    "analysis_half_window_frames",
    "result_types",
    "movie_shape",
    "n_valid_centers",
})


def _remove_confirmed_artifacts(settings: Settings, canonical_id: str, pipeline_neuron_id: str) -> None:
    base = path_for(PathKind.SCIENTIFIC_DATABASE_ROOT, settings.project_root).resolve()
    target = path_for(PathKind.CONFIRMED_CELL_DIR, settings.project_root, canonical_id=canonical_id, cell=pipeline_neuron_id).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return
    if target.exists():
        shutil.rmtree(target)


def unconfirm_neuron(settings: Settings, neuron_id: int) -> bool:
    """Remove the confirmation of a cell and its saved copies; False when the cell was not confirmed."""
    return confirmed_repository.delete_confirmed_cell(settings, neuron_id, lambda canonical_id, cell: _remove_confirmed_artifacts(settings, canonical_id, cell))


def _spike_record(settings: Settings, spike_run: dict, cell: str, canonical_id: str) -> dict:
    """The peaks row of a confirmed cell, read from the spike run's saved files (which are also copied into the confirmed-cell folder)."""
    spike_out = run_dir(settings, int(spike_run["id"]), bool(spike_run["is_test"]))
    labels_path = path_for(PathKind.SPIKE_LABELS, spike_out, cell=cell)
    peaks_path = path_for(PathKind.SPIKE_PEAKS, spike_out, cell=cell)
    labels = _load_npz_arrays(labels_path)
    peaks = _load_npz_arrays(peaks_path)
    params = _load_npz_arrays(path_for(PathKind.SPIKE_PEAK_PARAMS, spike_out, cell=cell))
    all_peaks = labels.get("peaks_all", peaks.get("peaks_all", peaks.get("all_peak_frames", [])))
    simple_peaks = labels.get("simple_peaks", labels.get("simple_peak_frames", labels.get("peaks_simple", peaks.get("simple_peak_frames", []))))
    complex_peaks = labels.get("complex_peaks", labels.get("complex_peak_frames", labels.get("peaks_complex", peaks.get("complex_peak_frames", []))))
    spikes_dir = path_for(PathKind.CONFIRMED_SPIKES_DIR, settings.project_root, canonical_id=canonical_id, cell=cell)
    return {
        "all_peak_frames_json": _as_json_array(all_peaks),
        "simple_peak_frames_json": _as_json_array(simple_peaks),
        "complex_peak_frames_json": _as_json_array(complex_peaks),
        "n_all_peaks": int(np.asarray(all_peaks).size),
        "n_simple_peaks": int(np.asarray(simple_peaks).size),
        "n_complex_peaks": int(np.asarray(complex_peaks).size),
        "peak_detection_params_json": json.dumps({k: _jsonable(v) for k, v in params.items()}),
        "source_artifact_path": _copy_source_artifact(labels_path if labels_path.exists() else peaks_path, spikes_dir),
    }


def _sta_record(settings: Settings, row, canonical_id: str, cell: str) -> dict:
    """The confirmed_sta_results row (with its clusters) of one STA result file; copies the file and writes the cluster masks next to it."""
    source_npz = Path(row.path)
    sta_copy_dir = path_for(PathKind.CONFIRMED_STA_DIR, settings.project_root, canonical_id=canonical_id, cell=cell)
    masks_path = path_for(PathKind.CONFIRMED_CLUSTER_MASKS, settings.project_root, canonical_id=canonical_id, cell=cell, name=source_npz.stem)
    copied_npz = _copy_source_artifact(source_npz, sta_copy_dir)
    with np.load(source_npz, allow_pickle=True) as d:
        effect = np.asarray(d["observed_effect_movie"], dtype=np.float32)
        # stored traces are dF/F-like: use the raw effect, not the tested z-statistic
        trace_effect = np.asarray(d["observed_effect_raw_movie"], dtype=np.float32) if "observed_effect_raw_movie" in d else effect
        aw = np.asarray(d["analysis_window"], dtype=int)
        params = {
            key: _jsonable(np.asarray(d[key]).item() if np.asarray(d[key]).shape == () else np.asarray(d[key]))
            for key in d.files
            if key in CONFIRMED_PARAMETER_KEYS
        }
        clusters = extract_clusters(effect_movie=effect, **saved_cluster_args(d))  # honours per_sign_null when the file says so
    t_rel = np.arange(int(aw[0]), int(aw[1]) + 1)
    masks = {}
    cluster_rows = []
    for cluster in clusters:
        mask = cluster_mask2d_from_cluster(cluster)
        if mask is None:
            continue
        cluster_id = int(cluster.get("cluster_id", len(masks) + 1))
        masks[f"cluster_{cluster_id}_mask2d"] = np.asarray(mask, dtype=bool)
        trace = np.asarray(trace_effect[:, mask].mean(axis=1), dtype=np.float32)
        yyxx = np.argwhere(mask)
        cluster_rows.append({
            "cluster_id": cluster_id,
            "sign": str(cluster.get("sign", "")),
            "is_significant": int(bool(cluster.get("is_significant", False))),
            "p_value": float(cluster["p_value"]) if "p_value" in cluster else None,
            "mass": float(cluster.get("mass", 0.0)),
            "size_voxels": int(cluster.get("size", 0)),
            "peak_value": float(cluster.get("peak_value", 0.0)),
            "centroid_y": float(np.mean(yyxx[:, 0])) if yyxx.size else None,
            "centroid_x": float(np.mean(yyxx[:, 1])) if yyxx.size else None,
            "area_px": int(mask.sum()),
            "mean_trace_json": _as_json_array(trace),
            "t_rel_json": _as_json_array(t_rel),
            "mask_artifact_path": str(masks_path),
        })
    if masks:
        masks_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(masks_path, **masks)
    return {
        "result_type": str(row.result_type),
        "version": int(row.version),
        "n_peaks": _row_int(row, "n_peaks"),
        "n_significant_clusters": int(row.n_significant_clusters),
        "has_significant_cluster": int(bool(row.has_significant_cluster)),
        "pre_window": _row_int(row, "pre_window"),
        "post_window": _row_int(row, "post_window"),
        "analysis_window_json": _row_json_array(row, "analysis_window"),
        "baseline_window_json": _row_json_array(row, "baseline_window"),
        "spatial_bin": _row_int(row, "spatial_bin"),
        "voxel_alpha": _row_float(row, "voxel_alpha"),
        "cluster_alpha": _row_float(row, "cluster_alpha"),
        "n_permutations": _row_int(row, "n_permutations"),
        "parameters_json": json.dumps(params),
        "source_npz_path": copied_npz,
        "clusters": cluster_rows,
    }


def confirm_neuron(settings: Settings, neuron_id: int, reviewer_note: str = "") -> int:
    """Confirm a cell: copy its saved spike and STA result files into runtime/scientific_database and record their numbers in the database."""
    inputs = confirmed_repository.confirmation_inputs(settings, neuron_id)
    if inputs is None:
        raise ValueError("Neuron not found.")
    neuron, spike_run, behavior_run, sta_run = inputs["neuron"], inputs["spike_run"], inputs["behavior_run"], inputs["sta_run"]
    if sta_run is None:
        raise ValueError("Confirming a neuron requires a completed STA Maps run.")
    cell, canonical_id = str(neuron["pipeline_neuron_id"]), str(neuron["canonical_id"])

    peaks = _spike_record(settings, spike_run, cell, canonical_id) if spike_run is not None else None
    sta_dir = sta_dir_of_run(run_dir(settings, int(sta_run["id"]), bool(sta_run["is_test"])))
    result_df = collect_cluster_result_metadata(sta_dir, [int(cell)])
    if result_df.empty:
        raise ValueError("No STA result files were found for this neuron.")
    result_df = result_df.sort_values("version").groupby("result_type", as_index=False).tail(1)
    sta_results = [_sta_record(settings, row, canonical_id, cell) for row in result_df.itertuples()]
    return confirmed_repository.save_confirmed_cell(
        settings, neuron, int(spike_run["id"]) if spike_run else None, int(behavior_run["id"]) if behavior_run else None, int(sta_run["id"]),
        reviewer_note, peaks, sta_results,
    )
