"""Spike labels of a cell: read/write the label files, detection parameters, stale-Suite2p checks."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np

from app.services.runs import run_dir
from utils.paths_utils import PathKind, path_for
from utils.suite2p_utils import load_suite2p_context

def spikes_edited_after_sta(settings, spike_run, sta_run, pipeline_neuron_id) -> datetime | None:
    """When the cell's spike labels were last written, if that is after the STA run started (so the STA used older spikes)."""
    if spike_run is None or sta_run is None or not sta_run.get("started_at"):
        return None
    labels = path_for(PathKind.SPIKE_LABELS, run_dir(settings, int(spike_run["id"]), bool(spike_run["is_test"])), cell=str(pipeline_neuron_id))
    try:
        edited = datetime.fromtimestamp(labels.stat().st_mtime)
        started = datetime.fromisoformat(str(sta_run["started_at"]).replace(" ", "T"))
    except (OSError, ValueError):
        return None
    return edited if edited > started else None


def suite2p_cell_count(source_path: str | Path | None) -> int | None:
    """Number of accepted Suite2p cells, or None when the Suite2p files cannot be read (then nothing can be said about stale cell ids)."""
    if not source_path:
        return None
    try:
        return int(load_suite2p_context(path_for(PathKind.SUITE2P_PLANE0, source_path))["n_cells"])
    except (OSError, ValueError, IndexError):
        return None


def stale_suite2p_cell_message(source_path: str | Path | None, pipeline_neuron_id: str | int) -> str | None:
    try:
        cell_id = int(pipeline_neuron_id)
    except (TypeError, ValueError):
        return f"Cell {pipeline_neuron_id} is not a valid Suite2p cell id."
    n_cells = suite2p_cell_count(source_path)
    if n_cells is None:
        return None
    if cell_id < 0 or cell_id >= n_cells:
        valid = f"0..{n_cells - 1}" if n_cells else "none"
        return (
            f"Cell {cell_id} is stale in the app database. The current Suite2p accepted-cell list "
            f"has {n_cells} cells, so valid ids are {valid}. Refresh Suite2p Cells, rerun Spike Detection, "
            "and then edit spikes from the refreshed cell list."
        )
    return None


def current_suite2p_neurons(source_path: str | Path | None, neurons: list[dict]) -> list[dict]:
    n_cells = suite2p_cell_count(source_path)
    if n_cells is None:
        return neurons
    current = []
    for neuron in neurons:
        try:
            cell_id = int(neuron["pipeline_neuron_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= cell_id < n_cells:
            current.append(neuron)
    return current


def spike_review_info(settings, spike_run, pipeline_neuron_id: str) -> dict:
    if spike_run is None:
        return {"reviewed": False, "n_peaks": 0, "n_simple": 0, "n_complex": 0, "n_excluded": 0}
    run_folder = run_dir(settings, int(spike_run["id"]), bool(spike_run["is_test"]))
    try:
        payload = load_spike_payload(path_for(PathKind.SPIKE_LABELS, run_folder, cell=pipeline_neuron_id))
    except FileNotFoundError:  # no label file yet: the cell has no spikes to count
        return {"reviewed": False, "n_peaks": 0, "n_simple": 0, "n_complex": 0, "n_excluded": 0}
    labels = payload["labels"].astype(str)
    return {
        "reviewed": path_for(PathKind.SPIKE_MANUAL_REVIEW, run_folder, cell=pipeline_neuron_id).exists(),
        "n_peaks": int(payload["peaks_all"].size),
        "n_simple": int(np.sum(labels == "simple")),
        "n_complex": int(np.sum(labels == "complex")),
        "n_excluded": int(np.sum(labels == "exclude")),
    }


def load_spike_payload(path: Path) -> dict:
    """Peaks and labels (sorted by frame) of a cell's saved spike label file. FileNotFoundError says which step writes the file."""
    try:
        saved = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"The spike labels file {path} is missing: the Spike Detection step writes it, so rerun Spike Detection for this experiment.") from None
    with saved as d:
        peaks = np.asarray(d["peaks_all"], dtype=np.int32) if "peaks_all" in d else np.empty(0, dtype=np.int32)
        labels = np.asarray(d["labels"]).astype(str) if "labels" in d else np.full(peaks.shape, "simple")
    order = np.argsort(peaks)
    return {"peaks_all": peaks[order], "labels": labels[order]}


# name -> (kind, default, lo, hi): the nine detector settings of the spike editor, with the limits the server enforces
SPIKE_PARAM_SPECS = {
    "mad_factor": (float, 3.5, 0.5, 8.0),
    "min_amplitude": (float, 0.30, 0.0, 2.0),
    "min_width": (float, 18.0, 1.0, 120.0),
    "min_subthreshold": (float, 0.10, 0.0, 1.0),
    "min_ratio": (float, 0.30, 0.0, 2.0),
    "duplicate_window_frames": (int, 30, 1, 240),
    "align_radius_frames": (int, 5, 0, 80),
    "min_distance": (int, 0, 0, 500),
    "highpass_cutoff_hz": (float, 0.07, 0.0, 1.0),
}


def bounded_spike_params(data: dict) -> dict:
    """The detector settings found in `data`, each clamped to its limits (missing ones get the default)."""
    return {
        name: (bounded_int if kind is int else bounded_float)(data, name, default, lo, hi)
        for name, (kind, default, lo, hi) in SPIKE_PARAM_SPECS.items()
    }


def spike_detection_params(run_row, params_path: Path | None = None) -> dict:
    """Detector settings of a spike run; the ones saved with the cell (params_path, written by the editor's save) win."""
    try:
        params = json.loads(run_row["parameters_json"] or "{}") if run_row is not None else {}
    except json.JSONDecodeError:
        params = {}
    result = {name: kind(params.get(name, default)) for name, (kind, default, _, _) in SPIKE_PARAM_SPECS.items()}
    if params_path is not None:
        try:
            saved = np.load(params_path, allow_pickle=True)
        except FileNotFoundError:  # the file only exists once the cell's settings were saved
            return result
        with saved as d:
            if "mad_threshold_factor" in d:
                result["mad_factor"] = float(np.asarray(d["mad_threshold_factor"]).item())
            for name, (kind, _, _, _) in SPIKE_PARAM_SPECS.items():
                if name != "mad_factor" and name in d:
                    result[name] = kind(np.asarray(d[name]).item())
    return result


def bounded_float(data: dict, name: str, default: float, lo: float, hi: float) -> float:
    value = float(data.get(name, default))
    if not np.isfinite(value):
        value = default
    return float(min(hi, max(lo, value)))


def bounded_int(data: dict, name: str, default: int, lo: int, hi: int) -> int:
    value = int(data.get(name, default))
    return int(min(hi, max(lo, value)))


def save_spike_payload(spike_run_folder: Path, pipeline_neuron_id: str, peaks: np.ndarray, labels: np.ndarray, params: dict | None = None) -> None:
    """Write the edited peaks and labels of a cell into its folder of the spike run (the other keys of the old label file are kept)."""
    path_for(PathKind.SPIKE_CELL_DIR, spike_run_folder, cell=pipeline_neuron_id).mkdir(parents=True, exist_ok=True)
    order = np.argsort(peaks)
    peaks = np.asarray(peaks, dtype=np.int32)[order]
    labels = np.asarray(labels).astype("<U8")[order]
    payload = {}
    try:
        existing = np.load(path_for(PathKind.SPIKE_LABELS, spike_run_folder, cell=pipeline_neuron_id), allow_pickle=True)
    except FileNotFoundError:  # a cell without a label file yet: the first save creates it
        existing = None
    if existing is not None:
        with existing as d:
            payload = {key: np.asarray(d[key]) for key in d.files}
    payload.update({
        "peaks_all": peaks,
        "labels": labels,
        "simple_peaks": peaks[labels == "simple"],
        "complex_peaks": peaks[labels == "complex"],
        "excluded_peaks": peaks[labels == "exclude"],
        "manual_reviewed": np.asarray(True),
    })
    for kind in (PathKind.SPIKE_LABELS, PathKind.SPIKE_PEAKS):
        np.savez_compressed(path_for(kind, spike_run_folder, cell=pipeline_neuron_id), **payload)
    saved_params = dict(params or {})
    if saved_params:
        full = bounded_spike_params(saved_params)  # one default per setting, as in the editor
        np.savez(
            path_for(PathKind.SPIKE_PEAK_PARAMS, spike_run_folder, cell=pipeline_neuron_id),
            threshold_method="mad",
            mad_threshold_factor=full["mad_factor"],
            **{name: full[name] for name in SPIKE_PARAM_SPECS if name != "mad_factor"},
        )
    path_for(PathKind.SPIKE_MANUAL_REVIEW, spike_run_folder, cell=pipeline_neuron_id).write_text(
        json.dumps(
            {
                "peaks_all": peaks.tolist(),
                "labels": labels.tolist(),
                "detection_params": saved_params,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def review_cache_key(settings, experiment_id: int, spike_run, neurons: list[dict], source_path) -> tuple | None:
    """Key under which the cell-review summary of an experiment can be reused (PERF-01).

    It changes when any cell's label file or review mark is written or removed, when the spike run changes, or when
    Suite2p's iscell.npy changes. None (= do not cache) when there is no spike run.
    """
    if spike_run is None:
        return None
    newest = labels_seen = marks_seen = 0
    run_folder = run_dir(settings, int(spike_run["id"]), bool(spike_run["is_test"]))
    for neuron in neurons:
        for kind in (PathKind.SPIKE_LABELS, PathKind.SPIKE_MANUAL_REVIEW):
            try:
                stamp = path_for(kind, run_folder, cell=str(neuron["pipeline_neuron_id"])).stat().st_mtime_ns
            except OSError:
                continue
            newest = max(newest, stamp)
            if kind is PathKind.SPIKE_LABELS:
                labels_seen += 1
            else:
                marks_seen += 1
    try:
        iscell_stamp = path_for(PathKind.SUITE2P_ISCELL, source_path).stat().st_mtime_ns
    except (OSError, TypeError):
        iscell_stamp = 0
    return (int(experiment_id), int(spike_run["id"]), tuple(int(n["id"]) for n in neurons), newest, labels_seen, marks_seen, iscell_stamp)
