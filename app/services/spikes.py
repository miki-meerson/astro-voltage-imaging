"""Spike labels of a cell: read/write the label files, detection parameters, stale-Suite2p checks."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np

from app.services.runs import run_dir

def spike_cell_dir(settings, run_row, pipeline_neuron_id: str) -> Path:
    return run_dir(settings, int(run_row["id"]), bool(run_row["is_test"])) / f"cell_{pipeline_neuron_id}"


def spikes_edited_after_sta(settings, spike_run, sta_run, pipeline_neuron_id) -> datetime | None:
    """When the cell's spike labels were last written, if that is after the STA run started (so the STA used older spikes)."""
    if spike_run is None or sta_run is None or not sta_run.get("started_at"):
        return None
    labels = spike_cell_dir(settings, spike_run, str(pipeline_neuron_id)) / "simple_complex_spike_labels.npz"
    try:
        edited = datetime.fromtimestamp(labels.stat().st_mtime)
        started = datetime.fromisoformat(str(sta_run["started_at"]).replace(" ", "T"))
    except (OSError, ValueError):
        return None
    return edited if edited > started else None


def suite2p_plane0(source_path: str | Path) -> Path:
    """Folder of the Suite2p outputs (iscell.npy, stat.npy, F.npy, ...) of an experiment."""
    return Path(source_path) / "pipeline_results" / "for_suite2p" / "suite2p" / "plane0"


def suite2p_stat_cells(source_path: str | Path) -> np.ndarray:
    """Suite2p ROI stats of the accepted cells only (the cell ids of the app are indices into this array)."""
    plane0 = suite2p_plane0(source_path)
    iscell = np.load(plane0 / "iscell.npy", allow_pickle=True)[:, 0].astype(bool)
    return np.load(plane0 / "stat.npy", allow_pickle=True)[iscell]


def suite2p_cell_count(source_path: str | Path | None) -> int | None:
    if not source_path:
        return None
    try:
        from utils.suite2p_utils import load_suite2p_context

        return int(load_suite2p_context(suite2p_plane0(source_path))["n_cells"])
    except Exception:
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
    cell_dir = spike_cell_dir(settings, spike_run, pipeline_neuron_id)
    payload = load_spike_payload(cell_dir / "simple_complex_spike_labels.npz")
    labels = payload["labels"].astype(str)
    return {
        "reviewed": (cell_dir / "manual_spike_review.json").exists(),
        "n_peaks": int(payload["peaks_all"].size),
        "n_simple": int(np.sum(labels == "simple")),
        "n_complex": int(np.sum(labels == "complex")),
        "n_excluded": int(np.sum(labels == "exclude")),
    }


def load_spike_payload(path: Path) -> dict:
    if not path.exists():
        return {"peaks_all": np.empty(0, dtype=np.int32), "labels": np.empty(0, dtype="<U8")}
    with np.load(path, allow_pickle=True) as d:
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


def spike_detection_params(run_row, cell_dir: Path | None = None) -> dict:
    try:
        params = json.loads(run_row["parameters_json"] or "{}") if run_row is not None else {}
    except json.JSONDecodeError:
        params = {}
    result = {name: kind(params.get(name, default)) for name, (kind, default, _, _) in SPIKE_PARAM_SPECS.items()}
    params_path = cell_dir / "peak_detection_params.npz" if cell_dir is not None else None
    if params_path is not None and params_path.exists():
        with np.load(params_path, allow_pickle=True) as d:
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


def save_spike_payload(cell_dir: Path, peaks: np.ndarray, labels: np.ndarray, params: dict | None = None) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(peaks)
    peaks = np.asarray(peaks, dtype=np.int32)[order]
    labels = np.asarray(labels).astype("<U8")[order]
    existing_path = cell_dir / "simple_complex_spike_labels.npz"
    payload = {}
    if existing_path.exists():
        with np.load(existing_path, allow_pickle=True) as d:
            payload = {key: np.asarray(d[key]) for key in d.files}
    payload.update({
        "peaks_all": peaks,
        "labels": labels,
        "simple_peaks": peaks[labels == "simple"],
        "complex_peaks": peaks[labels == "complex"],
        "excluded_peaks": peaks[labels == "exclude"],
        "manual_reviewed": np.asarray(True),
    })
    for name in ("simple_complex_spike_labels.npz", "simple_complex_peaks_for_analysis.npz"):
        np.savez_compressed(cell_dir / name, **payload)
    saved_params = dict(params or {})
    if saved_params:
        full = bounded_spike_params(saved_params)  # one default per setting, as in the editor
        np.savez(
            cell_dir / "peak_detection_params.npz",
            threshold_method="mad",
            mad_threshold_factor=full["mad_factor"],
            **{name: full[name] for name in SPIKE_PARAM_SPECS if name != "mad_factor"},
        )
    (cell_dir / "manual_spike_review.json").write_text(
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
    for neuron in neurons:
        cell_dir = spike_cell_dir(settings, spike_run, str(neuron["pipeline_neuron_id"]))
        for name in ("simple_complex_spike_labels.npz", "manual_spike_review.json"):
            try:
                stamp = (cell_dir / name).stat().st_mtime_ns
            except OSError:
                continue
            newest = max(newest, stamp)
            if name.endswith(".npz"):
                labels_seen += 1
            else:
                marks_seen += 1
    try:
        iscell_stamp = (suite2p_plane0(source_path) / "iscell.npy").stat().st_mtime_ns
    except (OSError, TypeError):
        iscell_stamp = 0
    return (int(experiment_id), int(spike_run["id"]), tuple(int(n["id"]) for n in neurons), newest, labels_seen, marks_seen, iscell_stamp)
