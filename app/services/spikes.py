"""Spike labels of a cell: read/write the label files, detection parameters, stale-Suite2p checks."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from app.db import experiment_repository, run_repository
from app.services import caches
from app.services.runs import run_dir
from app.services.suite2p_data import experiment_stat_cells
from utils import globals_utils as g
from utils.file_utils import read_npz_arrays, write_json
from utils.paths_utils import PathKind, path_for
from utils.spike_params import SPIKE_PARAM_SPECS, spike_params_with_defaults

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
        return len(experiment_stat_cells(source_path))
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
        return {"reviewed": False, "n_peaks": 0, "n_excluded": 0}
    run_folder = run_dir(settings, int(spike_run["id"]), bool(spike_run["is_test"]))
    try:
        payload = load_spike_payload(path_for(PathKind.SPIKE_LABELS, run_folder, cell=pipeline_neuron_id))
    except FileNotFoundError:  # no label file yet: the cell has no spikes to count
        return {"reviewed": False, "n_peaks": 0, "n_excluded": 0}
    labels = payload["labels"].astype(str)
    return {
        "reviewed": path_for(PathKind.SPIKE_MANUAL_REVIEW, run_folder, cell=pipeline_neuron_id).exists(),
        "n_peaks": int(payload["peaks_all"].size),
        "n_excluded": int(np.sum(labels == "exclude")),
    }


def load_spike_payload(path: Path) -> dict:
    """Peaks and labels (sorted by frame) of a cell's saved spike label file. FileNotFoundError says which step writes the file."""
    missing = f"The spike labels file {path} is missing: the Spike Detection step writes it, so rerun Spike Detection for this experiment."
    saved = read_npz_arrays(path, "The spike labels file", missing=missing)
    peaks = np.asarray(saved["peaks_all"], dtype=np.int32)
    labels = np.asarray(saved["labels"]).astype(str)
    order = np.argsort(peaks)
    return {"peaks_all": peaks[order], "labels": labels[order]}


def spike_detection_params(run_row, params_path: Path | None = None) -> dict:
    """Detector settings of a spike run; the ones saved with the cell (params_path, written by the editor's save) win."""
    result = spike_params_with_defaults(json.loads(run_row["parameters_json"]) if run_row is not None else {})
    if params_path is not None:
        saved = read_npz_arrays(params_path, "The saved spike settings", default={})  # the file only exists once the cell's settings were saved
        if "mad_threshold_factor" in saved:
            result["mad_factor"] = float(saved["mad_threshold_factor"].item())
        for name, (kind, _) in SPIKE_PARAM_SPECS.items():
            if name != "mad_factor" and name in saved:
                result[name] = kind(saved[name].item())
    return result


def save_spike_payload(spike_run_folder: Path, pipeline_neuron_id: str, peaks: np.ndarray, labels: np.ndarray, params: dict | None = None) -> None:
    """Write the edited peaks and labels of a cell into its folder of the spike run (the other keys of the old label file are kept)."""
    path_for(PathKind.SPIKE_CELL_DIR, spike_run_folder, cell=pipeline_neuron_id).mkdir(parents=True, exist_ok=True)
    order = np.argsort(peaks)
    peaks = np.asarray(peaks, dtype=np.int32)[order]
    labels = np.asarray(labels).astype("<U8")[order]
    # the other keys of the old label file are kept; a cell without a label file yet: the first save creates it
    payload = read_npz_arrays(path_for(PathKind.SPIKE_LABELS, spike_run_folder, cell=pipeline_neuron_id), "The spike labels file", default={})
    payload.update({
        "peaks_all": peaks,
        "labels": labels,
        "manual_reviewed": np.asarray(True),
    })
    np.savez_compressed(path_for(PathKind.SPIKE_LABELS, spike_run_folder, cell=pipeline_neuron_id), **payload)
    saved_params = dict(params or {})
    if saved_params:
        full = spike_params_with_defaults(saved_params)  # one default per setting, as in the editor
        np.savez(
            path_for(PathKind.SPIKE_PEAK_PARAMS, spike_run_folder, cell=pipeline_neuron_id),
            threshold_method="mad",
            mad_threshold_factor=full["mad_factor"],
            **{name: full[name] for name in SPIKE_PARAM_SPECS if name != "mad_factor"},
        )
    write_json(
        path_for(PathKind.SPIKE_MANUAL_REVIEW, spike_run_folder, cell=pipeline_neuron_id),
        {
            "peaks_all": peaks.tolist(),
            "labels": labels.tolist(),
            "detection_params": saved_params,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def review_cache_key(experiment_id: int, spike_run, neurons: list[dict]) -> tuple | None:
    """Key under which the cell-review summary of an experiment is kept: the experiment, its spike run and its cells (None, not kept, when there is no spike run). No file is
    looked at: the summary is made again when the spike run or the cells change, when Suite2p is confirmed (the caches of the experiment are dropped) and, for the one cell,
    when it is saved or marked reviewed (refresh_review_entry)."""
    if spike_run is None:
        return None
    return (int(experiment_id), int(spike_run["id"]), tuple(int(n["id"]) for n in neurons))


def experiment_spike_review(settings, experiment_id: int) -> dict:
    """What the Spikes tab lists: {has_spike_run, reviewed_count, total_count, cells: [{id, cell, reviewed, n_peaks, n_excluded}]} for the current Suite2p
    cells of an experiment. Kept (review_cache_key) so that going between cells does not read the files of every cell again."""
    source_path = experiment_repository.experiment_source_path(settings, experiment_id)
    spike_run = run_repository.latest_run(settings, experiment_id, g.SPIKE_STEP)
    neurons = experiment_repository.experiment_neurons(settings, experiment_id)
    key = review_cache_key(experiment_id, spike_run, neurons)
    cached = caches.CELL_REVIEW.get(key) if key is not None else None
    if cached is not None:
        return cached
    cells = [{"id": int(n["id"]), "cell": int(n["pipeline_neuron_id"]), **spike_review_info(settings, spike_run, str(n["pipeline_neuron_id"]))}
             for n in current_suite2p_neurons(source_path, neurons)]
    result = {"has_spike_run": spike_run is not None, "reviewed_count": sum(c["reviewed"] for c in cells), "total_count": len(cells), "cells": cells}
    if key is not None:
        caches.CELL_REVIEW.put(key, result)
    return result


def refresh_review_entry(settings, experiment_id: int, spike_run, pipeline_neuron_id: str) -> dict:
    """A cell was saved or marked reviewed: its entry in the kept review of the experiment (when there is one) is made again from its files and the others stay as they are.
    Returns the cell's {reviewed, n_peaks, n_excluded}."""
    info = spike_review_info(settings, spike_run, str(pipeline_neuron_id))
    key = review_cache_key(experiment_id, spike_run, experiment_repository.experiment_neurons(settings, experiment_id))
    kept = caches.CELL_REVIEW.get(key) if key is not None else None
    if kept is not None:
        cells = [{**c, **info} if c["cell"] == int(pipeline_neuron_id) else c for c in kept["cells"]]
        caches.CELL_REVIEW.put(key, {**kept, "cells": cells, "reviewed_count": sum(c["reviewed"] for c in cells)})
    return info
