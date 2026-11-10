"""The data, detect and save endpoints of the spike editor (the editor itself is on the Spikes tab, workspace.py)."""
from __future__ import annotations

from flask import jsonify, request
import numpy as np

from app.db import neuron_repository
from app.services.runs import run_dir
from app.services.spikes import (
    load_spike_payload,
    save_spike_payload,
    spike_detection_params,
    refresh_review_entry,
    stale_suite2p_cell_message,
)
from app.services.suite2p_data import experiment_suite2p_context
from app.web.common import FlatBlueprint, get_settings
from utils import globals_utils as g
from utils.movie_utils import read_frame_rate
from utils.paths_utils import PathKind, path_for
from utils.spike_params import spike_params_with_defaults
from utils.spikes_utils import detect_peaks_mad, highpass_traces
from utils.suite2p_utils import suite2p_trace_dff

bp = FlatBlueprint("spikes", __name__)


@bp.get("/neurons/<int:neuron_id>/spikes/data")
def spike_data(neuron_id):
    settings = get_settings()
    n, spike_run = neuron_repository.neuron_with_latest_run(settings, neuron_id, g.SPIKE_STEP)
    if n is None:
        return jsonify({"error": "Cell not found"}), 404
    if spike_run is None:
        return jsonify({"error": "No completed spike run."}), 400
    suite2p_ctx = experiment_suite2p_context(n["source_path"])
    cell_id = int(n["pipeline_neuron_id"])
    if cell_id < 0 or cell_id >= int(suite2p_ctx["n_cells"]):
        stale_message = stale_suite2p_cell_message(n["source_path"], n["pipeline_neuron_id"])
        return jsonify({"error": stale_message or f"Cell {cell_id} is outside Suite2p traces."}), 400
    trace = np.asarray(suite2p_trace_dff(cell_id, suite2p_ctx=suite2p_ctx), dtype=float)
    try:
        fps = read_frame_rate(n["source_path"])
    except ValueError as exc:  # no frame rate in Experiment.xml
        return jsonify({"error": str(exc)}), 400
    run_folder = run_dir(settings, int(spike_run["id"]), bool(spike_run["is_test"]))
    cell = n["pipeline_neuron_id"]
    try:
        payload = load_spike_payload(path_for(PathKind.SPIKE_LABELS, run_folder, cell=cell))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    params = spike_detection_params(spike_run, path_for(PathKind.SPIKE_PEAK_PARAMS, run_folder, cell=cell))
    return jsonify(
        {
            "trace": np.round(trace, 5).tolist(),
            "fps": fps,
            "peaks": payload["peaks_all"].astype(int).tolist(),
            "labels": payload["labels"].astype(str).tolist(),
            "params": params,
        }
    )


@bp.post("/neurons/<int:neuron_id>/spikes/detect")
def detect_spike_data(neuron_id):
    settings = get_settings()
    data = request.get_json(force=True) or {}
    params = spike_params_with_defaults(data)
    n, spike_run = neuron_repository.neuron_with_latest_run(settings, neuron_id, g.SPIKE_STEP)
    if n is None:
        return jsonify({"error": "Cell not found"}), 404
    if spike_run is None:
        return jsonify({"error": "No completed spike run."}), 400
    suite2p = experiment_suite2p_context(n["source_path"])
    cell_id = int(n["pipeline_neuron_id"])
    try:
        fps = int(read_frame_rate(n["source_path"]))
    except ValueError as exc:  # no frame rate in Experiment.xml
        return jsonify({"error": str(exc)}), 400
    traces = np.asarray(suite2p["F_norm"], dtype=np.float32)
    if cell_id < 0 or cell_id >= traces.shape[0]:
        stale_message = stale_suite2p_cell_message(n["source_path"], n["pipeline_neuron_id"])
        return jsonify({"error": stale_message or f"Cell {cell_id} is outside Suite2p traces."}), 400
    trace = np.asarray(traces[cell_id], dtype=np.float32)
    detection = highpass_traces(trace[np.newaxis, :], fps, params["highpass_cutoff_hz"])[0]
    peaks = detect_peaks_mad(
        trace,
        detection,
        mad_factor=params["mad_factor"],
        duplicate_window_frames=params["duplicate_window_frames"],
        align_radius_frames=params["align_radius_frames"],
    )
    return jsonify({"peaks": peaks.astype(int).tolist(), "labels": ["spike"] * int(peaks.size), "n_total": int(peaks.size), "params": params})


@bp.post("/neurons/<int:neuron_id>/spikes/save")
def save_spike_data(neuron_id):
    settings = get_settings()
    data = request.get_json(force=True)
    peaks = np.asarray(data.get("peaks", []), dtype=np.int32)
    labels = np.asarray(data.get("labels", [])).astype("<U8")
    raw_params = data.get("params") or {}
    params = spike_params_with_defaults(raw_params)
    if peaks.size != labels.size:
        return jsonify({"error": "peaks and labels must have the same length"}), 400
    keep = labels != "delete"
    peaks = peaks[keep]
    labels = labels[keep]
    n, spike_run = neuron_repository.neuron_with_latest_run(settings, neuron_id, g.SPIKE_STEP)
    if n is None:
        return jsonify({"error": "Cell not found"}), 404
    if spike_run is None:
        return jsonify({"error": "No completed spike run."}), 400
    stale_message = stale_suite2p_cell_message(n["source_path"], n["pipeline_neuron_id"])
    if stale_message:
        return jsonify({"error": stale_message}), 400
    save_spike_payload(run_dir(settings, int(spike_run["id"]), bool(spike_run["is_test"])), n["pipeline_neuron_id"], peaks, labels, params)
    refresh_review_entry(settings, int(n["experiment_id"]), spike_run, n["pipeline_neuron_id"])
    return jsonify({"ok": True, "n_peaks": int(peaks.size), "params": params})
