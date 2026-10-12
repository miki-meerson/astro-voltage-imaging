"""Spike editor page and its data / detect / save endpoints."""
from __future__ import annotations

from pathlib import Path

from flask import jsonify, render_template, request
import numpy as np

from app.db.core import connect
from app.services.movies import experiment_fps
from app.services.run_views import latest_completed_run
from app.services.spikes import (
    bounded_spike_params,
    load_spike_payload,
    save_spike_payload,
    spike_cell_dir,
    spike_detection_params,
    stale_suite2p_cell_message,
    suite2p_plane0,
)
from app.web.common import FlatBlueprint, get_neuron, get_settings

bp = FlatBlueprint("spikes", __name__)


@bp.get("/neurons/<int:neuron_id>/spikes")
def spike_editor(neuron_id):
    settings = get_settings()
    with connect(settings) as conn:
        n = get_neuron(conn, neuron_id)
        if n is None:
            return render_template("error.html", title="Missing Cell", message="Cell not found"), 404
        spike_run = latest_completed_run(conn, int(n["experiment_id"]), "spike")
    if spike_run is None:
        return render_template("error.html", title="No Spike Run", message="Run spike detection before editing spikes."), 400
    stale_message = stale_suite2p_cell_message(n["source_path"], n["pipeline_neuron_id"])
    if stale_message:
        return render_template("error.html", title="Stale Suite2p Cell", message=stale_message), 400
    return render_template("spikes.html", neuron=dict(n), embedded=request.args.get("embedded") == "1")


@bp.get("/neurons/<int:neuron_id>/spikes/data")
def spike_data(neuron_id):
    settings = get_settings()
    with connect(settings) as conn:
        n = get_neuron(conn, neuron_id)
        if n is None:
            return jsonify({"error": "Cell not found"}), 404
        spike_run = latest_completed_run(conn, int(n["experiment_id"]), "spike")
    if spike_run is None:
        return jsonify({"error": "No completed spike run."}), 400
    from utils.spikes_utils import _suite2p_trace_dff
    from utils.suite2p_utils import load_suite2p_context

    suite2p_ctx = load_suite2p_context(suite2p_plane0(n["source_path"]))
    cell_id = int(n["pipeline_neuron_id"])
    if cell_id < 0 or cell_id >= int(suite2p_ctx["n_cells"]):
        stale_message = stale_suite2p_cell_message(n["source_path"], n["pipeline_neuron_id"])
        return jsonify({"error": stale_message or f"Cell {cell_id} is outside Suite2p traces."}), 400
    trace = np.asarray(_suite2p_trace_dff(cell_id, suite2p_ctx=suite2p_ctx), dtype=float)
    fps = experiment_fps(n["source_path"])
    labels_path = spike_cell_dir(settings, spike_run, n["pipeline_neuron_id"]) / "simple_complex_spike_labels.npz"
    payload = load_spike_payload(labels_path)
    params = spike_detection_params(spike_run, labels_path.parent)
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
    params = bounded_spike_params(data)
    with connect(settings) as conn:
        n = get_neuron(conn, neuron_id)
        if n is None:
            return jsonify({"error": "Cell not found"}), 404
        spike_run = latest_completed_run(conn, int(n["experiment_id"]), "spike")
    if spike_run is None:
        return jsonify({"error": "No completed spike run."}), 400
    from utils.spikes_utils import classify_spikes, detect_peaks_mad, highpass_traces
    from utils.suite2p_utils import load_suite2p_context

    home = Path(n["source_path"])
    suite2p = load_suite2p_context(suite2p_plane0(home))
    cell_id = int(n["pipeline_neuron_id"])
    fps = int(experiment_fps(home))
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
        min_distance=params["min_distance"],
    )
    labels, features = classify_spikes(
        trace,
        peaks,
        fps,
        min_amplitude=params["min_amplitude"],
        min_width=params["min_width"],
        min_subthreshold=params["min_subthreshold"],
        min_ratio=params["min_ratio"],
        subthreshold_window_seconds=0.2,
        baseline_window_seconds=1.5,
    )
    return jsonify(
        {
            "peaks": labels.peaks_all.astype(int).tolist(),
            "labels": labels.labels.astype(str).tolist(),
            "n_simple": int(labels.simple_peaks.size),
            "n_complex": int(labels.complex_peaks.size),
            "n_total": int(labels.peaks_all.size),
            "params": params,
            "score": np.round(features.get("score", np.empty(0)), 4).tolist(),
        }
    )


@bp.post("/neurons/<int:neuron_id>/spikes/save")
def save_spike_data(neuron_id):
    settings = get_settings()
    data = request.get_json(force=True)
    peaks = np.asarray(data.get("peaks", []), dtype=np.int32)
    labels = np.asarray(data.get("labels", [])).astype("<U8")
    raw_params = data.get("params") or {}
    params = bounded_spike_params(raw_params)
    if peaks.size != labels.size:
        return jsonify({"error": "peaks and labels must have the same length"}), 400
    keep = labels != "delete"
    peaks = peaks[keep]
    labels = labels[keep]
    with connect(settings) as conn:
        n = get_neuron(conn, neuron_id)
        if n is None:
            return jsonify({"error": "Cell not found"}), 404
        spike_run = latest_completed_run(conn, int(n["experiment_id"]), "spike")
    if spike_run is None:
        return jsonify({"error": "No completed spike run."}), 400
    stale_message = stale_suite2p_cell_message(n["source_path"], n["pipeline_neuron_id"])
    if stale_message:
        return jsonify({"error": stale_message}), 400
    save_spike_payload(spike_cell_dir(settings, spike_run, n["pipeline_neuron_id"]), peaks, labels, params)
    return jsonify({"ok": True, "n_peaks": int(peaks.size), "params": params})
