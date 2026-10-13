"""Cell (neuron) page: summary images, cropped STA movie, confirm / unconfirm, review mark."""
from __future__ import annotations

from io import BytesIO

from flask import jsonify, redirect, render_template, request, send_file, url_for

from app.db import neuron_repository
from app.services.cell_views import cell_movie_files, sta_crop_stack_payload
from app.services.run_views import STEP_LABELS, run_view_for_neuron
from app.services.runs import run_dir
from app.services.scientific_db import confirm_neuron, unconfirm_neuron
from app.services.spikes import (
    load_spike_payload,
    save_spike_payload,
    spike_review_info,
    spikes_edited_after_sta,
    stale_suite2p_cell_message,
)
from app.services.visualizations import render_motion_sta, render_sta_cell_summary
from app.web.common import FlatBlueprint, get_settings
from utils.paths_utils import PathKind, path_for

bp = FlatBlueprint("neurons", __name__)


@bp.get("/neurons/<int:neuron_id>")
def neuron(neuron_id):
    settings = get_settings()
    n, run_rows, confirmed = neuron_repository.neuron_page_data(settings, neuron_id)
    if n is None:
        return render_template("error.html", title="Missing Cell", message="Cell not found"), 404
    runs = [run_view_for_neuron(r, settings, n["pipeline_neuron_id"]) for r in run_rows]
    active = any(r["status"] in {"pending", "queued", "running"} for r in runs)
    runs_by_step = {}
    for r in runs:
        runs_by_step.setdefault(r["step"], r)
    sta_stale = None
    sta_view, spike_view = runs_by_step.get("sta"), runs_by_step.get("spike")
    if sta_view and spike_view and sta_view["status"] == "completed" and spike_view["status"] == "completed":
        edited = spikes_edited_after_sta(settings, spike_view, sta_view, n["pipeline_neuron_id"])
        if edited is not None:
            sta_stale = f"Spikes of this cell were edited on {edited:%Y-%m-%d %H:%M}, after this STA run started ({str(sta_view['started_at'])[:16].replace('T', ' ')}). These STA results used the old spikes: rerun STA Maps."
    return render_template(
        "neuron.html",
        sta_stale=sta_stale,
        neuron=dict(n),
        runs_by_step=runs_by_step,
        active=active,
        step_labels=STEP_LABELS,
        confirmed=confirmed,
        embedded=request.args.get("embedded") == "1",
    )


@bp.get("/neurons/<int:neuron_id>/sta-summary.png")
def neuron_sta_summary(neuron_id):
    settings = get_settings()
    n = neuron_repository.neuron_row(settings, neuron_id)
    if n is None:
        return render_template("error.html", title="Missing Cell", message="Cell not found"), 404
    return send_file(render_sta_cell_summary(settings, n), mimetype="image/png")


@bp.get("/neurons/<int:neuron_id>/motion-sta.png")
def neuron_motion_sta(neuron_id):
    settings = get_settings()
    n = neuron_repository.neuron_row(settings, neuron_id)
    if n is None:
        return render_template("error.html", title="Missing Cell", message="Cell not found"), 404
    return send_file(render_motion_sta(settings, n), mimetype="image/png")


@bp.get("/neurons/<int:neuron_id>/anatomy-crop/stack")
def neuron_anatomy_crop_stack(neuron_id):
    settings = get_settings()
    try:
        files = cell_movie_files(settings, neuron_id)
        payload = sta_crop_stack_payload(files.movie_tif, files.anatomy_npz, files.source_path, files.pipeline_neuron_id)
        return send_file(BytesIO(payload), mimetype="application/octet-stream", max_age=300)
    except FileNotFoundError as exc:  # a saved file is missing: the text says which step writes it
        return jsonify({"error": str(exc)}), 404
    except (ValueError, KeyError, OSError) as exc:  # unknown cell, movie of another STA run, no fps in Experiment.xml, a file that cannot be read
        return jsonify({"error": str(exc)}), 400


@bp.post("/neurons/<int:neuron_id>/confirm")
def confirm_cell(neuron_id):
    settings = get_settings()
    try:
        confirm_neuron(settings, neuron_id, request.form.get("reviewer_note", ""))
        return redirect(url_for("neuron", neuron_id=neuron_id, embedded=request.form.get("embedded") or None, confirmed="1"))
    except (ValueError, OSError) as exc:  # no STA run / no result files for this cell, a file that cannot be copied
        return render_template("error.html", title="Confirm Failed", message=str(exc)), 400


@bp.post("/neurons/<int:neuron_id>/unconfirm")
def unconfirm_cell(neuron_id):
    settings = get_settings()
    try:
        unconfirm_neuron(settings, neuron_id)
        return redirect(url_for("neuron", neuron_id=neuron_id, embedded=request.form.get("embedded") or None, unconfirmed="1"))
    except OSError as exc:  # the saved copies could not be removed
        return render_template("error.html", title="Unconfirm Failed", message=str(exc)), 400


@bp.post("/neurons/<int:neuron_id>/review/mark")
def mark_cell_reviewed(neuron_id):
    settings = get_settings()
    n, spike_run = neuron_repository.neuron_with_latest_run(settings, neuron_id, "spike")
    if n is None:
        return jsonify({"error": "Cell not found"}), 404
    if spike_run is None:
        return jsonify({"error": "No completed spike run."}), 400
    stale_message = stale_suite2p_cell_message(n["source_path"], n["pipeline_neuron_id"])
    if stale_message:
        return jsonify({"error": stale_message}), 400
    run_folder = run_dir(settings, int(spike_run["id"]), bool(spike_run["is_test"]))
    cell = n["pipeline_neuron_id"]
    try:
        payload = load_spike_payload(path_for(PathKind.SPIKE_LABELS, run_folder, cell=cell))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    save_spike_payload(run_folder, cell, payload["peaks_all"], payload["labels"])
    return jsonify({"ok": True, **spike_review_info(settings, spike_run, cell)})
