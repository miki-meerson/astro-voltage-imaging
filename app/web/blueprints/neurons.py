"""Cell (neuron) page: summary images, cropped STA movie, confirm / unconfirm, review mark."""
from __future__ import annotations

from io import BytesIO

from flask import jsonify, redirect, render_template, request, send_file, url_for

from app.db.core import connect
from app.services.cell_views import sta_crop_stack_payload
from app.services.run_views import STEP_LABELS, latest_completed_run, run_view_for_neuron
from app.services.scientific_db import confirm_neuron, unconfirm_neuron
from app.services.spikes import (
    load_spike_payload,
    save_spike_payload,
    spike_cell_dir,
    spike_review_info,
    spikes_edited_after_sta,
    stale_suite2p_cell_message,
)
from app.services.visualizations import render_motion_sta, render_sta_cell_summary
from app.web.common import FlatBlueprint, get_neuron, get_settings

bp = FlatBlueprint("neurons", __name__)


@bp.get("/neurons/<int:neuron_id>")
def neuron(neuron_id):
    settings = get_settings()
    with connect(settings) as conn:
        n = get_neuron(conn, neuron_id)
        if n is None:
            return render_template("error.html", title="Missing Cell", message="Cell not found"), 404
        runs = [
            run_view_for_neuron(r, settings, n["pipeline_neuron_id"])
            for r in conn.execute("SELECT * FROM analysis_runs WHERE experiment_id=? ORDER BY step,id DESC", (n["experiment_id"],))
        ]
        confirmed = conn.execute("SELECT * FROM confirmed_neurons WHERE experiment_id=? AND pipeline_neuron_id=?", (n["experiment_id"], n["pipeline_neuron_id"])).fetchone()
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
        confirmed=dict(confirmed) if confirmed else None,
        embedded=request.args.get("embedded") == "1",
    )


@bp.get("/neurons/<int:neuron_id>/sta-summary.png")
def neuron_sta_summary(neuron_id):
    settings = get_settings()
    with connect(settings) as conn:
        n = get_neuron(conn, neuron_id)
        if n is None:
            return render_template("error.html", title="Missing Cell", message="Cell not found"), 404
        img = render_sta_cell_summary(settings, conn, n)
    return send_file(img, mimetype="image/png")


@bp.get("/neurons/<int:neuron_id>/motion-sta.png")
def neuron_motion_sta(neuron_id):
    settings = get_settings()
    with connect(settings) as conn:
        n = get_neuron(conn, neuron_id)
        if n is None:
            return render_template("error.html", title="Missing Cell", message="Cell not found"), 404
        img = render_motion_sta(settings, conn, n)
    return send_file(img, mimetype="image/png")


@bp.get("/neurons/<int:neuron_id>/anatomy-crop/stack")
def neuron_anatomy_crop_stack(neuron_id):
    settings = get_settings()
    try:
        return send_file(BytesIO(sta_crop_stack_payload(settings, neuron_id)), mimetype="application/octet-stream", max_age=300)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@bp.post("/neurons/<int:neuron_id>/confirm")
def confirm_cell(neuron_id):
    settings = get_settings()
    try:
        confirm_neuron(settings, neuron_id, request.form.get("reviewer_note", ""))
        return redirect(url_for("neuron", neuron_id=neuron_id, embedded=request.form.get("embedded") or None, confirmed="1"))
    except Exception as exc:
        return render_template("error.html", title="Confirm Failed", message=str(exc)), 400


@bp.post("/neurons/<int:neuron_id>/unconfirm")
def unconfirm_cell(neuron_id):
    settings = get_settings()
    try:
        unconfirm_neuron(settings, neuron_id)
        return redirect(url_for("neuron", neuron_id=neuron_id, embedded=request.form.get("embedded") or None, unconfirmed="1"))
    except Exception as exc:
        return render_template("error.html", title="Unconfirm Failed", message=str(exc)), 400


@bp.post("/neurons/<int:neuron_id>/review/mark")
def mark_cell_reviewed(neuron_id):
    settings = get_settings()
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
    cell_dir = spike_cell_dir(settings, spike_run, n["pipeline_neuron_id"])
    payload = load_spike_payload(cell_dir / "simple_complex_spike_labels.npz")
    save_spike_payload(cell_dir, payload["peaks_all"], payload["labels"])
    return jsonify({"ok": True, **spike_review_info(settings, spike_run, n["pipeline_neuron_id"])})
