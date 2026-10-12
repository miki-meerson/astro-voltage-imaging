"""Experiment pages: registry (index, add, scan), analysis page, notes, review data, experiment movie, run launch."""
from __future__ import annotations

import json
import threading

from flask import jsonify, redirect, render_template, request, url_for

from app.db.core import connect, rows, tx
from app.services import caches
from app.services.anatomy_cluster_summary import anatomy_cluster_summary
from app.services.behavior_view import behavior_payload
from app.services.experiments import (
    add_experiment,
    launch_default_steps,
    remove_experiment_from_registry,
    reset_experiment_app_data,
    scan_parent,
    validate_experiment,
)
from app.services.movies import pb_movie_path
from app.services.run_views import (
    STEP_LABELS,
    experiment_notes,
    latest_completed_run,
    run_view,
    build_validation_rows,
    validation_all_ok,
)
from app.services.runs import create_run, launch_run
from app.services.spikes import current_suite2p_neurons, review_cache_key, spike_review_info, spikes_edited_after_sta
from app.services.sta_results import STA_RESULT_TYPES, sta_significant_summary
from app.web.common import FlatBlueprint, get_settings, movie_frame_response, movie_meta_response
from app.web.security import safe_next

bp = FlatBlueprint("experiments", __name__)


@bp.get("/")
def index():
    settings = get_settings()
    favorites_only = request.args.get("favorites") == "1"
    with connect(settings) as conn:
        experiments = rows(conn, """
            SELECT e.*,
                   COALESCE(n.neuron_count, 0) AS neuron_count,
                   r.running_step,
                   r.pending_step,
                   r.failed_step,
                   COALESCE(r.completed_steps, 0) AS completed_steps,
                   CASE
                     WHEN r.running_step IS NOT NULL THEN 'running'
                     WHEN r.pending_step IS NOT NULL THEN 'pending'
                     WHEN r.failed_step IS NOT NULL THEN 'failed'
                     WHEN COALESCE(r.completed_steps, 0) >= 4 THEN 'completed'
                     ELSE 'none'
                   END AS job_status,
                   CASE
                     WHEN r.running_step IS NOT NULL THEN r.running_step
                     WHEN r.pending_step IS NOT NULL THEN r.pending_step
                     WHEN r.failed_step IS NOT NULL THEN r.failed_step
                     ELSE ''
                   END AS job_step
            FROM experiments e
            LEFT JOIN (
                SELECT experiment_id, COUNT(*) AS neuron_count
                FROM neurons
                GROUP BY experiment_id
            ) n ON n.experiment_id=e.id
            LEFT JOIN (
                SELECT r.experiment_id,
                       MIN(CASE WHEN r.status='running' THEN r.step END) AS running_step,
                       MIN(CASE WHEN r.status IN ('pending','queued') THEN r.step END) AS pending_step,
                       MIN(CASE WHEN r.status='failed' THEN r.step END) AS failed_step,
                       COUNT(DISTINCT CASE WHEN r.status='completed' THEN r.step END) AS completed_steps
                FROM analysis_runs r
                JOIN (
                    SELECT experiment_id,step,MAX(id) AS run_id
                    FROM analysis_runs
                    WHERE is_test=0
                    GROUP BY experiment_id,step
                ) latest ON latest.run_id=r.id
                GROUP BY r.experiment_id
            ) r ON r.experiment_id=e.id
            WHERE (? = 0 OR e.is_favorite = 1)
            ORDER BY substr(e.recording_date, 7, 4) DESC,
                     substr(e.recording_date, 4, 2) DESC,
                     substr(e.recording_date, 1, 2) DESC,
                     e.canonical_id ASC
        """, (1 if favorites_only else 0,))
        last = conn.execute("SELECT * FROM scan_history ORDER BY id DESC LIMIT 1").fetchone()
    active = any(e.get("job_status") in {"running", "pending"} for e in experiments)
    return render_template("index.html", experiments=experiments, settings=settings, last_scan=dict(last) if last else None, active=active, favorites_only=favorites_only, step_labels=STEP_LABELS)


@bp.post("/experiments/add")
def add():
    settings = get_settings()
    try:
        exp_id = add_experiment(request.form["path"].strip().strip('"'), settings)
        with connect(settings) as conn:
            exp = conn.execute("SELECT 1 FROM experiments WHERE id=?", (exp_id,)).fetchone()
            has_runs = conn.execute("SELECT 1 FROM analysis_runs WHERE experiment_id=? LIMIT 1", (exp_id,)).fetchone() is not None
        if not has_runs and exp is not None:
            launch_default_steps(settings, exp_id)
            return redirect(url_for("experiment", experiment_id=exp_id))
        return redirect(url_for("experiment", experiment_id=exp_id, notice="already_registered"))
    except Exception as exc: return render_template("error.html", title="Add Failed", message=str(exc)), 400


@bp.post("/scan")
def scan():
    settings = get_settings()
    try:
        parent = request.form.get("parent") or str(settings.scan_parent)
        threading.Thread(target=scan_parent, args=(parent, settings), daemon=True, name="astro-scan").start()
        return redirect(url_for("index"))
    except Exception as exc: return render_template("error.html", title="Scan Failed", message=str(exc)), 400


@bp.post("/experiments/<int:experiment_id>/remove")
def remove_experiment(experiment_id):
    settings = get_settings()
    try:
        remove_experiment_from_registry(experiment_id, settings)
        return redirect(url_for("index"))
    except Exception as exc:
        return render_template("error.html", title="Remove Failed", message=str(exc)), 400


@bp.post("/experiments/<int:experiment_id>/reset-data")
def reset_experiment_data(experiment_id):
    settings = get_settings()
    try:
        result = reset_experiment_app_data(experiment_id, settings)
        caches.clear_experiment_result_caches()
        caches.MEAN_INTENSITY.clear()
        notice = f"data_reset_{result['counts']['runs']}_{result['counts']['neurons']}"
        return redirect(url_for("experiment", experiment_id=experiment_id, notice=notice))
    except Exception as exc:
        return render_template("error.html", title="Reset Data Failed", message=str(exc)), 400


@bp.post("/experiments/<int:experiment_id>/favorite")
def favorite_experiment(experiment_id):
    settings = get_settings()
    try:
        favorite = 1 if request.form.get("favorite") == "1" else 0
        with tx(settings) as conn:
            updated = conn.execute(
                "UPDATE experiments SET is_favorite=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (favorite, experiment_id),
            ).rowcount
            if not updated:
                return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
        return redirect(safe_next(request.form.get("next"), url_for("index")))
    except Exception as exc:
        return render_template("error.html", title="Favorite Failed", message=str(exc)), 400


@bp.get("/experiments/<int:experiment_id>")
def experiment(experiment_id):
    settings = get_settings()
    with connect(settings) as conn:
        exp = conn.execute("SELECT workflow_stage FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    if exp is None:
        return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
    if exp["workflow_stage"] == "analysis_ready":
        return redirect(url_for("experiment_analysis", experiment_id=experiment_id, **request.args))
    return redirect(url_for("experiment_preprocessing", experiment_id=experiment_id, **request.args))


@bp.get("/experiments/<int:experiment_id>/anatomy-clusters.json")
def experiment_anatomy_clusters(experiment_id):
    settings = get_settings()
    try:
        with connect(settings) as conn:
            return jsonify(anatomy_cluster_summary(settings, conn, experiment_id))
    except Exception as exc:
        return jsonify({"available": False, "message": str(exc)}), 400


@bp.get("/experiments/<int:experiment_id>/analysis")
def experiment_analysis(experiment_id):
    settings = get_settings()
    selected_neuron_id = request.args.get("neuron", "")
    with connect(settings) as conn:
        exp = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
        neurons = rows(conn, "SELECT id,pipeline_neuron_id FROM neurons WHERE experiment_id=? ORDER BY CAST(pipeline_neuron_id AS INTEGER)", (experiment_id,))
        neurons = current_suite2p_neurons(exp["source_path"], neurons)
        runs = [run_view(r, settings) for r in conn.execute("SELECT * FROM analysis_runs WHERE experiment_id=? ORDER BY created_at DESC, id ASC", (experiment_id,))]
        analysis_latest = {}
        for step in ("spike", "behavior", "anatomy", "sta"):
            row = conn.execute(
                """
                SELECT *
                FROM analysis_runs
                WHERE experiment_id=? AND step=? AND is_test=0
                ORDER BY id DESC
                LIMIT 1
                """,
                (experiment_id, step),
            ).fetchone()
            if row is not None:
                analysis_latest[step] = run_view(row, settings)
        confirmed_count = conn.execute("SELECT COUNT(*) AS n FROM confirmed_neurons WHERE experiment_id=?", (experiment_id,)).fetchone()["n"]
        notes = experiment_notes(conn, experiment_id)
    if selected_neuron_id and not any(str(n["id"]) == str(selected_neuron_id) for n in neurons):
        selected_neuron_id = ""
    active = any(r["status"] in {"pending", "queued", "running"} for r in runs)
    report = json.loads(exp["validation_report"] or "{}")
    validation_rows = build_validation_rows(report)
    return render_template(
        "experiment.html",
        exp=dict(exp),
        neurons=neurons,
        runs=runs,
        validation_rows=validation_rows,
        validation_all_ok=validation_all_ok(validation_rows) and exp["validation_state"] == "valid",
        active=active,
        step_labels=STEP_LABELS,
        analysis_latest=analysis_latest,
        sta_result_types=STA_RESULT_TYPES,
        sta_defaults=dict(settings.raw.get("defaults", {}).get("sta", {})),
        notice=request.args.get("notice"),
        confirmed_count=confirmed_count,
        selected_neuron_id=selected_neuron_id,
        notes=notes,
    )


@bp.post("/experiments/<int:experiment_id>/notes")
def add_experiment_note(experiment_id):
    settings = get_settings()
    note_text = (request.form.get("note_text") or "").strip()
    next_url = safe_next(request.form.get("next"), url_for("experiment", experiment_id=experiment_id))
    if not note_text:
        return redirect(next_url)
    if len(note_text) > 10000:
        return render_template("error.html", title="Note Too Long", message="Experiment notes must be 10,000 characters or fewer."), 400
    with tx(settings) as conn:
        exists = conn.execute("SELECT 1 FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exists is None:
            return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
        conn.execute(
            "INSERT INTO experiment_notes(experiment_id,note_text) VALUES(?,?)",
            (experiment_id, note_text),
        )
    return redirect(next_url)


@bp.post("/experiments/<int:experiment_id>/notes/<int:note_id>/delete")
def delete_experiment_note(experiment_id, note_id):
    settings = get_settings()
    next_url = safe_next(request.form.get("next"), url_for("experiment", experiment_id=experiment_id))
    with tx(settings) as conn:
        deleted = conn.execute(
            "DELETE FROM experiment_notes WHERE id=? AND experiment_id=?",
            (note_id, experiment_id),
        ).rowcount
        if not deleted:
            return render_template("error.html", title="Missing Note", message="Note not found"), 404
    return redirect(next_url)


@bp.post("/experiments/<int:experiment_id>/validate")
def validate_again(experiment_id):
    settings = get_settings()
    with tx(settings) as conn:
        exp = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone(); _, report = validate_experiment(exp["source_path"])
        conn.execute("UPDATE experiments SET validation_state=?,validation_report=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (report["state"], json.dumps(report), experiment_id))
    return redirect(url_for("experiment", experiment_id=experiment_id))


@bp.post("/experiments/<int:experiment_id>/run")
def run_step(experiment_id):
    settings = get_settings()
    try:
        step = request.form["step"]
        with connect(settings) as conn:
            exp = conn.execute("SELECT workflow_stage FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            raise ValueError("Experiment not found.")
        if step in {"spike", "behavior", "anatomy", "sta"} and exp["workflow_stage"] != "analysis_ready":
            raise ValueError("Analysis steps can run only after preprocessing and Suite2p have been confirmed.")
        params = json.loads(request.form.get("params") or "{}") if request.form.get("params") else settings.raw.get("defaults", {}).get(step, {})
        params = dict(params)
        if step == "sta":
            requested_types = request.form.getlist("result_types")
            valid_types = {name for name, _ in STA_RESULT_TYPES}
            result_types = [name for name in requested_types if name in valid_types]
            if not result_types:
                raise ValueError("Choose at least one STA result type.")
            params["result_types"] = result_types
            if request.form.get("spatial_bin"):
                spatial_bin = int(request.form["spatial_bin"])
                if spatial_bin < 1:
                    raise ValueError("STA spatial bin must be at least 1.")
                params["spatial_bin"] = spatial_bin
            if request.form.get("anatomy_spatial_bin"):
                anatomy_spatial_bin = int(request.form["anatomy_spatial_bin"])
                if anatomy_spatial_bin < 1:
                    raise ValueError("Anatomy spatial bin must be at least 1.")
                params["anatomy_spatial_bin"] = anatomy_spatial_bin
            # opt-in statistic option; the form checkbox is pre-ticked only when the config default has it on
            params["per_sign_null"] = request.form.get("per_sign_null") == "1"
        backend = "cluster" if step == "sta" else "local"
        rid = create_run(settings, experiment_id, step, params, backend=backend); launch_run(settings, rid)
        return redirect(url_for("experiment_analysis", experiment_id=experiment_id, new_run=rid) + "#runs")
    except Exception as exc: return render_template("error.html", title="Run Failed", message=str(exc)), 400


@bp.get("/experiments/<int:experiment_id>/behavior/data")
def behavior_data(experiment_id):
    settings = get_settings()
    with connect(settings) as conn:
        run = latest_completed_run(conn, experiment_id, "behavior")
    return jsonify(behavior_payload(settings, run))


@bp.get("/experiments/<int:experiment_id>/movie/meta")
def experiment_movie_meta(experiment_id):
    return movie_meta_response(experiment_id, pb_movie_path)


@bp.get("/experiments/<int:experiment_id>/movie/frame")
def experiment_movie_frame(experiment_id):
    return movie_frame_response(experiment_id, pb_movie_path)


@bp.get("/experiments/<int:experiment_id>/sta-significant-cells/data")
def sta_significant_cells_data(experiment_id):
    settings = get_settings()
    with connect(settings) as conn:
        exists = conn.execute("SELECT 1 FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exists is None:
            return jsonify({"error": "Experiment not found"}), 404
        summary = dict(sta_significant_summary(settings, conn, experiment_id))
        sta_run, spike_run = latest_completed_run(conn, experiment_id, "sta"), latest_completed_run(conn, experiment_id, "spike")
        stale = []
        if sta_run is not None and spike_run is not None:
            sta_d, spike_d = dict(sta_run), dict(spike_run)
            for r in conn.execute("SELECT pipeline_neuron_id FROM neurons WHERE experiment_id=?", (experiment_id,)):
                if spikes_edited_after_sta(settings, spike_d, sta_d, r["pipeline_neuron_id"]) is not None:
                    stale.append(int(r["pipeline_neuron_id"]))
        summary["stale_cells"] = sorted(stale)
    return jsonify(summary)


@bp.get("/experiments/<int:experiment_id>/cell-review/data")
def cell_review_data(experiment_id):
    settings = get_settings()
    with connect(settings) as conn:
        exp = conn.execute("SELECT source_path FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            return jsonify({"error": "Experiment not found"}), 404
        spike_run = latest_completed_run(conn, experiment_id, "spike")
        neurons = rows(conn, "SELECT id,pipeline_neuron_id FROM neurons WHERE experiment_id=? ORDER BY CAST(pipeline_neuron_id AS INTEGER)", (experiment_id,))
    # the answer only depends on the label files, the review marks and iscell.npy: reuse it while none of them changed (PERF-01)
    key = review_cache_key(settings, experiment_id, spike_run, neurons, exp["source_path"])
    if key is not None:
        cached = caches.CELL_REVIEW.get(key)
        if cached is not None:
            return jsonify(cached)
    neurons = current_suite2p_neurons(exp["source_path"], neurons)
    review = {}
    for n in neurons:
        review[str(n["id"])] = spike_review_info(settings, spike_run, str(n["pipeline_neuron_id"]))
    reviewed_count = sum(1 for info in review.values() if info["reviewed"])
    payload = {
        "has_spike_run": spike_run is not None,
        "reviewed_count": reviewed_count,
        "total_count": len(neurons),
        "review": review,
    }
    if key is not None:
        caches.CELL_REVIEW.put(key, payload)
    return jsonify(payload)
