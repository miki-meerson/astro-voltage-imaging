"""Experiment pages: registry (index, add, scan), analysis page, notes, review data, experiment movie, run launch."""
from __future__ import annotations

import json
import threading

from flask import jsonify, redirect, render_template, request, url_for

from app.db import experiment_repository as repo
from app.db.run_repository import latest_step_statuses
from app.services import caches
from app.services.anatomy_cluster_summary import anatomy_cluster_summary
from app.services.behavior_view import behavior_payload
from app.services.experiments import (
    ANALYSIS_STEPS,
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
    run_view,
    build_validation_rows,
    validation_all_ok,
)
from app.services.runs import create_run, launch_run, pipeline_status
from app.services.spikes import current_suite2p_neurons, review_cache_key, spike_review_info, spikes_edited_after_sta
from app.services.sta_results import STA_RESULT_TYPES, sta_significant_summary
from app.web.common import FlatBlueprint, get_settings, movie_frame_response, movie_meta_response
from app.web.security import safe_next

bp = FlatBlueprint("experiments", __name__)

@bp.get("/")
def index():
    settings = get_settings()
    favorites_only = request.args.get("favorites") == "1"
    experiments, last = repo.experiment_list(settings, favorites_only)
    statuses = latest_step_statuses(settings)
    for e in experiments:
        e.update(pipeline_status(statuses.get(int(e["id"]), {})))
    active = any(e.get("job_status") in {"running", "pending"} for e in experiments)
    return render_template("index.html", experiments=experiments, settings=settings, last_scan=last, active=active, favorites_only=favorites_only, step_labels=STEP_LABELS)


@bp.post("/experiments/add")
def add():
    settings = get_settings()
    try:
        exp_id = add_experiment(request.form["path"].strip().strip('"'), settings)
        if not repo.experiment_has_runs(settings, exp_id) and repo.experiment_exists(settings, exp_id):
            launch_default_steps(settings, exp_id)
            return redirect(url_for("experiment", experiment_id=exp_id))
        return redirect(url_for("experiment", experiment_id=exp_id, notice="already_registered"))
    except (ValueError, OSError) as exc:
        return render_template("error.html", title="Add Failed", message=str(exc)), 400


@bp.post("/scan")
def scan():
    settings = get_settings()
    try:
        parent = request.form.get("parent") or str(settings.scan_parent)
        threading.Thread(target=scan_parent, args=(parent, settings), daemon=True, name="astro-scan").start()
        return redirect(url_for("index"))
    except (ValueError, OSError) as exc:
        return render_template("error.html", title="Scan Failed", message=str(exc)), 400


@bp.post("/experiments/<int:experiment_id>/remove")
def remove_experiment(experiment_id):
    settings = get_settings()
    try:
        remove_experiment_from_registry(experiment_id, settings)
        return redirect(url_for("index"))
    except (ValueError, OSError) as exc:
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
    except (ValueError, OSError) as exc:
        return render_template("error.html", title="Reset Data Failed", message=str(exc)), 400


@bp.post("/experiments/<int:experiment_id>/favorite")
def favorite_experiment(experiment_id):
    settings = get_settings()
    try:
        if not repo.set_favorite(settings, experiment_id, request.form.get("favorite") == "1"):
            return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
        return redirect(safe_next(request.form.get("next"), url_for("index")))
    except (ValueError, OSError) as exc:
        return render_template("error.html", title="Favorite Failed", message=str(exc)), 400


@bp.get("/experiments/<int:experiment_id>")
def experiment(experiment_id):
    settings = get_settings()
    stage = repo.workflow_stage_of(settings, experiment_id)
    if stage is None:
        return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
    if stage == repo.STAGE_ANALYSIS_READY:
        return redirect(url_for("experiment_analysis", experiment_id=experiment_id, **request.args))
    return redirect(url_for("experiment_preprocessing", experiment_id=experiment_id, **request.args))


@bp.get("/experiments/<int:experiment_id>/anatomy-clusters.json")
def experiment_anatomy_clusters(experiment_id):
    settings = get_settings()
    try:
        return jsonify(anatomy_cluster_summary(settings, experiment_id))
    except FileNotFoundError as exc:
        return jsonify({"available": False, "message": str(exc)}), 404
    except (ValueError, OSError, KeyError) as exc:
        return jsonify({"available": False, "message": str(exc)}), 400


@bp.get("/experiments/<int:experiment_id>/analysis")
def experiment_analysis(experiment_id):
    settings = get_settings()
    selected_neuron_id = request.args.get("neuron", "")
    data = repo.experiment_page_data(settings, experiment_id, ANALYSIS_STEPS)
    if data is None:
        return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
    exp = data["exp"]
    neurons = current_suite2p_neurons(exp["source_path"], data["neurons"])
    runs = [run_view(r, settings) for r in data["runs"]]
    analysis_latest = {step: run_view(row, settings) for step, row in data["latest_real_runs"].items()}
    confirmed_count = data["confirmed_count"]
    notes = data["notes"]
    if selected_neuron_id and not any(str(n["id"]) == str(selected_neuron_id) for n in neurons):
        selected_neuron_id = ""
    active = any(r["status"] in {"pending", "queued", "running"} for r in runs)
    report = json.loads(exp["validation_report"] or "{}")
    validation_rows = build_validation_rows(report)
    return render_template(
        "experiment.html",
        exp=exp,
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
    if not repo.add_note(settings, experiment_id, note_text):
        return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
    return redirect(next_url)


@bp.post("/experiments/<int:experiment_id>/notes/<int:note_id>/delete")
def delete_experiment_note(experiment_id, note_id):
    settings = get_settings()
    next_url = safe_next(request.form.get("next"), url_for("experiment", experiment_id=experiment_id))
    if not repo.delete_note(settings, experiment_id, note_id):
        return render_template("error.html", title="Missing Note", message="Note not found"), 404
    return redirect(next_url)


@bp.post("/experiments/<int:experiment_id>/validate")
def validate_again(experiment_id):
    settings = get_settings()
    source_path = repo.experiment_source_path(settings, experiment_id)
    if source_path is None:
        return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
    _, report = validate_experiment(source_path)
    repo.save_validation(settings, experiment_id, report["state"], report)
    return redirect(url_for("experiment", experiment_id=experiment_id))


@bp.post("/experiments/<int:experiment_id>/run")
def run_step(experiment_id):
    settings = get_settings()
    try:
        step = request.form["step"]
        stage = repo.workflow_stage_of(settings, experiment_id)
        if stage is None:
            raise ValueError("Experiment not found.")
        if step in ANALYSIS_STEPS and stage != repo.STAGE_ANALYSIS_READY:
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
    except (ValueError, OSError) as exc:
        return render_template("error.html", title="Run Failed", message=str(exc)), 400


@bp.get("/experiments/<int:experiment_id>/behavior/data")
def behavior_data(experiment_id):
    settings = get_settings()
    return jsonify(behavior_payload(settings, repo.latest_completed_run(settings, experiment_id, "behavior")))


@bp.get("/experiments/<int:experiment_id>/movie/meta")
def experiment_movie_meta(experiment_id):
    return movie_meta_response(experiment_id, pb_movie_path)


@bp.get("/experiments/<int:experiment_id>/movie/frame")
def experiment_movie_frame(experiment_id):
    return movie_frame_response(experiment_id, pb_movie_path)


@bp.get("/experiments/<int:experiment_id>/sta-significant-cells/data")
def sta_significant_cells_data(experiment_id):
    settings = get_settings()
    if not repo.experiment_exists(settings, experiment_id):
        return jsonify({"error": "Experiment not found"}), 404
    summary = dict(sta_significant_summary(settings, experiment_id))
    sta_run, spike_run = repo.latest_completed_run(settings, experiment_id, "sta"), repo.latest_completed_run(settings, experiment_id, "spike")
    stale = []
    if sta_run is not None and spike_run is not None:
        for pipeline_neuron_id in repo.pipeline_neuron_ids(settings, experiment_id):
            if spikes_edited_after_sta(settings, spike_run, sta_run, pipeline_neuron_id) is not None:
                stale.append(int(pipeline_neuron_id))
    summary["stale_cells"] = sorted(stale)
    return jsonify(summary)


@bp.get("/experiments/<int:experiment_id>/cell-review/data")
def cell_review_data(experiment_id):
    settings = get_settings()
    source_path = repo.experiment_source_path(settings, experiment_id)
    if source_path is None:
        return jsonify({"error": "Experiment not found"}), 404
    spike_run = repo.latest_completed_run(settings, experiment_id, "spike")
    neurons = repo.experiment_neurons(settings, experiment_id)
    # the answer only depends on the label files, the review marks and iscell.npy: reuse it while none of them changed (PERF-01)
    key = review_cache_key(settings, experiment_id, spike_run, neurons, source_path)
    if key is not None:
        cached = caches.CELL_REVIEW.get(key)
        if cached is not None:
            return jsonify(cached)
    neurons = current_suite2p_neurons(source_path, neurons)
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
