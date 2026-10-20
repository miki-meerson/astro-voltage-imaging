"""Experiment registry (index, add, scan, remove, reset, star), notes, validation, the analysis launch and the data endpoints of the experiment pages."""
from __future__ import annotations

import threading

from flask import jsonify, redirect, render_template, request, url_for

from app.db import experiment_repository as repo
from app.db.run_repository import latest_run, latest_step_statuses
from app.services import caches
from app.services.behavior_view import behavior_payload
from app.services.border_view import border_snippets
from app.services.experiments import (
    ANALYSIS_STEPS,
    add_experiment,
    launch_default_steps,
    remove_experiment_from_registry,
    reset_experiment_app_data,
    scan_parent,
    start_analysis,
    validate_experiment,
)
from app.services.global_trends import global_trends
from app.services.hypothesis_summary import anatomy_cluster_summary, chosen_step
from app.services.run_views import STEP_LABELS
from app.services.runs import pipeline_status
from app.web.common import FlatBlueprint, get_settings, movie_chunk_response, movie_meta_response
from app.web.security import safe_next
from utils import globals_utils as g

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
    return render_template("index.html", experiments=experiments, settings=settings, last_scan=last, active=active, favorites_only=favorites_only, step_labels=STEP_LABELS,
                           rerun_steps=ANALYSIS_STEPS)


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
    return redirect(url_for("experiment_overview", experiment_id=experiment_id, **request.args))


@bp.get("/experiments/<int:experiment_id>/anatomy-clusters.json")
def experiment_anatomy_clusters(experiment_id):
    settings = get_settings()
    try:
        return jsonify(anatomy_cluster_summary(settings, experiment_id, chosen_step(request.args.get("h"))))
    except FileNotFoundError as exc:
        return jsonify({"available": False, "message": str(exc)}), 404
    except (ValueError, OSError, KeyError) as exc:
        return jsonify({"available": False, "message": str(exc)}), 400


@bp.get("/experiments/<int:experiment_id>/border-snippets.json")
def experiment_border_snippets(experiment_id):
    settings = get_settings()
    try:
        return jsonify(border_snippets(settings, experiment_id))
    except FileNotFoundError as exc:
        return jsonify({"message": str(exc)}), 404
    except (ValueError, OSError) as exc:
        return jsonify({"message": str(exc)}), 400


@bp.get("/experiments/<int:experiment_id>/global-trends.json")
def experiment_global_trends(experiment_id):
    settings = get_settings()
    try:
        return jsonify(global_trends(settings, experiment_id, chosen_step(request.args.get("h"))))
    except FileNotFoundError as exc:
        return jsonify({"message": str(exc)}), 404
    except (ValueError, OSError, KeyError) as exc:
        return jsonify({"message": str(exc)}), 400


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


def _form_int(name: str) -> dict:
    """{name: the integer of that form field} when the field is filled in (the field's own min attribute keeps it at 1 or more)."""
    return {name: int(request.form[name])} if request.form.get(name) else {}


@bp.post("/experiments/<int:experiment_id>/run-analysis")
def run_analysis(experiment_id):
    settings = get_settings()
    try:
        # the STA parameters the form can set besides the result types; per_sign_null is an opt-in statistic option, pre-ticked only when the config default has it on
        sta_options = {**_form_int("spatial_bin"), **_form_int("anatomy_spatial_bin"), "per_sign_null": request.form.get("per_sign_null") == "1"}
        hypothesis = request.form.get("hypothesis")  # the rerun button of one hypothesis names it; the main button runs them all
        start_analysis(settings, experiment_id, request.form.getlist("result_types"), sta_options, [hypothesis] if hypothesis else None)
        return redirect(url_for("experiment_overview", experiment_id=experiment_id))
    except (ValueError, OSError) as exc:
        return render_template("error.html", title="Run Analysis Failed", message=str(exc)), 400


@bp.get("/experiments/<int:experiment_id>/behavior/data")
def behavior_data(experiment_id):
    settings = get_settings()
    return jsonify(behavior_payload(settings, latest_run(settings, experiment_id, g.BEHAVIOR_STEP)))


@bp.get("/experiments/<int:experiment_id>/movie/meta")
def experiment_movie_meta(experiment_id):
    return movie_meta_response(experiment_id)


@bp.get("/experiments/<int:experiment_id>/movie/chunk")
def experiment_movie_chunk(experiment_id):
    return movie_chunk_response(experiment_id)
