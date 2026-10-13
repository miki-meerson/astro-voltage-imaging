"""Preprocessing page and its actions, data and movie endpoints."""
from __future__ import annotations

from pathlib import Path

from flask import jsonify, redirect, render_template, request, send_file, url_for

from app.db import preprocessing_repository as repo
from app.services import caches
from app.services.experiments import confirm_preprocessing, confirm_suite2p, launch_analysis_defaults
from app.services.movies import (
    mean_intensity_payload,
    PB_KINDS,
    pb_movie_path,
)
from app.services.preprocessing import PREPROCESSING_STEPS, latest_preprocessing_params, rerun_preprocessing_step
from app.services.run_views import STEP_LABELS, run_view
from app.services.runs import create_run, launch_run
from app.web.common import FlatBlueprint, get_settings
from utils.movie_utils import read_frame_rate
from utils.paths_utils import PathKind, path_for

bp = FlatBlueprint("preprocessing", __name__)


@bp.get("/experiments/<int:experiment_id>/preprocessing")
def experiment_preprocessing(experiment_id):
    settings = get_settings()
    data = repo.preprocessing_page_data(settings, experiment_id)
    if data is None:
        return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
    exp = data["experiment"]
    runs = [run_view(r, settings) for r in data["runs"]]
    pre_steps = data["steps"]
    preprocessing_params = latest_preprocessing_params(settings, experiment_id)
    notes = data["notes"]
    active = any(r["status"] in {"pending", "queued", "running"} for r in runs)
    pb_available = any(
        path is not None and path.exists()
        for path in (pb_movie_path(exp["source_path"], "astro"), pb_movie_path(exp["source_path"], "neuron"))
    )
    pb_qc_plots = [
        {"kind": kind, "label": kind.title(), "available": path_for(PathKind.PB_QC_PLOT, exp["source_path"], pb_kind=kind).exists()}
        for kind in PB_KINDS
    ]
    pb_review_available = pb_available or any(plot["available"] for plot in pb_qc_plots)
    return render_template(
        "preprocessing.html",
        exp=exp,
        runs=runs,
        pre_steps=pre_steps,
        active=active,
        pb_available=pb_available,
        pb_review_available=pb_review_available,
        pb_qc_plots=pb_qc_plots,
        preprocessing_params=preprocessing_params,
        notes=notes,
        step_labels=STEP_LABELS,
        known_steps=PREPROCESSING_STEPS,
        notice=request.args.get("notice"),
    )


@bp.post("/experiments/<int:experiment_id>/preprocessing/run")
def run_preprocessing(experiment_id):
    settings = get_settings()
    try:
        rid = create_run(settings, experiment_id, "preprocessing", settings.raw.get("defaults", {}).get("preprocessing", {}), backend="cluster")
        launch_run(settings, rid)
        return redirect(url_for("experiment_preprocessing", experiment_id=experiment_id, new_run=rid))
    except Exception as exc:
        return render_template("error.html", title="Preprocessing Failed", message=str(exc)), 400


@bp.post("/experiments/<int:experiment_id>/preprocessing/steps/<step_name>/rerun")
def rerun_preprocessing_step_route(experiment_id, step_name):
    settings = get_settings()
    try:
        extra_params = {}
        if step_name == "photobleaching_correction" and request.form.get("trimmed") not in {None, ""}:
            extra_params["trimmed"] = int(request.form["trimmed"])
        rerun_preprocessing_step(settings, experiment_id, step_name, extra_params=extra_params)
        return redirect(url_for("experiment_preprocessing", experiment_id=experiment_id, notice=f"rerun_{step_name}"))
    except Exception as exc:
        return render_template("error.html", title="Step Rerun Failed", message=str(exc)), 400


@bp.post("/experiments/<int:experiment_id>/preprocessing/confirm")
def confirm_preprocessing_route(experiment_id):
    settings = get_settings()
    try:
        confirm_preprocessing(settings, experiment_id)
        return redirect(url_for("experiment_preprocessing", experiment_id=experiment_id, notice="preprocessing_confirmed"))
    except Exception as exc:
        return render_template("error.html", title="Confirm Failed", message=str(exc)), 400


@bp.post("/experiments/<int:experiment_id>/suite2p/confirm")
def confirm_suite2p_route(experiment_id):
    settings = get_settings()
    try:
        n = confirm_suite2p(settings, experiment_id)
        caches.clear_experiment_result_caches()
        launch_analysis_defaults(settings, experiment_id)
        return redirect(url_for("experiment_analysis", experiment_id=experiment_id, notice=f"suite2p_confirmed_{n}"))
    except Exception as exc:
        return render_template("error.html", title="Suite2p Confirm Failed", message=str(exc)), 400


@bp.get("/experiments/<int:experiment_id>/preprocessing/mean-intensity/data")
def preprocessing_mean_intensity_data(experiment_id):
    settings = get_settings()
    source_path = repo.experiment_source_path(settings, experiment_id)
    if source_path is None:
        return jsonify({"available": False, "message": "Experiment not found."}), 404
    try:
        max_points = min(1600, max(200, int(request.args.get("max_points", "350"))))
        return jsonify(mean_intensity_payload(source_path, read_frame_rate(source_path), max_points=max_points))
    except Exception as exc:
        return jsonify({"available": False, "message": str(exc)}), 500


@bp.get("/experiments/<int:experiment_id>/preprocessing/photobleaching-qc/<kind>.png")
def preprocessing_photobleaching_qc_plot(experiment_id, kind):
    settings = get_settings()
    source_path = repo.experiment_source_path(settings, experiment_id)
    if source_path is None or kind not in PB_KINDS:
        return render_template("error.html", title="Missing Experiment", message="Experiment not found."), 404
    source_root = Path(source_path).resolve()
    target = path_for(PathKind.PB_QC_PLOT, source_path, pb_kind=kind).resolve()
    if source_root not in target.parents and target != source_root:
        return render_template("error.html", title="Unsafe Path", message="QC plot path is outside the experiment folder."), 400
    if not target.exists() or not target.is_file():
        return render_template("error.html", title="Missing QC Plot", message=str(target)), 404
    return send_file(target, mimetype="image/png", max_age=30)
