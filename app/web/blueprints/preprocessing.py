"""Preprocessing page and its actions, data and movie endpoints."""
from __future__ import annotations

from io import BytesIO

from flask import jsonify, redirect, render_template, request, send_file, url_for

from app.db import preprocessing_repository as repo
from app.services import caches
from app.services.experiments import PAST_PREPROCESSING, confirm_preprocessing, confirm_suite2p, start_preprocessing
from app.services.movies import (
    mean_intensity_payload,
    PB_KINDS,
    pb_movie_path,
)
from app.services.preprocessing import PREPROCESSING_STEPS, latest_preprocessing_params, rerun_preprocessing_step
from app.services.roi_classes import export_manual_labels, roi_class_payload, roi_fov_png, run_auto_classification, set_manual_class
from app.services.run_views import run_view
from app.services.workspace import page_data
from app.web.common import FlatBlueprint, get_settings
from utils import globals_utils as g
from utils.file_utils import path_within
from utils.movie_utils import read_frame_rate
from utils.paths_utils import PathKind, path_for

bp = FlatBlueprint("preprocessing", __name__)


@bp.get("/experiments/<int:experiment_id>/preprocessing")
def experiment_preprocessing(experiment_id):
    settings = get_settings()
    data = page_data(settings, experiment_id, "preprocessing")
    if data is None:
        return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
    exp = data["exp"]
    runs = [run_view(r, settings) for r in data["runs"] if r["step"] == g.PREPROCESSING_STEP]
    pb_available = any(pb_movie_path(exp["source_path"], channel).exists() for channel in PB_KINDS)
    pb_qc_plots = [
        {"kind": kind, "label": kind.title(), "available": path_for(PathKind.PB_QC_PLOT, exp["source_path"], pb_kind=kind).exists()}
        for kind in PB_KINDS
    ]
    return render_template(
        "preprocessing.html", **{**data, "runs": runs},
        pre_steps=repo.preprocessing_page_data(settings, experiment_id)["steps"],
        pb_available=pb_available,
        pb_review_available=pb_available or any(plot["available"] for plot in pb_qc_plots),
        pb_qc_plots=pb_qc_plots,
        preprocessing_params=latest_preprocessing_params(settings, experiment_id),
        past_preprocessing=data["exp"]["workflow_stage"] in PAST_PREPROCESSING,
        default_pw_rigid=bool(settings.defaults.get(g.PREPROCESSING_STEP, {}).get("pw_rigid", False)),
        known_steps=PREPROCESSING_STEPS,
        notice=request.args.get("notice"),
    )


@bp.post("/experiments/<int:experiment_id>/preprocessing/run")
def run_preprocessing(experiment_id):
    settings = get_settings()
    try:
        params = {"pw_rigid": request.form["pw_rigid"] == "1"} if request.form.get("pw_rigid") in {"0", "1"} else {}
        rid = start_preprocessing(settings, experiment_id, params)
        caches.clear_experiment_result_caches()  # the movies and the results made from them are about to change
        caches.MEAN_INTENSITY.clear()
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
        if step_name == "motion_correction" and request.form.get("pw_rigid") in {"0", "1"}:
            extra_params["pw_rigid"] = request.form["pw_rigid"] == "1"
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
        return redirect(url_for("experiment_spikes", experiment_id=experiment_id, notice=f"suite2p_confirmed_{n}"))
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
    target = path_for(PathKind.PB_QC_PLOT, source_path, pb_kind=kind).resolve()
    if not path_within(target, source_path):
        return render_template("error.html", title="Unsafe Path", message="QC plot path is outside the experiment folder."), 400
    if not target.exists() or not target.is_file():
        return render_template("error.html", title="Missing QC Plot", message=str(target)), 404
    return send_file(target, mimetype="image/png", max_age=30)


ROI_CLASS_ACTIONS = {"data": roi_class_payload, "auto": run_auto_classification, "export": export_manual_labels}  # the requests with nothing but the experiment


@bp.get("/experiments/<int:experiment_id>/preprocessing/roi-classes.json", defaults={"action": "data", "roi": None})
@bp.get("/experiments/<int:experiment_id>/preprocessing/roi-fov.png", defaults={"action": "fov", "roi": None})
@bp.post("/experiments/<int:experiment_id>/preprocessing/roi-classes/<any(auto, export):action>", defaults={"roi": None})
@bp.post("/experiments/<int:experiment_id>/preprocessing/roi-classes/<int:roi>", defaults={"action": "set"})
def roi_classes_request(experiment_id, action, roi):
    """The ROI types tab's requests: json of what the action returns (the field of view as a PNG), or {"message"} with the status of what went wrong."""
    settings = get_settings()
    exp = repo.experiment_row(settings, experiment_id)
    if exp is None:
        return jsonify({"message": "Experiment not found."}), 404
    try:
        if action == "fov":
            return send_file(BytesIO(roi_fov_png(settings, exp, request.args.get("channel", "neuron"))), mimetype="image/png", max_age=300)
        if action == "set":
            return jsonify(set_manual_class(settings, exp, roi, (request.get_json(silent=True) or {}).get("label")))
        return jsonify(ROI_CLASS_ACTIONS[action](settings, exp))
    except FileNotFoundError as exc:
        return jsonify({"message": str(exc)}), 404
    except (ValueError, OSError) as exc:
        return jsonify({"message": str(exc)}), 400
